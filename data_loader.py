from huggingface_hub import snapshot_download
from pathlib import Path
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.preprocessing.object_detection import ObjectDetector
from src.preprocessing.depth_estimation import DepthEstimator
from src.preprocessing.fusion_utils import create_3d_object_representations

local_folder = snapshot_download(
    repo_id="EmbodiedBench/EB-Man_trajectory_dataset",
    repo_type="dataset",
    local_dir="./data/EB-Man_trajectory_dataset",     # or any folder you choose
    local_dir_use_symlinks=False
)

def resolve_data_root():
    """
    Resolve the data root directory.

    Priority:
      1. EMBODIEDBENCH_DATA env var
      2. ./data relative to cwd
      3. ../data relative to cwd
      4. ~/data
    """
    env_path = os.environ.get("EMBODIEDBENCH_DATA", None)
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p.resolve()
    candidates = [
        Path.cwd() / "data",
        Path.cwd().parent / "data",
        Path.home() / "data",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        "Could not find data folder. Set EMBODIEDBENCH_DATA in your .env or environment\n"
        "Examples:\n"
        "  export EMBODIEDBENCH_DATA=/absolute/path/to/your/data\n"
        "  or create ./data or ../data relative to the repo root\n"
    )

class EmbodiedDataset(Dataset):
    """
    Minimal dataset scaffold. Replace loading logic with your actual format.
    Uses self.data_root to find files.
    """
    def __init__(self, data_root=None, debug=False):
        self.data_root = Path(data_root) if data_root else resolve_data_root()
        self.debug = debug
        # Example: expect a folder 'examples' with JSON metadata; adapt as needed
        examples_dir = self.data_root / "examples"
        if examples_dir.exists():
            self.examples = sorted(list(examples_dir.glob("*.json")))
        else:
            # fallback: any json files in data root
            self.examples = sorted(list(self.data_root.glob("*.json")))
        if debug:
            self.examples = self.examples[:20]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Placeholder item. Replace with actual image/depth/bbox/action loading & preprocessing.
        meta_path = str(self.examples[idx]) if self.examples else ""
        return {
            "instruction": "stack the star",
            "image": torch.zeros(3, 224, 224, dtype=torch.float32),
            "objects": torch.zeros(10, 128, dtype=torch.float32),   # object embeddings placeholder
            "action_labels": torch.zeros(7, dtype=torch.long),
            "meta_path": meta_path
        }

def collate_fn_3d(batch, device="cpu"):
    """
    Collate function that performs 3D preprocessing on images.
    
    Args:
        batch: List of dicts from EmbodiedDataset with keys:
            - instruction: str
            - demo_images: (num_demos, 3, H, W) tensor
            - current_image: (3, H, W) tensor
            - demo_actions: list of (max_steps, 7) tensors
        device: Device to run preprocessing on
        
    Returns:
        dict with keys:
            - instructions: List[str]
            - demo_3d_objects: List[List[torch.Tensor]] - outer list is batch, inner is demos
            - current_3d_objects: List[torch.Tensor] - one per batch element
            - demo_actions: List[torch.Tensor] - one per batch element (last action from each demo)
            - targets: (B, 7) tensor
    """
    # Initialize preprocessing models (lazy initialization - could be cached)
    detector = ObjectDetector(device=device)
    estimator = DepthEstimator(device=device)
    
    instructions = []
    demo_3d_objects_list = []
    current_3d_objects_list = []
    demo_actions_list = []
    targets = []
    
    for b in batch:
        instructions.append(b["instruction"])
        
        # Process demonstration images
        demo_3d_objs_per_example = []
        if "demo_images" in b and b["demo_images"] is not None:
            demo_images = b["demo_images"]  # (num_demos, 3, H, W)
            num_demos = demo_images.shape[0]
            
            for demo_idx in range(num_demos):
                # Average frames if needed, or take first frame
                demo_img = demo_images[demo_idx]  # (3, H, W)
                
                # Convert to numpy RGB format (H, W, 3) in [0, 255]
                if demo_img.max() <= 1.0:
                    demo_img = (demo_img * 255).clamp(0, 255)
                demo_img_np = demo_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                h, w = demo_img_np.shape[:2]
                
                # Detect objects
                objects = detector.detect_objects(demo_img_np, conf_threshold=0.5)
                
                # Estimate depth
                depth_map = estimator.estimate_depth(demo_img_np)
                
                # Create 3D representations
                obj_3d = create_3d_object_representations(objects, depth_map, h, w)
                demo_3d_objs_per_example.append(obj_3d)
        else:
            # No demos - add empty list
            demo_3d_objs_per_example = []
        
        demo_3d_objects_list.append(demo_3d_objs_per_example)
        
        # Process current image
        if "current_image" in b and b["current_image"] is not None:
            current_img = b["current_image"]  # (3, H, W)
            
            # Convert to numpy RGB format
            if current_img.max() <= 1.0:
                current_img = (current_img * 255).clamp(0, 255)
            current_img_np = current_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            h, w = current_img_np.shape[:2]
            
            # Detect objects
            objects = detector.detect_objects(current_img_np, conf_threshold=0.5)
            
            # Estimate depth
            depth_map = estimator.estimate_depth(current_img_np)
            
            # Create 3D representations
            obj_3d = create_3d_object_representations(objects, depth_map, h, w)
            current_3d_objects_list.append(obj_3d)
        else:
            # No current image - add empty tensor
            current_3d_objects_list.append(torch.zeros((0, 7), dtype=torch.float32))
        
        # Extract demo actions (last valid action from each demo)
        demo_actions_per_example = []
        if "demo_actions" in b and b["demo_actions"] is not None:
            for demo_action_seq in b["demo_actions"]:  # (max_steps, 7)
                # Find last valid action
                valid = (demo_action_seq != -1).all(dim=1)
                idxs = valid.nonzero(as_tuple=False)
                if len(idxs) > 0:
                    last_action = demo_action_seq[idxs[-1].item()]  # (7,)
                    demo_actions_per_example.append(last_action)
                else:
                    demo_actions_per_example.append(torch.full((7,), -1, dtype=torch.long))
        else:
            demo_actions_per_example = []
        
        demo_actions_list.append(demo_actions_per_example)
        
        # Extract target (last valid action from first demo)
        if "demo_actions" in b and b["demo_actions"] is not None and len(b["demo_actions"]) > 0:
            seq = b["demo_actions"][0]  # (max_steps, 7)
            valid = (seq != -1).all(dim=1)
            idxs = valid.nonzero(as_tuple=False)
            if len(idxs) == 0:
                targets.append([-1] * 7)
            else:
                last = idxs[-1].item()
                targets.append(seq[last].tolist())
        else:
            targets.append([-1] * 7)
    
    # Convert demo_actions to list of tensors (one per demo, not per batch element)
    # This is a bit complex - we need to reorganize
    num_demos = max(len(demo_actions_list[b]) for b in range(len(demo_actions_list))) if demo_actions_list else 0
    demo_actions_by_demo = []
    for demo_idx in range(num_demos):
        actions_for_this_demo = []
        for b in range(len(demo_actions_list)):
            if demo_idx < len(demo_actions_list[b]):
                actions_for_this_demo.append(demo_actions_list[b][demo_idx])
            else:
                actions_for_this_demo.append(torch.full((7,), -1, dtype=torch.long))
        demo_actions_by_demo.append(torch.stack(actions_for_this_demo, dim=0))  # (B, 7)
    
    return {
        'instructions': instructions,
        'demo_3d_objects': demo_3d_objects_list,
        'current_3d_objects': current_3d_objects_list,
        'demo_actions': demo_actions_by_demo if demo_actions_by_demo else None,
        'targets': torch.tensor(targets, dtype=torch.long),
    }


def build_dataloader(batch_size=4, debug=False, data_root=None, num_workers=0, use_3d_preprocessing=True, device="cpu"):
    """
    Build dataloader with optional 3D preprocessing.
    
    Args:
        batch_size: Batch size
        debug: Use debug mode (smaller dataset)
        data_root: Root directory for data
        num_workers: Number of worker processes
        use_3d_preprocessing: If True, use collate_fn_3d for 3D preprocessing
        device: Device for preprocessing (if use_3d_preprocessing=True)
    """
    ds = EmbodiedDataset(data_root=data_root, debug=debug)
    collate_fn = collate_fn_3d if use_3d_preprocessing else None
    if collate_fn is not None:
        # Create a lambda that passes device
        collate_fn_with_device = lambda batch: collate_fn_3d(batch, device=device)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn_with_device)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)