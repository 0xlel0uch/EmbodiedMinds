from huggingface_hub import snapshot_download
from pathlib import Path
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

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

def build_dataloader(batch_size=4, debug=False, data_root=None, num_workers=0):
    ds = EmbodiedDataset(data_root=data_root, debug=debug)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)