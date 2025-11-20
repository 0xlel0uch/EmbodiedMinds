# Architecture Implementation Guide: Visual In-Context Learning with 3D Perception

## Overview
Your existing codebase has a strong foundation with most core components in place. This guide outlines the best practices for implementing the proposed architecture and identifying gaps that need to be addressed.

---

## Current State Analysis

### ✅ What's Already in Place

1. **Text Encoder** (`src/encoders/text_encoder.py`)
   - Uses frozen BERT for text instruction encoding
   - Properly frozen (requires_grad = False)
   - Returns pooled CLS embedding of shape (B, 768)

2. **Vision Encoder** (`src/encoders/vision_encoder.py`)
   - Uses frozen ResNet18 backbone
   - Properly frozen parameters
   - Returns (B, 512) embeddings

3. **Policy Transformer** (`src/policy/policy_transformer.py`)
   - TransformerEncoder implementation
   - Takes sequence of tokens (B, T, token_dim)
   - Pools and projects to decision vector

4. **Output Heads** (`src/heads/output_heads.py`)
   - 7 classification heads for 7D action space
   - Proper loss computation with ignore_index=-1
   - Binning structure: [101, 101, 101, 121, 121, 121, 2]

5. **Training Pipeline** (`src/training/train.py`)
   - Solid epoch-based training loop
   - Loss computation and optimization
   - Device management

### ⚠️ Major Gaps to Address

1. **3D Perception Preprocessing Module** - INCOMPLETE
   - `ObjectDetector` exists but uses basic CV2 DNN (should use PyTorch/modern models)
   - `DepthEstimator` is stubbed out (needs actual implementation)
   - `FusionModule` is only skeleton code (needs 3D representation fusion)
   - **CRITICAL**: No integration of these preprocessing steps into the main pipeline

2. **Multimodal Embedding Sequence** - OVERSIMPLIFIED
   - Current approach: Simple [instr, demo, cur] token stacking
   - **Missing**: 3D object embeddings from preprocessing
   - **Missing**: Multiple demonstration support
   - **Missing**: Example actions in the sequence

3. **Data Loading** - INCOMPLETE
   - Current collate_fn only provides basic image stacking
   - **Missing**: Integration with preprocessing (object detection, depth)
   - **Missing**: Multiple demonstrations per example
   - **Missing**: Proper 3D representation creation

4. **Vision Encoder** - Limited
   - Uses ResNet18 for spatial feature extraction
   - **Issue**: ResNet produces spatial feature maps, but code flattens them
   - **Better approach**: Use ViT from CLIP for dense object-level embeddings

---

## Recommended Implementation Strategy

### Phase 1: Establish the 3D Perception Pipeline

#### 1.1 Create a Robust 3D Object Representation System

**File: `src/preprocessing/object_detection.py`** (Upgrade)

```python
# Recommendation: Use ultralytics YOLOv8 instead of OpenCV DNN
# pip install ultralytics

from ultralytics import YOLO
import torch
import numpy as np

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt", device="cpu"):
        self.model = YOLO(model_name)
        self.device = device
        
    def detect_objects(self, image: np.ndarray):
        """
        Returns: List[Dict] with keys:
            - box: [x1, y1, x2, y2] normalized coords
            - confidence: float
            - class_id: int
            - center: [cx, cy] normalized
        """
        results = self.model(image, verbose=False)
        detections = []
        
        h, w = image.shape[:2]
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() / np.array([w, h, w, h])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'center': [cx, cy],
                    'confidence': float(box.conf[0].cpu()),
                    'class_id': int(box.cls[0].cpu()),
                })
        return detections
```

**File: `src/preprocessing/depth_estimation.py`** (Upgrade)

```python
# Recommendation: Use MiDaS (pre-trained depth model)
# pip install timm

import torch
import numpy as np
import cv2

class DepthEstimator:
    def __init__(self, model_type="DPT_Large", device="cpu"):
        """
        Uses MiDaS for monocular depth estimation
        model_type: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
        """
        self.device = device
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(device)
        self.midas.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform
        
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        image: HxWx3 numpy array
        Returns: HxW depth map (normalized 0-1)
        """
        input_batch = self.transform(image).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth = prediction.cpu().numpy()
        # Normalize to 0-1
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            depth = (depth - depth_min) / (depth_max - depth_min)
        
        return depth
```

#### 1.2 Create 3D Object Representations

**File: `src/preprocessing/fusion_utils.py`** (NEW)

```python
import torch
import numpy as np
from typing import List, Dict

def create_3d_object_representations(
    objects: List[Dict],  # from ObjectDetector
    depth_map: np.ndarray,  # from DepthEstimator
    image_h: int,
    image_w: int,
    focal_length: float = 500.0,  # camera intrinsic
) -> torch.Tensor:
    """
    Fuses 2D bounding boxes with depth to create 3D object representations.
    
    Returns: (num_objects, 8) tensor
        Format: [center_x, center_y, center_z, width, height, depth, confidence, class_id]
    """
    representations = []
    
    for obj in objects:
        x1, y1, x2, y2 = obj['box']
        cx_norm, cy_norm = obj['center']
        
        # Convert to pixel coordinates
        cx_px = int(cx_norm * image_w)
        cy_px = int(cy_norm * image_h)
        
        # Sample depth at bounding box region
        x1_px, y1_px = int(x1 * image_w), int(y1 * image_h)
        x2_px, y2_px = int(x2 * image_w), int(y2 * image_h)
        
        # Get average depth in bounding box
        bbox_region = depth_map[y1_px:y2_px, x1_px:x2_px]
        if bbox_region.size > 0:
            z = float(np.mean(bbox_region))
        else:
            z = 0.5
        
        w = x2_norm - x1_norm
        h = y2_norm - y1_norm
        
        representations.append([
            cx_norm,  # center_x (normalized)
            cy_norm,  # center_y (normalized)
            z,        # depth (normalized)
            w,        # width (normalized)
            h,        # height (normalized)
            obj['confidence'],
            float(obj['class_id']),
        ])
    
    if representations:
        return torch.tensor(representations, dtype=torch.float32)
    else:
        return torch.zeros((0, 7), dtype=torch.float32)
```

#### 1.3 Upgrade Vision Encoder for Object-Level Embeddings

**File: `src/encoders/vision_encoder.py`** (Recommended Enhancement)

Instead of using ResNet18, consider using CLIP's ViT which naturally produces token embeddings:

```python
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

class CLIPVisionEncoder(nn.Module):
    """Uses CLIP's Vision Transformer for dense object embeddings"""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Freeze the model
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.device = device
        self.model.to(device)
        self.out_dim = self.model.config.vision_config.hidden_size  # 512
        
    def encode(self, images):
        """
        images: tensor (B, 3, H, W) or list of PIL Images
        Returns: (B, seq_len, out_dim) - sequence of patch embeddings
        """
        with torch.no_grad():
            outputs = self.model.vision_model(images)
            # last_hidden_state: (B, num_patches+1, hidden_dim)
            return outputs.last_hidden_state  # (B, 50, 512) for ViT-B/32
```

---

### Phase 2: Restructure the Data Pipeline

#### 2.1 Update Data Collation

**File: `src/datasets/dataloader.py`** (Refactor collate_fn)

```python
def collate_fn_3d(batch):
    """
    Enhanced collation that includes 3D preprocessing.
    
    batch: list of dicts with:
        - instruction: str
        - demo_images: list of (num_frames, 3, H, W) - multiple demos
        - current_image: (3, H, W)
        - demo_actions: list of (max_steps, 7) - actions for each demo
    """
    from src.preprocessing.object_detection import ObjectDetector
    from src.preprocessing.depth_estimation import DepthEstimator
    from src.preprocessing.fusion_utils import create_3d_object_representations
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize preprocessing models (cache these outside batch loop in practice)
    obj_detector = ObjectDetector(device=device)
    depth_estimator = DepthEstimator(device=device)
    
    instructions = [b["instruction"] for b in batch]
    
    # Process multiple demonstrations
    demo_3d_objects_list = []  # list[list[tensors]] - one list per example
    demo_images_avg = []
    
    for b in batch:
        # b["demo_images"] is list of arrays (num_demos, num_frames, 3, H, W)
        example_3d_objects = []
        
        for demo_idx, demo_frames in enumerate(b["demo_images"]):
            # Average frames within demonstration
            avg_frame = demo_frames.mean(axis=0)  # (3, H, W)
            
            # Preprocess
            objects = obj_detector.detect_objects(avg_frame)
            depth = depth_estimator.estimate_depth(avg_frame)
            obj_3d = create_3d_object_representations(
                objects, depth, *avg_frame.shape[1:]
            )
            example_3d_objects.append(obj_3d)  # (num_objects, 7)
        
        demo_3d_objects_list.append(example_3d_objects)
        demo_images_avg.append(torch.from_numpy(demo_frames.mean(axis=0)))
    
    # Process current images
    current_3d_objects_list = []
    current_images = []
    
    for b in batch:
        cur_img = b["current_image"]  # (3, H, W)
        current_images.append(torch.from_numpy(cur_img))
        
        objects = obj_detector.detect_objects(cur_img)
        depth = depth_estimator.estimate_depth(cur_img)
        obj_3d = create_3d_object_representations(
            objects, depth, *cur_img.shape[1:]
        )
        current_3d_objects_list.append(obj_3d)
    
    # Extract action targets (from first demo's last valid action)
    targets = []
    for b in batch:
        if len(b["demo_actions"]) > 0:
            seq = b["demo_actions"][0]  # First demo
            valid = (seq != -1).all(axis=1)
            inds = np.where(valid)[0]
            if len(inds) > 0:
                targets.append(seq[inds[-1]])
            else:
                targets.append([-1] * 7)
        else:
            targets.append([-1] * 7)
    
    return {
        'instructions': instructions,
        'demo_3d_objects': demo_3d_objects_list,
        'demo_images': torch.stack(demo_images_avg),
        'current_3d_objects': current_3d_objects_list,
        'current_images': torch.stack(current_images),
        'targets': torch.tensor(targets, dtype=torch.long),
    }
```

---

### Phase 3: Build the Multimodal Sequence

#### 3.1 Create Object Embedding Module

**File: `src/encoders/object_encoder.py`** (NEW)

```python
import torch
import torch.nn as nn

class ObjectEncoder(nn.Module):
    """
    Encodes 3D object representations into embeddings.
    Input: (num_objects, 7) - 3D object features
    Output: (num_objects, embedding_dim)
    """
    
    def __init__(self, object_feature_dim=7, embedding_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(object_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
    
    def forward(self, objects):
        """
        objects: (num_objects, 7) or list of variable-sized tensors
        Returns: (num_objects, embedding_dim)
        """
        if isinstance(objects, list):
            # Handle variable-sized object lists
            embeddings = []
            for obj_set in objects:
                if obj_set.size(0) > 0:
                    embeddings.append(self.encoder(obj_set))
                else:
                    embeddings.append(torch.zeros(0, self.embedding_dim))
            return embeddings
        else:
            return self.encoder(objects)
```

#### 3.2 Build Sequence Constructor

**File: `src/fusion/sequence_builder.py`** (NEW)

```python
import torch
import torch.nn as nn
from typing import List, Dict

class MultimodalSequenceBuilder(nn.Module):
    """
    Constructs the multimodal token sequence for the policy transformer.
    
    Sequence structure (per example in batch):
    [instruction_embedding, 
     demo1_instructions, demo1_actions, demo1_objects,
     demo2_instructions, demo2_actions, demo2_objects,
     ...,
     current_objects]
    """
    
    def __init__(self, token_dim: int = 256):
        super().__init__()
        self.token_dim = token_dim
        
        # Projection layers
        self.instr_proj = nn.Linear(768, token_dim)  # BERT output
        self.action_proj = nn.Linear(7, token_dim)   # 7D action
        self.obj_proj = nn.Linear(256, token_dim)    # object embeddings
    
    def forward(
        self,
        instr_embedding: torch.Tensor,  # (B, 768) from BERT
        demo_object_embeddings: List[torch.Tensor],  # list of (B, num_obj, 256)
        demo_actions: List[torch.Tensor],  # list of (B, num_demos, 7)
        current_object_embeddings: torch.Tensor,  # (B, num_obj, 256)
    ) -> torch.Tensor:
        """
        Returns: (B, max_seq_len, token_dim) tensor
        """
        B = instr_embedding.size(0)
        device = instr_embedding.device
        
        sequences = []
        max_seq_len = 0
        
        for b in range(B):
            seq = []
            
            # 1. Instruction token
            seq.append(self.instr_proj(instr_embedding[b:b+1]))  # (1, token_dim)
            
            # 2. Demo sequences
            for demo_idx in range(len(demo_object_embeddings)):
                demo_objs = demo_object_embeddings[demo_idx][b]  # (num_obj, 256)
                if demo_objs.size(0) > 0:
                    seq.append(self.obj_proj(demo_objs))  # (num_obj, token_dim)
                
                # Add demo action if available
                demo_action = demo_actions[demo_idx][b:b+1]  # (1, 7)
                seq.append(self.action_proj(demo_action.float()))  # (1, token_dim)
            
            # 3. Current objects
            cur_objs = current_object_embeddings[b]  # (num_obj, 256)
            if cur_objs.size(0) > 0:
                seq.append(self.obj_proj(cur_objs))  # (num_obj, token_dim)
            
            # Concatenate all tokens
            seq = torch.cat(seq, dim=0)  # (total_tokens, token_dim)
            sequences.append(seq)
            max_seq_len = max(max_seq_len, seq.size(0))
        
        # Pad sequences to same length
        padded_sequences = []
        for seq in sequences:
            padding = torch.zeros(
                max_seq_len - seq.size(0), 
                self.token_dim, 
                device=device
            )
            padded = torch.cat([seq, padding], dim=0)
            padded_sequences.append(padded)
        
        return torch.stack(padded_sequences, dim=0)  # (B, max_seq_len, token_dim)
```

---

### Phase 4: Integrate into Training

#### 4.1 Update AgentModel

**File: `src/datasets/dataloader.py`** (Update AgentModel class)

```python
class AgentModel(nn.Module):
    def __init__(
        self,
        token_dim: int = 256,
        out_dim: int = 512,
        bins: list = None,
        text_model_name: str = "bert-base-uncased",
        vision_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
    ):
        super().__init__()
        if bins is None:
            bins = [101, 101, 101, 121, 121, 121, 2]
        
        self.device = device
        
        # Frozen encoders
        self.text_enc = TextEncoder(model_name=text_model_name, device=device)
        self.vision_enc = CLIPVisionEncoder(model_name=vision_model_name, device=device)
        
        # Object encoder (trainable)
        self.object_enc = ObjectEncoder(embedding_dim=256)
        
        # Sequence builder (trainable)
        self.seq_builder = MultimodalSequenceBuilder(token_dim=token_dim)
        
        # Policy & heads
        self.policy = PolicyTransformer(token_dim=token_dim, out_dim=out_dim)
        self.heads = OutputHeads(in_dim=out_dim, bins=bins)
        
        # Trainable params to device
        self.to(device)
    
    def forward(
        self,
        instr_texts: List[str],
        demo_images: torch.Tensor,
        demo_3d_objects: List[List[torch.Tensor]],
        current_images: torch.Tensor,
        current_3d_objects: List[torch.Tensor],
        demo_actions: List[torch.Tensor] = None,
    ):
        """
        Returns: list of logits from output heads
        """
        # Encode instruction
        instr_embed = self.text_enc.encode(instr_texts)  # (B, 768)
        
        # Encode demo images for visual context (if using CLIP patches)
        # Note: Could also use for keyframe extraction
        
        # Encode 3D objects
        demo_obj_embeds = []
        for demo_idx in range(len(demo_3d_objects[0])):  # num demos
            batch_objs = [demo_3d_objects[b][demo_idx] for b in range(len(demo_3d_objects))]
            # Need to handle variable-sized object sets
            # Option: Pad or use set-based encoding
            demo_obj_embeds.append(self._encode_object_batch(batch_objs))
        
        current_obj_embeds = self._encode_object_batch(current_3d_objects)
        
        # Build sequence
        tokens = self.seq_builder(
            instr_embed,
            demo_obj_embeds,
            demo_actions if demo_actions else [],
            current_obj_embeds,
        )  # (B, seq_len, token_dim)
        
        # Policy reasoning
        decision = self.policy(tokens)  # (B, out_dim)
        
        # Output heads
        logits = self.heads(decision)  # list of (B, bins_i)
        
        return logits
    
    def _encode_object_batch(self, object_list: List[torch.Tensor]) -> torch.Tensor:
        """Handle variable-sized object sets per batch element"""
        B = len(object_list)
        max_objs = max(o.size(0) if o.size(0) > 0 else 0 for o in object_list)
        
        if max_objs == 0:
            return torch.zeros(B, 1, 256, device=self.device)
        
        padded_objs = []
        for objs in object_list:
            if objs.size(0) > 0:
                encoded = self.object_enc(objs)  # (num_obj, 256)
            else:
                encoded = torch.zeros(1, 256, device=self.device)
            
            # Pad or pool to fixed size
            if encoded.size(0) < max_objs:
                pad = torch.zeros(max_objs - encoded.size(0), 256, device=self.device)
                encoded = torch.cat([encoded, pad], dim=0)
            
            padded_objs.append(encoded)
        
        return torch.stack(padded_objs, dim=0)  # (B, max_objs, 256)
```

---

## Implementation Checklist

### Priority 1: Critical Components
- [ ] Implement modern ObjectDetector (YOLOv8)
- [ ] Implement DepthEstimator (MiDaS)
- [ ] Create 3D object representation fusion
- [ ] Create ObjectEncoder

### Priority 2: Integration
- [ ] Create MultimodalSequenceBuilder
- [ ] Update collate_fn to include 3D preprocessing
- [ ] Update AgentModel to use new components
- [ ] Handle variable-sized object sets (padding/pooling strategy)

### Priority 3: Training
- [ ] Add attention mask creation for variable-length sequences
- [ ] Add visualization of 3D representations
- [ ] Add metrics specific to 3D alignment

---

## Key Design Decisions

### 1. **Frozen vs Trainable Components**
- ✅ Keep BERT and ViT encoders frozen (as per architecture)
- ✅ Make object encoder, sequence builder, policy, and heads trainable

### 2. **Handling Variable Object Counts**
- **Option A**: Pad all object sets to max (simpler, more wasteful)
- **Option B**: Use Set Abstraction networks (complex but efficient)
- **Recommendation**: Start with padding, optimize later

### 3. **Demonstration Incorporation**
- Include last action from each demo in the sequence
- Use attention to weight demonstrations based on similarity
- Consider learned positional embeddings for demo ordering

### 4. **3D vs 2D Representations**
- Use normalized 3D coordinates (0-1 range)
- Include confidence scores
- Consider including object class embeddings

---

## Testing Strategy

1. **Unit tests** for each preprocessing component
2. **Integration tests** for the full pipeline
3. **Ablation study**: Test impact of 3D preprocessing vs 2D
4. **Visualization**: Plot 3D representations to verify correctness

---

## Performance Considerations

- **Bottleneck**: Object detection + depth estimation
- **Optimization**: Cache preprocessed features or run asynchronously
- **Memory**: Variable-sized sequences will complicate batch processing
- **Recommendation**: Implement efficient padding/masking strategy

---

## Next Steps

1. Start with Priority 1 components
2. Test each independently before integration
3. Use debugging visualizations to verify 3D representations
4. Run ablation studies to validate architectural choices
