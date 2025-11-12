# File Implementation Guide

## Where to Find Everything

### 📍 Documentation Files (Start Here!)

All new documentation is at your workspace root:

```
/Users/sameermemon/Desktop/gradStuff/classwork/11-777/EmbodiedMinds/
├── EXECUTIVE_SUMMARY.md                     ← What you need to know
├── DOCUMENTATION_INDEX.md                   ← How to navigate
├── QUICK_IMPLEMENTATION_GUIDE.md            ← Start implementing
├── ARCHITECTURE_VISUAL_SUMMARY.md           ← Visual overview
├── IMPLEMENTATION_ROADMAP.md                ← 3-day schedule
├── IMPLEMENTATION_TEMPLATES.py              ← Code to copy
├── CURRENT_VS_PROPOSED.md                   ← Why this design
└── ARCHITECTURE_IMPLEMENTATION_GUIDE.md     ← Deep reference
```

---

## Code Implementation Order

### Step 1: Replace Object Detection (2-3 hours)

**File:** `src/preprocessing/object_detection.py`

Replace entire file with:
```python
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt", device="cpu"):
        self.model = YOLO(model_name)
        self.device = device
        self.model.to(device)
        
    def detect_objects(self, image, conf_threshold=0.5):
        # See IMPLEMENTATION_TEMPLATES.py for full code
```

**Template location:** `IMPLEMENTATION_TEMPLATES.py` lines 10-75

**Test:**
```python
detector = ObjectDetector()
objects = detector.detect_objects(image)
assert len(objects) > 0, "No objects detected"
```

---

### Step 2: Replace Depth Estimation (2-3 hours)

**File:** `src/preprocessing/depth_estimation.py`

Replace entire file with:
```python
import torch
from torch import nn

class DepthEstimator:
    def __init__(self, model_type="DPT_Large", device="cpu"):
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(device)
        self.midas.eval()
        
    def estimate_depth(self, image):
        # See IMPLEMENTATION_TEMPLATES.py for full code
```

**Template location:** `IMPLEMENTATION_TEMPLATES.py` lines 85-155

**Test:**
```python
estimator = DepthEstimator()
depth = estimator.estimate_depth(image)
assert depth.min() >= 0 and depth.max() <= 1, "Depth not normalized"
```

---

### Step 3: Create Fusion Utils (1-2 hours)

**File:** `src/preprocessing/fusion_utils.py` (NEW)

Create new file with:
```python
import torch
import numpy as np

def create_3d_object_representations(objects, depth_map, image_h, image_w):
    # See IMPLEMENTATION_TEMPLATES.py for full code
    representations = []
    for obj in objects:
        # Combine 2D bbox with depth
        obj_3d = [cx, cy, z, w, h, conf, class_id]
        representations.append(obj_3d)
    return torch.tensor(representations)
```

**Template location:** `IMPLEMENTATION_TEMPLATES.py` lines 165-230

**Test:**
```python
obj_3d = create_3d_object_representations(objects, depth, H, W)
assert obj_3d.shape[1] == 7, "Wrong number of features"
assert obj_3d.min() >= 0 and obj_3d.max() <= 1, "Values not normalized"
```

---

### Step 4: Create Object Encoder (1 hour)

**File:** `src/encoders/object_encoder.py` (NEW)

Create new file with:
```python
import torch
import torch.nn as nn

class ObjectEncoder(nn.Module):
    def __init__(self, object_feature_dim=7, embedding_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(object_feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
    
    def forward(self, objects):
        return self.encoder(objects)
```

**Template location:** `IMPLEMENTATION_TEMPLATES.py` lines 240-275

**Test:**
```python
encoder = ObjectEncoder()
embeddings = encoder(obj_3d)  # (N, 7) -> (N, 256)
assert embeddings.shape == (obj_3d.shape[0], 256)
```

---

### Step 5: Create Sequence Builder (2-3 hours)

**File:** `src/fusion/sequence_builder.py` (NEW)

Create new file with comprehensive sequence building logic.

**Template location:** `ARCHITECTURE_IMPLEMENTATION_GUIDE.md` Phase 3

**Key concept:**
```python
# Build sequence: [instr] + [demo1_objs, demo1_action] + [demo2_objs, demo2_action] + [current_objs]

sequences = []
for b in batch:
    seq = []
    seq.append(project_instruction(instr[b]))  # (256,)
    
    for demo in demos:
        seq.extend(encode_objects(demo_objects[b]))  # Multiple tokens
        seq.append(encode_action(demo_actions[b]))    # (256,)
    
    seq.extend(encode_objects(current_objects[b]))   # Multiple tokens
    sequences.append(torch.stack(seq))
```

---

### Step 6: Update Data Loading (2-3 hours)

**File:** `src/datasets/dataloader.py`

**Section 1:** Enhance collate_fn

Find the existing `collate_fn` and replace with:
```python
def collate_fn_3d(batch):
    from src.preprocessing.object_detection import ObjectDetector
    from src.preprocessing.depth_estimation import DepthEstimator
    from src.preprocessing.fusion_utils import create_3d_object_representations
    
    # Initialize preprocessing
    detector = ObjectDetector()
    estimator = DepthEstimator()
    
    # Process each example
    instructions = []
    demo_3d_objects_list = []
    current_3d_objects_list = []
    targets = []
    
    for b in batch:
        # Process demonstrations
        for demo_idx, demo_images in enumerate(b["demo_images"]):
            avg_frame = demo_images.mean(axis=0)
            objects = detector.detect_objects(avg_frame)
            depth = estimator.estimate_depth(avg_frame)
            obj_3d = create_3d_object_representations(objects, depth, H, W)
            demo_3d_objects_list.append(obj_3d)
        
        # Process current image
        objects = detector.detect_objects(b["current_image"])
        depth = estimator.estimate_depth(b["current_image"])
        obj_3d = create_3d_object_representations(objects, depth, H, W)
        current_3d_objects_list.append(obj_3d)
        
        # Extract target action
        if len(b["demo_actions"]) > 0:
            targets.append(last_valid_action(b["demo_actions"][0]))
    
    return {
        'instructions': instructions,
        'demo_3d_objects': demo_3d_objects_list,
        'current_3d_objects': current_3d_objects_list,
        'targets': torch.tensor(targets),
    }
```

**Section 2:** Update AgentModel

Find the `AgentModel` class and add:
```python
class AgentModel(nn.Module):
    def __init__(self, ...):
        # Existing initialization
        super().__init__()
        
        # ADD THESE:
        self.object_enc = ObjectEncoder(embedding_dim=256)
        self.seq_builder = MultimodalSequenceBuilder(token_dim=256)
        # Keep existing:
        self.text_enc = TextEncoder(...)
        self.vis_enc = VisionEncoder(...)
        self.policy = PolicyTransformer(...)
        self.heads = OutputHeads(...)
    
    def forward(self, instr_texts, demo_images, demo_3d_objects, 
                current_images, current_3d_objects, demo_actions=None):
        # Encode instruction
        instr_embed = self.text_enc.encode(instr_texts)  # (B, 768)
        
        # Encode 3D objects
        demo_obj_embeds = [self.object_enc(obj) for obj in demo_3d_objects]
        current_obj_embeds = self.object_enc(current_3d_objects)
        
        # Build sequence
        tokens = self.seq_builder(instr_embed, demo_obj_embeds, 
                                  demo_actions, current_obj_embeds)
        
        # Policy reasoning
        decision = self.policy(tokens)  # (B, out_dim)
        
        # Output heads
        logits = self.heads(decision)  # list of (B, bins_i)
        
        return logits
```

---

## Implementation Checklist

### Before Starting
- [ ] `pip install ultralytics timm`
- [ ] Have sample images ready
- [ ] Backup your code: `git commit -m "backup"`

### Phase 1: Preprocessing (6-8 hours)
- [ ] Replace `object_detection.py` with YOLOv8
  - [ ] Test: detector finds 3-5 objects
  - [ ] Test: confidence scores reasonable
- [ ] Replace `depth_estimation.py` with MiDaS
  - [ ] Test: output is normalized (0-1)
  - [ ] Test: spatial structure makes sense
- [ ] Create `fusion_utils.py`
  - [ ] Test: output shape (N, 7)
  - [ ] Test: all values normalized (0-1)
- [ ] Create `object_encoder.py`
  - [ ] Test: (N, 7) → (N, 256)
  - [ ] Test: no NaN values

### Phase 2: Integration (6-8 hours)
- [ ] Create `sequence_builder.py`
  - [ ] Test: sequence construction works
  - [ ] Test: handles variable object counts
- [ ] Update `dataloader.py` collate_fn
  - [ ] Test: collate_fn returns correct dict
  - [ ] Test: preprocessing runs without errors
- [ ] Update `AgentModel` in dataloader.py
  - [ ] Test: forward pass completes
  - [ ] Test: output shapes correct
- [ ] Test one training iteration
  - [ ] Test: no shape mismatches
  - [ ] Test: loss is finite

### Phase 3: Validation (2-4 hours)
- [ ] Run 10 training iterations
  - [ ] Test: loss decreasing
  - [ ] Test: no NaN/Inf values
- [ ] Visualize results
  - [ ] Test: 3D representations make sense
  - [ ] Test: sequence has expected length
- [ ] Run full training epoch
  - [ ] Test: completes without error
  - [ ] Test: validation metrics computed

---

## File Size Reference

Estimated lines of code to write:

```
object_encoder.py               ~50 lines (new)
fusion_utils.py                 ~80 lines (new)
sequence_builder.py            ~100 lines (new)
object_detection.py             ~50 lines (replacement)
depth_estimation.py             ~40 lines (replacement)
dataloader.py                  ~100 lines (modifications)

Total NEW code:                ~420 lines
Total MODIFIED code:           ~100 lines
```

This is very manageable - about 1-2 hours per file with testing.

---

## Testing Each Component

### Test ObjectDetector
```python
from src.preprocessing.object_detection import ObjectDetector
import cv2

detector = ObjectDetector()
image = cv2.imread("test.jpg")
objects = detector.detect_objects(image)

print(f"Detected {len(objects)} objects")
for obj in objects:
    print(f"  Box: {obj['box']}, Conf: {obj['confidence']:.2f}")
```

### Test DepthEstimator
```python
from src.preprocessing.depth_estimation import DepthEstimator
import matplotlib.pyplot as plt

estimator = DepthEstimator()
image = cv2.imread("test.jpg")
depth = estimator.estimate_depth(image)

print(f"Depth range: {depth.min():.2f} - {depth.max():.2f}")
plt.imshow(depth, cmap='viridis')
plt.show()
```

### Test Fusion
```python
from src.preprocessing.fusion_utils import create_3d_object_representations

obj_3d = create_3d_object_representations(objects, depth, h, w)
print(f"3D representations shape: {obj_3d.shape}")
print(f"Min/max values: {obj_3d.min():.3f} - {obj_3d.max():.3f}")
```

### Test ObjectEncoder
```python
from src.encoders.object_encoder import ObjectEncoder

encoder = ObjectEncoder()
embeddings = encoder(obj_3d)
print(f"Embeddings shape: {embeddings.shape}")
```

### Test Full Pipeline
```python
batch = next(iter(dataloader))
output = model(batch['instructions'], batch['demo_images'], 
               batch['demo_3d_objects'], batch['current_images'],
               batch['current_3d_objects'])
print(f"Output: {len(output)} logit tensors")
for i, logits in enumerate(output):
    print(f"  Dim {i}: shape {logits.shape}")
```

---

## Common Issues & Solutions

### Issue: ObjectDetector finds no objects
**Solution:** Try larger model: `"yolov8m.pt"` instead of `"yolov8n.pt"`

### Issue: DepthEstimator returns wrong range
**Solution:** Check normalization. Should be 0-1 after `(depth - min) / (max - min)`

### Issue: Shape mismatch in sequence building
**Solution:** Print shapes at each stage, handle variable object counts with padding

### Issue: Training diverges (loss → NaN)
**Solution:** Check for invalid 3D coordinates, reduce learning rate

### Issue: Out of memory
**Solution:** Reduce batch_size or run on CPU for testing

---

## Success Verification

After each phase, run:

```python
# Phase 1 complete?
detector = ObjectDetector()
estimator = DepthEstimator()
obj_3d = create_3d_object_representations(...)
encoder = ObjectEncoder()
embeddings = encoder(obj_3d)
print("✅ Phase 1 OK" if embeddings.shape[1] == 256 else "❌ Phase 1 FAILED")

# Phase 2 complete?
builder = MultimodalSequenceBuilder()
tokens = builder(instr_embed, demo_objs, demo_actions, current_objs)
output = model(...)
print("✅ Phase 2 OK" if len(output) == 7 else "❌ Phase 2 FAILED")

# Phase 3 complete?
for epoch in range(1):
    for batch in dataloader:
        loss = train_step(batch)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
print("✅ Phase 3 OK" if not torch.isnan(loss) else "❌ Phase 3 FAILED")
```

---

## You're Ready to Start!

1. Read: `QUICK_IMPLEMENTATION_GUIDE.md`
2. Copy: Code templates from `IMPLEMENTATION_TEMPLATES.py`
3. Create/Modify: Files listed above
4. Test: Each component using test code above
5. Integrate: Full pipeline
6. Validate: Training works

**Estimated total time: 3-5 days focused work**

Good luck! 🚀
