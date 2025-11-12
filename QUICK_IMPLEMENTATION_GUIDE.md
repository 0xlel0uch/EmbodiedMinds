# Quick Implementation Summary

## What You Need to Build

Your proposed architecture has **5 main components**. Here's the status:

### ✅ Component 1: Text Encoder (DONE)
Your `TextEncoder` with BERT is correctly implemented and frozen.

### ✅ Component 2: Vision Encoder (PARTIALLY DONE)
Your `VisionEncoder` uses ResNet18. **Consider upgrading to CLIP's ViT** for better object-level understanding.

### ⚠️ Component 3: 3D Perception Preprocessing (NEEDS WORK)
**This is your main implementation focus.** Three sub-components:

1. **Object Detection** → Use YOLOv8 instead of OpenCV
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   results = model(image)
   ```

2. **Depth Estimation** → Use MiDaS
   ```python
   model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
   depth = model(image_tensor)
   ```

3. **3D Fusion** → Create object representations
   ```python
   # Combine 2D bboxes + depth → 3D coordinates
   obj_3d = [center_x, center_y, depth, width, height, confidence, class_id]
   ```

### ⚠️ Component 4: Trainable Policy Network (NEEDS UPDATE)
Your `PolicyTransformer` is good, but the **sequence construction needs work**:

**Current (too simple):**
```
[instruction_embedding, demo_image_embedding, current_image_embedding]
```

**What it should be:**
```
[instruction_embedding,
 demo1_objects, demo1_action,
 demo2_objects, demo2_action,
 ...,
 current_objects]
```

### ✅ Component 5: Output Heads (DONE)
Your 7 classification heads are correctly implemented.

---

## The 4-Phase Implementation Plan

### Phase 1: Build 3D Perception (Days 1-2)
1. Upgrade `ObjectDetector` → YOLOv8
2. Implement `DepthEstimator` → MiDaS  
3. Create `fusion_utils.py` → 3D representations
4. Test each module independently

**Files to create/modify:**
- Modify: `src/preprocessing/object_detection.py`
- Modify: `src/preprocessing/depth_estimation.py`
- Create: `src/preprocessing/fusion_utils.py`

### Phase 2: Data Pipeline (Days 1-2, in parallel)
1. Create `ObjectEncoder` class
2. Update collate function to run preprocessing
3. Handle variable-sized object sets

**Files to create/modify:**
- Create: `src/encoders/object_encoder.py`
- Modify: `src/datasets/dataloader.py` → enhance collate_fn

### Phase 3: Multimodal Sequence (Day 1-2)
1. Create `MultimodalSequenceBuilder` class
2. Properly structure instruction + demos + current scene

**Files to create/modify:**
- Create: `src/fusion/sequence_builder.py`

### Phase 4: Integration (Day 3)
1. Update `AgentModel` to use new components
2. Update training loop if needed
3. Add debugging/visualization

**Files to modify:**
- Modify: `src/datasets/dataloader.py` → AgentModel class
- Modify: `src/training/train.py` → if needed

---

## Most Important Design Decisions

### 1. Which Vision Encoder?
**Current:** ResNet18 → flattened features  
**Recommended:** CLIP ViT → token embeddings per patch

**Why?** ViT naturally produces per-patch embeddings, perfect for per-object encoding.

### 2. How to Handle Variable Object Counts?
**Simplest approach:** Pad all to max, use attention masks  
**Better approach:** Pool or use set operations  
**Start with:** Padding (simpler, good enough for now)

### 3. How to Include Demonstrations?
**Architecture says:** Include object embeddings AND actions from demos  
**Implementation:**
- Extract last valid action from each demo
- Encode objects detected in demo
- Include both in sequence to Transformer
- Transformer learns which demos are relevant

### 4. Action Space Encoding
**Your bins:** [101, 101, 101, 121, 121, 121, 2]  
**Meaning:** First 3 dims have 101 bins (continuous space discretized), next 3 have 121, last is binary (gripper)  
**Include in sequence:** Encode demo actions as additional tokens

---

## Code Structure Recommendation

```
src/
├── encoders/
│   ├── text_encoder.py      ✅ Keep as-is
│   ├── vision_encoder.py    ✅ Keep, optionally upgrade
│   └── object_encoder.py    🆕 Create
├── preprocessing/
│   ├── object_detection.py  ♻️ Upgrade to YOLOv8
│   ├── depth_estimation.py  ♻️ Implement properly
│   └── fusion_utils.py      🆕 Create
├── fusion/
│   ├── fusion_module.py     🗑️ Remove (old approach)
│   └── sequence_builder.py  🆕 Create
├── policy/
│   └── policy_transformer.py ✅ Keep as-is
├── heads/
│   └── output_heads.py      ✅ Keep as-is
├── datasets/
│   ├── dataloader.py        ♻️ Update collate_fn
│   └── transforms.py        ✅ Keep
└── training/
    └── train.py             ✅ Keep (minimal changes)
```

---

## Example: What the Data Flow Should Look Like

```
Input: Text instruction + 2 demo videos + current camera frame
        ↓
[Text] → BERT → instr_embedding (768)
        ↓
[Demo Video 1]
  ├─→ Object Detection → 5 objects detected
  ├─→ Depth Estimation → depth map
  ├─→ Fusion → 5 × (center_x, center_y, depth, w, h, conf, cls)
  ├─→ Object Encoder → 5 × 256-dim embeddings
  └─→ Extract Action → [x_bin=45, y_bin=32, ..., gripper=1]
        ↓
[Demo Video 2] (same process)
        ↓
[Current Frame]
  ├─→ Object Detection → 3 objects
  ├─→ Depth Estimation → depth map
  ├─→ Fusion → 3 × (center_x, center_y, depth, w, h, conf, cls)
  └─→ Object Encoder → 3 × 256-dim embeddings
        ↓
Sequence Builder:
  [instr_embed (256),
   demo1_obj1 (256), demo1_obj2, demo1_obj3, demo1_action (256),
   demo2_obj1 (256), demo2_obj2, ..., demo2_action (256),
   cur_obj1 (256), cur_obj2, cur_obj3]
        ↓
Transformer Policy: seq_len×256 → 512-dim decision vector
        ↓
Output Heads: 512-dim → [45, 32, ..., 1] (7D action)
```

---

## Recommended Package Additions

```bash
pip install ultralytics  # YOLOv8
pip install timm         # For MiDaS backbone
pip install opencv-python-headless  # Better than cv2
pip install pillow       # Image operations
```

---

## Testing Approach

**Test in isolation first:**
1. Test ObjectDetector on sample images
2. Test DepthEstimator on sample images  
3. Test 3D fusion (visualize to verify)
4. Test ObjectEncoder
5. Test MultimodalSequenceBuilder
6. Test full pipeline integration

**Debugging tools:**
- Visualize 3D bboxes projected onto images
- Check sequence shape at each layer
- Monitor attention patterns in Transformer
- Compare performance: with vs without 3D preprocessing

---

## Success Metrics

After implementation, validate:
- [ ] 3D object representations are spatially accurate
- [ ] Variable object counts handled correctly
- [ ] Sequence length manageable (not exploding)
- [ ] Attention patterns show demonstration relevance
- [ ] Model learns to predict reasonable actions
- [ ] Performance improvement over 2D baseline

