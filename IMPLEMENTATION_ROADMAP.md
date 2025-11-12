# Implementation Priority & Timeline

## TL;DR - What You Need to Do

Your codebase is **60% complete**. Here's the breakdown:

| Component | Status | Priority | Est. Time |
|-----------|--------|----------|-----------|
| Text Encoder (BERT) | ✅ Done | - | - |
| Vision Encoder (ResNet) | ✅ Done | Optional | 2hrs (to upgrade to CLIP) |
| Object Detector | ❌ Broken | **CRITICAL** | 2-3hrs |
| Depth Estimator | ❌ Stubbed | **CRITICAL** | 2-3hrs |
| 3D Fusion | ❌ Stubbed | **CRITICAL** | 1-2hrs |
| Object Encoder | ❌ Missing | High | 1hr |
| Sequence Builder | ❌ Missing | High | 2-3hrs |
| Policy Transformer | ✅ Done | - | - |
| Output Heads | ✅ Done | - | - |
| Data Pipeline | ⚠️ Partial | High | 2-3hrs |
| **TOTAL** | **60%** | - | **~14-18 hours** |

---

## Implementation Roadmap

### Day 1: Build 3D Perception Pipeline (6-8 hours)

**Morning:**
1. **Upgrade ObjectDetector** (1-2 hrs)
   - Install: `pip install ultralytics`
   - Replace `src/preprocessing/object_detection.py`
   - Use template from `IMPLEMENTATION_TEMPLATES.py`
   - Test on sample images

2. **Implement DepthEstimator** (1-2 hrs)
   - Uses PyTorch Hub (no new install needed)
   - Replace `src/preprocessing/depth_estimation.py`
   - Use template from `IMPLEMENTATION_TEMPLATES.py`
   - Test on sample images

**Afternoon:**
3. **Create fusion_utils.py** (1-2 hrs)
   - Create `src/preprocessing/fusion_utils.py`
   - Implement `create_3d_object_representations`
   - Test: objects + depth → 3D coords

4. **Create ObjectEncoder** (1 hr)
   - Create `src/encoders/object_encoder.py`
   - Simple 2-layer MLP with LayerNorm
   - Test: (N, 7) → (N, 256)

**Deliverables:** Can run preprocessing on any image

---

### Day 2: Data Pipeline & Integration (6-8 hours)

**Morning:**
1. **Enhance collate_fn** (2-3 hrs)
   - Update `src/datasets/dataloader.py`
   - Integrate preprocessing pipeline
   - Handle variable object counts (padding)
   - Test: batch loads correctly

2. **Create MultimodalSequenceBuilder** (2 hrs)
   - Create `src/fusion/sequence_builder.py`
   - Constructs token sequence with demos + actions
   - Test: sequence shapes and lengths

**Afternoon:**
3. **Update AgentModel** (1-2 hrs)
   - Integrate all components
   - Update forward pass
   - Test: full forward pass runs

4. **Minimal Training Test** (1 hr)
   - Run 1 epoch on small debug dataset
   - No errors, shapes correct

**Deliverables:** Can do end-to-end forward pass with 3D preprocessing

---

### Day 3: Testing & Debugging (4-6 hours)

1. **Unit test each component** (2 hrs)
   - Verify object detector accuracy
   - Verify depth estimator output
   - Verify 3D representation correctness

2. **Integration testing** (1 hr)
   - Run full pipeline on sample batch
   - Visualize 3D objects on images
   - Check sequence construction

3. **Training validation** (1-2 hrs)
   - Train for 10 epochs
   - Monitor loss progression
   - Check for NaNs or divergence

4. **Optimization** (1 hr)
   - Profile computational cost
   - Cache preprocessing if too slow
   - Reduce model sizes if too slow

**Deliverables:** Working system ready for full training

---

## Quick Start Checklist

### Prerequisites (30 mins)
- [ ] Create backup of current code: `git commit -m "backup before 3D architecture"`
- [ ] Install packages: `pip install ultralytics timm`
- [ ] Have sample images ready for testing

### Phase 1: Preprocessing (2-3 hours)

**File 1: `src/preprocessing/object_detection.py`**
```bash
1. Copy template from IMPLEMENTATION_TEMPLATES.py
2. Test: detector = ObjectDetector(); objects = detector.detect_objects(img)
3. Verify: 3-5 objects detected with reasonable confidence
```

**File 2: `src/preprocessing/depth_estimation.py`**
```bash
1. Copy template from IMPLEMENTATION_TEMPLATES.py
2. Test: estimator = DepthEstimator(); depth = estimator.estimate_depth(img)
3. Verify: depth values in [0, 1], shows reasonable spatial structure
```

**File 3: `src/preprocessing/fusion_utils.py`** (NEW)
```bash
1. Create file with create_3d_object_representations()
2. Test: obj_3d = create_3d_representations(objects, depth, H, W)
3. Verify: shape (N, 7), values normalized
```

**File 4: `src/encoders/object_encoder.py`** (NEW)
```bash
1. Create ObjectEncoder class
2. Test: encoder = ObjectEncoder(); embeds = encoder(obj_3d)
3. Verify: shape (N, 256)
```

### Phase 2: Data Pipeline (2-3 hours)

**File 5: `src/datasets/dataloader.py`** - Enhance collate_fn
```bash
1. Update collate_fn to call preprocessing
2. Return: instructions, demo_3d_objects, current_3d_objects, targets
3. Test: batch = next(iter(dataloader)); batch shapes correct
```

**File 6: `src/fusion/sequence_builder.py`** (NEW)
```bash
1. Create MultimodalSequenceBuilder class
2. Test: builder = MultimodalSequenceBuilder(); tokens = builder(...)
3. Verify: shape (B, seq_len, 256)
```

**File 7: `src/datasets/dataloader.py`** - Update AgentModel
```bash
1. Add ObjectEncoder and MultimodalSequenceBuilder
2. Update forward() to use new components
3. Test: output = model(instrs, demo_imgs, demo_3d, cur_imgs, cur_3d)
4. Verify: output is list of 7 logit tensors
```

### Phase 3: Validation (2 hours)

```bash
1. Run: python -c "
from src.datasets.dataloader import build_dataloader
dl = build_dataloader(batch_size=2, debug=True)
for batch in dl:
    print('Batch shapes correct')
    break
"

2. Run: python src/training/train.py --debug --epochs 1
   - Should complete without errors
   - Loss should be reasonable (~1.0-2.0)

3. Check: git diff to review all changes
```

---

## File-by-File Implementation Guide

### Create/Modify Priority Order

1. **First** (enables everything else):
   - ✏️ Modify `src/preprocessing/object_detection.py` 
   - ✏️ Modify `src/preprocessing/depth_estimation.py`
   - ✏️ Create `src/preprocessing/fusion_utils.py`

2. **Second** (uses above):
   - ✏️ Create `src/encoders/object_encoder.py`
   - ✏️ Create `src/fusion/sequence_builder.py`

3. **Third** (integrates everything):
   - ✏️ Modify `src/datasets/dataloader.py` → collate_fn
   - ✏️ Modify `src/datasets/dataloader.py` → AgentModel

4. **Don't touch** (already working):
   - ✅ `src/encoders/text_encoder.py`
   - ✅ `src/encoders/vision_encoder.py` (optional upgrade)
   - ✅ `src/policy/policy_transformer.py`
   - ✅ `src/heads/output_heads.py`
   - ✅ `src/training/train.py`

---

## Success Criteria

After implementation, you should be able to:

1. ✅ Load an image and run full preprocessing pipeline
2. ✅ Get 3D object representations with proper coordinates
3. ✅ Create embedding sequence with demonstrations and actions
4. ✅ Run forward pass through policy transformer
5. ✅ Train for 10 epochs without errors
6. ✅ See loss decreasing
7. ✅ Verify attention attends to relevant objects

---

## Debugging Tips

### If object detector finds too many objects:
```python
# Increase confidence threshold
objects = detector.detect_objects(image, conf_threshold=0.7)
```

### If object detector finds no objects:
```python
# Check model is correct for your domain
# Try larger model: "yolov8m.pt" instead of "yolov8n.pt"
```

### If depth looks inverted:
```python
# MiDaS sometimes needs inversion
depth_inverted = 1.0 - depth_map
```

### If sequence shapes don't match:
```python
# Debug: print shapes at each stage
print(f"Instr: {instr_embed.shape}")
print(f"Demo objs: {demo_obj_embeds[0].shape}")
print(f"Current objs: {current_obj_embeds.shape}")
print(f"Final sequence: {tokens.shape}")
```

### If training diverges (loss → NaN):
```python
# Check for:
# 1. Gradients exploding: print(max_grad) in training loop
# 2. Learning rate too high: reduce from 1e-4 to 1e-5
# 3. Invalid 3D coordinates: check normalization (should be 0-1)
```

---

## Reference Documents

You now have 4 comprehensive guides in your workspace:

1. **`QUICK_IMPLEMENTATION_GUIDE.md`** ← Start here!
   - High-level overview
   - What needs to be built
   - Design decisions

2. **`ARCHITECTURE_IMPLEMENTATION_GUIDE.md`**
   - Detailed component-by-component guide
   - Code templates for each module
   - Testing strategy

3. **`CURRENT_VS_PROPOSED.md`**
   - Visual comparison of old vs new architecture
   - Why each component matters
   - Common mistakes to avoid

4. **`IMPLEMENTATION_TEMPLATES.py`**
   - Copy-paste ready code
   - Usage examples
   - Test code

---

## Questions You Might Have

**Q: Do I need to upgrade to CLIP ViT?**
A: Not required. Your ResNet18 works fine. CLIP is better for object-level reasoning but optional.

**Q: How much GPU memory needed?**
A: ~1GB for batch_size=8. Reduce batch_size if insufficient.

**Q: Can I precompute 3D representations to save training time?**
A: Yes! Store preprocessed features in dataset. This is recommended for large datasets.

**Q: What if I don't have GPU?**
A: Everything works on CPU but will be ~10× slower. Start with debug=True and small batches.

**Q: Should I freeze the object encoder?**
A: No, keep it trainable. It learns to extract meaningful embeddings for your task.

**Q: How do I know if architecture is working?**
A: If training loss decreases smoothly and validation accuracy improves, you're good!

---

## Final Note

Your architecture is well-designed. The existing code quality is good. You're in a strong position to implement this successfully.

The key insight: **You have the frozen perception → trainable reasoning pipeline correct. Now add explicit 3D spatial understanding.**

Start with Day 1's object detection and depth estimation. Everything else follows naturally from there.

Good luck! 🚀
