# Current vs Proposed Architecture Comparison

## Side-by-Side Comparison

### CURRENT STATE

```
Text Input → BERT (frozen) → text_embed (768)
                               ↓
Demo Image → ResNet18 (frozen) → flatten → vis_embed (512)
                                            ↓
Current Image → ResNet18 (frozen) → flatten → vis_embed (512)
                                              ↓
Simple Linear Projections
                                              ↓
Sequence: [instr_token, demo_token, current_token]  (3 tokens)
                                              ↓
PolicyTransformer (4 layers, 8 heads)
                                              ↓
Decision Vector (512 dim)
                                              ↓
7 Output Heads → [45, 32, 78, 119, 55, 98, 1]
```

**Issues with current approach:**
1. ❌ No explicit 3D understanding - just using image embeddings
2. ❌ No object-level information - treating entire image as single token
3. ❌ Demonstrations not structured - no action information in sequence
4. ❌ No spatial reasoning - ResNet features are too abstract

---

### PROPOSED STATE

```
Text Input
    ↓
BERT (frozen) → instr_embed (768)
    ↓
[project to token_dim]
    ↓ instr_token (256)

Demo Images #1
    ├→ ObjectDetector (YOLOv8) → 5 objects detected
    ├→ DepthEstimator (MiDaS) → depth_map
    ├→ Fusion → 3D representations (5, 7)
    ├→ ObjectEncoder → object_embeds (5, 256)
    │   ↓ demo1_objects
    │
    ├→ Extract Action [45, 32, ..., 1]
    │   ↓ [project to token_dim]
    │   ↓ demo1_action_token (256)

Demo Images #2 (same process)
    ↓ demo2_objects + demo2_action_token

Current Images
    ├→ ObjectDetector (YOLOv8) → 3 objects detected
    ├→ DepthEstimator (MiDaS) → depth_map
    ├→ Fusion → 3D representations (3, 7)
    ├→ ObjectEncoder → object_embeds (3, 256)
    ↓ current_objects

Sequence Assembly:
    [instr_token,
     demo1_obj1, demo1_obj2, demo1_obj3, demo1_obj4, demo1_obj5, demo1_action_token,
     demo2_obj1, demo2_obj2, demo2_obj3, demo2_obj4, demo2_obj5, demo2_action_token,
     current_obj1, current_obj2, current_obj3]
    
    Total: 1 + (5+1) + (5+1) + 3 = 16 tokens (vs 3 currently)
    
    ↓ PolicyTransformer (sees ~16 tokens now instead of 3)
    
Decision Vector (512 dim)
    
    ↓ 7 Output Heads
    
[45, 32, 78, 119, 55, 98, 1]  ← Same output, but richer reasoning
```

**Benefits of proposed approach:**
1. ✅ Explicit 3D spatial understanding via depth
2. ✅ Per-object features - can attend to relevant objects
3. ✅ Demonstrations explicitly include actions - can learn action patterns
4. ✅ Rich spatial reasoning - Transformer has more context

---

## Detailed Component Mapping

| Layer | Current | Proposed | Status | Notes |
|-------|---------|----------|--------|-------|
| **Input Processing** | Images only | Images + Text | ⚠️ Partial | Text already done |
| **Object Detection** | None | YOLOv8 | 🔴 TODO | Key missing component |
| **Depth Estimation** | None | MiDaS | 🔴 TODO | Key missing component |
| **3D Fusion** | None | Bbox+Depth fusion | 🔴 TODO | Creates 3D representations |
| **Text Encoder** | BERT (frozen) | BERT (frozen) | ✅ Done | Keep as-is |
| **Vision Encoder** | ResNet18 (frozen) | CLIP ViT (frozen) | ⚠️ Optional | Current works, ViT is better |
| **Object Encoder** | None | MLPs | 🔴 TODO | Encodes 3D features |
| **Sequence Building** | Simple concat | MultimodalSequenceBuilder | 🔴 TODO | Must include demos + actions |
| **Policy Network** | PolicyTransformer | PolicyTransformer | ✅ Done | Works for variable seq lengths |
| **Output Heads** | 7 classifiers | 7 classifiers | ✅ Done | Keep as-is |

---

## Key Architectural Changes Required

### Change #1: Input Processing Pipeline

**Before:**
```python
instr_embed = text_enc.encode(instrs)
demo_embed = vis_enc.encode(demo_images)
current_embed = vis_enc.encode(current_images)
```

**After:**
```python
# Text: same as before
instr_embed = text_enc.encode(instrs)

# Demo processing (multiple demos)
demo_objects = []
for demo_img in demo_images:
    detections = obj_detector.detect_objects(demo_img)
    depth = depth_estimator.estimate_depth(demo_img)
    obj_3d = create_3d_representations(detections, depth)
    obj_embed = object_encoder(obj_3d)
    demo_objects.append(obj_embed)
    
    # Extract action for this demo (NEW)
    demo_action = extract_last_valid_action(demo_actions[i])

# Current processing
detections = obj_detector.detect_objects(current_img)
depth = depth_estimator.estimate_depth(current_img)
obj_3d = create_3d_representations(detections, depth)
current_objects = object_encoder(obj_3d)
```

### Change #2: Sequence Construction

**Before:**
```python
tokens = torch.stack([t_instr, t_demo, t_cur], dim=1)  # (B, 3, 256)
```

**After:**
```python
# Build rich sequence with objects and actions
tokens = []
tokens.append(instr_embedding)  # (256,)

for demo_idx in range(num_demos):
    tokens.extend(demo_objects[demo_idx])  # multiple object tokens
    tokens.append(demo_action_tokens[demo_idx])  # (256,)

tokens.extend(current_objects)  # multiple object tokens

tokens = torch.stack(tokens)  # (total_tokens, 256)
# Pad or handle variable lengths
```

### Change #3: Data Flow Through Transformer

**Before:**
```
PolicyTransformer: 3 tokens → decision (512 dim)
Sequence too short, limited reasoning
```

**After:**
```
PolicyTransformer: ~16 tokens → decision (512 dim)
Transformer can:
  - Learn which objects are relevant
  - Learn which demonstrations are similar
  - Learn temporal action patterns
  - Perform spatial reasoning
```

---

## Information Richness Comparison

### Current Approach (Minimal Information)

```
Instruction: "Stack the star on top of the cube"
↓
BERT encoding → loses spatial information, focuses on language

Demo Image: [full image as single embedding]
↓
ResNet → generic visual features, no object awareness

Current Image: [full image as single embedding]
↓
ResNet → generic visual features, no object awareness

Reasoning: Does instruction mention "cube"? 
           But I only have global image feature, 
           can't locate the cube specifically!

Output: [guess action]
```

### Proposed Approach (Rich Information)

```
Instruction: "Stack the star on top of the cube"
↓
BERT + Transformer attention → can attend to "stack", "star", "cube"

Demo Image 1:
  - Detects: [star_center_x=0.3, star_center_y=0.4, star_depth=0.8],
             [cube_center_x=0.7, cube_center_y=0.5, cube_depth=0.9],
             [background_depth=0.95], ...
  - Previous Action: [move_to_x=0.3, move_to_y=0.4, ...] ← LEARN THIS PATTERN

Demo Image 2:
  - Detects: [star_center_x=0.2, star_center_y=0.3, star_depth=0.7],
             [cube_center_x=0.6, cube_center_y=0.4, cube_depth=0.85],
             ...
  - Previous Action: [move_to_x=0.2, move_to_y=0.3, ...] ← LEARN THIS PATTERN

Current Image:
  - Detects: [star_center_x=0.35, star_center_y=0.45, star_depth=0.75],
             [cube_center_x=0.65, cube_center_y=0.55, cube_depth=0.88],
             ...

Reasoning: "Stack star on cube"
  - Attention to: "star" token ↔ star_object ↔ 0.35, 0.45 position
  - Attention to: "cube" token ↔ cube_object ↔ 0.65, 0.55 position
  - Attention to: demo_action patterns ↔ moving to object positions
  - Decision: "Move to 0.35, 0.45" (the star's position!)

Output: [move_to_x=0.35, move_to_y=0.45, move_to_z=0.75, ...]
        ↑ Much more spatially accurate!
```

---

## Impact on Model Capacity

| Metric | Current | Proposed | Change |
|--------|---------|----------|--------|
| Input tokens per example | 3 | ~16 | 5.3× more context |
| Information per token | Low (global image) | High (specific object) | 10× richer |
| Trainable parameters | ~1M (projections + transformer + heads) | ~2-3M (adds object encoder) | +1-2M |
| Computational cost | ~100ms/batch | ~500ms/batch | 5× (mostly preprocessing) |
| Memory per batch | ~500MB | ~1GB | 2× |

**Trade-off:** Slightly more computation and memory, but significantly richer reasoning.

---

## Why This Architecture Works

### 1. **3D Perception Bottleneck**
- Original paper noted: MLLMs struggle with precise 3D manipulation
- Our approach: Explicitly provides 3D coordinates to model
- Transformer can learn spatial relationships: "object A is above object B at depth Z"

### 2. **Demonstration Learning**
- Including demo actions in sequence allows Transformer to:
  - Learn: "When star is at position X, move gripper to X"
  - Generalize: Apply learned mapping to current scene
  - This is in-context learning!

### 3. **Object-Level Attention**
- With per-object tokens, Transformer can:
  - Focus on relevant objects (star, cube) vs irrelevant (background, hand)
  - Learn that certain object properties matter (size, position, depth)
  - Ignore visual clutter (lighting, textures)

### 4. **Modular Frozen Components**
- Keeps BERT and ViT frozen → stable language/vision understanding
- Only trains: object encoder, transformer, output heads
- Reduces overfitting, speeds training, better generalization

---

## Common Implementation Mistakes to Avoid

### ❌ Mistake #1: Not normalizing 3D coordinates
```python
# Wrong:
z = depth_map[y, x]  # Range: 0-∞, varies by scene

# Right:
z_norm = (z - z_min) / (z_max - z_min)  # Range: 0-1, consistent
```

### ❌ Mistake #2: Using ResNet for object embeddings
```python
# Wrong: ResNet output is (B, 2048) global feature
# Can't tell which feature is about which object

# Right: Per-object encoding
for obj in objects:
    obj_embed = object_encoder(obj_3d)  # (256,) per object
```

### ❌ Mistake #3: Forgetting about variable object counts
```python
# Wrong: Assume always 5 objects
objects = detect(image)  # 3 objects this time!
tokens = stack(objects)  # Crash! Expected 5, got 3

# Right: Pad or use masking
max_objects = 10
padded = torch.zeros(max_objects, 256)
padded[:objects.shape[0]] = objects
attention_mask = torch.arange(max_objects) < objects.shape[0]
```

### ❌ Mistake #4: Not including demonstration actions
```python
# Wrong: Only include objects, no action learning
sequence = [instr, demo_objs, current_objs]

# Right: Include what actions were taken
sequence = [instr, demo_objs, demo_action, current_objs]
# Now Transformer can learn: "demo took action X with these objects"
# "Current has similar objects, so take similar action"
```

### ❌ Mistake #5: Forgetting that preprocessing is expensive
```python
# Wrong: Run object detection in forward pass during training
# This makes training 10× slower!

# Right: Precompute during data loading
# Store detected objects, depths in dataset
# Only run neural network components during training
```

---

## Validation Checklist

Before running full training, validate:

- [ ] Object detector finds objects with reasonable confidence
- [ ] Depth estimator produces smooth, reasonable depth maps
- [ ] 3D coordinates are normalized consistently (0-1 range)
- [ ] Object encoder runs without shape mismatches
- [ ] Variable object counts handled by padding/masking
- [ ] Sequence built with correct token order
- [ ] Attention masks created for padded sequences
- [ ] PolicyTransformer accepts variable seq lengths
- [ ] Output heads produce correct action bins
- [ ] Training loop runs 1 batch without errors
- [ ] Validation metrics computed correctly

---

## Expected Training Dynamics

### Phase 1: Learning object recognition (Epoch 1-5)
- Object encoder learns to embed 3D coordinates meaningfully
- Transformer learns basic attention patterns
- Loss decreases slowly as model learns fundamentals

### Phase 2: Learning spatial reasoning (Epoch 5-15)
- Transformer learns to relate: "star at 0.3, 0.4" ↔ "move to 0.3, 0.4"
- Learns from demonstrations: patterns in demo actions
- Accuracy improves notably

### Phase 3: Fine-tuning (Epoch 15+)
- Learns edge cases and specific task nuances
- Performance plateaus as model saturates capacity
- Early stopping recommended

**Typical progression:**
- Epoch 1: Loss ~2.0, Accuracy ~14% (random for 7 dims)
- Epoch 5: Loss ~1.2, Accuracy ~30%
- Epoch 10: Loss ~0.8, Accuracy ~45%
- Epoch 20: Loss ~0.5, Accuracy ~60%
- Epoch 50+: Loss ~0.4, Accuracy ~70-80%

(These are estimates - actual values depend on dataset difficulty)

