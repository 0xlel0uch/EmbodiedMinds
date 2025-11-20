# Visual Architecture Summary

## Your Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VISUAL IN-CONTEXT LEARNING                       │
│                  3D Spatial Reasoning for Manipulation               │
└─────────────────────────────────────────────────────────────────────┘

STAGE 1: PREPROCESSING (On raw images)
═══════════════════════════════════════

Demo Image #1 / Demo Image #2 / Current Image
        ↓                ↓              ↓
    ┌───────────────────────────────────────┐
    │    1. Object Detection (YOLOv8)       │ ← TO BUILD
    │    "What objects are in the scene?"   │
    └───────────────────────────────────────┘
              ↓
    ┌───────────────────────────────────────┐
    │    2. Depth Estimation (MiDaS)        │ ← TO BUILD
    │    "How far is each object?"          │
    └───────────────────────────────────────┘
              ↓
    ┌───────────────────────────────────────┐
    │    3. 3D Fusion (fusion_utils)         │ ← TO BUILD
    │    Bboxes + Depth → 3D Coordinates    │
    │    (center_x, center_y, depth, w, h) │
    └───────────────────────────────────────┘
              ↓
    ┌───────────────────────────────────────┐
    │    4. Object Encoding (ObjectEncoder) │ ← TO BUILD
    │    3D coords → 256-dim embeddings     │
    └───────────────────────────────────────┘

STAGE 2: FROZEN ENCODERS (Pre-trained, frozen weights)
═══════════════════════════════════════════════════════

Text Instruction         Image Objects
"Stack the star"         [obj_embed_1, obj_embed_2, ...]
        ↓                         ↓
    ┌──────────────┐      ┌─────────────────┐
    │ BERT         │      │ ObjectEncoder   │ ← Already done above
    │ (frozen)     │      │ (just encoded)  │
    └──────────────┘      └─────────────────┘
        ↓                         ↓
    768-dim                  256-dim each
    embedding           (for each detected object)

STAGE 3: SEQUENCE CONSTRUCTION (Prepare for reasoning)
═══════════════════════════════════════════════════════

        ┌─────────────────────────────────────────────────────────────┐
        │ Build sequence for Transformer (from demonstrations + current)│
        └─────────────────────────────────────────────────────────────┘

        ┌─ Instruction embedding (256-dim)
        │
        ├─ Demo 1 Objects (multiple 256-dim tokens)
        │ └─ Demo 1 Action (256-dim, e.g., [x_bin=45, y_bin=32, ...])
        │
        ├─ Demo 2 Objects (multiple 256-dim tokens)  
        │ └─ Demo 2 Action (256-dim)
        │
        └─ Current Scene Objects (multiple 256-dim tokens)
                    ↓
          Total: ~16 tokens of 256-dims each
          Sequence: (16, 256)

STAGE 4: TRAINABLE REASONING (Policy network)
═══════════════════════════════════════════════

    ┌──────────────────────────────────────┐
    │  Transformer (PolicyTransformer)     │ ← ALREADY DONE
    │  ∘ Self-attention over all tokens    │
    │  ∘ Learns what to attend to          │
    │  ∘ Produces decision vector (512-dim)│
    └──────────────────────────────────────┘
                    ↓
            Decision Vector
                  (512-dim)

STAGE 5: ACTION GENERATION (Output heads)
═══════════════════════════════════════════

    ┌──────────────────────────────────────┐
    │  Output Head 1: x-position (101 bins)│
    │  Output Head 2: y-position (101 bins)│
    │  Output Head 3: z-position (101 bins)│
    │  Output Head 4: rx-rotation(121 bins)│
    │  Output Head 5: ry-rotation(121 bins)│
    │  Output Head 6: rz-rotation(121 bins)│
    │  Output Head 7: gripper (2 bins)     │
    └──────────────────────────────────────┘
                    ↓
            Final 7D Action
        [45, 32, 78, 119, 55, 98, 1]
        ← ALREADY DONE
```

---

## Component Status Dashboard

```
┌──────────────────────┬────────┬──────────┬────────────────────┐
│ Component            │ Status │ Priority │ File               │
├──────────────────────┼────────┼──────────┼────────────────────┤
│ PREPROCESSING        │        │          │                    │
│  ├─ Object Detector  │   ❌   │ CRITICAL │ object_detection   │
│  ├─ Depth Estimator  │   ❌   │ CRITICAL │ depth_estimation   │
│  ├─ 3D Fusion        │   ❌   │ CRITICAL │ fusion_utils (new) │
│  └─ Object Encoder   │   ❌   │ HIGH     │ object_encoder(new)│
├──────────────────────┼────────┼──────────┼────────────────────┤
│ FROZEN ENCODERS      │        │          │                    │
│  ├─ Text (BERT)      │   ✅   │ DONE     │ text_encoder       │
│  └─ Vision (ResNet)  │   ✅   │ DONE     │ vision_encoder     │
├──────────────────────┼────────┼──────────┼────────────────────┤
│ SEQUENCE BUILDING    │        │          │                    │
│  └─ Seq Builder      │   ❌   │ HIGH     │ sequence_builder   │
│                      │        │          │ (new)              │
├──────────────────────┼────────┼──────────┼────────────────────┤
│ REASONING            │        │          │                    │
│  └─ Policy Transformer│  ✅   │ DONE     │ policy_transformer │
├──────────────────────┼────────┼──────────┼────────────────────┤
│ ACTION GENERATION    │        │          │                    │
│  └─ Output Heads     │   ✅   │ DONE     │ output_heads       │
├──────────────────────┼────────┼──────────┼────────────────────┤
│ DATA PIPELINE        │   ⚠️   │ HIGH     │ dataloader.py      │
│  ├─ collate_fn       │   ⚠️   │ HIGH     │ (needs update)     │
│  └─ AgentModel       │   ⚠️   │ HIGH     │ (needs update)     │
└──────────────────────┴────────┴──────────┴────────────────────┘

Legend:
  ✅ = Complete and working
  ⚠️  = Partial/needs update
  ❌  = Missing or broken
```

---

## Information Flow Diagram

```
INPUT SPACE (Raw sensory data)
┌─────────────────────────────────────────────────────────────┐
│ Images: 480×640×3 RGB pixels                               │
│ Text: "Stack the star on the cube"                         │
│ Actions: [x_bin, y_bin, z_bin, rx_bin, ry_bin, rz_bin, grip]
└─────────────────────────────────────────────────────────────┘
  ↓         ↓              ↓
  │         │              └─ Extract target action
  │         │
  │         └─ ObjectDetector (YOLOv8)
  │             ↓
  │             Lists objects: [{x, y, w, h, conf, cls}, ...]
  │             Removes clutter, finds structure
  │
  └─ DepthEstimator (MiDaS)
      ↓
      Depth map: 480×640 array with distances
      Answers: "How far is each object?"

FEATURE SPACE (Learned representations)
┌─────────────────────────────────────────────────────────────┐
│ 3D Object Representations:                                  │
│   (center_x, center_y, depth, width, height, conf, cls_id) │
│ ← Combines what (detection) + where (2D+depth) + why (class)
│                                                              │
│ Object Embeddings: 256-dimensional vectors                 │
│ ← Learned by ObjectEncoder to be useful for the task        │
│                                                              │
│ Text Embedding: 768-dimensional (from BERT)                │
│ ← Semantic meaning of instruction                           │
│                                                              │
│ Sequence: 16 tokens × 256-dim                              │
│ ← Rich multimodal context with demonstrations              │
└─────────────────────────────────────────────────────────────┘
  ↓
  PolicyTransformer (4 layers, 8 heads)
  ↓ (learns attention patterns)
  ↓
  Decision Vector: 512-dimensional
  ← "What should the robot do?"

DECISION SPACE (Discretized actions)
┌─────────────────────────────────────────────────────────────┐
│ Output Heads classify each dimension independently:         │
│   x-position:  bin 45 out of 101  (→ ~44.6% of x range)   │
│   y-position:  bin 32 out of 101  (→ ~31.7% of y range)   │
│   z-position:  bin 78 out of 101  (→ ~77.2% of z range)   │
│   rx-rotation: bin 119 out of 121 (→ ~98.3% of rx range)  │
│   ry-rotation: bin 55 out of 121  (→ ~45.5% of ry range)  │
│   rz-rotation: bin 98 out of 121  (→ ~81.0% of rz range)  │
│   gripper:     bin 1 out of 2     (→ Open/Close)          │
└─────────────────────────────────────────────────────────────┘
  ↓
ACTION EXECUTION
Robot executes: "Move to (x=44.6%, y=31.7%, z=77.2%), rotate and close gripper"
```

---

## Data Flow Through Training

```
BATCH LOADING
═════════════
5 examples
    ↓
collate_fn processes each:

  Example 1:
    Image 1 (demo) ──→ Detect 5 objects ──→ Get 3D coords ──→ [5, 7]
    Image 2 (demo) ──→ Detect 4 objects ──→ Get 3D coords ──→ [4, 7]
    Image 3 (curr) ──→ Detect 6 objects ──→ Get 3D coords ──→ [6, 7]
    
    Action target: [45, 32, 78, 119, 55, 98, 1]

  Example 2, 3, 4, 5: (same process)
    ↓
    Pad all to max objects (6) within batch:
    [5, 7] ──pad──→ [6, 7]
    [4, 7] ──pad──→ [6, 7]
    etc.

BATCHED DATA
═════════════
{
  'instructions': ["Stack star...", "Place cube...", ...],  (B, text)
  'demo_3d_objs': [
    [tensor(6,7), tensor(6,7)],  # Example 1: 2 demos, 6 objs each
    [tensor(6,7), tensor(6,7)],  # Example 2
    ...
  ],
  'current_3d_objs': [tensor(6,7), tensor(6,7), ...],  # (B, 6, 7)
  'targets': [
    [45, 32, 78, 119, 55, 98, 1],
    [50, 35, 80, 115, 52, 96, 0],
    ...
  ],  # (B, 7)
}

FORWARD PASS
═════════════
Step 1: Encode instruction (frozen BERT)
  instructions → BERT → [instr_embed_1, ..., instr_embed_B]
                         Shape: (B, 768)

Step 2: Encode 3D objects (trainable)
  objects_3d → ObjectEncoder → object_embeddings
               Shape: (B, max_objs, 256) per demo

Step 3: Build sequence (trainable)
  instruction + demo_objs + demo_actions + current_objs
    → MultimodalSequenceBuilder
    → [tokens_1, tokens_2, ...]
       Each: (seq_len_i, 256)
       After padding: (B, 16, 256)

Step 4: Policy reasoning (trainable)
  tokens → PolicyTransformer → decision
           Shape: (B, 512)

Step 5: Action prediction (trainable)
  decision → 7 Output Heads → logits
             [(B, 101), (B, 101), ..., (B, 2)]

LOSS & BACKPROP
═══════════════
Loss = Average CrossEntropy over 7 dimensions
       targets are the ground truth action bins

Gradient flows:
  Loss ← Output Heads ← Transformer ← Seq Builder ← Obj Encoder
                                       ↖ Frozen BERT (no grad)
                                       ↖ Frozen ResNet (no grad)

Update weights:
  Object Encoder, Seq Builder, Transformer, Output Heads
  (Frozen components not updated)
```

---

## Key Insights

### 1️⃣ Why 3D Representations?
```
❌ Image embedding: "Here's a 512-dim vector"
   Problem: Where is the star? Which pixel? Unknown!

✅ 3D object representation: "Star is at (0.35, 0.45, 0.8)"
   Benefit: Robot can explicitly reason about location!
```

### 2️⃣ Why Multiple Objects?
```
❌ Single image embedding: "Here's global visual feature"
   Problem: Can't separate star from cube

✅ Per-object embeddings: [star_embed, cube_embed, background_embed]
   Benefit: Transformer learns: "attend to star for stacking task"
```

### 3️⃣ Why Include Demo Actions?
```
❌ Demo objects only: "Previous scene had star and cube"
   Problem: What was the robot doing?

✅ Demo objects + action: "Star was here, robot moved to star position"
   Benefit: In-context learning: "Current star here, move there too!"
```

### 4️⃣ Why Frozen Encoders?
```
Frozen BERT: Language understanding is stable, general
Frozen ResNet/ViT: Visual understanding is stable, general
Trainable Transformer: Task-specific policy learning

Benefits:
  ✓ Use pre-trained knowledge
  ✓ Reduce overfitting
  ✓ Faster training
  ✓ Better generalization
```

---

## Success Indicators

After implementation, look for:

```
✅ Training Loss Curve
   Epoch 1:  Loss ~2.0
   Epoch 5:  Loss ~1.2
   Epoch 10: Loss ~0.8
   Epoch 20: Loss ~0.5
   
   → Smooth decrease = Good!
   → Erratic/NaN = Debug!

✅ Validation Accuracy
   Epoch 1:  ~15% (near random for 7D)
   Epoch 5:  ~30%
   Epoch 10: ~45%
   Epoch 20: ~60-70%
   
   → Each dimension improves = Good!

✅ Attention Visualization
   - Attention focuses on relevant objects
   - Weights highest for demonstrations
   - Task-relevant objects get more attention
   
   → Makes intuitive sense = Good!

✅ Action Quality
   - Predicted actions place hand near objects
   - Gripper action reasonable (open/close)
   - Motion smooth across frames
   
   → Robot can follow instructions = Good!
```

---

## Common Questions Answered

**Q: Why 7 dimensions for action?**
A: 3D position (x,y,z) + 3D rotation (rx,ry,rz) + 1 gripper = 7D continuous space
   Discretized into bins for classification (easier than regression)

**Q: Why freeze text/vision encoders?**
A: Pre-trained on huge datasets. More data = better features.
   Freezing saves GPU memory and prevents overfitting.

**Q: Why normalize 3D coordinates?**
A: Scenes vary in size/scale. Normalizing (0-1 range) makes model generalize.
   E.g., different table sizes won't confuse the model.

**Q: Why include demo actions in sequence?**
A: In-context learning! Model learns: "When object is here, move gripper there"
   Can apply pattern to current scene even if positions differ.

**Q: Why not just use image pixels?**
A: Paper showed: MLLMs struggle with precise 3D manipulation.
   Explicit 3D representations help model understand spatial relationships.

---

## You're Ready!

```
📊 Status: 60% Complete
🎯 Effort: ~14-18 hours remaining
💡 Complexity: Medium (no PhD needed!)
✨ Impact: High (significantly improves model performance)

Next Step: Start with IMPLEMENTATION_ROADMAP.md Day 1
            Object Detection + Depth Estimation
```

Good luck! Your architecture is solid. 🚀
