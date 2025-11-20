# Quick Reference Card

Print this or keep it open while implementing!

---

## 🎯 Your Mission

Transform:
```
❌ Image → Global Feature → Guess Action
→ 
✅ Image → 3D Objects → Precise Spatial Reasoning → Accurate Action
```

---

## 📊 Component Checklist

```
IMPLEMENT IN THIS ORDER:
┌─────────────────────────────────────────────────┐
│ 1. ObjectDetector (YOLOv8)       [ ] 2-3 hrs   │
│    Input: Image  |  Output: Objects with boxes  │
├─────────────────────────────────────────────────┤
│ 2. DepthEstimator (MiDaS)        [ ] 2-3 hrs   │
│    Input: Image  |  Output: Depth map (0-1)    │
├─────────────────────────────────────────────────┤
│ 3. create_3d_representations()   [ ] 1-2 hrs   │
│    Input: Objects+Depth | Output: (N,7) tensor │
├─────────────────────────────────────────────────┤
│ 4. ObjectEncoder                 [ ] 1 hr      │
│    Input: (N,7)  |  Output: (N,256) embeddings │
├─────────────────────────────────────────────────┤
│ 5. MultimodalSequenceBuilder     [ ] 2-3 hrs   │
│    Input: All components | Output: (B,~16,256) │
├─────────────────────────────────────────────────┤
│ 6. Update Data Pipeline          [ ] 2-3 hrs   │
│    Integrate preprocessing + AgentModel update  │
├─────────────────────────────────────────────────┤
│ 7. Validate & Test               [ ] 2-4 hrs   │
│    Unit tests + Integration tests + Training   │
└─────────────────────────────────────────────────┘
```

---

## 🐍 Code Snippets You'll Need

### ObjectDetector
```python
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt", device="cpu"):
        self.model = YOLO(model_name)
        self.device = device
        self.model.to(device)
        
    def detect_objects(self, image, conf_threshold=0.5):
        results = self.model(image, conf=conf_threshold, verbose=False)
        # Return list of dicts with 'box', 'center', 'confidence', 'class_id'
```

### DepthEstimator
```python
import torch

class DepthEstimator:
    def __init__(self, model_type="DPT_Large", device="cpu"):
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        
    def estimate_depth(self, image):
        # Return normalized (0-1) depth map
```

### 3D Fusion
```python
def create_3d_object_representations(objects, depth_map, h, w):
    representations = []
    for obj in objects:
        x1, y1, x2, y2 = obj['box']
        cx, cy = obj['center']
        z = np.mean(depth_map[...])  # Sample depth at bbox
        w_norm = x2 - x1
        h_norm = y2 - y1
        representations.append([cx, cy, z, w_norm, h_norm, 
                               obj['confidence'], obj['class_id']])
    return torch.tensor(representations)
```

### ObjectEncoder
```python
import torch.nn as nn

class ObjectEncoder(nn.Module):
    def __init__(self, in_dim=7, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )
    
    def forward(self, x):
        return self.net(x)
```

---

## 📁 Files to Create/Modify

| File | Action | Time |
|------|--------|------|
| `src/preprocessing/object_detection.py` | Replace | 2-3h |
| `src/preprocessing/depth_estimation.py` | Replace | 2-3h |
| `src/preprocessing/fusion_utils.py` | Create | 1-2h |
| `src/encoders/object_encoder.py` | Create | 1h |
| `src/fusion/sequence_builder.py` | Create | 2-3h |
| `src/datasets/dataloader.py` | Update | 2-3h |
| **TOTAL** | | **12-17h** |

---

## ✅ Testing Checklist

After each component:

```
ObjectDetector:
  [ ] Finds 3-5 objects in test image
  [ ] Confidence scores 0.5-0.95
  [ ] Boxes normalized (0-1)
  
DepthEstimator:
  [ ] Output range 0-1
  [ ] Reasonable spatial structure
  [ ] No NaN values
  
3D Fusion:
  [ ] Output shape (N, 7)
  [ ] All values 0-1
  [ ] No NaN values
  
ObjectEncoder:
  [ ] Input (N, 7) → Output (N, 256)
  [ ] No gradient issues
  [ ] Trainable weights
  
SequenceBuilder:
  [ ] Constructs (B, seq_len, 256)
  [ ] Handles variable objects
  [ ] No shape mismatches
  
Data Pipeline:
  [ ] collate_fn returns correct dict
  [ ] Preprocessing runs
  [ ] AgentModel forward pass works
  
Training:
  [ ] No NaN loss
  [ ] Loss decreases
  [ ] No OOM errors
```

---

## 🐛 Debugging Quick Fixes

| Problem | Solution |
|---------|----------|
| No objects detected | Try larger YOLOv8: "yolov8m.pt" |
| Depth values wrong range | Check normalization: `(d - min) / (max - min)` |
| Shape mismatch | Print shapes at each step, check padding |
| Training diverges (NaN) | Reduce learning rate, check coord normalization |
| Out of memory | Reduce batch_size or run on CPU |
| Attention weird | Verify sequence structure is correct |

---

## 📈 Success Metrics by Epoch

```
Epoch 1:   Loss ~2.0   Accuracy ~15%  (Random baseline)
Epoch 5:   Loss ~1.2   Accuracy ~30%  ✓ Getting better
Epoch 10:  Loss ~0.8   Accuracy ~45%  ✓ Good progress
Epoch 20:  Loss ~0.5   Accuracy ~65%  ✓ Converging
Epoch 50:  Loss ~0.4   Accuracy ~75%  ✓ Excellent
```

---

## 🔑 Key Insights

| What | Why | How |
|------|-----|-----|
| 3D coords | Model needs to know WHERE | Combine 2D detection + depth |
| Per-object | Can't squeeze all info into 1 vector | Encode each object separately |
| Demo actions | Enable in-context learning | Include action tokens in sequence |
| Frozen encoders | Leverage pre-trained knowledge | Keep BERT & Vision encoder frozen |
| Token sequence | Transformer needs structured input | [instr] + [objs+action]×N + [objs] |

---

## 📚 Documentation Map

```
START HERE!
    ↓
START_HERE.md (5 min)
    ↓
EXECUTIVE_SUMMARY.md (10 min)
    ↓
FILE_IMPLEMENTATION_GUIDE.md (10 min)
    ↓
IMPLEMENTATION_TEMPLATES.py (reference while coding)
    ↓
IMPLEMENTATION_ROADMAP.md (if you need detailed schedule)
    ↓
ARCHITECTURE_VISUAL_SUMMARY.md (if confused)
    ↓
ARCHITECTURE_IMPLEMENTATION_GUIDE.md (deep reference)
    ↓
CURRENT_VS_PROPOSED.md (if questioning design)
```

---

## 💡 Pro Tips

1. **Test each component alone first**
   Don't integrate until each works independently

2. **Use debug=True in dataloader**
   Load small debug dataset for quick testing

3. **Print shapes obsessively**
   Most bugs are shape mismatches

4. **Visualize 3D representations**
   Plot bboxes on images to verify correctness

5. **Cache preprocessing results**
   Save object detections + depth to disk for faster iteration

6. **Use Git frequently**
   Commit after each working component

7. **Monitor GPU memory**
   Use `nvidia-smi` to watch memory usage

---

## 🚀 Your Timeline

```
DAY 1 (6-8 hours)
├─ Morning: Preprocessing pipeline
│  ├─ ObjectDetector (2h)
│  ├─ DepthEstimator (2h)
│  └─ 3D Fusion (1h)
└─ Afternoon: Encoding
   ├─ ObjectEncoder (1h)
   └─ Testing (1-2h)

DAY 2 (6-8 hours)
├─ Morning: Sequence building
│  ├─ SequenceBuilder (2-3h)
│  └─ Testing (1h)
└─ Afternoon: Integration
   ├─ Update Data Pipeline (2-3h)
   └─ AgentModel integration (1-2h)

DAY 3 (2-4 hours)
├─ Unit tests (1h)
├─ Integration tests (1h)
└─ Training validation (1-2h)
```

---

## 📞 Quick Help

**Stuck on X?** Check:
- ObjectDetector → `IMPLEMENTATION_TEMPLATES.py` line 10-75
- DepthEstimator → `IMPLEMENTATION_TEMPLATES.py` line 85-155
- 3D Fusion → `IMPLEMENTATION_TEMPLATES.py` line 165-230
- ObjectEncoder → `IMPLEMENTATION_TEMPLATES.py` line 240-275
- SequenceBuilder → `ARCHITECTURE_IMPLEMENTATION_GUIDE.md` Phase 3
- Data Pipeline → `FILE_IMPLEMENTATION_GUIDE.md` Step 6
- Debugging → `IMPLEMENTATION_ROADMAP.md` Debugging section

---

## ✨ You've Got This!

```
┌──────────────────────────────────┐
│  Architecture: ✅ Solid          │
│  Codebase: ✅ 60% Complete       │
│  Documentation: ✅ Comprehensive │
│  Your ability: ✅ More than enough│
│                                  │
│  Result: 🚀 Success Incoming!    │
└──────────────────────────────────┘
```

---

**Next Step:** Open `START_HERE.md` and begin!

**Questions:** Check `DOCUMENTATION_INDEX.md`

**Reference:** Keep this card open while coding!

Good luck! 🎉
