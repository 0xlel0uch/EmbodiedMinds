# Loss Function and Predictions Documentation

## Loss Function

### Type: **CrossEntropyLoss** (Multi-class Classification)

The model uses **CrossEntropyLoss** for each of the 7 action dimensions independently.

### Implementation Details

**Location**: `src/heads/output_heads.py`

```python
def loss(self, logits, targets):
    # logits: list of (B, bins_i), targets: (B,7) longs
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    total = 0.0
    for i, log in enumerate(logits):
        target_dim = targets[:, i].clone()
        # Clamp targets to valid range [0, bins[i]-1] or keep -1 for invalid
        valid_mask = (target_dim >= 0) & (target_dim < self.bins[i])
        target_dim[~valid_mask] = -1  # Mark out-of-range targets as invalid
        total = total + loss_fn(log, target_dim)
    return total / len(logits)
```

### Key Features

1. **Per-Dimension Loss**: Each of the 7 action dimensions (x, y, z, roll, pitch, yaw, gripper) has its own classification head
2. **Ignore Invalid Targets**: Uses `ignore_index=-1` to ignore invalid/missing targets
3. **Target Validation**: Automatically clamps out-of-range targets to -1 (invalid)
4. **Averaged Loss**: Returns the average loss across all 7 dimensions

### Action Dimensions and Bins

- **x, y, z**: 101 bins each (range: 0-100)
- **roll, pitch, yaw**: 121 bins each (range: 0-120)
- **gripper**: 2 bins (0=closed, 1=open)

### Loss Calculation

For each dimension `i`:
- **Input**: Logits of shape `(batch_size, bins[i])`
- **Target**: Target class index of shape `(batch_size,)`
- **Loss**: `CrossEntropyLoss(logits[i], targets[:, i])`
- **Final Loss**: Average of all 7 dimension losses

## Predictions Storage

### During Training

**Location**: `logs/predictions_epoch_{epoch}.npz`

Saved after each validation epoch:
- **predictions**: `(num_samples, 7)` numpy array - Predicted action values
- **targets**: `(num_samples, 7)` numpy array - Expected/ground truth action values
- **instructions**: List of instruction strings

**Format**: NumPy compressed format (.npz)

### During Evaluation

**Location**: `logs/evaluation_results_{split}.pt` or `logs/trajectory_evaluation_*.json`

Saved by evaluation scripts:
- **predictions**: `(num_samples, 7)` tensor
- **targets**: `(num_samples, 7)` tensor
- Additional metrics and analysis

## Viewing Predictions

### Method 1: Using view_predictions.py Script

```bash
# View latest predictions
python3 view_predictions.py

# View specific epoch
python3 view_predictions.py --epoch 5

# View specific file
python3 view_predictions.py --file logs/predictions_epoch_10.npz

# View more samples
python3 view_predictions.py --epoch 0 --num-samples 50
```

### Method 2: Direct Python Access

```python
import numpy as np

# Load predictions
data = np.load('logs/predictions_epoch_0.npz', allow_pickle=True)
predictions = data['predictions']  # (N, 7)
targets = data['targets']          # (N, 7)
instructions = data['instructions'] # List of strings

# View first sample
print(f"Instruction: {instructions[0]}")
print(f"Predicted: {predictions[0]}")
print(f"Expected:  {targets[0]}")
```

### Method 3: From Evaluation Results

```python
import torch

# Load evaluation results
results = torch.load('logs/evaluation_results_val.pt')
predictions = results['predictions'].numpy()  # (N, 7)
targets = results['targets'].numpy()          # (N, 7)

# Compare
for i in range(10):
    print(f"Sample {i}:")
    print(f"  Predicted: {predictions[i]}")
    print(f"  Expected:  {targets[i]}")
    print(f"  Match: {(predictions[i] == targets[i]).all()}")
```

## Prediction Format

### Action Vector (7 dimensions)

Each prediction/target is a 7-dimensional vector:

```python
[x, y, z, roll, pitch, yaw, gripper]
```

**Example**:
```python
predicted = [45, 32, 78, 119, 55, 98, 1]
expected  = [45, 32, 78, 120, 55, 98, 1]
#           ✓   ✓   ✓   ✗    ✓   ✓   ✓
#           All match except roll (predicted 119, expected 120)
```

### Dimension Meanings

1. **x, y, z** (0-100): 3D position coordinates
   - Normalized to [0, 100] range
   - Each bin represents ~1% of the workspace

2. **roll, pitch, yaw** (0-120): Rotation angles
   - Each bin represents 3 degrees (120 bins × 3° = 360°)
   - Normalized to [0, 120] range

3. **gripper** (0-1): Gripper state
   - 0 = Closed
   - 1 = Open

## Analysis Tools

### Calculate Accuracy Per Dimension

```python
import numpy as np

data = np.load('logs/predictions_epoch_0.npz')
predictions = data['predictions']
targets = data['targets']

dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
mask = (targets != -1)

for i, name in enumerate(dim_names):
    valid = mask[:, i]
    if valid.sum() > 0:
        correct = (predictions[valid, i] == targets[valid, i]).sum()
        accuracy = correct / valid.sum()
        print(f"{name}: {accuracy:.2%}")
```

### Calculate Mean Absolute Error

```python
errors = np.abs(predictions - targets)
mask = (targets != -1)
errors[~mask] = 0  # Ignore invalid targets

for i, name in enumerate(dim_names):
    valid = mask[:, i]
    if valid.sum() > 0:
        mae = errors[valid, i].mean()
        print(f"{name} MAE: {mae:.2f}")
```

### Find Worst Predictions

```python
# Calculate per-sample accuracy
sample_accuracy = (predictions == targets).all(axis=1)
worst_indices = np.argsort(sample_accuracy)[:10]  # 10 worst

for idx in worst_indices:
    print(f"Sample {idx}:")
    print(f"  Instruction: {instructions[idx]}")
    print(f"  Predicted: {predictions[idx]}")
    print(f"  Expected:  {targets[idx]}")
    print()
```

## Files Generated

### Training Predictions
- `logs/predictions_epoch_0.npz`
- `logs/predictions_epoch_1.npz`
- ...
- `logs/predictions_epoch_{N}.npz`

### Evaluation Predictions
- `logs/evaluation_results_val.pt`
- `logs/trajectory_evaluation_*.json`

### Metrics
- `logs/training_metrics_epoch_{N}.json` - Contains accuracy metrics
- `logs/metrics_*.json` - Comprehensive task-level metrics

## Quick Access Commands

```bash
# View latest predictions
python3 view_predictions.py

# View specific epoch
python3 view_predictions.py --epoch 5

# View from EC2
ssh -i key.pem ec2-user@3.139.95.113
cd ~/EmbodiedMinds
python3 view_predictions.py --epoch 0

# Download predictions locally
scp -i key.pem ec2-user@3.139.95.113:~/EmbodiedMinds/logs/predictions_epoch_*.npz ./
```

