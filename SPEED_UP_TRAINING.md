# Speed Up Training Guide

## Current Training Time
- **Per epoch**: ~30-40 minutes
- **50 epochs**: ~25-33 hours
- **With early stopping (patience=5)**: ~2-5 hours (typically stops after 10-15 epochs)

## Options to Speed Up Training

### 1. Early Stopping (✅ Already Implemented)
Early stopping will automatically stop training when validation loss stops improving.

**Current settings:**
- Patience: 5 epochs
- Min delta: 0.001

**To adjust:**
```bash
# More aggressive (stops faster)
--early-stopping-patience 3

# Less aggressive (trains longer)
--early-stopping-patience 10
```

### 2. Reduce Number of Epochs
```bash
# Train for fewer epochs
--epochs 20  # Instead of 50
```

### 3. Increase Batch Size (Faster, but needs more GPU memory)
```bash
# Increase batch size (if GPU memory allows)
--batch-size 16  # Instead of 8
```

**Note**: You'll need to check GPU memory first:
```bash
nvidia-smi
```

### 4. Reduce Dataset Size (For faster iteration)
Edit `train_with_metrics.sh`:
```bash
# Use smaller subset for faster training
BATCH_SIZE=16  # Larger batches = faster
EPOCHS=20      # Fewer epochs
```

### 5. Use Mixed Precision Training (Future enhancement)
Would require code changes but could speed up by 2x.

### 6. Upgrade GPU (If budget allows)
- Current: NVIDIA A10G (23.7 GB)
- Options:
  - A100 (40GB or 80GB) - 2-3x faster
  - Multiple GPUs - Parallel training

## Recommended Quick Settings

For fastest training with good results:

```bash
# In train_with_metrics.sh, change:
BATCH_SIZE=16
EPOCHS=30
EARLY_STOPPING_PATIENCE=5
```

**Expected time**: ~3-6 hours (with early stopping typically stopping around epoch 10-15)

## Monitor Training Speed

Check current speed:
```bash
ssh -i key.pem ec2-user@3.139.95.113
tail -f ~/EmbodiedMinds/logs/training.log | grep "it/s"
```

## Early Stopping Status

Early stopping is **enabled by default** and will:
- Monitor validation loss
- Stop if no improvement for 5 epochs
- Save best checkpoint automatically
- Typically stops after 10-15 epochs (2-5 hours)

To disable early stopping:
```bash
--no-early-stopping
```

