# AWS EC2 Setup Complete ✅

## Instance Details

- **Public IP:** 3.17.26.129
- **Username:** ec2-user
- **Key File:** /Users/abhivakil/Desktop/11777.pem
- **GPU:** NVIDIA A10G (23GB)
- **CUDA:** 12.1
- **PyTorch:** 2.5.1+cu121

## What Was Set Up

### ✅ Repository
- Cloned `EmbodiedMinds` repository
- Checked out `Abhi_vakil_completed_code` branch
- All code files present

### ✅ Dependencies
- PyTorch with CUDA 12.1 support
- YOLOv8 (ultralytics)
- MiDaS support (timm)
- Transformers (BERT)
- AWS SDK (boto3, s3fs)
- TensorBoard
- All other required packages

### ✅ Project Structure
- `src/` - All source code
- `checkpoints/` - For model checkpoints
- `data/` - For training data
- `logs/` - For training logs

### ✅ Verification
- GPU detected and accessible
- CUDA working correctly
- All project modules import successfully
- No circular import issues

## Quick Commands

### Connect to Instance
```bash
ssh -i /Users/abhivakil/Desktop/11777.pem ec2-user@3.17.26.129
```

### Navigate to Project
```bash
cd ~/EmbodiedMinds
```

### Check GPU
```bash
nvidia-smi
```

### Verify Setup
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Next Steps

### 1. Upload Data to S3 (if not done)
```bash
# From your local machine
aws s3 sync ./data/train s3://embodied-minds-data/raw/train/
aws s3 sync ./data/val s3://embodied-minds-data/raw/val/
```

### 2. Download Data to EC2
```bash
# SSH into instance first
ssh -i /Users/abhivakil/Desktop/11777.pem ec2-user@3.17.26.129

# Then download data
cd ~/EmbodiedMinds
aws s3 sync s3://embodied-minds-data/raw/train/ ./data/train/
aws s3 sync s3://embodied-minds-data/raw/val/ ./data/val/
```

### 3. Start Training
```bash
# Option 1: Direct training
python3 train_aws.py \
    --data-root ./data \
    --batch-size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --s3-checkpoint-bucket embodied-minds-checkpoints

# Option 2: Using screen (recommended for long training)
screen -S training
python3 train_aws.py --batch-size 8 --epochs 50
# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

### 4. Monitor Training
```bash
# View logs
tail -f training.log

# Monitor GPU
watch -n 1 nvidia-smi

# Start TensorBoard
tensorboard --logdir=./logs --port=6006 --host=0.0.0.0
# Access via SSH tunnel: ssh -i key.pem -L 6006:localhost:6006 ec2-user@3.17.26.129
```

### 5. Evaluate Model
```bash
# Download checkpoint from S3
aws s3 cp s3://embodied-minds-checkpoints/models/agent_epoch50.pt ./checkpoints/

# Evaluate
python3 evaluate_aws.py \
    --checkpoint ./checkpoints/agent_epoch50.pt \
    --data-root ./data \
    --split val
```

## Important Notes

1. **Data Location:** Make sure your data is in `~/EmbodiedMinds/data/` or update `--data-root` path
2. **S3 Buckets:** Create these buckets if not already created:
   - `embodied-minds-data`
   - `embodied-minds-checkpoints`
   - `embodied-minds-results`
3. **IAM Permissions:** Ensure EC2 instance has IAM role with S3 read/write permissions
4. **Cost:** Instance is running and incurring charges (~$1/hour for g5.xlarge)
5. **Persistent Sessions:** Use `screen` or `tmux` for long-running training jobs

## Troubleshooting

### If imports fail:
```bash
# Make sure you're in the project directory
cd ~/EmbodiedMinds

# Verify Python path
python3 -c "import sys; print(sys.path)"
```

### If GPU not detected:
```bash
# Check NVIDIA driver
nvidia-smi

# Verify CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### If out of memory:
```bash
# Reduce batch size
python3 train_aws.py --batch-size 4  # Instead of 8
```

## Status

✅ **Setup Complete** - Ready for training!

All components are installed and verified. You can now proceed with data preparation and training.

