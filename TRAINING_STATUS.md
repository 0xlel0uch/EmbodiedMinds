# Training Status

## Training Started

**Status**: ✅ Training is running in background

**Instance**: `3.139.95.113`  
**Session**: Screen session named "training"  
**Script**: `train_with_metrics.sh`

## Configuration

- **Data**: `./data/EB-Man_trajectory_dataset`
- **Batch Size**: 8
- **Epochs**: 50
- **Learning Rate**: 1e-4
- **Log Directory**: `./logs`
- **Checkpoint Directory**: `./checkpoints`

## Monitoring

### View Training Progress
```bash
ssh -i /path/to/key.pem ec2-user@3.139.95.113
cd ~/EmbodiedMinds
tail -f logs/training.log
```

### View Full Output
```bash
ssh -i /path/to/key.pem ec2-user@3.139.95.113
cd ~/EmbodiedMinds
tail -f training_output.log
```

### Attach to Screen Session
```bash
ssh -i /path/to/key.pem ec2-user@3.139.95.113
screen -r training
# Detach: Ctrl+A, then D
```

### Check GPU Usage
```bash
ssh -i /path/to/key.pem ec2-user@3.139.95.113
watch -n 1 nvidia-smi
```

### Check Training Process
```bash
ssh -i /path/to/key.pem ec2-user@3.139.95.113
ps aux | grep python3
```

## Metrics Tracking

### During Training
- Per-epoch metrics saved to: `logs/training_metrics_epoch_*.json`
- Training log: `logs/training.log`

### After Training
The script automatically runs comprehensive trajectory evaluation which generates:
- **Task Success Rate**: `logs/metrics_*.json` → `summary.task_success_rate`
- **Subgoal Success Rate**: `logs/metrics_*.json` → `summary.avg_subgoal_success_rate`
- **Planner Steps**: `logs/metrics_*.json` → `summary.avg_planner_steps`
- **Environment Steps**: `logs/metrics_*.json` → `summary.avg_environment_steps`
- **Error Analysis**: `logs/metrics_*.json` → `summary.error_analysis`

All metrics are also saved as CSV: `logs/metrics_summary_*.csv`

## Checkpoints

Checkpoints are saved after each epoch:
- Location: `checkpoints/agent_epoch{epoch}.pt`
- Latest checkpoint will be used for evaluation

## S3 Upload

Metrics are automatically uploaded to:
- `s3://11777-h1/metrics/`

## Estimated Time

- **Per Epoch**: ~10-30 minutes (depending on dataset size and GPU)
- **Total Training**: ~8-25 hours for 50 epochs
- **Evaluation**: ~30-60 minutes (100 episodes)

## Troubleshooting

### Training Stopped?
```bash
# Check if screen session exists
screen -ls

# Restart if needed
cd ~/EmbodiedMinds
./train_with_metrics.sh
```

### Out of Memory?
- Reduce batch size in `train_with_metrics.sh`
- Check GPU memory: `nvidia-smi`

### View Errors
```bash
tail -100 logs/training.log | grep -i error
```

## Next Steps

1. Monitor training progress
2. Wait for training to complete (50 epochs)
3. Evaluation will run automatically
4. Check metrics in `logs/` directory
5. Metrics will be uploaded to S3 automatically

