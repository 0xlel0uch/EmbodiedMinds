# Metrics Checklist - Verification

## Requested Metrics

You asked for these 5 metrics:
1. ✅ **Task Success Rate**: The percentage of tasks where the agent successfully completes the full instruction.
2. ✅ **Subgoal Success Rate**: For high-level tasks, the fraction of intermediate goals completed correctly.
3. ✅ **Planner Steps**: The number of model inferences (planning calls) required to produce a complete executable plan.
4. ✅ **Environment Steps**: The number of interactions the agent performs within the environment while executing a task.
5. ✅ **Error Analysis**: Qualitative categorization of failures (e.g., perception errors, reasoning errors, or planning errors).

## Current Status

### ✅ All 5 Metrics ARE Being Tracked

**Location**: Saved by `evaluate_trajectory.py` after training completes

**Files Generated**:
1. `logs/metrics_YYYYMMDD_HHMMSS.json` - Contains ALL 5 metrics
2. `logs/metrics_summary_YYYYMMDD_HHMMSS.csv` - Summary with ALL 5 metrics
3. `logs/trajectory_evaluation_*.json` - Detailed evaluation results

### When Are They Saved?

1. **During Training**: Basic accuracy metrics only (per-dimension accuracy)
   - Saved to: `logs/training_metrics_epoch_*.json`
   - Does NOT include the 5 comprehensive metrics

2. **After Training**: ALL 5 comprehensive metrics
   - Saved automatically by `train_with_metrics.sh`
   - Runs `evaluate_trajectory.py` which generates:
     - Task Success Rate
     - Subgoal Success Rate
     - Planner Steps
     - Environment Steps
     - Error Analysis

## Verification

### Check if Metrics Are Being Saved

```bash
# On EC2
ssh -i key.pem ec2-user@3.139.95.113
cd ~/EmbodiedMinds

# Check if comprehensive metrics exist
ls -la logs/metrics_*.json
ls -la logs/metrics_summary_*.csv

# View summary
cat logs/metrics_summary_*.csv
```

### View All 5 Metrics

```bash
# View JSON with all metrics
python3 << 'EOF'
import json
import glob

files = glob.glob('logs/metrics_*.json')
if files:
    with open(files[-1]) as f:
        data = json.load(f)
        summary = data['summary']
        print("="*60)
        print("COMPREHENSIVE METRICS SUMMARY")
        print("="*60)
        print(f"1. Task Success Rate: {summary['task_success_rate']:.2%}")
        print(f"2. Subgoal Success Rate: {summary['avg_subgoal_success_rate']:.2%}")
        print(f"3. Avg Planner Steps: {summary['avg_planner_steps']:.2f}")
        print(f"4. Avg Environment Steps: {summary['avg_environment_steps']:.2f}")
        print(f"5. Error Analysis:")
        for error_type, count in summary['error_analysis'].items():
            print(f"   - {error_type}: {count}")
        print("="*60)
else:
    print("No metrics files found yet. Metrics will be saved after training completes.")
EOF
```

## Metrics Storage Locations

### 1. Task Success Rate
- **File**: `logs/metrics_*.json` → `summary.task_success_rate`
- **CSV**: `logs/metrics_summary_*.csv` → "Task Success Rate" row
- **S3**: `s3://11777-h1/metrics/metrics_*.json`

### 2. Subgoal Success Rate
- **File**: `logs/metrics_*.json` → `summary.avg_subgoal_success_rate`
- **CSV**: `logs/metrics_summary_*.csv` → "Avg Subgoal Success Rate" row
- **S3**: `s3://11777-h1/metrics/metrics_*.json`

### 3. Planner Steps
- **File**: `logs/metrics_*.json` → `summary.avg_planner_steps`
- **CSV**: `logs/metrics_summary_*.csv` → "Avg Planner Steps" row
- **S3**: `s3://11777-h1/metrics/metrics_*.json`

### 4. Environment Steps
- **File**: `logs/metrics_*.json` → `summary.avg_environment_steps`
- **CSV**: `logs/metrics_summary_*.csv` → "Avg Environment Steps" row
- **S3**: `s3://11777-h1/metrics/metrics_*.json`

### 5. Error Analysis
- **File**: `logs/metrics_*.json` → `summary.error_analysis`
- **CSV**: `logs/metrics_summary_*.csv` → "Error Type" and "Count" columns
- **S3**: `s3://11777-h1/metrics/metrics_*.json`

## What's Currently Saved During Training

### Per-Epoch Training Metrics
- **File**: `logs/training_metrics_epoch_{N}.json`
- **Contains**:
  - Train loss
  - Validation loss
  - Per-dimension accuracy (x, y, z, roll, pitch, yaw, gripper)
  - Average accuracy
  - **Does NOT include**: Task Success Rate, Subgoal Success Rate, Planner Steps, Environment Steps, Error Analysis

### Predictions vs Targets
- **File**: `logs/predictions_epoch_{N}.npz`
- **Contains**:
  - Predictions array
  - Targets array
  - Instructions

## What's Saved After Training (Comprehensive Metrics)

### Comprehensive Task Metrics
- **File**: `logs/metrics_YYYYMMDD_HHMMSS.json`
- **Contains ALL 5 metrics**:
  1. ✅ Task Success Rate
  2. ✅ Subgoal Success Rate
  3. ✅ Planner Steps
  4. ✅ Environment Steps
  5. ✅ Error Analysis

### Summary CSV
- **File**: `logs/metrics_summary_YYYYMMDD_HHMMSS.csv`
- **Contains ALL 5 metrics in CSV format**

## Automatic Saving

The `train_with_metrics.sh` script automatically:
1. Trains the model
2. Runs comprehensive evaluation (`evaluate_trajectory.py`)
3. Saves ALL 5 metrics to JSON and CSV
4. Uploads to S3

**No manual action needed** - all metrics are saved automatically after training completes.

## Quick Access After Training

```bash
# View all 5 metrics summary
cat logs/metrics_summary_*.csv

# View detailed metrics
python3 view_predictions.py  # Shows predictions vs targets
cat logs/metrics_*.json | python3 -m json.tool | less  # View full metrics
```

## Summary

✅ **All 5 requested metrics ARE being tracked and saved**
✅ **Saved automatically after training completes**
✅ **Available in both JSON and CSV formats**
✅ **Uploaded to S3 automatically**

**Note**: The comprehensive metrics are calculated during trajectory evaluation (after training), not during training validation. This is because they require full task trajectory evaluation, not just single-step predictions.

