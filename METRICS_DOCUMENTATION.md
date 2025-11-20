# Metrics Tracking Documentation

## Overview

This document describes where and how metrics are stored during training and evaluation. The system tracks comprehensive task-level metrics including Task Success Rate, Subgoal Success Rate, Planner Steps, Environment Steps, and Error Analysis.

## Metrics Storage Locations

### 1. Local Storage (EC2 Instance)

All metrics are stored in the `logs/` directory by default:

```
~/EmbodiedMinds/
├── logs/
│   ├── training_metrics_epoch_*.json      # Per-epoch training metrics
│   ├── metrics_YYYYMMDD_HHMMSS.json       # Comprehensive task metrics
│   ├── metrics_summary_YYYYMMDD_HHMMSS.csv # Summary CSV for analysis
│   └── trajectory_evaluation_*.json       # Detailed trajectory evaluation
```

### 2. S3 Storage (Optional)

If configured, metrics are automatically uploaded to S3:

```
s3://11777-h1/
└── metrics/
    ├── metrics_YYYYMMDD_HHMMSS.json
    ├── metrics_summary_YYYYMMDD_HHMMSS.csv
    └── trajectory_evaluation_*.json
```

## Tracked Metrics

### 1. Task Success Rate
**Definition**: The percentage of tasks where the agent successfully completes the full instruction.

**Storage**:
- Location: `logs/metrics_*.json` → `summary.task_success_rate`
- CSV: `logs/metrics_summary_*.csv` → "Task Success Rate" row
- Format: Float (0.0 to 1.0)

**Example**:
```json
{
  "summary": {
    "task_success_rate": 0.75,
    "successful_tasks": 150,
    "failed_tasks": 50,
    "total_tasks": 200
  }
}
```

### 2. Subgoal Success Rate
**Definition**: For high-level tasks, the fraction of intermediate goals completed correctly.

**Storage**:
- Location: `logs/metrics_*.json` → `summary.avg_subgoal_success_rate`
- CSV: `logs/metrics_summary_*.csv` → "Avg Subgoal Success Rate" row
- Per-task: `task_results[].metrics.subgoal_success_rate`
- Format: Float (0.0 to 1.0)

**Example**:
```json
{
  "summary": {
    "avg_subgoal_success_rate": 0.82
  },
  "task_results": [
    {
      "metrics": {
        "subgoal_success_rate": 0.85,
        "subgoals": [
          {"subgoal_id": 0, "success": true},
          {"subgoal_id": 1, "success": false},
          {"subgoal_id": 2, "success": true}
        ]
      }
    }
  ]
}
```

### 3. Planner Steps
**Definition**: The number of model inferences (planning calls) required to produce a complete executable plan.

**Storage**:
- Location: `logs/metrics_*.json` → `summary.avg_planner_steps`
- CSV: `logs/metrics_summary_*.csv` → "Avg Planner Steps" row
- Per-task: `task_results[].planner_steps`
- Format: Integer

**Example**:
```json
{
  "summary": {
    "avg_planner_steps": 12.5
  },
  "task_results": [
    {
      "planner_steps": 10,
      "metrics": {
        "planner_steps": 10
      }
    }
  ]
}
```

### 4. Environment Steps
**Definition**: The number of interactions the agent performs within the environment while executing a task.

**Storage**:
- Location: `logs/metrics_*.json` → `summary.avg_environment_steps`
- CSV: `logs/metrics_summary_*.csv` → "Avg Environment Steps" row
- Per-task: `task_results[].environment_steps`
- Format: Integer

**Example**:
```json
{
  "summary": {
    "avg_environment_steps": 8.3
  },
  "task_results": [
    {
      "environment_steps": 7,
      "metrics": {
        "environment_steps": 7
      }
    }
  ]
}
```

### 5. Error Analysis
**Definition**: Qualitative categorization of failures (e.g., perception errors, reasoning errors, or planning errors).

**Error Types**:
- `perception`: Errors in position (x, y, z coordinates)
- `orientation`: Errors in rotation (roll, pitch, yaw)
- `gripper`: Errors in gripper state
- `planning`: Multiple dimension errors (planning failure)
- `invalid_target`: Invalid target values

**Storage**:
- Location: `logs/metrics_*.json` → `summary.error_analysis`
- CSV: `logs/metrics_summary_*.csv` → "Error Type" and "Count" columns
- Per-task: `task_results[].errors[]` and `task_results[].metrics.error_counts`

**Example**:
```json
{
  "summary": {
    "error_analysis": {
      "perception": 45,
      "orientation": 23,
      "gripper": 12,
      "planning": 8
    },
    "total_errors": 88
  },
  "task_results": [
    {
      "errors": [
        {
          "type": "perception",
          "message": "Prediction mismatch at step 3",
          "step": 3
        }
      ],
      "metrics": {
        "error_counts": {
          "perception": 1
        },
        "total_errors": 1
      }
    }
  ]
}
```

## Additional Metrics

### Training Metrics (Per Epoch)
Stored in `logs/training_metrics_epoch_*.json`:

```json
{
  "epoch": 5,
  "loss": 0.0234,
  "val_accuracies": {
    "x": 0.89,
    "y": 0.87,
    "z": 0.91,
    "roll": 0.85,
    "pitch": 0.83,
    "yaw": 0.88,
    "gripper": 0.95
  },
  "avg_accuracy": 0.88
}
```

### Per-Dimension Accuracy
Stored in task metrics:

```json
{
  "metrics": {
    "per_dimension_accuracy": [0.89, 0.87, 0.91, 0.85, 0.83, 0.88, 0.95]
  }
}
```

## Usage

### During Training

Metrics are automatically saved during training:

```bash
python3 src/encoders/text_encoder.py \
    --data-root ./data/EB-Man_trajectory_dataset \
    --batch-size 8 \
    --epochs 50
```

Metrics saved to: `logs/training_metrics_epoch_*.json`

### After Training (Trajectory Evaluation)

Run comprehensive trajectory evaluation:

```bash
python3 evaluate_trajectory.py \
    --checkpoint checkpoints/agent_epoch_50.pt \
    --data-root ./data/EB-Man_trajectory_dataset \
    --max-episodes 100 \
    --success-threshold 0.8 \
    --log-dir ./logs \
    --s3-bucket 11777-h1 \
    --s3-prefix metrics/
```

This generates:
- `logs/metrics_*.json` - Full metrics with all task results
- `logs/metrics_summary_*.csv` - Summary for easy analysis
- `logs/trajectory_evaluation_*.json` - Detailed evaluation results

### Viewing Metrics

#### View Summary CSV:
```bash
cat logs/metrics_summary_*.csv
```

#### View Full Metrics JSON:
```bash
cat logs/metrics_*.json | python3 -m json.tool | less
```

#### Extract Specific Metric:
```bash
python3 << 'EOF'
import json
with open('logs/metrics_20241120_120000.json') as f:
    data = json.load(f)
    print(f"Task Success Rate: {data['summary']['task_success_rate']:.2%}")
    print(f"Avg Planner Steps: {data['summary']['avg_planner_steps']:.2f}")
EOF
```

## S3 Sync

Metrics are automatically uploaded to S3 if configured:

```bash
# Manual upload
aws s3 sync ./logs/ s3://11777-h1/metrics/ --region us-east-2

# Or use sync script
./sync_s3.sh upload
```

## Metrics File Structure

### Full Metrics JSON (`metrics_*.json`)
```json
{
  "timestamp": "2024-11-20T12:00:00",
  "summary": {
    "task_success_rate": 0.75,
    "avg_subgoal_success_rate": 0.82,
    "avg_planner_steps": 12.5,
    "avg_environment_steps": 8.3,
    "total_tasks": 200,
    "successful_tasks": 150,
    "failed_tasks": 50,
    "error_analysis": {...},
    "total_errors": 88
  },
  "task_results": [
    {
      "task_id": "episode_0",
      "episode_id": 0,
      "instruction": "Pick up the star...",
      "start_time": "2024-11-20T12:00:00",
      "end_time": "2024-11-20T12:00:05",
      "planner_steps": 10,
      "environment_steps": 7,
      "subgoals": [...],
      "actions": [...],
      "predictions": [...],
      "targets": [...],
      "errors": [...],
      "completed": true,
      "success": true,
      "metrics": {...}
    }
  ]
}
```

## Integration with Training Scripts

The metrics tracker is integrated into:
- `src/encoders/text_encoder.py` - Basic training metrics
- `evaluate_trajectory.py` - Comprehensive trajectory evaluation
- `evaluate_aws.py` - AWS-optimized evaluation

## Best Practices

1. **Regular Evaluation**: Run trajectory evaluation after each epoch or every N epochs
2. **S3 Backup**: Always configure S3 upload for important metrics
3. **CSV Analysis**: Use CSV files for quick analysis in Excel/Python
4. **JSON Details**: Use JSON files for detailed error analysis
5. **Version Control**: Tag metrics files with model version/checkpoint name

## Troubleshooting

### Metrics not saving?
- Check `logs/` directory exists and is writable
- Verify disk space: `df -h`

### S3 upload failing?
- Check IAM role has S3 write permissions
- Verify bucket name: `11777-h1`
- Check region: `us-east-2`

### Missing metrics?
- Ensure evaluation script completed successfully
- Check for errors in console output
- Verify dataset loaded correctly

