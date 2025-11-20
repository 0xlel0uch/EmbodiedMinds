# S3 Sync Status

## ✅ Completed

1. **IAM Role Verified**
   - Role: `11777-Cursor`
   - S3 access: ✅ Working

2. **S3 Bucket Contents**
   - Bucket: `s3://11777-h1`
   - Region: `us-east-2`
   - Files:
     - ✅ `eb-man_dataset_single_step.json` (32.8 MB) - **Downloaded**
     - ✅ `eb-man_dataset_multi_step.json` (17.0 MB) - **Downloaded**
     - ✅ `README.md` (4.1 KB) - **Downloaded**
     - ⏳ `images.zip` (9.5 GB) - **Downloading in background**

3. **EC2 Setup**
   - Instance: `3.139.95.113`
   - Data directory: `~/EmbodiedMinds/data/EB-Man_trajectory_dataset/`
   - Sync script: `~/sync_s3.sh` (uploaded and executable)

## ⏳ In Progress

- **images.zip download**: Running in background (9.5 GB)
  - Estimated time: ~2-3 minutes at ~50-100 MB/s
  - Location: `~/EmbodiedMinds/data/EB-Man_trajectory_dataset/images.zip`

## 📋 Next Steps

### 1. Wait for images.zip Download
```bash
ssh -i /path/to/key.pem ec2-user@3.139.95.113
cd ~/EmbodiedMinds/data/EB-Man_trajectory_dataset
ls -lh images.zip  # Check if download complete
```

### 2. Extract Images
```bash
cd ~/EmbodiedMinds/data/EB-Man_trajectory_dataset
unzip -q images.zip
# This will create: images/ directory with all episode folders
```

### 3. Verify Dataset
```bash
cd ~/EmbodiedMinds
python3 -c "
from data_loader import EmbodiedDataset
ds = EmbodiedDataset(data_root='./data/EB-Man_trajectory_dataset', debug=True)
print(f'Dataset loaded: {len(ds)} examples')
item = ds[0]
print(f'Demo images: {item[\"demo_images\"].shape}')
print(f'Current image: {item[\"current_image\"].shape}')
"
```

### 4. Start Training
```bash
cd ~/EmbodiedMinds
python3 train_aws.py \
    --data-root ./data/EB-Man_trajectory_dataset \
    --batch-size 8 \
    --epochs 50 \
    --lr 1e-4
```

## 🔄 Sync Script Usage

The sync script is available at `~/sync_s3.sh`:

```bash
# List S3 contents
./sync_s3.sh list

# Download from S3
./sync_s3.sh download

# Upload checkpoints to S3
./sync_s3.sh upload

# Both download and upload
./sync_s3.sh both
```

## 📊 Disk Space

- Total: 70 GB
- Used: 58 GB
- Available: 13 GB
- **Note**: After extracting images.zip (~9.5 GB), you'll have ~3.5 GB free. Consider:
  - Extracting only needed episodes
  - Using S3 for checkpoint storage
  - Cleaning up after training

## ⚠️ Important Notes

1. **Extraction Space**: The images.zip will expand to ~10-15 GB when extracted
2. **Checkpoint Storage**: Configure training to upload checkpoints to S3 automatically
3. **Data Access**: The dataset loader will automatically find images in the `images/` directory after extraction

