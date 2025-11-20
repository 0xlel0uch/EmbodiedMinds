# S3 Sync Setup Instructions

## Current Status
- **S3 Bucket:** `11777-h1`
- **Region:** `us-east-2` (US East Ohio)
- **EC2 Instance:** `3.139.95.113`
- **Status:** ⚠️ AWS credentials not configured

## Required: Configure AWS Access

You have **two options** to enable S3 access:

### Option 1: Attach IAM Role (Recommended)

1. **Create IAM Role with S3 Permissions:**
   - Go to AWS Console → IAM → Roles → Create Role
   - Select "EC2" as the service
   - Attach policy: `AmazonS3FullAccess` (or create custom policy for `11777-h1` bucket only)
   - Name the role (e.g., `EC2-S3-Access-Role`)

2. **Attach Role to EC2 Instance:**
   - Go to EC2 Console → Select your instance (`3.139.95.113`)
   - Actions → Security → Modify IAM role
   - Select the role you created
   - Save

3. **Verify:**
   ```bash
   ssh -i /path/to/key.pem ec2-user@3.139.95.113
   aws sts get-caller-identity
   ```

### Option 2: Configure AWS Credentials Manually

1. **Get AWS Access Keys:**
   - Go to AWS Console → IAM → Users → Your User
   - Security Credentials → Create Access Key
   - Save Access Key ID and Secret Access Key

2. **Configure on EC2:**
   ```bash
   ssh -i /path/to/key.pem ec2-user@3.139.95.113
   aws configure
   # Enter:
   # - AWS Access Key ID: [your key]
   # - AWS Secret Access Key: [your secret]
   # - Default region: us-east-2
   # - Default output format: json
   ```

## After Configuration

### 1. Upload Sync Script
```bash
scp -i /path/to/key.pem sync_s3.sh ec2-user@3.139.95.113:~/
ssh -i /path/to/key.pem ec2-user@3.139.95.113 "chmod +x ~/sync_s3.sh"
```

### 2. List S3 Bucket Contents
```bash
ssh -i /path/to/key.pem ec2-user@3.139.95.113 "./sync_s3.sh list"
```

### 3. Download Data from S3
```bash
ssh -i /path/to/key.pem ec2-user@3.139.95.113 "./sync_s3.sh download"
```

### 4. Upload Checkpoints to S3
```bash
ssh -i /path/to/key.pem ec2-user@3.139.95.113 "./sync_s3.sh upload"
```

## S3 Bucket Structure (Expected)

```
s3://11777-h1/
├── data/
│   └── EB-Man_trajectory_dataset/
│       ├── eb-man_dataset_single_step.json
│       ├── eb-man_dataset_multi_step.json
│       └── images/  (or images.zip)
├── checkpoints/
│   └── agent_epoch*.pt
└── logs/
    └── training.log
```

## Quick Test

Once credentials are configured, test access:
```bash
ssh -i /path/to/key.pem ec2-user@3.139.95.113
aws s3 ls s3://11777-h1/ --region us-east-2
```

## Troubleshooting

### "Access Denied"
- Check IAM role permissions
- Verify bucket name is correct: `11777-h1`
- Verify region: `us-east-2`

### "Bucket does not exist"
- Verify bucket name
- Check region
- Ensure bucket is in `us-east-2`

### "No credentials found"
- IAM role not attached, or
- AWS credentials not configured
- Run `aws configure` or attach IAM role

