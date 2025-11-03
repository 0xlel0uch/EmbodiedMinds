# Embodied Minds Project Setup Guide

## 1. GitHub Repository Structure

```
embodied-minds/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── configs/
│   ├── baseline_config.yaml
│   ├── hypothesis1_visual_icl.yaml
│   ├── hypothesis2_navigation.yaml
│   └── hypothesis3_graph_rag.yaml
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_models.py
│   │   ├── visual_icl_model.py
│   │   ├── navigation_model.py
│   │   └── graph_rag_agent.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── embodiedbench_loader.py
│   │   └── preprocessing.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── evaluator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── visualization.py
│   │   └── logging_utils.py
│   └── training/
│       ├── __init__.py
│       └── trainer.py
├── experiments/
│   ├── hypothesis1_visual_icl/
│   ├── hypothesis2_navigation/
│   └── hypothesis3_graph_rag/
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── baseline_analysis.ipynb
│   └── results_visualization.ipynb
├── scripts/
│   ├── setup_embodiedbench.sh
│   ├── train_baseline.py
│   ├── run_hypothesis1.py
│   ├── run_hypothesis2.py
│   └── run_hypothesis3.py
└── results/
    ├── baselines/
    ├── hypothesis1/
    ├── hypothesis2/
    └── hypothesis3/
```

## 2. EC2 Instance Setup

### Instance Requirements

**For Baseline & Hypothesis Testing:**
- **Instance Type**: `g5.xlarge` or `g5.2xlarge`
  - g5.xlarge: 1x A10G (24GB), 4 vCPUs, 16GB RAM (~$1.00/hr)
  - g5.2xlarge: 1x A10G (24GB), 8 vCPUs, 32GB RAM (~$1.21/hr)
  
**For Larger Models (7B-13B quantized):**
- **Instance Type**: `g5.4xlarge` or `g5.8xlarge`
  - g5.4xlarge: 1x A10G (24GB), 16 vCPUs, 64GB RAM
  - g5.8xlarge: 1x A10G (24GB), 32 vCPUs, 128GB RAM

**For Multi-GPU or 34B+ Models:**
- **Instance Type**: `g5.12xlarge` or `p3.8xlarge`
  - Consider for final experiments if needed

### Launch Configuration

```bash
# AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
# Storage: 100GB - 200GB EBS (gp3)
# Security Group: Allow SSH (22), Jupyter (8888), Custom ports as needed
```

### Initial Setup Script

```bash
#!/bin/bash
# save as setup_ec2.sh

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install essential packages
sudo apt-get install -y build-essential git wget curl htop tmux

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init

# Create Python environment
conda create -n embodied python=3.10 -y
conda activate embodied

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Clone your repository
cd ~
git clone https://github.com/YOUR_USERNAME/embodied-minds.git
cd embodied-minds

# Install project dependencies
pip install -r requirements.txt

# Install additional ML tools
pip install transformers accelerate bitsandbytes einops
pip install wandb tensorboard
pip install jupyter jupyterlab

# Setup EmbodiedBench dataset
bash scripts/setup_embodiedbench.sh

echo "Setup complete! Activate environment with: conda activate embodied"
```

## 3. Requirements.txt

```python
# Core ML frameworks
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.36.0
accelerate>=0.25.0
bitsandbytes>=0.41.0

# Vision and multimodal
timm>=0.9.0
clip @ git+https://github.com/openai/CLIP.git
pillow>=10.0.0
opencv-python>=4.8.0
einops>=0.7.0

# EmbodiedBench dependencies
ai2thor>=5.0.0
habitat-sim>=0.2.4
pybullet>=3.2.5

# Graph database for Hypothesis 3
neo4j>=5.14.0
py2neo>=2021.2.3

# Object detection (for Hypothesis 1 & 2)
ultralytics>=8.0.0  # YOLO
detectron2 @ git+https://github.com/facebookresearch/detectron2.git

# Depth estimation
transformers[depth-estimation]

# Experiment tracking
wandb>=0.16.0
tensorboard>=2.15.0

# Utilities
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.66.0
pyyaml>=6.0
scipy>=1.11.0
scikit-learn>=1.3.0

# Development
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
ipython>=8.12.0
jupyter>=1.0.0
```

## 4. Environment Variables (.env)

```bash
# API Keys (if using proprietary models for comparison)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Weights & Biases
WANDB_API_KEY=your_key_here
WANDB_PROJECT=embodied-minds

# HuggingFace
HF_TOKEN=your_token_here

# Neo4j (for Hypothesis 3)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Paths
EMBODIEDBENCH_DATA=/path/to/embodiedbench/data
MODEL_CACHE=/path/to/model/cache
RESULTS_DIR=/path/to/results
```

## 5. Getting Started Commands

```bash
# 1. SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# 2. Run setup script
bash setup_ec2.sh

# 3. Activate environment
conda activate embodied

# 4. Setup WandB
wandb login

# 5. Test installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"

# 6. Download EmbodiedBench
cd ~/embodied-minds
python scripts/download_embodiedbench.py

# 7. Run baseline experiments
python scripts/train_baseline.py --config configs/baseline_config.yaml

# 8. Launch Jupyter (for development)
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
# Access via: http://your-ec2-ip:8888
```

## 6. Git Workflow

```bash
# Initialize repository
git init
git add .
git commit -m "Initial project structure"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/embodied-minds.git
git push -u origin main

# Create branch for each hypothesis
git checkout -b hypothesis1-visual-icl
git checkout -b hypothesis2-navigation
git checkout -b hypothesis3-graph-rag

# Regular workflow
git checkout main
git pull origin main
git checkout hypothesis1-visual-icl
# make changes
git add .
git commit -m "Implement visual ICL architecture"
git push origin hypothesis1-visual-icl
# create PR on GitHub
```

## 7. Cost Management Tips

1. **Use Spot Instances**: Save 70-90% for non-critical experiments
2. **Stop instances when not in use**: Only pay for storage
3. **Use tmux/screen**: Keep training running after SSH disconnect
4. **Monitor with CloudWatch**: Set billing alerts
5. **Offload data to S3**: Cheaper long-term storage

```bash
# Example: Using tmux for long-running jobs
tmux new -s training
python scripts/train_baseline.py
# Press Ctrl+B, then D to detach
# Reconnect later with: tmux attach -t training
```

## 8. Next Steps Priority

1. ✅ Set up EC2 instance
2. ✅ Create GitHub repository with structure
3. ✅ Install dependencies and verify GPU
4. ✅ Download EmbodiedBench dataset
5. 🔄 Implement baseline models (verify paper results)
6. 🔄 Implement Hypothesis 1: Visual ICL model
7. 🔄 Implement Hypothesis 2: Single-box navigation
8. 🔄 Implement Hypothesis 3: Graph-RAG memory
9. 🔄 Run experiments and collect results
10. 🔄 Analyze and visualize results
11. 🔄 Write final report

## 9. Monitoring & Debugging

```bash
# Check GPU usage
watch -n 1 nvidia-smi

# Monitor training
tensorboard --logdir=results/ --port=6006

# Check disk space
df -h

# Monitor system resources
htop

# View logs
tail -f logs/training.log
```

## 10. Backup Strategy

```bash
# Regular backups to S3
aws s3 sync results/ s3://your-bucket/embodied-minds/results/
aws s3 sync checkpoints/ s3://your-bucket/embodied-minds/checkpoints/

# Commit code regularly
git add . && git commit -m "Progress update" && git push
```