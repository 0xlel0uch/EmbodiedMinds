# Instructions to Clone and Setup the Repository

## Quick Method (if you have access)

Run the setup script:
```bash
./setup_repo.sh
```

## Manual Method

### Step 1: Clone the Repository

**Option A: Using HTTPS (requires authentication for private repos)**
```bash
cd /Users/abhivakil/Desktop/Multi_Model_ML/Final_Project
git clone https://github.com/Oxlelouch/EmbodiedMinds.git .
cd EmbodiedMinds
```

**Option B: Using SSH (if you have SSH keys set up)**
```bash
cd /Users/abhivakil/Desktop/Multi_Model_ML/Final_Project
git clone git@github.com:Oxlelouch/EmbodiedMinds.git .
cd EmbodiedMinds
```

**Option C: Using GitHub CLI (if authenticated)**
```bash
cd /Users/abhivakil/Desktop/Multi_Model_ML/Final_Project
gh repo clone Oxlelouch/EmbodiedMinds .
cd EmbodiedMinds
```

### Step 2: Checkout the 'sameer' Branch

Once cloned, checkout the branch with incomplete code:
```bash
git checkout sameer
```

If the branch doesn't exist locally, fetch it first:
```bash
git fetch origin
git checkout sameer
```

### Step 3: Verify Files

Check what files are in the branch:
```bash
ls -la
git status
```

## If Repository is Private

### Authenticate with GitHub CLI:
```bash
gh auth login
```

### Or use Personal Access Token with HTTPS:
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Create a new token with `repo` permissions
3. Use it when prompted for password:
```bash
git clone https://github.com/Oxlelouch/EmbodiedMinds.git .
# When prompted, use your username and the token as password
```

## After Making Changes

Once you've completed the code:

1. **Check status:**
   ```bash
   git status
   ```

2. **Add your changes:**
   ```bash
   git add .
   ```

3. **Commit:**
   ```bash
   git commit -m "Completed incomplete code sections"
   ```

4. **Push to the sameer branch:**
   ```bash
   git push origin sameer
   ```

## Troubleshooting

- **"Repository not found"**: The repo might be private or the URL is incorrect. Verify the repository name and ensure you have access.
- **"Authentication failed"**: Set up GitHub authentication using `gh auth login` or use a personal access token.
- **"Branch not found"**: Make sure the branch name is correct. Check available branches with `git branch -r`.

