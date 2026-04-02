# GitHub Setup Instructions

Your local Git repository is initialized and ready! Follow these steps to push to GitHub.

## Step 1: Create a GitHub Repository

1. Go to https://github.com/new
2. Enter repository name: `IsomorphicDataSet`
3. Add description: "Framework for proving latent space isomorphism between LLMs using Procrustes analysis"
4. Choose visibility: **Public** (recommended for open science)
5. **Do NOT** initialize with README, .gitignore, or license (we already have those)
6. Click **Create repository**

## Step 2: Connect Local Repo to GitHub

Copy and run the commands from GitHub (they'll look similar to this):

```bash
# Navigate to your project
cd c:\Users\MPC\Documents\code\IsomophicDataSet

# Remove default branch name (if needed)
git branch -M main

# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/IsomorphicDataSet.git

# Push to GitHub
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 3: Verify Push

Visit `https://github.com/YOUR_USERNAME/IsomorphicDataSet` to see your repository online!

## Next Steps (Optional)

### Add Branch Protection Rules
1. Go to **Settings** → **Branches**
2. Click **Add rule**
3. Set up protection for `main` branch

### Enable Actions
1. Go to **Actions**
2. Your CI/CD workflow should appear automatically
3. It will run tests on push/PR

### Add Topics
Add these topics to help discovery:
- `llm`
- `latent-space`
- `neural-networks`
- `machine-learning`
- `procrustes-analysis`
- `semantic-alignment`

## Troubleshooting

### Authentication Error
If you get an authentication error, use a Personal Access Token (PAT):

```bash
# Create a PAT at https://github.com/settings/tokens
# Then use this format:
git remote set-url origin https://YOUR_USERNAME:YOUR_TOKEN@github.com/YOUR_USERNAME/IsomorphicDataSet.git
```

### Already Have Commits
If the repository isn't empty on GitHub, use:
```bash
git push -u origin main --force  # Use carefully!
```

## Project Structure for Reference

```
IsomorphicDataSet/
├── .github/
│   └── workflows/
│       └── python-tests.yml        # CI/CD pipeline
├── source/
│   ├── generator.py                # ConceptGenerator class
│   ├── alignment.py                # Anchor-based alignment
│   └── alignment_utils.py          # Procrustes SVD solver
├── main.py                         # Main workflow
├── example_cross_model_alignment.py # Detailed example
├── pyproject.toml                  # Project config (uv/pip)
├── requirements.txt                # Dependencies
├── README.md                       # Full documentation
├── LICENSE                         # MIT License
└── .gitignore                      # Git ignore rules
```

## Git Commands Cheat Sheet

```bash
# View git status
git status

# View recent commits
git log --oneline -10

# Create new branch
git checkout -b feature/new-feature

# Commit changes
git add .
git commit -m "Description of changes"

# Push changes
git push origin main

# Create pull request
# (Do this on GitHub website)

# Update local from remote
git pull origin main
```

---

Once you've pushed to GitHub, share the repository link: `https://github.com/YOUR_USERNAME/IsomorphicDataSet`
