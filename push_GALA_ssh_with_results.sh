#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/tahiti/ARTeccv"
REPO_SSH="git@github.com:cataug/GALA.git"

cd "$PROJECT_DIR"

# Git identity for this repo
git config user.name "Ivan Ivan"
git config user.email "catauggie@gmail.com"

# Correct SSH remote
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_SSH"

# Keep logs and outputs, ignore only heavy/raw stuff
cat > .gitignore <<'EOF'
# Python/cache/env
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
*.egg-info/
.eggs/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
.venv/
venv/
env/

# Models / datasets / HF cache
models/
hf_cache/
data/
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
*.onnx
*.h5
*.msgpack

# Keep experiment results and logs
!outputs/
!outputs/**
!logs/
!logs/**
!runs/
!runs/**

# OS/editor/secrets
.DS_Store
Thumbs.db
.vscode/
.idea/
.env
.env.*
!.env.example

# LaTeX build trash, but keep logs/ experiment logs due rules above
*.aux
*.bbl
*.bcf
*.blg
*.fdb_latexmk
*.fls
*.out
*.run.xml
*.synctex.gz
*.toc
*.lof
*.lot
*.nav
*.snm
*.vrb
EOF

# No README for now
rm -f README.md README.rst README.txt readme.md

# Make sure heavy folders are not tracked
git rm -r --cached models hf_cache data .venv 2>/dev/null || true

# Add everything allowed by .gitignore
git add .

echo ""
echo "Staged files:"
git status --short

echo ""
echo "Checking staged files > 95 MB, GitHub hard limit is 100 MB:"
BIG=$(git diff --cached --name-only | while read -r f; do
  if [ -f "$f" ]; then
    size_mb=$(du -m "$f" | cut -f1)
    if [ "$size_mb" -gt 95 ]; then
      echo "${size_mb} MB  $f"
    fi
  fi
done)

if [ -n "$BIG" ]; then
  echo "$BIG"
  echo ""
  echo "ERROR: these files are too large for normal GitHub push."
  echo "Either remove/compress them or use Git LFS."
  exit 1
fi

if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  git commit -m "Add GALA code, logs, and results"
fi

git branch -M main

echo ""
echo "Testing SSH connection..."
ssh -T git@github.com || true

echo ""
echo "Pushing via SSH..."
git push -u origin main

echo ""
echo "DONE: pushed to $REPO_SSH"
