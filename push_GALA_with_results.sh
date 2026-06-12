#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/tahiti/ARTeccv"
REPO_URL="https://github.com/cataug/GALA.git"

cd "$PROJECT_DIR"

# ------------------------------------------------------------
# Git ignore for heavy folders (keep logs/outputs)
# ------------------------------------------------------------

cat > .gitignore <<'EOF'
# Python
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

# Virtual environments
.venv/
venv/
env/

# Models and datasets
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

# Large generated files (except outputs/logs)
*.npz
*.npy
*.csv
*.tsv
*.parquet
*.pkl
*.pickle

# LaTeX build garbage
*.aux
*.bbl
*.bcf
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.run.xml
*.synctex.gz
*.toc
*.lof
*.lot
*.nav
*.snm
*.vrb

# OS / editor
.DS_Store
Thumbs.db
.vscode/
.idea/

# Secrets
.env
.env.*
!.env.example
EOF

# ------------------------------------------------------------
# Remove README temporarily
# ------------------------------------------------------------

rm -f README.md README.rst README.txt readme.md

# ------------------------------------------------------------
# Init git if needed
# ------------------------------------------------------------

if [ ! -d .git ]; then
  git init
fi
git branch -M main

# ------------------------------------------------------------
# Set remote
# ------------------------------------------------------------

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

# ------------------------------------------------------------
# Remove cached heavy folders
# ------------------------------------------------------------

git rm -r --cached models hf_cache data 2>/dev/null || true

# ------------------------------------------------------------
# Add all other files (logs and outputs included)
# ------------------------------------------------------------

git add .

# ------------------------------------------------------------
# Commit & push
# ------------------------------------------------------------

if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  git commit -m "GALA code, logs, and outputs"
fi

git push -u origin main

echo "DONE: pushed to $REPO_URL"
