#!/usr/bin/env bash
set -euo pipefail

cd /home/tahiti/ARTeccv

REPO_SSH="git@github.com:cataug/GALA.git"
TODAY="$(date +%F)"
TOMORROW="$(date -d "$TODAY + 1 day" +%F)"
COMMIT_MSG="Add GALA files updated on ${TODAY}"

echo "============================================================"
echo "GALA today's files push"
echo "============================================================"
echo "TODAY     = $TODAY"
echo "TOMORROW  = $TOMORROW"
echo "REPO_SSH  = $REPO_SSH"
echo ""

# ------------------------------------------------------------
# Git init / identity / SSH remote
# ------------------------------------------------------------

if [ ! -d .git ]; then
  git init
fi

git config user.name "Ivan Ivan"
git config user.email "catauggie@gmail.com"

git branch -M main

git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_SSH"

# ------------------------------------------------------------
# .gitignore: exclude only heavy/raw stuff, keep outputs/logs/results
# ------------------------------------------------------------

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

# Heavy model/data folders
models/
hf_cache/
data/

# Model weights / huge binaries
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
*.onnx
*.h5
*.msgpack

# Backups are duplicated old outputs; keep final outputs instead
outputs/backups/

# OS/editor/secrets
.DS_Store
Thumbs.db
.vscode/
.idea/
.env
.env.*
!.env.example

# LaTeX build trash
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

# ------------------------------------------------------------
# Make list of today's files only
# ------------------------------------------------------------

LIST_FILE=".git/today_files_${TODAY}.zlist"
: > "$LIST_FILE"

find . -type f \
  -newermt "${TODAY} 00:00:00" ! -newermt "${TOMORROW} 00:00:00" \
  ! -path "./.git/*" \
  ! -path "./.venv/*" \
  ! -path "./venv/*" \
  ! -path "./env/*" \
  ! -path "./models/*" \
  ! -path "./hf_cache/*" \
  ! -path "./data/*" \
  ! -path "./outputs/backups/*" \
  ! -name "*.safetensors" \
  ! -name "*.bin" \
  ! -name "*.pt" \
  ! -name "*.pth" \
  ! -name "*.ckpt" \
  ! -name "*.onnx" \
  ! -name "*.h5" \
  ! -name "*.msgpack" \
  -print0 > "$LIST_FILE"

# Ensure .gitignore is included
printf "%s\0" ".gitignore" >> "$LIST_FILE"

echo "Today's files selected:"
tr '\0' '\n' < "$LIST_FILE" | sed 's#^\./##' | sort | sed -n '1,240p'

COUNT="$(tr '\0' '\n' < "$LIST_FILE" | sed '/^$/d' | wc -l)"
echo ""
echo "Selected files count: $COUNT"

if [ "$COUNT" -eq 0 ]; then
  echo "ERROR: no files modified today were found."
  exit 1
fi

# ------------------------------------------------------------
# Hard GitHub size check
# ------------------------------------------------------------

echo ""
echo "Checking selected files > 95 MB..."
BIG_FILES="$(tr '\0' '\n' < "$LIST_FILE" | while read -r f; do
  [ -z "$f" ] && continue
  [ -f "$f" ] || continue
  size_mb="$(du -m "$f" | cut -f1)"
  if [ "$size_mb" -gt 95 ]; then
    echo "${size_mb} MB  ${f#./}"
  fi
done)"

if [ -n "$BIG_FILES" ]; then
  echo "ERROR: these files are too large for normal GitHub push:"
  echo "$BIG_FILES"
  echo ""
  echo "Remove/compress them or use Git LFS."
  exit 1
fi

# ------------------------------------------------------------
# Add today's files only
# ------------------------------------------------------------

echo ""
echo "Adding today's files..."
git add -f --pathspec-from-file="$LIST_FILE" --pathspec-file-nul

echo ""
echo "Staged changes:"
git status --short

if git diff --cached --quiet; then
  echo ""
  echo "Nothing to commit."
else
  echo ""
  git commit -m "$COMMIT_MSG"
fi

echo ""
echo "SSH remote:"
git remote -v

echo ""
echo "Testing SSH..."
ssh -T git@github.com || true

echo ""
echo "Pushing..."
git push -u origin main

echo ""
echo "DONE: pushed today's GALA files to $REPO_SSH"
