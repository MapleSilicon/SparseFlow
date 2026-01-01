#!/bin/bash
set -e

# Go to repo root
cd "$(git rev-parse --show-toplevel)"

echo "══════════════════════════════════════════════════════════════"
echo "🚀 Triggering SparseFlow v0.2.0 CI from master"
echo "══════════════════════════════════════════════════════════════"

# Update a simple CI trigger file
date -u +"Last CI trigger (UTC): %Y-%m-%d %H:%M:%S" > CI_TRIGGER_v0_2.txt

git add CI_TRIGGER_v0_2.txt
git commit -m "ci: trigger v0.2.0 full GitHub workflow"
git push origin master

echo ""
echo "✅ Push complete. Check GitHub → Actions for the new run."
