#!/bin/bash
set -e

echo "══════════════════════════════════════════════════════════════"
echo "🧹 Cleaning up GitHub Actions workflows"
echo "══════════════════════════════════════════════════════════════"
echo ""

if [ ! -d ".github/workflows" ]; then
  echo "❌ No .github/workflows directory found. Run from repo root."
  exit 1
fi

echo "Existing workflow files:"
ls .github/workflows
echo ""

# Find the main v0.2 Complete Validation workflow
FULL_CI_FILE=$(grep -l "SparseFlow v0.2 - Complete Validation" .github/workflows/*.yml 2>/dev/null || true)

if [ -n "$FULL_CI_FILE" ]; then
  echo "✅ Found full CI workflow: $FULL_CI_FILE"
  if [ "$FULL_CI_FILE" != ".github/workflows/ci-full-validation.yml" ]; then
    echo "→ Renaming to .github/workflows/ci-full-validation.yml"
    git mv "$FULL_CI_FILE" .github/workflows/ci-full-validation.yml
  else
    echo "→ Already named ci-full-validation.yml"
  fi
else
  echo "⚠️ Could not find workflow with name 'SparseFlow v0.2 - Complete Validation'"
fi

# Keep demo workflow (if present)
DEMO_FILE=$(grep -l "SparseFlow Demo (Working)" .github/workflows/*.yml 2>/dev/null || true)
if [ -n "$DEMO_FILE" ]; then
  echo "✅ Keeping demo workflow: $DEMO_FILE"
else
  echo "ℹ️ No 'SparseFlow Demo (Working)' workflow found (that's fine)"
fi

echo ""
echo "🔎 Looking for old/experimental workflows to remove..."

REMOVE_PATTERNS=(
  "SparseFlow v0.2 - REAL Build & Test"
  "Quick Demo"
  "Diagnose Build"
)

TO_REMOVE=()

for PATTERN in "${REMOVE_PATTERNS[@]}"; do
  FILES=$(grep -l "$PATTERN" .github/workflows/*.yml 2>/dev/null || true)
  if [ -n "$FILES" ]; then
    echo "→ Marking for removal (pattern: '$PATTERN'):"
    echo "$FILES"
    TO_REMOVE+=($FILES)
  fi
done

if [ ${#TO_REMOVE[@]} -eq 0 ]; then
  echo "ℹ️ No extra workflows matched removal patterns."
else
  echo ""
  echo "❗ Removing these workflow files with git rm:"
  for F in "${TO_REMOVE[@]}"; do
    echo "   $F"
  done
  git rm -f "${TO_REMOVE[@]}"
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "✅ Workflow cleanup staged."
echo "Next:"
echo "  git status"
echo "  git commit -m \"ci: clean up workflows\""
echo "  git push origin master"
echo "══════════════════════════════════════════════════════════════"
