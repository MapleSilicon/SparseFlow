#!/bin/bash

set -e

echo "══════════════════════════════════════════════════════════════"
echo "🔄 Syncing local master with origin/master (rebase)"
echo "══════════════════════════════════════════════════════════════"
echo ""

# Show current branch & status
echo "📋 Current branch and status:"
git branch --show-current
git status -s
echo ""

echo "📥 Fetching latest from origin..."
git fetch origin
echo "✅ Fetch complete"
echo ""

echo "🔍 Comparing local master vs origin/master..."
echo "→ Commits on local master not on origin/master:"
git log --oneline origin/master..master || true
echo ""
echo "→ Commits on origin/master not on local master:"
git log --oneline master..origin/master || true
echo ""

echo "⚙️  Rebasing local master onto origin/master..."
git pull --rebase origin master || {
  echo ""
  echo "❌ Rebase stopped due to conflicts."
  echo "   Fix conflicts, then run:"
  echo "     git add <files>"
  echo "     git rebase --continue"
  echo ""
  echo "   To abort rebase (dangerous if you have new work):"
  echo "     git rebase --abort"
  exit 1
}

echo ""
echo "✅ Rebase completed successfully."
echo ""

echo "📤 Pushing updated master to origin..."
git push origin master

echo ""
echo "✅ Push complete. Local and remote master are now in sync."
echo "══════════════════════════════════════════════════════════════"
