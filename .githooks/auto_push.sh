#!/bin/sh

set -e

branch="$(git rev-parse --abbrev-ref HEAD)"

# Skip detached HEAD states.
if [ "$branch" = "HEAD" ]; then
  exit 0
fi

if git remote get-url origin >/dev/null 2>&1; then
  git push origin "$branch"
fi

if git remote get-url hf >/dev/null 2>&1; then
  git push hf "$branch"
fi
