#!/usr/bin/env python3
"""Master build script — run before every commit.

What it does:
  1. Regenerates data/stats.js   (node/snippet/concept counts)
  2. Regenerates sitemap.xml     (all concept page URLs)

Usage:
  python3 scripts/build.py

Add as a pre-commit hook:
  cp scripts/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
"""
import subprocess, sys, pathlib

ROOT    = pathlib.Path(__file__).parent.parent
PYTHON  = sys.executable

steps = [
    ('Counting nodes, snippets & concept pages', ['gen-stats.py']),
    ('Building sitemap.xml',                     ['gen-sitemap.py']),
]

ok = True
for label, script_parts in steps:
    script = ROOT / 'scripts' / script_parts[0]
    print(f'\n▸ {label}')
    result = subprocess.run([PYTHON, str(script)], cwd=ROOT)
    if result.returncode != 0:
        print(f'  ERROR: {script.name} failed (exit {result.returncode})')
        ok = False

print()
if ok:
    print('✓ Build complete. Ready to commit.')
else:
    print('✗ Build finished with errors.')
    sys.exit(1)
