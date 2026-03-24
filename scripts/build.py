#!/usr/bin/env python3
"""Master build script — run before every commit.

What it does:
  1. Counts nodes, snippets & concept pages  →  data/stats.js
  2. Regenerates sitemap.xml from concepts/ folder
  3. Injects Prev/Next nav into every concept page
  4. Injects JSON-LD Article schema into every concept page

Usage:
  python3 scripts/build.py

Install as a pre-commit hook so it runs automatically:
  cp scripts/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
"""
import subprocess, sys, pathlib

ROOT   = pathlib.Path(__file__).parent.parent
PYTHON = sys.executable

steps = [
    ('Counting nodes, snippets & concept pages', 'gen-stats.py'),
    ('Building sitemap.xml',                     'gen-sitemap.py'),
    ('Injecting prev/next navigation',            'gen-nav.py'),
    ('Injecting JSON-LD Article schema',          'gen-jsonld.py'),
]

ok = True
for label, script_name in steps:
    script = ROOT / 'scripts' / script_name
    print(f'\n▸ {label}')
    result = subprocess.run([PYTHON, str(script)], cwd=ROOT)
    if result.returncode != 0:
        print(f'  ERROR: {script_name} failed (exit {result.returncode})')
        ok = False

print()
if ok:
    print('✓ Build complete. Ready to commit.')
else:
    print('✗ Build finished with errors.')
    sys.exit(1)
