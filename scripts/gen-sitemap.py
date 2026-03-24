#!/usr/bin/env python3
"""Auto-generates sitemap.xml by scanning the repository.

Sources:
  - Root pages:   index.html, map.html
  - Directory:    concepts/index.html
  - Concept pages: concepts/*.html  (excludes index.html)

Last-modified date is taken from git log for each file.
Falls back to today's date if git is unavailable or file is untracked.

Run from repo root:
  python3 scripts/gen-sitemap.py
"""
import pathlib, subprocess, datetime, re

ROOT    = pathlib.Path(__file__).parent.parent
BASE    = 'https://genai4a11.github.io'
TODAY   = datetime.date.today().isoformat()

EXCLUDED_CONCEPTS = {'index.html'}   # directory page gets its own priority


def git_lastmod(path: pathlib.Path) -> str:
    """Return ISO date of the last git commit touching *path*, or TODAY."""
    try:
        rel = path.relative_to(ROOT)
        out = subprocess.check_output(
            ['git', 'log', '-1', '--format=%cs', '--', str(rel)],
            cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out if out else TODAY
    except Exception:
        return TODAY


def entry(loc: str, lastmod: str, priority: str) -> str:
    return (
        f'  <url>'
        f'<loc>{loc}</loc>'
        f'<lastmod>{lastmod}</lastmod>'
        f'<priority>{priority}</priority>'
        f'</url>\n'
    )


lines = ['<?xml version="1.0" encoding="UTF-8"?>\n',
         '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n']

# ── root pages ────────────────────────────────────────────────────────────────
lines.append(entry(BASE + '/',
                   git_lastmod(ROOT / 'index.html'), '1.0'))
lines.append(entry(BASE + '/map.html',
                   git_lastmod(ROOT / 'map.html'), '0.9'))

# ── concepts directory ────────────────────────────────────────────────────────
lines.append(entry(BASE + '/concepts/',
                   git_lastmod(ROOT / 'concepts/index.html'), '0.9'))

# ── individual concept pages ──────────────────────────────────────────────────
concept_files = sorted(
    f for f in (ROOT / 'concepts').glob('*.html')
    if f.name not in EXCLUDED_CONCEPTS
)
for f in concept_files:
    lines.append(entry(
        f'{BASE}/concepts/{f.name}',
        git_lastmod(f),
        '0.8'
    ))

lines.append('</urlset>\n')

out = ''.join(lines)
(ROOT / 'sitemap.xml').write_text(out)
print(f"sitemap.xml  →  {len(concept_files)} concept pages + 3 root/dir URLs")
