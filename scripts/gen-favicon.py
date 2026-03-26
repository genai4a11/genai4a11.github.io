#!/usr/bin/env python3
"""Inject favicon <link> tags into every concept page.

Adds (or replaces) three favicon link tags before </head>:
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <link rel="icon" type="image/x-icon" href="/favicon.ico">
  <link rel="apple-touch-icon" href="/assets/apple-touch-icon.png">

Run from repo root:
  python3 scripts/gen-favicon.py
"""
import re
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
CONCEPTS_DIR = ROOT / 'concepts'

FAVICON_BLOCK = (
    '<link rel="icon" type="image/svg+xml" href="/favicon.svg"/>'
    '<link rel="icon" type="image/x-icon" href="/favicon.ico"/>'
    '<link rel="apple-touch-icon" href="/assets/apple-touch-icon.png"/>'
)

FAVICON_PATTERN = re.compile(
    r'(<link[^>]+rel="icon"[^>]*/?>|<link[^>]+apple-touch-icon[^>]*/?>)',
    re.IGNORECASE
)


def inject_favicon(html: str) -> str:
    """Remove existing favicon links then insert the canonical block before </head>."""
    # Strip any existing favicon links
    cleaned = FAVICON_PATTERN.sub('', html)
    # Insert before </head>
    if FAVICON_BLOCK not in cleaned:
        cleaned = cleaned.replace('</head>', FAVICON_BLOCK + '\n</head>', 1)
    return cleaned


def main():
    updated = 0
    for filepath in sorted(CONCEPTS_DIR.glob('*.html')):
        original = filepath.read_text(encoding='utf-8')
        updated_html = inject_favicon(original)
        if updated_html != original:
            filepath.write_text(updated_html, encoding='utf-8')
            updated += 1

    print(f"Favicon links injected/updated in {updated} pages")


if __name__ == '__main__':
    main()
