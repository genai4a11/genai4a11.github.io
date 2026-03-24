#!/usr/bin/env python3
"""Inject JSON-LD Article structured data into the <head> of every concept page.

This helps Google show rich results (article date, author, breadcrumbs).

For each concept page:
  1. Extract title from <title> tag (strip " — GenAI Mindmap" suffix)
  2. Extract description from <meta name="description"> content attribute
  3. Extract canonical URL from <link rel="canonical"> href attribute
  4. Get last-modified date from git log
  5. Inject (or replace) a <script type="application/ld+json"> block

For concepts/index.html, use CollectionPage schema instead.

Run from repo root:
  python3 scripts/gen-jsonld.py
"""
import re
import json
import pathlib
import subprocess
from datetime import datetime

ROOT = pathlib.Path(__file__).parent.parent
CONCEPTS_DIR = ROOT / 'concepts'

# ── Extract metadata from HTML ─────────────────────────────────────────────────
def extract_metadata(html_content, filepath):
    """Extract title, description, and canonical URL from HTML."""
    
    # Extract title from <title> tag
    title_match = re.search(r'<title>(.+?)</title>', html_content)
    title = title_match.group(1) if title_match else None
    if title:
        title = title.replace(' — GenAI Mindmap', '').strip()
    
    # Extract description from <meta name="description">
    desc_match = re.search(
        r'<meta\s+name="description"\s+content="([^"]+)"',
        html_content
    )
    description = desc_match.group(1) if desc_match else None
    
    # Extract canonical URL from <link rel="canonical">
    canonical_match = re.search(
        r'<link\s+rel="canonical"\s+href="([^"]+)"',
        html_content
    )
    canonical_url = canonical_match.group(1) if canonical_match else None
    
    return title, description, canonical_url

# ── Get last-modified date from git ────────────────────────────────────────────
def get_last_modified_date(filepath):
    """Get ISO 8601 date from git log, or use current date as fallback."""
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%aI', str(filepath)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    
    # Fallback: use current date
    return datetime.now().isoformat()

# ── Build JSON-LD for Article ──────────────────────────────────────────────────
def build_article_jsonld(title, description, canonical_url, last_modified):
    """Build JSON-LD Article schema."""
    schema = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": title,
        "description": description,
        "url": canonical_url,
        "dateModified": last_modified,
        "datePublished": "2025-01-01",
        "author": {
            "@type": "Organization",
            "name": "GenAI Mindmap",
            "url": "https://genai4a11.github.io"
        },
        "publisher": {
            "@type": "Organization",
            "name": "GenAI Mindmap",
            "url": "https://genai4a11.github.io",
            "logo": {
                "@type": "ImageObject",
                "url": "https://genai4a11.github.io/favicon.ico"
            }
        },
        "breadcrumb": {
            "@type": "BreadcrumbList",
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": 1,
                    "name": "Home",
                    "item": "https://genai4a11.github.io/"
                },
                {
                    "@type": "ListItem",
                    "position": 2,
                    "name": "Deep Dives",
                    "item": "https://genai4a11.github.io/concepts/"
                },
                {
                    "@type": "ListItem",
                    "position": 3,
                    "name": title,
                    "item": canonical_url
                }
            ]
        }
    }
    return schema

# ── Build JSON-LD for CollectionPage ──────────────────────────────────────────
def build_collectionpage_jsonld(title, description, canonical_url, last_modified):
    """Build JSON-LD CollectionPage schema."""
    schema = {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        "headline": title,
        "description": description,
        "url": canonical_url,
        "dateModified": last_modified,
        "datePublished": "2025-01-01",
        "author": {
            "@type": "Organization",
            "name": "GenAI Mindmap",
            "url": "https://genai4a11.github.io"
        },
        "publisher": {
            "@type": "Organization",
            "name": "GenAI Mindmap",
            "url": "https://genai4a11.github.io",
            "logo": {
                "@type": "ImageObject",
                "url": "https://genai4a11.github.io/favicon.ico"
            }
        },
        "breadcrumb": {
            "@type": "BreadcrumbList",
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": 1,
                    "name": "Home",
                    "item": "https://genai4a11.github.io/"
                },
                {
                    "@type": "ListItem",
                    "position": 2,
                    "name": "Deep Dives",
                    "item": "https://genai4a11.github.io/concepts/"
                }
            ]
        }
    }
    return schema

# ── Build script tag with JSON-LD ──────────────────────────────────────────────
def build_jsonld_script(schema_dict):
    """Build <script type="application/ld+json"> tag."""
    json_str = json.dumps(schema_dict, indent=2)
    script = (
        '<script type="application/ld+json">\n'
        f'{json_str}\n'
        '</script>'
    )
    return script

# ── Main processing ────────────────────────────────────────────────────────────
def main():
    updated_count = 0
    
    # Process all .html files in concepts/
    for filepath in sorted(CONCEPTS_DIR.glob('*.html')):
        html_content = filepath.read_text(encoding='utf-8')
        
        # Extract metadata
        title, description, canonical_url = extract_metadata(html_content, filepath)
        if not title or not description or not canonical_url:
            continue
        
        # Get last-modified date
        last_modified = get_last_modified_date(filepath)
        
        # Build appropriate schema
        if filepath.name == 'index.html':
            schema = build_collectionpage_jsonld(title, description, canonical_url, 
                                               last_modified)
        else:
            schema = build_article_jsonld(title, description, canonical_url, 
                                         last_modified)
        
        # Build script tag
        jsonld_script = build_jsonld_script(schema)
        
        # Replace or insert before </head>
        jsonld_pattern = r'<script type="application/ld\+json">.*?</script>'
        if re.search(jsonld_pattern, html_content, re.DOTALL):
            # Replace existing JSON-LD — use lambda so replacement isn't
            # interpreted as a regex template (avoids \u escape errors)
            html_content = re.sub(
                jsonld_pattern,
                lambda m: jsonld_script,
                html_content,
                count=1,
                flags=re.DOTALL,
            )
        else:
            # Insert before </head>
            html_content = html_content.replace('</head>',
                                                jsonld_script + '\n</head>')
        
        filepath.write_text(html_content, encoding='utf-8')
        updated_count += 1
    
    print(f"JSON-LD injected/updated in {updated_count} pages")

if __name__ == '__main__':
    main()
