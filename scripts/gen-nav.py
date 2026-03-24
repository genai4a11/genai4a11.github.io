#!/usr/bin/env python3
"""Inject a Prev/Next navigation bar at the bottom of every concept page.

This allows readers to navigate within a category without going back to the index.

How it works:
  1. Read all concept pages from concepts/*.html (exclude index.html)
  2. For each page, extract the title and map filename to category using CATEGORIES
  3. Group pages by category in reading order (alphabetical by title within category)
  4. For each page, compute prev and next within its category
  5. Inject (or replace) a <nav class="concept-nav"> block just before </body>

Run from repo root:
  python3 scripts/gen-nav.py
"""
import re
import pathlib

ROOT = pathlib.Path(__file__).parent.parent

# Category mapping (filename without .html → category label)
CATEGORIES = {
    # Inference & Architecture
    'attention':         ('Inference & Architecture', '#38bdf8'),
    'kv-cache':          ('Inference & Architecture', '#38bdf8'),
    'quantization':      ('Inference & Architecture', '#38bdf8'),
    'transformer-arch':  ('Inference & Architecture', '#38bdf8'),
    'llm-internals':     ('Inference & Architecture', '#38bdf8'),
    'advanced-reasoning':('Inference & Architecture', '#38bdf8'),
    'serving':           ('Inference & Architecture', '#38bdf8'),
    'pos-encoding':      ('Inference & Architecture', '#38bdf8'),
    # Training & Fine-Tuning
    'peft-methods':      ('Training & Fine-Tuning', '#f59e0b'),
    'alignment':         ('Training & Fine-Tuning', '#f59e0b'),
    'training-tech':     ('Training & Fine-Tuning', '#f59e0b'),
    'synthetic-data':    ('Training & Fine-Tuning', '#f59e0b'),
    'ft-tools':          ('Training & Fine-Tuning', '#f59e0b'),
    'data-centric':      ('Training & Fine-Tuning', '#f59e0b'),
    'data-labeling':     ('Training & Fine-Tuning', '#f59e0b'),
    'data-prep':         ('Training & Fine-Tuning', '#f59e0b'),
    'regularization':    ('Training & Fine-Tuning', '#f59e0b'),
    # Retrieval & RAG
    'embeddings':        ('Retrieval & RAG', '#22c55e'),
    'retrieval-tech':    ('Retrieval & RAG', '#22c55e'),
    'advanced-rag':      ('Retrieval & RAG', '#22c55e'),
    'vector-dbs':        ('Retrieval & RAG', '#22c55e'),
    'golden-datasets':   ('Retrieval & RAG', '#22c55e'),
    'chunking':          ('Retrieval & RAG', '#22c55e'),
    'data-ingestion':    ('Retrieval & RAG', '#22c55e'),
    'post-retrieval':    ('Retrieval & RAG', '#22c55e'),
    'unstructured':      ('Retrieval & RAG', '#22c55e'),
    'docling':           ('Retrieval & RAG', '#22c55e'),
    # Agents & Orchestration
    'agent-frameworks':  ('Agents & Orchestration', '#a78bfa'),
    'agent-planning':    ('Agents & Orchestration', '#a78bfa'),
    'agent-memory':      ('Agents & Orchestration', '#a78bfa'),
    'multi-agent':       ('Agents & Orchestration', '#a78bfa'),
    'tool-use':          ('Agents & Orchestration', '#a78bfa'),
    'compound-ai':       ('Agents & Orchestration', '#a78bfa'),
    'state-sessions':    ('Agents & Orchestration', '#a78bfa'),
    'execution-models':  ('Agents & Orchestration', '#a78bfa'),
    # Models & Prompting
    'frontier-models':   ('Models & Prompting', '#818cf8'),
    'basic-prompting':   ('Models & Prompting', '#818cf8'),
    'programmatic-prompting': ('Models & Prompting', '#818cf8'),
    'output-control':    ('Models & Prompting', '#818cf8'),
    'vision-language':   ('Models & Prompting', '#818cf8'),
    'open-models':       ('Models & Prompting', '#818cf8'),
    'integration-std':   ('Models & Prompting', '#818cf8'),
    'dev-frameworks':    ('Models & Prompting', '#818cf8'),
    # Safety & Evaluation
    'safety-tech':       ('Safety & Evaluation', '#f472b6'),
    'evals-practice':    ('Safety & Evaluation', '#f472b6'),
    'benchmarks':        ('Safety & Evaluation', '#f472b6'),
    'reliability':       ('Safety & Evaluation', '#f472b6'),
    'rag-eval':          ('Safety & Evaluation', '#f472b6'),
    'human-oversight':   ('Safety & Evaluation', '#f472b6'),
    # Production & Infra
    'mlops':             ('Production & Infra', '#fb923c'),
    'hardware':          ('Production & Infra', '#fb923c'),
    'monitoring':        ('Production & Infra', '#fb923c'),
    'cloud-deploy':      ('Production & Infra', '#fb923c'),
    'traffic-cost':      ('Production & Infra', '#fb923c'),
    'data-governance':   ('Production & Infra', '#fb923c'),
    # Multimodal
    'image-gen':         ('Multimodal', '#e879f9'),
    'audio-models':      ('Multimodal', '#e879f9'),
    'video-models':      ('Multimodal', '#e879f9'),
    # Foundations
    'neural-nets':       ('Foundations', '#94a3b8'),
    'optimization':      ('Foundations', '#94a3b8'),
    'pytorch-basics':    ('Foundations', '#94a3b8'),
    'math-foundations':  ('Foundations', '#94a3b8'),
    'python-ecosystem':  ('Foundations', '#94a3b8'),
    # Strategy
    'decision-frameworks':   ('Strategy & Design', '#34d399'),
    'frontier-implications': ('Strategy & Design', '#34d399'),
}

CONCEPTS_DIR = ROOT / 'concepts'
EXCLUDED_FILES = {'index.html'}

# ── Extract title from HTML ────────────────────────────────────────────────────
def extract_title(html_content):
    """Extract title from <title> tag and strip ' — GenAI Mindmap' suffix."""
    match = re.search(r'<title>(.+?)</title>', html_content)
    if match:
        title = match.group(1)
        title = title.replace(' — GenAI Mindmap', '')
        return title.strip()
    return None

# ── Build category map with titles ─────────────────────────────────────────────
def build_category_map():
    """Build a dict: category → [(filename, title), ...]"""
    categories = {}
    
    # Get all concept files in CATEGORIES
    for filepath in sorted(CONCEPTS_DIR.glob('*.html')):
        if filepath.name in EXCLUDED_FILES:
            continue
        
        filename_without_ext = filepath.stem
        if filename_without_ext not in CATEGORIES:
            continue  # Skip files not in CATEGORIES
        
        html_content = filepath.read_text(encoding='utf-8')
        title = extract_title(html_content)
        if not title:
            continue
        
        cat_label, _ = CATEGORIES[filename_without_ext]
        if cat_label not in categories:
            categories[cat_label] = []
        
        categories[cat_label].append((filename_without_ext, title, filepath.name))
    
    # Sort each category alphabetically by title
    for cat_label in categories:
        categories[cat_label].sort(key=lambda x: x[1].lower())
    
    return categories

# ── Build navigation HTML ──────────────────────────────────────────────────────
def build_nav_html(category_label, prev_file=None, prev_title=None, 
                   next_file=None, next_title=None):
    """Build the <nav class="concept-nav"> HTML."""
    prev_link = ''
    if prev_file and prev_title:
        prev_link = f'<a class="concept-nav-prev" href="{prev_file}">← {prev_title}</a>'
    else:
        prev_link = '<span></span>'
    
    next_link = ''
    if next_file and next_title:
        next_link = f'<a class="concept-nav-next" href="{next_file}">→ {next_title}</a>'
    else:
        next_link = '<span></span>'
    
    nav_html = (
        '<nav class="concept-nav">\n'
        f'  {prev_link}\n'
        f'  <span class="concept-nav-cat">{category_label}</span>\n'
        f'  {next_link}\n'
        '</nav>'
    )
    return nav_html

# ── CSS for concept-nav ────────────────────────────────────────────────────────
CONCEPT_NAV_CSS = '''<style id="concept-nav-css">
.concept-nav{display:flex;align-items:center;justify-content:space-between;padding:1.5rem 2rem;background:#0f172a;border-top:1px solid rgba(255,255,255,0.08);font-size:0.85rem;flex-wrap:wrap;gap:0.75rem;}
.concept-nav a{color:#818cf8;text-decoration:none;font-weight:600;transition:color 0.15s;}
.concept-nav a:hover{color:#a78bfa;}
.concept-nav-cat{color:#64748b;font-size:0.78rem;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;}
</style>'''

# ── Main processing ────────────────────────────────────────────────────────────
def main():
    categories = build_category_map()
    updated_count = 0
    
    for cat_label, pages in categories.items():
        for idx, (filename_without_ext, title, html_filename) in enumerate(pages):
            filepath = CONCEPTS_DIR / html_filename
            html_content = filepath.read_text(encoding='utf-8')
            
            # Determine prev and next
            prev_file = prev_title = None
            next_file = next_title = None
            
            if idx > 0:
                prev_file = pages[idx - 1][2]  # html_filename
                prev_title = pages[idx - 1][1]  # title
            
            if idx < len(pages) - 1:
                next_file = pages[idx + 1][2]  # html_filename
                next_title = pages[idx + 1][1]  # title
            
            # Build nav HTML
            nav_html = build_nav_html(cat_label, prev_file, prev_title, 
                                      next_file, next_title)
            
            # Inject/replace nav before </body>
            nav_pattern = r'<nav class="concept-nav">.*?</nav>'
            if re.search(nav_pattern, html_content, re.DOTALL):
                # Replace existing nav
                html_content = re.sub(nav_pattern, nav_html, html_content, 
                                    count=1, flags=re.DOTALL)
            else:
                # Insert before </body>
                html_content = html_content.replace('</body>', 
                                                   nav_html + '\n</body>')
            
            # Inject CSS if not present
            if '<style id="concept-nav-css">' not in html_content:
                # Insert before </head>
                html_content = html_content.replace('</head>', 
                                                   CONCEPT_NAV_CSS + '\n</head>')
            
            filepath.write_text(html_content, encoding='utf-8')
            updated_count += 1
    
    print(f"concept-nav injected/updated in {updated_count} pages")

if __name__ == '__main__':
    main()
