#!/usr/bin/env python3
"""Inject 'Related concepts' section into concept pages that are missing one.

For each concept page without a .related-concepts div:
  1. Find the matching node in data/nodes.js via the CONCEPT_PAGES map in app.js
  2. Collect related nodes: parent, children, siblings (same parent)
  3. Filter to nodes that have their own concept pages
  4. Pick up to 5, coloured by cluster
  5. Inject HTML before <footer or before the closing </div><nav> block

Already-present sections are left untouched (idempotent).

Run from repo root:
  python3 scripts/gen-related.py
"""
import re
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
CONCEPTS_DIR = ROOT / 'concepts'

# ── Cluster accent colours ────────────────────────────────────────────────────
CAT_COLORS = {
    'mc_foundations': '#94a3b8', 'foundations': '#94a3b8', 'ml_core': '#94a3b8',
    'transformers': '#94a3b8', 'llm_arch': '#94a3b8', 'multimodal': '#94a3b8',
    'mc_building': '#a78bfa', 'prompting': '#a78bfa', 'rag': '#a78bfa',
    'agents': '#a78bfa', 'finetuning': '#a78bfa', 'devtools': '#a78bfa',
    'data_eng': '#a78bfa',
    'mc_production': '#fb923c', 'sysdesign': '#fb923c', 'infra': '#fb923c',
    'prod_eng': '#fb923c',
    'mc_governance': '#f472b6', 'eval': '#f472b6', 'safety': '#f472b6',
    'mc_apps': '#34d399', 'apps': '#34d399',
}


def accent(cat):
    return CAT_COLORS.get(cat, '#64748b')


# ── Parse nodes.js ────────────────────────────────────────────────────────────
def load_nodes():
    text = (ROOT / 'data/nodes.js').read_text(encoding='utf-8')
    nodes = {}
    pat = re.compile(
        r"\{id:'([^']+)',label:'([^']+)',cat:'([^']+)',g:\d+(?:,meta:true)?,p:'?([^',}\s]+)'?"
    )
    for m in pat.finditer(text):
        nid, label, cat, parent = m.groups()
        nodes[nid] = {'id': nid, 'label': label, 'cat': cat, 'p': parent}
    return nodes


# ── Parse CONCEPT_PAGES from app.js ──────────────────────────────────────────
def load_concept_pages():
    text = (ROOT / 'app.js').read_text(encoding='utf-8')
    cp_match = re.search(r'const CONCEPT_PAGES=\{(.*?)\};', text, re.DOTALL)
    if not cp_match:
        return {}, {}
    cp = {}
    for m in re.finditer(r"(\w+):'(concepts/[^']+)'", cp_match.group(1)):
        cp[m.group(1)] = m.group(2).replace('concepts/', '')
    # reverse: filename → node_id
    rev = {v: k for k, v in cp.items()}
    return cp, rev


# ── Build related-concepts HTML ───────────────────────────────────────────────
def build_related_block(node_id, nodes, concept_pages):
    node = nodes.get(node_id)
    if not node:
        return None

    parent_id = node['p']
    candidates = []

    # 1. Children
    candidates += [n for n in nodes.values() if n['p'] == node_id]
    # 2. Siblings (same parent, not self)
    if parent_id:
        candidates += [n for n in nodes.values()
                       if n['p'] == parent_id and n['id'] != node_id]
    # 3. Parent itself
    if parent_id and parent_id in nodes:
        candidates.append(nodes[parent_id])

    # Deduplicate preserving order
    seen = set()
    picks = []
    for c in candidates:
        if c['id'] not in seen and c['id'] in concept_pages:
            seen.add(c['id'])
            picks.append(c)
        if len(picks) >= 5:
            break

    if not picks:
        return None

    cards = ''
    for c in picks:
        url   = concept_pages[c['id']]
        color = accent(c['cat'])
        label = c['label'].replace('&', '&amp;').replace('<', '&lt;')
        cards += (f'<a class="rc-card" href="{url}" style="--rc-accent:{color}">'
                  f'<span class="rc-label">{label}</span></a>\n')

    return (
        '\n<div class="related-concepts">\n'
        '<h3 class="rc-heading">Related concepts</h3>\n'
        '<div class="rc-grid">\n'
        + cards +
        '</div>\n</div>\n'
    )


# ── Inject into a single HTML file ───────────────────────────────────────────
def inject_related(filepath, block):
    html = filepath.read_text(encoding='utf-8')
    if 'related-concepts' in html:
        return False  # already present
    if '<footer' in html:
        html = html.replace('<footer', block + '<footer', 1)
    elif '</div>\n<nav class="concept-nav"' in html:
        html = html.replace('</div>\n<nav class="concept-nav"',
                            block + '</div>\n<nav class="concept-nav"', 1)
    else:
        return False  # can't find insertion point
    filepath.write_text(html, encoding='utf-8')
    return True


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    nodes = load_nodes()
    concept_pages, rev_map = load_concept_pages()

    updated = 0
    skipped = 0

    for filepath in sorted(CONCEPTS_DIR.glob('*.html')):
        if filepath.name == 'index.html':
            continue
        html = filepath.read_text(encoding='utf-8')
        if 'related-concepts' in html:
            continue  # already done

        # Find node id from filename
        node_id = rev_map.get(filepath.name)
        if not node_id:
            skipped += 1
            continue

        block = build_related_block(node_id, nodes, concept_pages)
        if not block:
            skipped += 1
            continue

        if inject_related(filepath, block):
            updated += 1
        else:
            skipped += 1

    print(f"Related concepts injected in {updated} pages ({skipped} skipped/already present)")


if __name__ == '__main__':
    main()
