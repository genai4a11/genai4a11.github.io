#!/usr/bin/env python3
"""Re-generates data/stats.js from live sources.

Counts:
  nodes        — from data/nodes.js
  snippets     — from data/snippets.js
  clusters     — hardcoded (update manually when topology changes)
  conceptPages — from concepts/*.html (excludes index.html)

Run from repo root:
  python3 scripts/gen-stats.py
"""
import re, pathlib

ROOT = pathlib.Path(__file__).parent.parent

# ── nodes ─────────────────────────────────────────────────────────────────────
nodes_src  = (ROOT / 'data/nodes.js').read_text()
all_ids    = re.findall(r"\{id:'([^']+)'", nodes_src)
node_count = len(all_ids) - 1          # minus the synthetic root node

# ── snippets ──────────────────────────────────────────────────────────────────
snip_src  = (ROOT / 'data/snippets.js').read_text()
snip_keys = re.findall(
    r'(?:^|,)([a-z_][a-z_0-9]*):\{(?:use|diag|code|tip|refs)',
    snip_src, re.MULTILINE
)
snip_count = len(snip_keys)

# ── concept pages ─────────────────────────────────────────────────────────────
concepts_dir   = ROOT / 'concepts'
EXCLUDED_FILES = {'index.html'}          # directory page, not a concept article
concept_files  = sorted(
    f for f in concepts_dir.glob('*.html')
    if f.name not in EXCLUDED_FILES
)
concept_count  = len(concept_files)

# ── clusters (static) ─────────────────────────────────────────────────────────
clusters = 5

# ── write ──────────────────────────────────────────────────────────────────────
out = (
    "// Auto-generated — do not edit by hand\n"
    "// Regenerate: python3 scripts/gen-stats.py\n"
    "const SITE_STATS = {\n"
    f"  nodes:        {node_count},\n"
    f"  snippets:     {snip_count},\n"
    f"  clusters:     {clusters},\n"
    f"  conceptPages: {concept_count},\n"
    "};\n"
)
(ROOT / 'data/stats.js').write_text(out)
print(
    f"data/stats.js  →  nodes={node_count}, snippets={snip_count}, "
    f"clusters={clusters}, conceptPages={concept_count}"
)
