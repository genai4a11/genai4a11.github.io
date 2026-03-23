#!/usr/bin/env python3
"""Re-generates data/stats.js from data/nodes.js and data/snippets.js.
Run from repo root: python3 scripts/gen-stats.py
"""
import re, pathlib

ROOT = pathlib.Path(__file__).parent.parent

nodes_src = (ROOT / 'data/nodes.js').read_text()
snip_src  = (ROOT / 'data/snippets.js').read_text()

all_ids    = re.findall(r"\{id:'([^']+)'", nodes_src)
node_count = len(all_ids) - 1  # minus root

snip_keys  = re.findall(r'(?:^|,)([a-z_][a-z_0-9]*):\{(?:use|diag|code|tip|refs)', snip_src, re.MULTILINE)
snip_count = len(snip_keys)
clusters   = 5

out = f"""// Auto-generated — do not edit by hand
// Regenerate: python3 scripts/gen-stats.py
const SITE_STATS = {{
  nodes:    {node_count},
  snippets: {snip_count},
  clusters: {clusters}
}};
"""
(ROOT / 'data/stats.js').write_text(out)
print(f"data/stats.js updated: nodes={node_count}, snippets={snip_count}, clusters={clusters}")
