
# extract_thesis.py
# Reads generate_thesis_v2.py, extracts the LaTeX content between the raw string markers,
# and writes it directly to thesis.tex

import re

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
generate_thesis_path = os.path.join(script_dir, "generate_thesis_v2.py")
src = open(generate_thesis_path, "r", encoding="utf-8").read()

# The raw string starts after 'THESIS = r"""' and ends just before the line 'output_path = r"...'
# We locate the start marker
marker_start = 'THESIS = r"""\n'
marker_end = '\n"""\n\nimport os\nscript_dir'

start_pos = src.index(marker_start) + len(marker_start)
end_pos   = src.index(marker_end, start_pos)

thesis_content = src[start_pos:end_pos]

out_path = os.path.join(script_dir, "thesis.tex")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(thesis_content)

print(f"thesis.tex written: {len(thesis_content.splitlines())} lines, {len(thesis_content.encode())} bytes")
