# fix_paths.py — replace old PROJECT path with new one in all parser notebooks
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent
OLD = r"C:\\Users\\shlok\\projects\\ddp-llm"
NEW = r"C:\\Users\\shlok\\projects\\ddp-llm\\parser"

for nb_path in NOTEBOOKS_DIR.glob("*.ipynb"):
    text = nb_path.read_text(encoding="utf-8")
    if OLD not in text:
        print(f"  skip: {nb_path.name}  (no match)")
        continue
    n = text.count(OLD)
    new_text = text.replace(OLD, NEW)
    nb_path.write_text(new_text, encoding="utf-8")
    print(f"  fixed: {nb_path.name}  ({n} occurrence(s))")

print("\ndone.")