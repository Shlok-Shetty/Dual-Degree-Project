"""All paths, model IDs, and hyperparams for the app.

Paths are resolved relative to this file so the folder works unchanged in both:
  C:\\Users\\shlok\\projects\\ddp-llm\\comparison-search-app\\
  C:\\Users\\shlok\\projects\\github\\Dual-Degree-Project\\comparison-search-llm\\comparison-search-app\\
"""
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_ROOT.parent  # ddp-llm/  or  comparison-search-llm/

# Sibling folders — same structure in both the working dir and the git repo.
SCENERY_ROOT = PROJECT_ROOT / "scenery-search"
SCENERY_DATA = SCENERY_ROOT / "data"
EMBEDDING_NPZ = SCENERY_DATA / "scenery_embedding.npz"
IMAGES_DIR = SCENERY_DATA / "intel_images"

# The 3B adapter lives in the parser project. In the working dir it's under
# ddp-llm/parser/checkpoints/qwen3b-qlora-v1/... — in the git repo the
# checkpoints/ folders are gitignored, so this path is only resolvable on
# the machine that has the adapter locally (or after downloading from HF).
PARSER_BASE = "Qwen/Qwen2.5-3B-Instruct"
PARSER_ADAPTER_LOCAL = PROJECT_ROOT / "parser" / "checkpoints" / "qwen3b-qlora-v1" / "checkpoint-240"
# Fallback: pull from HF Hub if the local adapter isn't there.
PARSER_ADAPTER_HF = "Nightshade2304/qwen3b-comparison-parser-v1"

VERBALIZER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SIGMA_EPS = 0.05
MAX_QUERIES = 50
DEFAULT_SEED = 42

# Placeholder — replace with your actual Drive folder URL after uploading images.
DRIVE_GALLERY_URL = "https://drive.google.com/drive/folders/PLACEHOLDER"

CLASS_COLORS = {
    "buildings": "#e74c3c",
    "forest":    "#27ae60",
    "glacier":   "#3498db",
    "mountain":  "#8b4513",
    "sea":       "#1abc9c",
    "street":    "#f39c12",
}

VERBALIZER_STYLES = [
    ("letter",     "Just say the letter: 'A' or 'B'. Nothing else."),
    ("ordinal",    "Say 'the first one' if they picked A, or 'the second one' if they picked B. Nothing else."),
    ("positional", "Say 'the left one' if they picked A, or 'the right one' if they picked B. Nothing else."),
    ("direct",     "Commit clearly to their choice in one short sentence. Examples: 'Option A.', 'I'll go with B.', 'A for sure.'"),
    ("casual",     "Casual and short, 3-6 words, but commit clearly. Examples: 'A works', 'B for me', 'gimme A'."),
]

PARSER_SYSTEM_PROMPT = """You are a parser that converts a user's natural language response into a subset of the shown options.

The user is shown 4 options labeled A, B, C, D and gives feedback about which are close to what they want. Your job is to output which options the user views favorably.

Output format (JSON only, nothing else):
- A JSON list of the favored labels, e.g. ["A", "B"]
- [] if the user explicitly rejects ALL options ("none of these", "all wrong")
- "*" if the utterance is off-topic OR expresses no usable preference ("I don't know", "they all look the same", "I love football")

Rules:
- Any positive signal about an option means it goes in the list.
- "X is better than Y" endorses only X, not Y.
- "X and Y are both good, X is better" endorses both X and Y.
- Negations like "not D" or "anything but B" mean the remaining options go in the list.
- Questions like "is it A?" are treated as tentative endorsement of A.

Output ONLY the JSON. No explanation, no prose."""


def resolve_parser_adapter() -> str:
    """Return a path or HF repo id that PeftModel.from_pretrained can consume."""
    if PARSER_ADAPTER_LOCAL.exists():
        return str(PARSER_ADAPTER_LOCAL)
    return PARSER_ADAPTER_HF
