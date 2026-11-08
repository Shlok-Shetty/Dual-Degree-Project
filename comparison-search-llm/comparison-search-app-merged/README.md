# comparison-search-app-merged

Streamlit interface tying together the parser (Qwen 3B + LoRA), both search
engines (GAUSSSEARCH and γ-CKLSearch), and the scenery embeddings.

Sits alongside `parser/`, `search-algorithm/`, and `scenery-search/`.

## Flow

The app opens on a **home screen** where you pick a search algorithm
(GAUSSSEARCH or γ-CKLSearch) and then one of two modes:

- **Human search** — you have a target image in mind. The system shows pairs
  and you type which one is closer, or click **Found it!** under whichever
  image matches. When you finish, the app reveals the actual target, tells you
  whether the two match, and shows the full query history.
- **LLM simulation** — a verbalizer LLM plays the user. You pick parameters
  (σ_ε for GAUSSSEARCH, γ for γ-CKL, plus seed and max queries), then step
  through or auto-run. Each step shows a 3-panel view: query A, query B,
  target.

Both modes stop automatically when the target appears in the query pair.
`Home` on any screen returns to the top.

## Search algorithms

- **GAUSSSEARCH** — Chumbalov et al. (2020). Probit oracle, Gaussian belief
  in embedding space. Noise parameter `σ_ε` (default 0.05).
- **γ-CKLSearch** — Chumbalov et al. (2024), Algorithm 3. Scale-free ratio
  oracle, discrete posterior over items. Sharpness parameter `γ`
  (default 5.0).

Both engines share the same interface, so the UI code and the LLM round-trip
work identically for either.

## Layout

```
comparison-search-app-merged/
├── README.md
├── requirements.txt
├── run_app.py
├── .gitignore
└── app/
    ├── __init__.py
    ├── config.py                 # paths, model IDs, sigma_eps, gamma, Drive link
    ├── embeddings.py             # loads scenery_embedding.npz + scale fix
    ├── search_engine.py          # SearchEngine — GAUSSSEARCH
    ├── gamma_ckl_engine.py       # GammaCKLEngine — γ-CKLSearch (Algorithm 3)
    ├── parser_llm.py             # ParserLLM — Qwen 3B + LoRA
    ├── verbalizer_llm.py         # VerbalizerLLM — Qwen 1.5B, prompted
    └── streamlit_app.py          # the UI (screen dispatcher)
```

## Setup

**Hardware.** The app picks a backend automatically (`app/devices.py`):

| Backend | Parser (Qwen 3B + LoRA) | Verbalizer (Qwen 1.5B) | Memory |
|---|---|---|---|
| NVIDIA GPU (CUDA) | 4-bit NF4, adapter separate | fp16 | ~2.2 GB VRAM human mode, ~5.3 GB LLM mode |
| Apple Silicon (MPS) | fp16, adapter merged | fp16 | ~9 GB unified memory for both |
| CPU only | bf16, adapter merged | bf16 | ~9 GB RAM for both; ~3 s per parse |

Force one with `COMPARISON_SEARCH_DEVICE=cuda|mps|cpu`. On MPS/CPU the
parser reuses the cached system-prompt KV between calls.

The parser adapter was trained on the 4-bit base, so on MPS/CPU it answers a
few hard cases differently: on the 210-example parser test set it scores
204/210 in fp16 versus 207/210 in 4-bit, mostly on ordinal negations such as
"the third one is off". Of 28 two-option app-style cases, only "yes" and
"bingo" differ, and both backends get those wrong. For a quick demo of the search algorithms alone, use
`comparison-search-app-lite`, which needs no LLMs.

Tested with: Python 3.12, torch 2.11.0+cu128, transformers 5.12.1,
peft 0.19.1, accelerate 1.14.0, bitsandbytes 0.49.2, streamlit 1.61.1
(Windows 11, RTX 5060 Laptop).

### 1. Install torch (per your platform)

torch is NOT in `requirements.txt` — the wheel is specific to your
CUDA / CPU setup, and pinning one here would break every other machine.
Pick the one that matches:

```powershell
# CUDA 12.8 (Blackwell: RTX 50-series)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1 (most common)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (older GPUs)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# macOS (Apple Silicon, uses MPS) or CPU-only
pip install torch
```

If unsure which CUDA you have, run `nvidia-smi` — the top-right corner shows
the maximum CUDA your driver supports. Pick the matching index.

### 2. Install the rest

```powershell
pip install -r requirements.txt
```

This covers transformers, peft, bitsandbytes, scipy, numpy, pillow,
streamlit, huggingface_hub.

### 3. Check paths in `config.py`

- `DRIVE_GALLERY_URL` — points to the HuggingFace dataset by default
  (`Nightshade2304/scenery-search-images`). Change if you have your own
  gallery link.
- `PARSER_ADAPTER_LOCAL` — defaults to
  `<project_root>/parser/checkpoints/qwen3b-qlora-v2/checkpoint-final`.
  If the local adapter isn't there, the app falls back to the HF Hub copy
  at `Nightshade2304/qwen3b-comparison-parser-v2` automatically.

## Run

```powershell
streamlit run run_app.py
```

The parser (~60s first load) is loaded lazily when you enter either mode,
with a visible progress bar. It's cached after that, so subsequent searches
start instantly.

## Design notes

- **Human mode does not show belief internals.** No σ_ε / γ slider, no
  tr(Σ), no distance-to-target — those would leak information about the
  target. You only see the query count and (on completion) the target
  reveal.
- **LLM mode shows everything.** The relevant algorithm parameter (σ_ε or γ),
  the seed, and the target class are all visible on the setup screen and
  the step-through view. For GAUSSSEARCH you see tr(Σ) and distance to
  target; for γ-CKL you see the current argmax class and `P[target]`.
- Under-image `Found it!` buttons are per-image, so it's unambiguous which
  one you picked. The completion screen shows both what you clicked and
  the actual target.
- Every search screen displays a small badge (`engine: SearchEngine (σε=…)`
  or `engine: GammaCKLEngine (γ=…)`) so there's no ambiguity about which
  engine is actually running.

## Troubleshooting

- **`RuntimeError: operator torchvision::nms does not exist`** — your
  torchvision was built against a different torch than you have installed.
  The app doesn't use torchvision; the quickest fix is
  `pip uninstall -y torchvision`. If you want to keep it, reinstall it
  from the same CUDA index you used for torch.
- **CUDA out of memory** on parser load — the parser is 4-bit quantized and
  uses ~2 GB VRAM. If something else is on the GPU, `nvidia-smi` →
  `taskkill /F /IM python.exe` (Windows) or `pkill python` (Linux).
- **Images missing in the UI** — the app auto-downloads them from
  HuggingFace on first run. If offline, point `IMAGES_DIR` in `config.py`
  at a local copy.

## Swapping internals

Each component is a class with a small surface. Swaps are one-import changes:

- **Parser** — replace `ParserLLM` with anything exposing
  `.parse(utterance, options) -> (parsed, raw_text)`.
- **Verbalizer** — replace `VerbalizerLLM` with anything exposing
  `.verbalize(picked_letter, rng) -> (utterance, style_name)`.
- **Search algorithm** — replace `SearchEngine` / `GammaCKLEngine` with
  anything exposing `.propose_query() -> (i, j)`,
  `.oracle_answer(i, j) -> int`,
  `.apply_answer(y, **meta) -> record`, `.done`, `.step`,
  `.step_records`, `.belief_stats()`.

## Notes

- σ_ε defaults to 0.05 (closed-loop notebook value); γ defaults to 5.0.
  Only exposed in LLM mode.
- RNG streams for search and verbalizer are separated so verbalizer
  generation doesn't shift which items the search picks.
- Parser prompt is the 4-option one you trained on; called with
  `options=("A", "B")`. OOD tests showed this generalizes cleanly.