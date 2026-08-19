# comparison-search-app

Streamlit interface tying together the parser (Qwen 3B + LoRA), GAUSSSEARCH,
and the scenery embeddings.

Sits alongside `parser/`, `search-algorithm/`, and `scenery-search/`.

## Flow

The app opens on a **home screen** where you pick one of two paths:

- **Human search** — you have a target image in mind. The system shows pairs
  and you type which one is closer, or click **Found it!** under whichever
  image matches. When you finish, the app reveals the actual target, tells you
  whether the two match, and shows the full query history.
- **LLM simulation** — a verbalizer LLM plays the user. You pick target and
  parameters (σ_ε, seed, max queries), then step through or auto-run.
  Each step shows a 3-panel view: query A, query B, target. The image the
  parser routed the reply to gets a green border; rejected gets a gray one.
  Below: verbalizer utterance, parser output, round-trip status.

Both modes stop automatically when the target appears in the query pair.
`← Home` on any screen returns to the top.

## Layout

```
comparison-search-app/
├── README.md
├── requirements.txt
├── run_app.py                    # streamlit entry point
├── .gitignore
└── app/
    ├── __init__.py
    ├── config.py                 # paths, model IDs, sigma_eps, Drive link
    ├── embeddings.py             # loads scenery_embedding.npz + scale fix
    ├── search_engine.py          # SearchEngine — GAUSSSEARCH, one step at a time
    ├── parser_llm.py             # ParserLLM — Qwen 3B + LoRA
    ├── verbalizer_llm.py         # VerbalizerLLM — Qwen 1.5B, prompted
    └── streamlit_app.py          # the UI (screen dispatcher)
```

## Setup

Reuses the main `ddp-llm` venv (torch, transformers, peft, bitsandbytes, scipy,
numpy). Just add streamlit:

```powershell
cd C:\Users\shlok\projects\ddp-llm
.venv\Scripts\activate
pip install -r comparison-search-app\requirements.txt
```

`config.py` resolves paths relative to itself, so the folder works unchanged in
both the working dir and the git repo — no path edits needed.

Two things to check in `config.py`:

- `DRIVE_GALLERY_URL` — replace `PLACEHOLDER` with your Drive folder link.
- `PARSER_ADAPTER_LOCAL` — defaults to
  `<project_root>/parser/checkpoints/qwen3b-qlora-v1/checkpoint-240`. If the
  local adapter isn't there, the app falls back to the HF Hub copy at
  `Nightshade2304/qwen3b-comparison-parser-v1` automatically.

## Run

```powershell
streamlit run comparison-search-app\run_app.py
```

The parser (~60s first load) is loaded lazily when you enter either mode, with
a visible progress bar. It's cached after that, so subsequent searches start
instantly.

## Design notes

- **Human mode does not show belief internals.** No sigma_eps slider, no tr(Sigma),
  no distance-to-target — those would leak information about the target. You
  only see the query count and (on completion) the target reveal.
- **LLM mode shows everything.** It's a simulation of the closed loop —
  sigma_eps, seed, and target class are all visible on the setup screen and the
  step-through view.
- Under-image `Found it!` buttons are per-image, so it's unambiguous which
  one you picked. The completion screen shows both what you clicked and the
  actual target, so it's obvious whether you matched or not.

## Swapping internals

Each component is a class with a small surface. Swaps are one-import changes:

- **Parser** — replace `ParserLLM` with anything exposing
  `.parse(utterance, options) -> (parsed, raw_text)`.
- **Verbalizer** — replace `VerbalizerLLM` with anything exposing
  `.verbalize(picked_letter, rng) -> (utterance, style_name)`.
- **Search algorithm** — replace `SearchEngine` with anything exposing
  `.propose_query() -> (i, j)`, `.apply_answer(y, **meta) -> record`,
  `.done`, `.step`, `.step_records`, `.belief_stats()`. This is where a
  gamma-CKL implementation slots in.

## Notes

- sigma_eps defaults to 0.05 (closed-loop notebook value). Only exposed in LLM mode.
- RNG streams for search and verbalizer are separated so verbalizer generation
  doesn't shift which items the search picks (session 5 handover, RNG drift).
- Parser prompt is the 4-option one you trained on; called with
  `options=("A", "B")`. OOD tests showed this generalizes cleanly (session 3).
- Image paths in `scenery_embedding.npz` are absolute Windows paths.
  `embeddings.py` falls back to the local `IMAGES_DIR` if the stored path
  doesn't exist on the current machine.
