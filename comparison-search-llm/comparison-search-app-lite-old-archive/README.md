# comparison-search-app-lite

The LLM-free version of the comparison-search demo. Same GAUSSSEARCH engine,
same scenery images, no parser and no verbalizer — you just click A, B, or
Skip on each pair.

Sits alongside `comparison-search-app/`, `parser/`, `search-algorithm/`, and
`scenery-search/`.

## Why this version exists

The main app needs a GPU to run the parser (Qwen 3B, 4-bit) and verbalizer
(Qwen 1.5B). This one doesn't need any of that. Just Python + streamlit.
Startup is instant.

## Flow

- **home** — start search
- **search** — pair of images with three buttons per pair: Pick A, Pick B,
  or Skip (neither is close). Click **Found it!** under whichever image
  matches what you had in mind.
- **result** — what you found + full query history

The search stops automatically when the target you were looking for appears
in the query pair.

## Layout

```
comparison-search-app-lite/
├── README.md
├── requirements.txt
├── run_app.py
└── app/
    ├── __init__.py
    ├── config.py           # paths, sigma_eps, Drive link
    ├── embeddings.py       # loads scenery_embedding.npz
    ├── search_engine.py    # GAUSSSEARCH — one step at a time
    └── streamlit_app.py    # the UI
```

## Setup

```
pip install -r comparison-search-app-lite/requirements.txt
```

That's it. No torch, no transformers, no CUDA. Runs on any machine.

## Run

```
streamlit run comparison-search-app-lite/run_app.py
```

Opens instantly since there's no model loading.

## Notes

- `sigma_eps` is fixed at 0.05, same as the main app.
- Uses the same scenery data as the main app — expects
  `scenery-search/data/scenery_embedding.npz` and `intel_images/` to be
  present in the sibling folder.
- Skip button applies no belief update but consumes a step, so the search
  advances to a fresh pair.