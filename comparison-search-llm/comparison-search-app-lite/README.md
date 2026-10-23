# comparison-search-app-lite

The LLM-free comparison-search demo on the scenery dataset. Pick an
algorithm on the home screen (GAUSSSEARCH or γ-CKLSearch), then either
drive the search by hand or watch the simulated-user auto-run.

Both algorithms are implemented from scratch:

- **GAUSSSEARCH** — Chumbalov et al. (2020). Probit oracle, Gaussian
  belief in embedding space.
- **γ-CKLSearch** — Chumbalov et al. (2024), Algorithm 3. Scale-free
  ratio oracle, discrete posterior over items.

The images are pulled from the public HuggingFace dataset
`Nightshade2304/scenery-search-images` on first run (~40 MB, cached
under `~/.cache/huggingface/`).

## Modes

- **Human search** — you pick A, B, or Skip, or click *Found it!* when
  you spot the image you had in mind.
- **Auto-run** — a simulated user answers via the selected algorithm's
  oracle so you can watch the search converge. Play / Pause / Step
  controls with Slow / Normal / Fast speed. The picked image is
  highlighted with a green border; the target sits on the right with a
  red border.

## Layout

```
comparison-search-app-lite/
├── README.md
├── requirements.txt
├── run_app.py
└── app/
    ├── __init__.py
    ├── config.py             # paths, sigma_eps, gamma, HF link
    ├── embeddings.py         # loads scenery_embedding.npz
    ├── search_engine.py      # GAUSSSEARCH engine
    ├── gamma_ckl_engine.py   # γ-CKL engine
    └── streamlit_app.py      # the UI
```

## Setup

```
pip install -r requirements.txt
```

No torch, no transformers, no CUDA. Runs on any machine.

## Run

```
streamlit run run_app.py
```

First launch downloads the scenery images from HuggingFace (a couple of
minutes), instant thereafter. The app opens in your browser.

## Config

`app/config.py`:

- `SIGMA_EPS = 0.05` — probit noise for GAUSSSEARCH.
- `GAMMA = 5.0` — oracle sharpness for γ-CKL. Higher = more
  deterministic oracle (closer item almost always picked); lower =
  noisier. γ=5 was the best-performing setting on the scenery
  embeddings.
- `DRIVE_GALLERY_URL` — HuggingFace dataset page for browsing the full
  gallery.

## Notes

- Uses the same scenery data as the main app — expects
  `scenery_embedding.npz` next to it in `scenery-search/data/`.
- Skip button applies no belief update but consumes a step so the
  search advances to a fresh pair. Available in human mode for both
  algorithms.
