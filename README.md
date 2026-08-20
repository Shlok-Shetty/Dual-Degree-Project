# Dual Degree Project — Interactive Comparison-Based Search

An LLM-powered interactive search system built on the algorithms from Chumbalov et al. (2020 / 2024). You find a target image not by describing it, but by comparing pairs — the system shows two images, you say which is closer, and the search narrows down until it finds what you had in mind.

## Repo layout

Everything lives under `comparison-search-llm/`:

```
comparison-search-llm/
├── parser/                    fine-tuned LLM that turns free-form user replies into structured answers
├── search-algorithm/          from-scratch GAUSSSEARCH (2020) implementation and validation
├── scenery-search/            search on real image data (Intel scenery, CLIP embeddings)
└── comparison-search-app/     Streamlit interface that wires all three together
```

Each subfolder is a working piece of the project:

- **`parser/`** — Qwen 2.5-3B fine-tuned with QLoRA on ~1500 examples. Converts things like "the left one" or "not really either" into a structured subset over the shown options.
- **`search-algorithm/`** — pure NumPy implementation of GAUSSSEARCH from the 2020 paper (Probit oracle, SAMPLEMIRROR, ADF update). 
- **`scenery-search/`** — puts the search on 600 real images (Intel Image Classification, embedded with CLIP ViT-B/32). Includes belief animations.
- **`comparison-search-app/`** — Streamlit app that runs the full loop. Two modes: a human types responses, or a verbalizer LLM simulates one.

## Quick start

For the app , see the setup instructions in [`comparison-search-llm/comparison-search-app/README.md`](comparison-search-llm/comparison-search-app/README.md).

For the individual pieces, each subfolder is a set of Jupyter notebooks you can run top-to-bottom.

## Reading order

If you want to understand how the project came together, work through it in this order:

1. `comparison-search-llm/parser/notebooks/` — start with `01_smoke_test.ipynb`, then `02_baseline.ipynb`, `03_synth_gen.ipynb`, and the two training notebooks. This is the LLM side.
2. `comparison-search-llm/search-algorithm/notebooks/01_gauss_search.ipynb` — the from-scratch algorithm with all the validation experiments.
3. `comparison-search-llm/scenery-search/notebooks/` — `01_scenery_setup.ipynb` (CLIP embeddings), then `03_gauss_search_3d_viz.ipynb` (real-data search + animations).
4. `comparison-search-llm/comparison-search-app/` — the app that ties it all together.

