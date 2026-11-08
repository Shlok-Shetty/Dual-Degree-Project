# Dual Degree Project — Interactive Comparison-Based Search

An LLM-powered interactive search system built on the algorithms from Chumbalov et al. (2020 / 2024). You find a target image not by describing it, but by comparing pairs — the system shows two images, you say which is closer, and the search narrows down until it finds what you had in mind.

## Repo layout

Everything lives under `comparison-search-llm/`:

```
comparison-search-llm/
├── parser/                          fine-tuned LLM that turns free-form user replies into structured answers
├── search-algorithm/                from-scratch GAUSSSEARCH (2020) and γ-CKLSearch (2024) implementations and validation
├── scenery-search/                  search on real image data (Intel scenery, CLIP embeddings), closed-loop LLM evaluation
├── comparison-search-app-merged/    Streamlit app: parser + verbalizer + both search engines (needs an NVIDIA GPU)
├── comparison-search-app-lite/      Streamlit app without LLMs: both search engines, runs on CPU
├── *-archive/                       earlier app versions kept for reference
└── requirements.txt                 packages for the notebooks (torch installed separately)
```

Each subfolder is a working piece of the project:

- **`parser/`** — Qwen 2.5-3B fine-tuned with QLoRA. Three training sets (`train_v1.jsonl` 1278, `train_v2.jsonl` 1451, `train_v3.jsonl` 1598 examples) and a fixed 210-example test set. Converts things like "the left one" or "not really either" into a structured subset over the shown options. The production adapter (v2) is on Hugging Face at `Nightshade2304/qwen3b-comparison-parser-v2`.
- **`search-algorithm/`** — pure NumPy implementations of GAUSSSEARCH from the 2020 paper (Probit oracle, SAMPLEMIRROR, ADF update) and γ-CKLSearch from the 2024 paper (Algorithm 3), on synthetic data.
- **`scenery-search/`** — puts both searches on 600 real images (Intel Image Classification, embedded with CLIP ViT-B/32, PCA to 10D). Includes belief animations, the closed-loop LLM evaluation, and a both-similar response experiment.
- **`comparison-search-app-merged/`** — Streamlit app that runs the full loop with a runtime choice of search engine. Two modes: a human types responses, or a verbalizer LLM simulates one.
- **`comparison-search-app-lite/`** — the same search engines with click-based answers and no LLMs, for a quick demo on any machine.

## Quick start

- Full app: see [`comparison-search-llm/comparison-search-app-merged/README.md`](comparison-search-llm/comparison-search-app-merged/README.md).
- Lite app: see [`comparison-search-llm/comparison-search-app-lite/README.md`](comparison-search-llm/comparison-search-app-lite/README.md).
- Notebooks: install torch for your platform first (see the merged app README), then `pip install -r comparison-search-llm/requirements.txt`. Each subfolder is a set of Jupyter notebooks you can run top-to-bottom. Scenery images are downloaded from `Nightshade2304/scenery-search-images` on Hugging Face.

## Reading order

If you want to understand how the project came together, work through it in this order:

1. `comparison-search-llm/parser/notebooks/` — run `01` to `11` in order: smoke test, 1.5B prompted baseline, synthetic data + split (`train_v1.jsonl`), 1.5B and 3B v1 training, 3B prompted baseline, adversarial set + v2 retrain (`train_v2.jsonl`), v2 eval + inference prompt, confirmation data v3 (`train_v3.jsonl`), model-size comparison, verbalizer audit. This is the LLM side.
2. `comparison-search-llm/search-algorithm/notebooks/` — `01_gauss_search.ipynb` (GAUSSSEARCH with its validation experiments), `02_visualization.ipynb` (belief animations), `03_gamma_ckl_search.ipynb` (γ-CKLSearch on synthetic data).
3. `comparison-search-llm/scenery-search/notebooks/` — `01_scenery_setup.ipynb` (CLIP embeddings), `02_gauss_search_3d_viz.ipynb` (GAUSSSEARCH on real images), `03_closed_loop_v1.ipynb` and `04_closed_loop_v2.ipynb` (closed-loop LLM evaluation, before and after the validation gate), `05_gamma_ckl_scenery.ipynb` (γ-CKLSearch on real images), `06_closed_loop_gckl.ipynb` (γ-CKL closed loop + robustness), `07_both_similar_gckl.ipynb` (both-similar responses). Superseded notebooks are in `_archive/`.
4. `comparison-search-llm/comparison-search-app-merged/` — the app that ties it all together.
