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

Requires Python 3.10 or newer.

### 1. Clone the repo and go into the lite app folder

```
git clone https://github.com/Shlok-Shetty/Dual-Degree-Project.git
cd Dual-Degree-Project/comparison-search-llm/comparison-search-app-lite
```

### 2. Create and activate a virtual environment

**Windows (cmd):**
```
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```
python3 -m venv .venv
source .venv/bin/activate
```

**Or with conda:**
```
conda create -n comparison-search-lite python=3.11 -y
conda activate comparison-search-lite
```

### 3. Install requirements

```
pip install -r requirements.txt
```

Only four packages: streamlit, numpy, scipy, pillow. No torch, no CUDA.
Install takes ~30 seconds.

### 4. Run

```
streamlit run run_app.py
```

Opens in your browser at `http://localhost:8501`. First load is instant —
no model loading.

## Notes

- `sigma_eps` is fixed at 0.05, same as the main app.
- Uses the same scenery data as the main app — expects
  `scenery-search/data/scenery_embedding.npz` and `intel_images/` to be
  present in the sibling folder.
- Skip button applies no belief update but consumes a step, so the search
  advances to a fresh pair.