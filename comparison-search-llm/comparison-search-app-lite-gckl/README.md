# comparison-search-app-lite-gckl

Interactive comparison-based image search on the scenery dataset, using the **γ-CKLSearch** algorithm from Chumbalov et al. (2024), *Fast Interactive Search under a Scale-Free Comparison Oracle* (UAI 2024, Algorithm 3).

This is a variant of `comparison-search-app-lite` with the search engine swapped from the 2020 GAUSSSEARCH (Probit oracle, Gaussian belief in embedding space) to γ-CKLSearch (scale-free ratio oracle, discrete posterior over items). No LLM in the loop — clicks A/B directly drive the search.

The images are pulled from the public HuggingFace dataset `Nightshade2304/scenery-search-images` on first run (~40MB, cached under `~/.cache/huggingface/`).

## Modes

- **Human search** — you pick A or B, or click *Found it!* when you spot the image you had in mind.
- **Auto-run** — a simulated user answers via the γ-CKL oracle, so you can watch the search converge. Play / Pause / Step controls with Slow / Normal / Fast speed. The picked image is highlighted with a green border; the target is on the right with a red border.

## Setup

```bash
git clone https://github.com/Shlok-Shetty/Dual-Degree-Project.git
cd Dual-Degree-Project/comparison-search-llm/comparison-search-app-lite-gckl

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

streamlit run run_app.py
```

First launch will download the scenery images from HuggingFace (a couple of minutes on the first run, instant thereafter).

## Configuration

`app/config.py`:

- `GAMMA = 5.0` — oracle sharpness for γ-CKL. Higher = more deterministic oracle (closer item almost always picked); lower = noisier. The paper uses γ=3 for their D=5 user study; γ=5 was the best-performing setting on the scenery embeddings.
- `SIGMA_EPS = 0.05` — kept from the original engine, unused by γ-CKL.
- `DRIVE_GALLERY_URL` — points to the HuggingFace dataset page for browsing the full gallery.

## How is this different from the original lite app?

| | `comparison-search-app-lite` (GAUSS) | `comparison-search-app-lite-gckl` (γ-CKL) |
|---|---|---|
| Search algorithm | GAUSSSEARCH (Chumbalov 2020) | γ-CKLSearch (Chumbalov 2024, Alg. 3) |
| Oracle model | Probit | γ-CKL (scale-free) |
| Belief | Gaussian (μ, Σ) in ℝ^d | Discrete posterior over items |
| Skip button | Yes | No (removed) |
| Mean queries on 100-target eval | 21.6 | 12.8 |

Both apps are shareable side-by-side.

## References

- Chumbalov, Maystre, Grossglauser (2020). *Scalable and Efficient Comparison-based Search without Features*. ICML.
- Chumbalov, Klein, Maystre, Grossglauser (2024). *Fast Interactive Search under a Scale-Free Comparison Oracle*. UAI. arXiv:2306.01814