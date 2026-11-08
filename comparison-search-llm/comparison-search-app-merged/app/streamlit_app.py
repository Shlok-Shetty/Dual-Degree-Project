"""Streamlit interface for comparison-based scenery search.

Supports two engines: GAUSSSEARCH (Chumbalov 2020) and γ-CKLSearch
(Chumbalov 2024, Algorithm 3). The user picks one on the home screen and
both modes (human search, LLM simulation) use the chosen engine.

Screen flow:
    home           -> pick algorithm, then Human search or LLM simulation
    human_search   -> 2 images + text input + "Found it!" under each image
    human_result   -> what you found + full history with images
    llm_setup      -> configure algo-specific params -> start
    llm_search     -> 3-panel view + step controls, single-pass render
    llm_result     -> completion + fidelity stats + history + New / Home
"""
import base64
import io
import time
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from . import config
from .answers import parsed_to_y
from .embeddings import Embeddings
from .search_engine import SearchEngine
from .gamma_ckl_engine import GammaCKLEngine


IMG_WIDTH_HUMAN = 340
IMG_WIDTH_LLM = 280
IMG_WIDTH_HISTORY = 180
HUMAN_MAX_QUERIES = 50
HUMAN_FEEDBACK_S = 1.0
HUMAN_FEEDBACK_SKIP_S = 1.8

# status from parsed_to_y -> (A highlight, B highlight, message, st alert kind).
# "clean" highlights are filled in from y at runtime. Everything except "clean"
# is a skip in the engines: both items are marked used, the belief is not
# updated, and the next pair is drawn from the same belief.
HUMAN_FEEDBACK = {
    "clean":      (None, None, "", "success"),
    "both":       ("chosen", "chosen",
                   "parser read **both** - no belief update, showing a new pair", "info"),
    "reject_all": ("rejected", "rejected",
                   "parser read **neither** - no belief update, showing a new pair", "info"),
    "uncertain":  (None, None,
                   "parser read **unsure** - no belief update, showing a new pair", "info"),
    "malformed":  (None, None,
                   "could not read that answer - no update, showing a new pair", "warning"),
}
LLM_AUTORUN_PAUSE_S = 2.5

ALGO_OPTIONS = [config.ALGO_GAUSS, config.ALGO_GCKL]


COMPACT_CSS = """
<style>
    header[data-testid="stHeader"] { height: 0rem; visibility: hidden; }
    div.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    button[title="View fullscreen"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    .algo-sub {
        text-align: center;
        font-size: 0.95rem;
        color: #aaa;
        margin-top: -8px;
        margin-bottom: 8px;
    }
    .algo-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        background: #2a2a2a;
        color: #ddd;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
"""


# ==================================================================
# Cached loaders
# ==================================================================

@st.cache_resource(show_spinner=False)
def load_embeddings() -> Embeddings:
    return Embeddings()


@st.cache_resource(show_spinner=False)
def load_parser():
    from .parser_llm import ParserLLM
    return ParserLLM()


@st.cache_resource(show_spinner=False)
def load_verbalizer():
    from .verbalizer_llm import VerbalizerLLM
    return VerbalizerLLM()


def backend_label() -> str:
    from .devices import describe, pick_device
    return describe(pick_device())


def ensure_parser_loaded():
    if st.session_state.get("_parser_loaded"):
        return load_parser()
    with st.spinner(
        f"Loading parser (Qwen 2.5-3B + LoRA{', 4-bit' if backend_label() == 'CUDA GPU' else ''}) "
        f"on {backend_label()}. "
        "The first run downloads the model and can take a few minutes."
    ):
        p = load_parser()
    st.session_state._parser_loaded = True
    return p


def ensure_verbalizer_loaded():
    if st.session_state.get("_verbalizer_loaded"):
        return load_verbalizer()
    with st.spinner(f"Loading verbalizer LLM (Qwen 1.5B) on {backend_label()}..."):
        v = load_verbalizer()
    st.session_state._verbalizer_loaded = True
    return v


# ==================================================================
# Session helpers
# ==================================================================

def go(screen: str):
    st.session_state.screen = screen
    st.rerun()


def reset_all():
    """Clear everything except model-loaded flags and the current algo pick."""
    keep = {"_parser_loaded", "_verbalizer_loaded", "algo"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]


def current_algo() -> str:
    return st.session_state.get("algo", config.DEFAULT_ALGO)


def new_engine(emb: Embeddings, *, target_idx: int | None = None,
               seed: int | None = None,
               sigma_eps: float = config.SIGMA_EPS,
               gamma: float = config.GAMMA,
               max_queries: int = HUMAN_MAX_QUERIES):
    """Factory: build whichever engine the user picked on the home screen."""
    algo = current_algo()
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 1_000_000))
    if algo == config.ALGO_GCKL:
        return GammaCKLEngine(
            X=emb.X,
            gamma=gamma,
            target_idx=target_idx,
            seed=seed,
            max_queries=max_queries,
        )
    return SearchEngine(
        X=emb.X,
        sigma_eps=sigma_eps,
        target_idx=target_idx,
        seed=seed,
        max_queries=max_queries,
    )


def algo_subtitle(engine=None) -> str:
    """Describe the active algorithm. If an engine is passed, use ITS type
    (bulletproof against session_state desync); otherwise read the current
    session pick."""
    if engine is not None:
        if isinstance(engine, GammaCKLEngine):
            return f"γ-CKLSearch (Chumbalov 2024), γ = {engine.gamma}"
        if isinstance(engine, SearchEngine):
            return f"GAUSSSEARCH (Chumbalov 2020), σε = {engine.sigma_eps}"
    algo = current_algo()
    if algo == config.ALGO_GCKL:
        return f"γ-CKLSearch (Chumbalov 2024), γ = {config.GAMMA}"
    return f"GAUSSSEARCH (Chumbalov 2020), σε = {config.SIGMA_EPS}"


def engine_pill(engine) -> str:
    """Short badge showing the engine class — unambiguous diagnostic."""
    cls = type(engine).__name__
    if isinstance(engine, GammaCKLEngine):
        return f'<span class="algo-pill">engine: {cls} (γ={engine.gamma})</span>'
    if isinstance(engine, SearchEngine):
        return f'<span class="algo-pill">engine: {cls} (σε={engine.sigma_eps})</span>'
    return f'<span class="algo-pill">engine: {cls}</span>'


# ==================================================================
# Image renderer
# ==================================================================

BORDERS = {
    "chosen":   (4, "#27ae60"),
    "rejected": (2, "#7f8c8d"),
    "target":   (4, "#e74c3c"),
    None:       (1, "#333333"),
}


@st.cache_data(show_spinner=False, max_entries=4000)
def render_image(path_str: str, width: int, highlight: str | None) -> str:
    """Draw the border into the pixels (Streamlit cannot wrap st.image in an
    HTML div), resizing once to the final width so nothing is resampled again.
    Returns an <img> tag with the JPEG inlined as base64, so all images of a
    step arrive in the same update (no separate /media fetches landing at
    different times) and an unchanged image keeps identical HTML."""
    image = Image.open(path_str).convert("RGB")
    border, color = BORDERS.get(highlight, BORDERS[None])
    inner = width - 2 * border
    h = int(round(image.height * inner / image.width))
    image = image.resize((inner, h), Image.Resampling.LANCZOS)
    image = ImageOps.expand(image, border=border, fill=color)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return (
        f'<img src="data:image/jpeg;base64,{b64}" width="{image.width}" '
        f'height="{image.height}" style="width:100%; max-width:{image.width}px; '
        f'height:auto; display:block; border-radius:0.5rem;">'
    )


def show_image(idx: int, emb: Embeddings, caption: str, *,
                highlight: str | None = None, width: int | None = None):
    path = emb.image_path(idx)
    w = width if width is not None else IMG_WIDTH_HUMAN
    if path.exists():
        st.markdown(render_image(str(path), w, highlight), unsafe_allow_html=True)
    else:
        st.warning(f"image missing: {path}")
    if caption:
        st.caption(caption)


def render_inline_history(emb: Embeddings):
    hist = st.session_state.get("history", [])
    if not hist:
        return
    with st.expander(f"Query history ({len(hist)})", expanded=False):
        for rec in hist:
            if "event" in rec:
                st.markdown(f"**step {rec['step']}**: `{rec['event']}`")
                continue
            i, j = rec.get("i"), rec.get("j")
            picked = rec.get("parsed")
            if picked == ["A"]:
                picked_str = "**A**"
            elif picked == ["B"]:
                picked_str = "**B**"
            elif picked == []:
                picked_str = "reject"
            elif picked == "*":
                picked_str = "unsure"
            elif isinstance(picked, list) and len(picked) == 2:
                picked_str = "both"
            else:
                picked_str = "-"
            utt = rec.get("utterance", "")
            st.markdown(
                f"**{rec['step'] + 1}.** {emb.label(i)} vs {emb.label(j)} "
                f"-> {picked_str}   _{utt}_"
            )


def render_history_with_images(emb: Embeddings):
    hist = st.session_state.get("history", [])
    if not hist:
        return

    st.markdown("### Full query history")
    for rec in hist:
        if "event" in rec:
            found_side = rec.get("found_side")
            i, j = rec.get("i"), rec.get("j")
            st.markdown(f"**step {rec['step']}** - you clicked **Found it!** on **{found_side}**")
            col1, col2 = st.columns(2)
            with col1:
                show_image(i, emb, f"A - {emb.label(i)}",
                            highlight="chosen" if found_side == "A" else None,
                            width=IMG_WIDTH_HISTORY)
            with col2:
                show_image(j, emb, f"B - {emb.label(j)}",
                            highlight="chosen" if found_side == "B" else None,
                            width=IMG_WIDTH_HISTORY)
            st.markdown("---")
            continue

        i, j = rec["i"], rec["j"]
        parsed = rec.get("parsed")
        status = rec.get("status", "")
        if parsed == ["A"]:
            a_hl, b_hl, verdict = "chosen", "rejected", "you picked A"
        elif parsed == ["B"]:
            a_hl, b_hl, verdict = "rejected", "chosen", "you picked B"
        elif parsed == []:
            a_hl, b_hl, verdict = None, None, "you rejected both"
        elif parsed == "*":
            a_hl, b_hl, verdict = None, None, "you were unsure"
        elif isinstance(parsed, list) and len(parsed) == 2:
            a_hl, b_hl, verdict = "chosen", "chosen", "you said both"
        else:
            a_hl, b_hl, verdict = None, None, f"unparsed ({status})"

        st.markdown(f"**step {rec['step'] + 1}** - {verdict}")
        col1, col2 = st.columns(2)
        with col1:
            show_image(i, emb, f"A - {emb.label(i)}",
                        highlight=a_hl, width=IMG_WIDTH_HISTORY)
        with col2:
            show_image(j, emb, f"B - {emb.label(j)}",
                        highlight=b_hl, width=IMG_WIDTH_HISTORY)
        if rec.get("utterance"):
            st.caption(f"you said: _{rec['utterance']}_ -> parser: `{parsed!r}`")
        st.markdown("---")


def render_llm_history_with_images(emb: Embeddings):
    hist = st.session_state.get("history", [])
    if not hist:
        return

    st.markdown("### Full query history")
    for rec in hist:
        if "y_true" not in rec:
            continue
        i, j = rec["i"], rec["j"]
        y_true = rec["y_true"]
        y_rec = rec.get("y")
        agrees = (y_rec == y_true)

        if y_rec == 0:
            a_hl, b_hl = "chosen", "rejected"
        elif y_rec == 1:
            a_hl, b_hl = "rejected", "chosen"
        else:
            a_hl, b_hl = None, None

        oracle_picked = "A" if y_true == 0 else "B"
        header = f"**step {rec['step'] + 1}** - oracle picked **{oracle_picked}**"
        if y_rec is not None:
            header += " - parser: " + ("match" if agrees else "flipped")
        else:
            header += f" - parser: skipped ({rec.get('status', '')})"

        st.markdown(header)
        col1, col2 = st.columns(2)
        with col1:
            show_image(i, emb, f"A - {emb.label(i)}",
                        highlight=a_hl, width=IMG_WIDTH_HISTORY)
        with col2:
            show_image(j, emb, f"B - {emb.label(j)}",
                        highlight=b_hl, width=IMG_WIDTH_HISTORY)
        st.caption(
            f"verbalizer (style=`{rec.get('style', '?')}`): _{rec.get('utterance', '')}_ "
            f"-> parser: `{rec.get('parsed')!r}`"
        )
        st.markdown("---")


# ==================================================================
# HOME
# ==================================================================

def screen_home(emb: Embeddings):
    st.title("comparison-based search")
    st.markdown(
        "Find a scenery image by comparing pairs. Pick an algorithm, then "
        "either drive the search yourself or watch the LLM simulation."
    )
    st.markdown("")

    # Manual-sync pattern: let the widget initialise from session_state.algo
    # via `index=`, but write back ourselves afterwards. Avoids Streamlit's
    # widget-state teardown that was clearing the pick on screen switches.
    picked = st.radio(
        "Algorithm",
        options=ALGO_OPTIONS,
        index=ALGO_OPTIONS.index(current_algo()),
        horizontal=True,
    )
    st.session_state.algo = picked

    st.markdown(f'<div class="algo-sub">{algo_subtitle()}</div>', unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### Human search")
        st.markdown(
            "You have an image in mind. The system shows pairs; you type which "
            "is closer. Click **Found it!** under an image when it matches "
            "what you were looking for."
        )
        if st.button("Start human search", type="primary", use_container_width=True, key="btn_human"):
            st.session_state.engine = new_engine(
                emb, target_idx=None, max_queries=HUMAN_MAX_QUERIES,
            )
            st.session_state.history = []
            st.session_state.mode = "human"
            ensure_parser_loaded()
            go("human_search")

    with col2:
        st.markdown("### LLM simulation")
        st.markdown(
            "Watch the closed loop run end-to-end. A verbalizer LLM plays the "
            "user; the parser recovers each answer; the search converges. "
            "Shows belief statistics and per-step diagnostics."
        )
        if st.button("Start LLM simulation", type="primary", use_container_width=True, key="btn_llm"):
            go("llm_setup")

    st.markdown("---")
    st.caption(
        f"[Browse the scenery gallery]({config.DRIVE_GALLERY_URL}) - "
        f"{emb.n} images across {len(emb.classes())} classes"
    )


# ==================================================================
# HUMAN MODE
# ==================================================================

def screen_human_search(emb: Embeddings):
    engine = st.session_state.engine
    parser = ensure_parser_loaded()

    top = st.columns([1, 4, 1])
    with top[0]:
        if st.button("Home", key="human_home_top"):
            reset_all()
            go("home")
    with top[1]:
        st.markdown(f"#### Query {engine.step + 1} / {HUMAN_MAX_QUERIES}")
    with top[2]:
        st.markdown(f"**{engine.step}** answered")

    st.markdown(engine_pill(engine), unsafe_allow_html=True)

    if engine.done:
        go("human_result")

    i, j = engine.propose_query()

    _, col_a, col_b, _ = st.columns([1, 3, 3, 1])
    with col_a:
        img_a = st.empty()
        with img_a.container():
            show_image(i, emb, "A (left)", width=IMG_WIDTH_HUMAN)
        if st.button("Found it! - this is A", key=f"found_a_{engine.step}",
                      use_container_width=True):
            engine.stop_manual("user_found_target")
            st.session_state.found_idx = i
            st.session_state.history.append({
                "step": engine.step, "event": "user_declared_found",
                "i": i, "j": j, "found_side": "A",
            })
            go("human_result")
    with col_b:
        img_b = st.empty()
        with img_b.container():
            show_image(j, emb, "B (right)", width=IMG_WIDTH_HUMAN)
        if st.button("Found it! - this is B", key=f"found_b_{engine.step}",
                      use_container_width=True):
            engine.stop_manual("user_found_target")
            st.session_state.found_idx = j
            st.session_state.history.append({
                "step": engine.step, "event": "user_declared_found",
                "i": i, "j": j, "found_side": "B",
            })
            go("human_result")

    feedback_holder = st.empty()
    with st.form(key=f"utt_form_{engine.step}", clear_on_submit=True):
        utt = st.text_input(
            "Which is closer to what you're looking for?",
            placeholder="e.g. 'the left one', 'A', 'go with B', 'neither'",
        )
        submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

    if submitted and utt.strip():
        with st.spinner("parsing..."):
            parsed, raw = parser.parse(utt, options=("A", "B"))
        y, status = parsed_to_y(parsed, options=("A", "B"))
        record = engine.apply_answer(
            y=y, utterance=utt, parsed=parsed, status=status,
        )
        record["raw_parser_output"] = raw
        st.session_state.history.append(record)

        # Flash what the parser understood on the same pair before moving on.
        a_hl, b_hl, msg, kind = HUMAN_FEEDBACK.get(status, HUMAN_FEEDBACK["malformed"])
        show = getattr(feedback_holder, kind)
        if status == "clean":
            a_hl, b_hl = ("chosen", "rejected") if y == 0 else ("rejected", "chosen")
            msg = f"parser read **{'A' if y == 0 else 'B'}**"
        with img_a.container():
            show_image(i, emb, "A (left)", highlight=a_hl, width=IMG_WIDTH_HUMAN)
        with img_b.container():
            show_image(j, emb, "B (right)", highlight=b_hl, width=IMG_WIDTH_HUMAN)
        show(f"{msg}  -  you said: _{utt}_")
        time.sleep(HUMAN_FEEDBACK_S if status == "clean" else HUMAN_FEEDBACK_SKIP_S)
        st.rerun()

    render_inline_history(emb)


def start_new_human_search(emb: Embeddings):
    for k in ("engine", "history", "found_idx"):
        st.session_state.pop(k, None)
    st.session_state.engine = new_engine(
        emb, target_idx=None, max_queries=HUMAN_MAX_QUERIES,
    )
    st.session_state.history = []
    st.session_state.mode = "human"
    go("human_search")


def start_new_simulation(emb: Embeddings):
    for k in ("engine", "target_idx", "history", "auto_run",
              "force_step", "rng_lang", "llm_sigma_eps", "llm_gamma"):
        st.session_state.pop(k, None)
    go("llm_setup")


def result_top_bar(emb: Embeddings, new_label: str, start_new, key: str):
    """Home (left) and new-run (right) at the top of a results page, so you
    don't have to scroll past the history to reach them."""
    left, _, right = st.columns([1, 3.5, 1.5])
    with left:
        if st.button("Home", key=f"{key}_home_top", use_container_width=True):
            reset_all()
            go("home")
    with right:
        if st.button(new_label, key=f"{key}_new_top", type="primary",
                     use_container_width=True):
            start_new(emb)


def screen_human_result(emb: Embeddings):
    engine = st.session_state.engine
    found_idx = st.session_state.get("found_idx")

    result_top_bar(emb, "New search", start_new_human_search, "human_result")
    st.title("Search complete")
    st.caption(algo_subtitle(engine))
    st.markdown(engine_pill(engine), unsafe_allow_html=True)
    if engine.stop_reason == "user_found_target":
        st.success(f"Found in {engine.step} queries.")
    elif engine.stop_reason == "max_queries":
        st.warning(
            f"Reached the query cap ({HUMAN_MAX_QUERIES}) without you finding "
            "the image. Try a new search."
        )
    else:
        st.info(f"Stopped: {engine.stop_reason}")

    st.markdown("---")

    if found_idx is not None:
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.markdown("### You found")
            show_image(found_idx, emb, f"class: **{emb.label(found_idx)}**",
                        width=IMG_WIDTH_HUMAN + 60)

    st.markdown("---")

    metrics = st.columns(2)
    metrics[0].metric("queries answered", engine.step)
    if found_idx is not None:
        metrics[1].metric("image class", emb.label(found_idx))

    st.markdown("---")
    render_history_with_images(emb)

    col1, col2 = st.columns(2)
    if col1.button("New search", type="primary", use_container_width=True):
        start_new_human_search(emb)
    if col2.button("Home", use_container_width=True):
        reset_all()
        go("home")


# ==================================================================
# LLM MODE
# ==================================================================

def screen_llm_setup(emb: Embeddings):
    if st.button("Home"):
        go("home")

    st.title("Configure LLM simulation")
    st.caption(algo_subtitle())
    st.markdown(
        "A verbalizer LLM will play a user searching for a random target image. "
        "You can watch each step and inspect the parser's output."
    )

    algo = current_algo()

    st.markdown("### Search parameters")
    col1, col2 = st.columns(2)
    with col1:
        if algo == config.ALGO_GCKL:
            gamma = st.slider(
                "γ (oracle sharpness)", 0.5, 20.0, config.GAMMA, 0.5,
                help="γ-CKL oracle sharpness. Higher = closer item almost always picked; "
                     "lower = noisier.",
            )
            sigma_eps = config.SIGMA_EPS
        else:
            sigma_eps = st.slider(
                "σε (oracle noise)", 0.01, 1.0, config.SIGMA_EPS, 0.01,
                help="Probit oracle noise. Higher = noisier answers.",
            )
            gamma = config.GAMMA
    with col2:
        max_queries = st.slider("max queries", 5, 200, config.MAX_QUERIES, 5)

    st.markdown("---")

    if st.button("Start simulation", type="primary", use_container_width=True):
        rng = np.random.default_rng()
        target_idx = int(rng.integers(0, emb.n))
        seed = int(rng.integers(0, 1_000_000))

        st.session_state.engine = new_engine(
            emb,
            target_idx=target_idx,
            seed=seed,
            sigma_eps=sigma_eps,
            gamma=gamma,
            max_queries=int(max_queries),
        )
        st.session_state.target_idx = target_idx
        st.session_state.history = []
        st.session_state.auto_run = False
        st.session_state.mode = "llm"
        st.session_state.llm_sigma_eps = sigma_eps
        st.session_state.llm_gamma = gamma

        ensure_parser_loaded()
        ensure_verbalizer_loaded()
        go("llm_search")


def _render_belief_stats(engine, emb: Embeddings):
    """Belief stat strip. Differs per algo since the quantities aren't the same."""
    stats = engine.belief_stats()
    scols = st.columns(4)
    scols[0].metric("step", stats["step"])
    if isinstance(engine, GammaCKLEngine):
        argmax_cls = emb.label(stats["argmax_idx"])
        scols[1].metric("argmax class", argmax_cls)
        if stats["p_target"] is not None:
            scols[2].metric("P[target]", f"{stats['p_target']:.4f}")
        else:
            scols[2].metric("max P", f"{stats['max_p']:.4f}")
    else:
        scols[1].metric("tr(Σ)", f"{stats['trace_sigma']:.4f}")
        if stats["dist_to_target"] is not None:
            scols[2].metric("dist µ-target", f"{stats['dist_to_target']:.4f}")
        else:
            scols[2].metric("dist µ-target", "-")
    scols[3].metric("target class", emb.label(st.session_state.target_idx))


def screen_llm_search(emb: Embeddings):
    engine = st.session_state.engine
    parser = ensure_parser_loaded()
    verbalizer = ensure_verbalizer_loaded()

    top = st.columns([1, 4, 1])
    with top[0]:
        if st.button("Home", key="llm_home_top"):
            reset_all()
            go("home")
    with top[1]:
        header_holder = st.empty()
        header_holder.markdown(f"#### Step {engine.step + 1}")
    with top[2]:
        if st.button("End search", key="llm_end"):
            if not engine.done:
                engine.stop_manual("user_stopped")
            go("llm_result")

    # Prominent algo indicator so you can see at a glance which engine is live.
    st.caption(algo_subtitle(engine))
    st.markdown(engine_pill(engine), unsafe_allow_html=True)

    # Refreshed after this run's step so the numbers match the images below.
    stats_holder = st.empty()
    with stats_holder.container():
        _render_belief_stats(engine, emb)

    st.markdown("---")

    if "rng_lang" not in st.session_state:
        st.session_state.rng_lang = np.random.default_rng(
            engine.rng.integers(0, 1_000_000_000)
        )
    rng_lang = st.session_state.rng_lang

    should_step = st.session_state.get("force_step") or st.session_state.get("auto_run")

    def last_record():
        return next(
            (r for r in reversed(st.session_state.get("history", []))
             if "i" in r and "y_true" in r),
            None,
        )

    def draw_panels(rec):
        with panels_holder.container():
            if rec is not None:
                _render_llm_panels(rec, emb)
            else:
                st.info("Press **Next step** to begin.")

    panels_holder = st.empty()
    info_holder = st.empty()

    # Keep the previous step on screen while the models work on the next one.
    draw_panels(last_record())

    if should_step and not engine.done:
        st.session_state.pop("force_step", None)
        i, j = engine.propose_query()
        y_true = engine.oracle_answer(i, j)
        picked = "A" if y_true == 0 else "B"
        info_holder.info("verbalizer generating...")
        utt, style = verbalizer.verbalize(picked, rng_lang, parser=parser)
        if verbalizer.last_parse is not None:
            parsed, raw = verbalizer.last_parse
        else:
            info_holder.info("parser...")
            parsed, raw = parser.parse(utt, options=("A", "B"))
        info_holder.empty()
        y_rec, status = parsed_to_y(parsed, options=("A", "B"))
        record = engine.apply_answer(
            y=y_rec, utterance=utt, style=style, parsed=parsed, status=status,
        )
        record["y_true"] = y_true
        record["picked_letter"] = picked
        record["raw_parser_output"] = raw
        st.session_state.history.append(record)
        draw_panels(record)
        with stats_holder.container():
            _render_belief_stats(engine, emb)
        if engine.done:
            header_holder.markdown(f"#### Finished after {engine.step} steps")
        else:
            header_holder.markdown(f"#### Step {engine.step + 1}")

    if engine.done:
        st.markdown("---")
        if st.button("See results", type="primary", use_container_width=True,
                     key="llm_see_results"):
            go("llm_result")
        if st.session_state.get("auto_run"):
            time.sleep(LLM_AUTORUN_PAUSE_S)
            go("llm_result")
        return

    if not engine.done:
        st.markdown("---")
        ccols = st.columns(2)
        if ccols[0].button("Next step", type="primary", use_container_width=True):
            st.session_state.force_step = True
            st.rerun()
        toggle_label = "Pause" if st.session_state.get("auto_run") else "Auto-run to completion"
        if ccols[1].button(toggle_label, use_container_width=True):
            st.session_state.auto_run = not st.session_state.get("auto_run", False)
            st.rerun()
        if st.session_state.get("auto_run"):
            time.sleep(LLM_AUTORUN_PAUSE_S)
            st.session_state.force_step = True
            st.rerun()


def _render_llm_panels(record: dict, emb: Embeddings):
    i, j = record["i"], record["j"]
    y_true = record["y_true"]
    y_rec = record.get("y")
    agrees = (y_rec == y_true)

    if y_rec == 0:
        a_hl, b_hl = "chosen", "rejected"
    elif y_rec == 1:
        a_hl, b_hl = "rejected", "chosen"
    else:
        a_hl, b_hl = None, None

    col_a, col_b, col_t = st.columns(3)
    with col_a:
        show_image(i, emb, f"A - {emb.label(i)}", highlight=a_hl, width=IMG_WIDTH_LLM)
    with col_b:
        show_image(j, emb, f"B - {emb.label(j)}", highlight=b_hl, width=IMG_WIDTH_LLM)
    with col_t:
        show_image(st.session_state.target_idx, emb,
                    f"target - {emb.label(st.session_state.target_idx)}",
                    highlight="target", width=IMG_WIDTH_LLM)

    style = record.get("style", "?")
    utt = record.get("utterance", "")
    parsed = record.get("parsed")
    status = record.get("status", "")
    st.markdown(
        f"**verbalizer** (style=`{style}`, oracle picked **{record['picked_letter']}**): _{utt}_"
    )
    match_icon = "match" if agrees else ("flipped" if y_rec is not None else "skipped")
    st.markdown(f"**parser**: `{parsed!r}` -> y={y_rec} `[{status}]` - {match_icon}")
    if record.get("target_in_query"):
        st.success("target appeared in query - search complete")


def screen_llm_result(emb: Embeddings):
    engine = st.session_state.engine
    target_idx = st.session_state.target_idx

    result_top_bar(emb, "New simulation", start_new_simulation, "llm_result")
    st.title("Simulation complete")
    st.caption(algo_subtitle(engine))
    st.markdown(engine_pill(engine), unsafe_allow_html=True)
    st.success(f"`{engine.stop_reason}` after {engine.step} queries.")

    st.markdown("---")
    _, col, _ = st.columns([1, 2, 1])
    with col:
        show_image(target_idx, emb,
                    f"target - {emb.label(target_idx)}",
                    highlight="target", width=IMG_WIDTH_LLM + 60)

    st.markdown("---")

    steps = [r for r in st.session_state.history if "y_true" in r]
    n_steps = len(steps)
    if n_steps > 0:
        n_match = sum(1 for r in steps if r["y"] == r["y_true"])
        n_skip = sum(1 for r in steps if r["y"] is None)
        n_flip = n_steps - n_match - n_skip
        m = st.columns(4)
        m[0].metric("total queries", engine.step)
        m[1].metric("round-trip match", f"{n_match}/{n_steps}")
        m[2].metric("skipped", n_skip)
        m[3].metric("parser flipped", n_flip)

    st.markdown("---")
    render_llm_history_with_images(emb)

    col1, col2 = st.columns(2)
    if col1.button("New simulation", type="primary", use_container_width=True):
        start_new_simulation(emb)
    if col2.button("Home", use_container_width=True):
        reset_all()
        go("home")


# ==================================================================
# Dispatcher
# ==================================================================

def main():
    st.set_page_config(
        page_title="comparison-search",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(COMPACT_CSS, unsafe_allow_html=True)

    if "screen" not in st.session_state:
        st.session_state.screen = "home"
    if "algo" not in st.session_state:
        st.session_state.algo = config.DEFAULT_ALGO

    emb = load_embeddings()

    screen = st.session_state.screen
    if screen == "home":
        screen_home(emb)
    elif screen == "human_search":
        screen_human_search(emb)
    elif screen == "human_result":
        screen_human_result(emb)
    elif screen == "llm_setup":
        screen_llm_setup(emb)
    elif screen == "llm_search":
        screen_llm_search(emb)
    elif screen == "llm_result":
        screen_llm_result(emb)
    else:
        st.error(f"unknown screen: {screen}")
        if st.button("Home"):
            go("home")


if __name__ == "__main__":
    main()