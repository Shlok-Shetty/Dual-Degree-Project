"""Streamlit interface for comparison-based scenery search.

Screen flow:
    home            → pick Human search or LLM simulation
    human_setup     → pick target (random / class / index) → start
    human_search    → 2 images + text input + "Found it!" under each image
    human_result    → target reveal + query history + New search / Home
    llm_setup       → pick target + configure σ_ε, seed, max_queries → start
    llm_search      → 3-panel view + step controls
    llm_result      → completion + history + New search / Home
"""
import time
import numpy as np
import streamlit as st

from . import config
from .embeddings import Embeddings
from .search_engine import SearchEngine
from .parser_llm import ParserLLM, parsed_to_y


# ==================================================================
# Cached loaders (fire once per Streamlit process)
# ==================================================================

@st.cache_resource(show_spinner=False)
def load_embeddings() -> Embeddings:
    return Embeddings()


@st.cache_resource(show_spinner=False)
def load_parser() -> ParserLLM:
    return ParserLLM()


@st.cache_resource(show_spinner=False)
def load_verbalizer():
    from .verbalizer_llm import VerbalizerLLM
    return VerbalizerLLM()


def ensure_parser_loaded():
    """Load the parser with a visible progress bar. Runs only once per session
    because @st.cache_resource memoizes."""
    if st.session_state.get("_parser_loaded"):
        return load_parser()

    progress_holder = st.empty()
    bar_holder = st.empty()
    with progress_holder:
        st.info("Loading parser — first time takes ~60 seconds while the model downloads and quantizes.")
    bar = bar_holder.progress(0, text="Starting up...")

    # The actual load is one blocking call; we can't get true progress, but we
    # can stage a plausible message sequence so the user sees motion.
    stages = [
        (10, "Loading tokenizer..."),
        (25, "Downloading base model (Qwen 2.5-3B)..."),
        (55, "Applying 4-bit quantization..."),
        (75, "Attaching LoRA adapter..."),
        (90, "Warming up..."),
    ]
    import threading
    parser_ref: dict = {}

    def _load():
        parser_ref["p"] = load_parser()

    t = threading.Thread(target=_load)
    t.start()

    idx = 0
    while t.is_alive():
        if idx < len(stages):
            pct, msg = stages[idx]
            bar.progress(pct, text=msg)
            idx += 1
        time.sleep(3)
    t.join()
    bar.progress(100, text="Ready.")
    time.sleep(0.3)
    bar_holder.empty()
    progress_holder.empty()

    st.session_state._parser_loaded = True
    return parser_ref["p"]


def ensure_verbalizer_loaded():
    if st.session_state.get("_verbalizer_loaded"):
        return load_verbalizer()
    with st.spinner("Loading verbalizer LLM (Qwen 1.5B, ~15s)..."):
        v = load_verbalizer()
    st.session_state._verbalizer_loaded = True
    return v


# ==================================================================
# Session state
# ==================================================================

def go(screen: str):
    st.session_state.screen = screen
    st.rerun()


def reset_all():
    keep = {"_parser_loaded", "_verbalizer_loaded"}  # keep loaded models
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]


# ==================================================================
# Common: image renderer
# ==================================================================

def show_image(idx: int, emb: Embeddings, caption: str, *,
                highlight: str | None = None):
    border_map = {
        "chosen":   "4px solid #27ae60",
        "rejected": "2px solid #7f8c8d",
        "target":   "4px solid #e74c3c",
        None:       "1px solid #333",
    }
    border = border_map[highlight]
    path = emb.image_path(idx)
    if path.exists():
        st.markdown(
            f'<div style="border:{border}; padding:6px; border-radius:6px;">',
            unsafe_allow_html=True,
        )
        st.image(str(path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"image missing: {path}")
    if caption:
        st.caption(caption)


def render_history_expander(emb: Embeddings, expanded: bool = False):
    hist = st.session_state.get("history", [])
    if not hist:
        return
    with st.expander(f"Query history — {len(hist)} steps", expanded=expanded):
        for rec in hist:
            if "event" in rec:
                st.markdown(f"— step {rec['step']}: {rec['event']}")
                continue
            i, j = rec.get("i"), rec.get("j")
            line = f"**step {rec['step'] + 1}** — A: `{emb.label(i)}` vs B: `{emb.label(j)}`"
            if rec.get("utterance"):
                line += f" · _{rec['utterance']}_ → `{rec.get('parsed')!r}`"
            if rec.get("target_in_query"):
                line += " · 🎯 target in query"
            st.markdown(line)


# ==================================================================
# HOME SCREEN
# ==================================================================

def screen_home(emb: Embeddings):
    st.title("comparison-based search")
    st.markdown(
        "Find a scenery image by comparing pairs. Say which is closer to what "
        "you have in mind; the search narrows down until your target appears."
    )
    st.markdown("")
    st.markdown("")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### Human search")
        st.markdown(
            "You have a target image in mind. The system shows pairs and you "
            "type which is closer. Click **Found it!** when your target appears."
        )
        if st.button("Start human search", type="primary", use_container_width=True, key="btn_human"):
            go("human_setup")

    with col2:
        st.markdown("### LLM simulation")
        st.markdown(
            "Watch the closed loop run end-to-end. A verbalizer LLM plays the "
            "user; the parser recovers each answer; the search converges. Shows "
            "belief statistics and per-step diagnostics."
        )
        if st.button("Start LLM simulation", type="primary", use_container_width=True, key="btn_llm"):
            go("llm_setup")

    st.markdown("---")
    st.caption(
        f"[Browse the scenery gallery ↗]({config.DRIVE_GALLERY_URL}) · "
        f"{emb.n} images across {len(emb.classes())} classes"
    )


# ==================================================================
# HUMAN MODE
# ==================================================================

def screen_human_setup(emb: Embeddings):
    if st.button("← Home"):
        go("home")

    st.title("Pick a target")
    st.markdown(
        "Choose an image to search for. Browse the "
        f"[gallery ↗]({config.DRIVE_GALLERY_URL}) if you want to pick by index."
    )

    target_mode = st.radio(
        "how to pick",
        ["Random", "Random from a class", "By index"],
        horizontal=True,
    )

    target_idx = None
    picker_class = None
    if target_mode == "Random from a class":
        picker_class = st.selectbox("class", emb.classes())
    elif target_mode == "By index":
        target_idx = st.number_input(
            "item index", min_value=0, max_value=emb.n - 1, value=437, step=1,
        )

    st.markdown("---")

    if st.button("Start search", type="primary", use_container_width=True):
        rng = np.random.default_rng()
        if target_idx is None:
            if picker_class:
                pool = emb.indices_by_class(picker_class)
                target_idx = int(rng.choice(pool))
            else:
                target_idx = int(rng.integers(0, emb.n))

        # human mode: fixed sensible defaults, no exposure to belief internals
        st.session_state.engine = SearchEngine(
            X=emb.X,
            sigma_eps=config.SIGMA_EPS,
            target_idx=int(target_idx),
            seed=int(rng.integers(0, 1_000_000)),
            max_queries=config.MAX_QUERIES,
        )
        st.session_state.target_idx = int(target_idx)
        st.session_state.history = []
        st.session_state.mode = "human"
        # ensure parser is loaded before we hit the search screen
        ensure_parser_loaded()
        go("human_search")


def screen_human_search(emb: Embeddings):
    engine: SearchEngine = st.session_state.engine
    parser = ensure_parser_loaded()

    # top bar: only info a human user should see
    top = st.columns([1, 3, 1])
    with top[0]:
        if st.button("← Home", key="human_home_top"):
            reset_all()
            go("home")
    with top[1]:
        st.markdown(f"### Query {engine.step + 1}")
    with top[2]:
        st.markdown(f"**{engine.step}** answered")

    st.markdown("---")

    if engine.done:
        go("human_result")

    i, j = engine.propose_query()

    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        show_image(i, emb, "A (left)")
        if st.button("Found it! — this is A", key=f"found_a_{engine.step}",
                      use_container_width=True):
            engine.stop_manual("user_found_target")
            st.session_state.found_idx = i
            st.session_state.history.append({
                "step": engine.step, "event": "user_declared_found",
                "i": i, "j": j, "found_side": "A",
            })
            go("human_result")
    with col_b:
        show_image(j, emb, "B (right)")
        if st.button("Found it! — this is B", key=f"found_b_{engine.step}",
                      use_container_width=True):
            engine.stop_manual("user_found_target")
            st.session_state.found_idx = j
            st.session_state.history.append({
                "step": engine.step, "event": "user_declared_found",
                "i": i, "j": j, "found_side": "B",
            })
            go("human_result")

    st.markdown("---")
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
        st.rerun()

    render_history_expander(emb, expanded=False)


def screen_human_result(emb: Embeddings):
    engine: SearchEngine = st.session_state.engine
    target_idx = st.session_state.target_idx

    # what to reveal: if the user clicked Found it!, show what they clicked;
    # otherwise the search converged and the target was in the last query pair.
    found_idx = st.session_state.get("found_idx")
    if found_idx is None:
        found_idx = target_idx

    correct = (found_idx == target_idx)

    st.title("Search complete")
    if correct:
        st.success(f"Found in {engine.step} queries.")
    else:
        st.warning(
            f"You picked a different image. The search would have found the "
            f"target in query {engine.step}."
        )

    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### You picked")
        show_image(found_idx, emb, f"class: **{emb.label(found_idx)}**")
    with col2:
        st.markdown("### Target")
        show_image(target_idx, emb, f"class: **{emb.label(target_idx)}**",
                    highlight="target")

    st.markdown("---")

    metrics = st.columns(3)
    metrics[0].metric("queries answered", engine.step)
    metrics[1].metric("target class", emb.label(target_idx))
    metrics[2].metric("match", "yes" if correct else "no")

    render_history_expander(emb, expanded=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    if col1.button("New search", type="primary", use_container_width=True):
        # keep mode, wipe search state
        for k in ("engine", "target_idx", "history", "found_idx"):
            st.session_state.pop(k, None)
        go("human_setup")
    if col2.button("Home", use_container_width=True):
        reset_all()
        go("home")


# ==================================================================
# LLM MODE
# ==================================================================

def screen_llm_setup(emb: Embeddings):
    if st.button("← Home"):
        go("home")

    st.title("Configure LLM simulation")
    st.markdown(
        "The verbalizer LLM will play a user searching for the target image. "
        "You can watch each step and inspect the parser's output."
    )

    st.markdown("### Target")
    target_mode = st.radio(
        "how to pick",
        ["Random", "Random from a class", "By index"],
        horizontal=True,
    )
    target_idx = None
    picker_class = None
    if target_mode == "Random from a class":
        picker_class = st.selectbox("class", emb.classes())
    elif target_mode == "By index":
        target_idx = st.number_input(
            "item index", min_value=0, max_value=emb.n - 1, value=437, step=1,
        )

    st.markdown("### Search parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        sigma_eps = st.slider("σ_ε (oracle noise)", 0.01, 1.0,
                               config.SIGMA_EPS, 0.01,
                               help="Probit oracle noise. Higher = noisier answers.")
    with col2:
        max_queries = st.slider("max queries", 5, 200, config.MAX_QUERIES, 5)
    with col3:
        seed = st.number_input("random seed", value=config.DEFAULT_SEED, step=1)

    st.markdown("---")

    if st.button("Start simulation", type="primary", use_container_width=True):
        rng = np.random.default_rng(seed)
        if target_idx is None:
            if picker_class:
                pool = emb.indices_by_class(picker_class)
                target_idx = int(rng.choice(pool))
            else:
                target_idx = int(rng.integers(0, emb.n))

        st.session_state.engine = SearchEngine(
            X=emb.X,
            sigma_eps=sigma_eps,
            target_idx=int(target_idx),
            seed=int(seed),
            max_queries=int(max_queries),
        )
        st.session_state.target_idx = int(target_idx)
        st.session_state.history = []
        st.session_state.auto_run = False
        st.session_state.mode = "llm"

        ensure_parser_loaded()
        ensure_verbalizer_loaded()
        go("llm_search")


def screen_llm_search(emb: Embeddings):
    engine: SearchEngine = st.session_state.engine
    parser = ensure_parser_loaded()
    verbalizer = ensure_verbalizer_loaded()

    # top bar
    top = st.columns([1, 3, 1])
    with top[0]:
        if st.button("← Home", key="llm_home_top"):
            reset_all()
            go("home")
    with top[1]:
        st.markdown(f"### Step {engine.step + 1}")
    with top[2]:
        if st.button("End search", key="llm_end"):
            engine.stop_manual("user_stopped")
            go("llm_result")

    # stats
    stats = engine.belief_stats()
    scols = st.columns(4)
    scols[0].metric("step", stats["step"])
    scols[1].metric("tr(Σ)", f"{stats['trace_sigma']:.4f}")
    if stats["dist_to_target"] is not None:
        scols[2].metric("‖μ − x_t‖", f"{stats['dist_to_target']:.4f}")
    scols[3].metric("target class", emb.label(st.session_state.target_idx))

    st.markdown("---")

    if engine.done:
        go("llm_result")

    # RNG stream for verbalizer, spawned once per search
    if "rng_lang" not in st.session_state:
        st.session_state.rng_lang = np.random.default_rng(
            engine.rng.integers(0, 1_000_000_000)
        )
    rng_lang = st.session_state.rng_lang

    # step advancement
    should_step = st.session_state.get("force_step") or st.session_state.get("auto_run")
    if should_step:
        st.session_state.pop("force_step", None)
        i, j = engine.propose_query()
        y_true = engine.oracle_answer(i, j)
        picked = "A" if y_true == 0 else "B"
        with st.spinner("verbalizer generating..."):
            utt, style = verbalizer.verbalize(picked, rng_lang)
        with st.spinner("parser..."):
            parsed, raw = parser.parse(utt, options=("A", "B"))
        y_rec, status = parsed_to_y(parsed, options=("A", "B"))
        record = engine.apply_answer(
            y=y_rec, utterance=utt, style=style, parsed=parsed, status=status,
        )
        record["y_true"] = y_true
        record["picked_letter"] = picked
        record["raw_parser_output"] = raw
        st.session_state.history.append(record)

    # render most recent step
    last = next(
        (r for r in reversed(st.session_state.get("history", []))
         if "i" in r and "y_true" in r),
        None,
    )
    if last is not None:
        _render_llm_panels(last, emb)
    else:
        st.info("Press **Next step** to begin.")

    # controls
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
            st.session_state.force_step = True
            st.rerun()

    render_history_expander(emb, expanded=False)


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
        show_image(i, emb, f"A — {emb.label(i)}", highlight=a_hl)
    with col_b:
        show_image(j, emb, f"B — {emb.label(j)}", highlight=b_hl)
    with col_t:
        show_image(st.session_state.target_idx, emb,
                    f"target — {emb.label(st.session_state.target_idx)}",
                    highlight="target")

    style = record.get("style", "?")
    utt = record.get("utterance", "")
    parsed = record.get("parsed")
    status = record.get("status", "")
    st.markdown(
        f"**verbalizer** (style=`{style}`, oracle picked **{record['picked_letter']}**): _{utt}_"
    )
    match_icon = "✓" if agrees else ("✗" if y_rec is not None else "—")
    st.markdown(f"**parser**: `{parsed!r}` → y={y_rec} `[{status}]` {match_icon}")
    if record.get("target_in_query"):
        st.success("target appeared in query — search complete")


def screen_llm_result(emb: Embeddings):
    engine: SearchEngine = st.session_state.engine
    target_idx = st.session_state.target_idx

    st.title("Simulation complete")
    st.success(f"`{engine.stop_reason}` after {engine.step} queries.")

    st.markdown("---")
    col_t, _ = st.columns([1, 2])
    with col_t:
        show_image(target_idx, emb,
                    f"target — {emb.label(target_idx)}",
                    highlight="target")

    st.markdown("---")

    # aggregate parser fidelity
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

    render_history_expander(emb, expanded=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    if col1.button("New simulation", type="primary", use_container_width=True):
        for k in ("engine", "target_idx", "history", "auto_run",
                  "force_step", "rng_lang"):
            st.session_state.pop(k, None)
        go("llm_setup")
    if col2.button("Home", use_container_width=True):
        reset_all()
        go("home")


# ==================================================================
# Main dispatcher
# ==================================================================

def main():
    st.set_page_config(page_title="comparison-search", layout="wide")

    if "screen" not in st.session_state:
        st.session_state.screen = "home"

    emb = load_embeddings()

    screen = st.session_state.screen
    if screen == "home":
        screen_home(emb)
    elif screen == "human_setup":
        screen_human_setup(emb)
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
