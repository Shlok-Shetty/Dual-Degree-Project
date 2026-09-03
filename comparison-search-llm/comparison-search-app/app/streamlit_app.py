"""Streamlit interface for comparison-based scenery search.

Screen flow:
    home           -> pick Human search or LLM simulation
    human_search   -> 2 images + text input + "Found it!" under each image
                      Collapsible sidebar shows running query history.
                      No pre-assigned target - user decides when done.
    human_result   -> what you found + full history with images
    llm_setup      -> configure sigma_eps, max_queries -> start
    llm_search     -> 3-panel view + step controls, single-pass render
    llm_result     -> completion + fidelity stats + history + New / Home
"""
import time
import numpy as np
import streamlit as st

from . import config
from .embeddings import Embeddings
from .search_engine import SearchEngine
from .parser_llm import ParserLLM, parsed_to_y


IMG_WIDTH_HUMAN = 340
IMG_WIDTH_LLM = 280
IMG_WIDTH_HISTORY = 180
HUMAN_MAX_QUERIES = 50
LLM_AUTORUN_PAUSE_S = 2.5


# CSS: kill the header's dead space and hide streamlit chrome.
# Also disables the fullscreen-zoom button on images (it never rendered
# usefully - it just center-crops to a fixed tiny size).
COMPACT_CSS = """
<style>
    header[data-testid="stHeader"] { height: 0rem; visibility: hidden; }
    div.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    button[title="View fullscreen"] { display: none !important; }
    /* no sidebar in this app - collapse to zero */
    section[data-testid="stSidebar"] { display: none !important; }
</style>
"""


# ==================================================================
# Cached loaders
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
    if st.session_state.get("_parser_loaded"):
        return load_parser()

    info_holder = st.empty()
    bar_holder = st.empty()
    info_holder.info("Loading parser - first time takes ~60 seconds while the model downloads and quantizes.")
    bar = bar_holder.progress(0, text="Starting up...")

    import threading
    parser_ref: dict = {}

    def _load():
        parser_ref["p"] = load_parser()

    t = threading.Thread(target=_load)
    t.start()

    stages = [
        (10, "Loading tokenizer..."),
        (25, "Downloading base model (Qwen 2.5-3B)..."),
        (55, "Applying 4-bit quantization..."),
        (75, "Attaching LoRA adapter..."),
        (90, "Warming up..."),
    ]
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
    info_holder.empty()

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
# Session helpers
# ==================================================================

def go(screen: str):
    st.session_state.screen = screen
    st.rerun()


def reset_all():
    keep = {"_parser_loaded", "_verbalizer_loaded"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]


# ==================================================================
# Image renderer
# ==================================================================

def show_image(idx: int, emb: Embeddings, caption: str, *,
                highlight: str | None = None, width: int | None = None):
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
            f'<div style="border:{border}; padding:6px; border-radius:6px; '
            f'display:inline-block;">',
            unsafe_allow_html=True,
        )
        if width is not None:
            st.image(str(path), width=width)
        else:
            st.image(str(path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"image missing: {path}")
    if caption:
        st.caption(caption)


def render_inline_history(emb: Embeddings):
    """Compact text-only history strip, rendered inline below the search query."""
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
    """Full history with image thumbnails, for the results page."""
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
        # what the parser routed to
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
        "Find a scenery image by comparing pairs. Say which is closer to what "
        "you have in mind; the search narrows down until you find your target."
    )
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
            rng = np.random.default_rng()
            st.session_state.engine = SearchEngine(
                X=emb.X,
                sigma_eps=config.SIGMA_EPS,
                target_idx=None,
                seed=int(rng.integers(0, 1_000_000)),
                max_queries=HUMAN_MAX_QUERIES,
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
    engine: SearchEngine = st.session_state.engine
    parser = ensure_parser_loaded()

    # compact top bar - no dead space
    top = st.columns([1, 4, 1])
    with top[0]:
        if st.button("Home", key="human_home_top"):
            reset_all()
            go("home")
    with top[1]:
        st.markdown(f"#### Query {engine.step + 1} / {HUMAN_MAX_QUERIES}")
    with top[2]:
        st.markdown(f"**{engine.step}** answered")

    if engine.done:
        go("human_result")

    i, j = engine.propose_query()

    _, col_a, col_b, _ = st.columns([1, 3, 3, 1])
    with col_a:
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

    st.markdown("")
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

    render_inline_history(emb)


def screen_human_result(emb: Embeddings):
    engine: SearchEngine = st.session_state.engine
    found_idx = st.session_state.get("found_idx")

    st.title("Search complete")
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
        for k in ("engine", "history", "found_idx"):
            st.session_state.pop(k, None)
        rng = np.random.default_rng()
        st.session_state.engine = SearchEngine(
            X=load_embeddings().X,
            sigma_eps=config.SIGMA_EPS,
            target_idx=None,
            seed=int(rng.integers(0, 1_000_000)),
            max_queries=HUMAN_MAX_QUERIES,
        )
        st.session_state.history = []
        st.session_state.mode = "human"
        go("human_search")
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
    st.markdown(
        "A verbalizer LLM will play a user searching for a random target image. "
        "You can watch each step and inspect the parser's output."
    )

    st.markdown("### Search parameters")
    col1, col2 = st.columns(2)
    with col1:
        sigma_eps = st.slider("sigma_eps (oracle noise)", 0.01, 1.0,
                               config.SIGMA_EPS, 0.01,
                               help="Probit oracle noise. Higher = noisier answers.")
    with col2:
        max_queries = st.slider("max queries", 5, 200, config.MAX_QUERIES, 5)

    st.markdown("---")

    if st.button("Start simulation", type="primary", use_container_width=True):
        # target is always random - user has no reason to pick it
        rng = np.random.default_rng()
        target_idx = int(rng.integers(0, emb.n))
        seed = int(rng.integers(0, 1_000_000))

        st.session_state.engine = SearchEngine(
            X=emb.X,
            sigma_eps=sigma_eps,
            target_idx=target_idx,
            seed=seed,
            max_queries=int(max_queries),
        )
        st.session_state.target_idx = target_idx
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
    top = st.columns([1, 4, 1])
    with top[0]:
        if st.button("Home", key="llm_home_top"):
            reset_all()
            go("home")
    with top[1]:
        st.markdown(f"#### Step {engine.step + 1}")
    with top[2]:
        if st.button("End search", key="llm_end"):
            engine.stop_manual("user_stopped")
            go("llm_result")

    stats = engine.belief_stats()
    scols = st.columns(4)
    scols[0].metric("step", stats["step"])
    scols[1].metric("tr(Sigma)", f"{stats['trace_sigma']:.4f}")
    if stats["dist_to_target"] is not None:
        scols[2].metric("dist mu-target", f"{stats['dist_to_target']:.4f}")
    scols[3].metric("target class", emb.label(st.session_state.target_idx))

    st.markdown("---")

    if engine.done:
        go("llm_result")

    if "rng_lang" not in st.session_state:
        st.session_state.rng_lang = np.random.default_rng(
            engine.rng.integers(0, 1_000_000_000)
        )
    rng_lang = st.session_state.rng_lang

    # single-pass render: run the step inline (blocking on LLM calls) using
    # placeholders so the images don't stagger. All three images render in one
    # write after the LLM work is done.
    should_step = st.session_state.get("force_step") or st.session_state.get("auto_run")

    panels_holder = st.empty()
    info_holder = st.empty()

    if should_step and not engine.done:
        st.session_state.pop("force_step", None)
        i, j = engine.propose_query()
        y_true = engine.oracle_answer(i, j)
        picked = "A" if y_true == 0 else "B"
        info_holder.info("verbalizer generating...")
        utt, style = verbalizer.verbalize(picked, rng_lang, parser=parser)
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

    # render most recent step inside a single container so all three images
    # appear together
    last = next(
        (r for r in reversed(st.session_state.get("history", []))
         if "i" in r and "y_true" in r),
        None,
    )
    with panels_holder.container():
        if last is not None:
            _render_llm_panels(last, emb)
        else:
            st.info("Press **Next step** to begin.")

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
            # comfortable reading pace before pushing the next step
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
    engine: SearchEngine = st.session_state.engine
    target_idx = st.session_state.target_idx

    st.title("Simulation complete")
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
        for k in ("engine", "target_idx", "history", "auto_run",
                  "force_step", "rng_lang"):
            st.session_state.pop(k, None)
        go("llm_setup")
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