"""Streamlit interface for the lite comparison-search app.

Supports two engines: GAUSSSEARCH (Chumbalov 2020) and γ-CKLSearch
(Chumbalov 2024, Algorithm 3). The user picks one on the home screen
and both modes (human search, auto-run) use the chosen engine.
"""
import base64
import io
import time
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from . import config
from .embeddings import Embeddings
from .search_engine import SearchEngine
from .gamma_ckl_engine import GammaCKLEngine

ALGO_GAUSS = "GAUSSSEARCH"
ALGO_GCKL = "gamma-CKL"
ALGO_OPTIONS = [ALGO_GAUSS, ALGO_GCKL]

IMG_WIDTH = 460
IMG_WIDTH_HISTORY = 200
AUTO_HIGHLIGHT_DELAY = 0.7
HUMAN_MAX_QUERIES = 50
AUTO_MAX_QUERIES = 50
SPEED_DELAYS = {"Slow": 2.0, "Normal": 1.0, "Fast": 0.4}
DEFAULT_SPEED = "Normal"

COMPACT_CSS = """
<style>
    header[data-testid="stHeader"] { height: 0rem; visibility: hidden; }
    div.block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 1200px !important;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    section[data-testid="stSidebar"] { display: none !important; }
    button[title="View fullscreen"],
    button[title="Fullscreen"],
    [data-testid="StyledFullScreenButton"],
    [data-testid="stImageToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }
    .query-counter {
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    .answered-badge {
        text-align: right;
        font-size: 0.95rem;
        margin-top: 0.6rem;
    }
    .col-header {
        text-align: center;
        font-weight: 600;
        margin-bottom: 6px;
        height: 24px;
    }
    .col-header-target { color: #e74c3c; }
    .algo-sub {
        text-align: center;
        font-size: 0.95rem;
        color: #aaa;
        margin-top: -8px;
        margin-bottom: 8px;
    }
</style>
"""


@st.cache_resource(show_spinner=False)
def load_embeddings() -> Embeddings:
    return Embeddings()


def go(screen: str):
    st.session_state.screen = screen
    st.rerun()


def reset_all():
    saved_algo = st.session_state.get("algo", config.DEFAULT_ALGO)
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state.algo = saved_algo


def current_algo() -> str:
    return st.session_state.get("algo", config.DEFAULT_ALGO)


def new_engine(emb: Embeddings, *, target_idx: int | None = None,
               max_queries: int = HUMAN_MAX_QUERIES):
    """Factory: build whichever engine the user picked on the home screen."""
    algo = current_algo()
    rng = np.random.default_rng()
    seed = int(rng.integers(0, 1_000_000))
    if algo == ALGO_GCKL:
        return GammaCKLEngine(
            X=emb.X,
            gamma=config.GAMMA,
            target_idx=target_idx,
            seed=seed,
            max_queries=max_queries,
        )
    return SearchEngine(
        X=emb.X,
        sigma_eps=config.SIGMA_EPS,
        target_idx=target_idx,
        seed=seed,
        max_queries=max_queries,
    )


def algo_subtitle() -> str:
    algo = current_algo()
    if algo == ALGO_GCKL:
        return f"γ-CKLSearch (Chumbalov 2024), γ = {config.GAMMA}"
    return f"GAUSSSEARCH (Chumbalov 2020), σε = {config.SIGMA_EPS}"


BORDERS = {
    "chosen": (4, "#27ae60"),
    "rejected": (2, "#7f8c8d"),
    "target": (3, "#e74c3c"),
}


@st.cache_data(show_spinner=False, max_entries=4000)
def render_image(path_str: str, width: int, highlight: str | None) -> str:
    """Resize to the final display width (border included) once and return an
    <img> tag with the JPEG inlined as base64. Inlining means all images of a
    step arrive in the same update as the page (no separate /media fetches
    landing at different times), and an unchanged image keeps identical HTML
    so the browser does not redraw it."""
    image = Image.open(path_str).convert("RGB")
    border, color = BORDERS.get(highlight, (0, None))
    inner = width - 2 * border
    h = int(round(image.height * inner / image.width))
    image = image.resize((inner, h), Image.Resampling.LANCZOS)
    if border:
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
    if not path.exists():
        st.warning(f"image missing: {path}")
        if caption:
            st.caption(caption)
        return
    w = width if width is not None else IMG_WIDTH
    try:
        tag = render_image(str(path), w, highlight)
    except Exception:
        st.warning(f"could not load image: {path}")
        if caption:
            st.caption(caption)
        return
    st.markdown(tag, unsafe_allow_html=True)
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
            choice = rec.get("choice", "-")
            st.markdown(
                f"**{rec['step'] + 1}.** {emb.label(i)} vs {emb.label(j)} -> **{choice}**"
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
            st.markdown(f"**step {rec['step']}** - **{found_side}** was picked")
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
        choice = rec.get("choice")
        if choice == "A":
            a_hl, b_hl, verdict = "chosen", "rejected", "picked A"
        elif choice == "B":
            a_hl, b_hl, verdict = "rejected", "chosen", "picked B"
        elif choice == "skip":
            a_hl, b_hl, verdict = None, None, "skipped"
        else:
            a_hl, b_hl, verdict = None, None, f"({choice})"

        st.markdown(f"**step {rec['step'] + 1}** - {verdict}")
        col1, col2 = st.columns(2)
        with col1:
            show_image(i, emb, f"A - {emb.label(i)}",
                        highlight=a_hl, width=IMG_WIDTH_HISTORY)
        with col2:
            show_image(j, emb, f"B - {emb.label(j)}",
                        highlight=b_hl, width=IMG_WIDTH_HISTORY)
        st.markdown("---")


def screen_home(emb: Embeddings):
    st.title("comparison-based search (lite)")
    st.markdown(
        "Find a scenery image by comparing pairs. Pick an algorithm, "
        "then either drive the search yourself or watch the simulated-user "
        "auto-run."
    )
    st.markdown("")

    picked = st.radio(
        "Algorithm",
        options=ALGO_OPTIONS,
        index=ALGO_OPTIONS.index(current_algo()),
        horizontal=True,
    )
    st.session_state.algo = picked
    st.markdown(f'<div class="algo-sub">{algo_subtitle()}</div>', unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start search (human)", type="primary", use_container_width=True):
            st.session_state.engine = new_engine(emb, target_idx=None,
                                                   max_queries=HUMAN_MAX_QUERIES)
            st.session_state.history = []
            go("human_search")
    with col2:
        if st.button("Auto-run (simulated user)", type="primary", use_container_width=True):
            rng = np.random.default_rng()
            target = int(rng.integers(0, emb.n))
            st.session_state.engine = new_engine(emb, target_idx=target,
                                                   max_queries=AUTO_MAX_QUERIES)
            st.session_state.history = []
            st.session_state.auto_target_idx = target
            st.session_state.auto_playing = False
            st.session_state.auto_speed = DEFAULT_SPEED
            st.session_state.auto_next_time = 0.0
            st.session_state.auto_showing_answer = False
            go("auto_search")

    st.markdown("---")
    st.caption(
        f"[Browse the scenery gallery]({config.DRIVE_GALLERY_URL}) - "
        f"{emb.n} images across {len(emb.classes())} classes"
    )


def screen_human_search(emb: Embeddings):
    engine = st.session_state.engine

    top = st.columns([1, 4, 1])
    with top[0]:
        if st.button("Home", key="human_home_top"):
            reset_all()
            go("home")
    with top[1]:
        st.markdown(
            f'<div class="query-counter">Query {engine.step + 1} / {HUMAN_MAX_QUERIES}</div>',
            unsafe_allow_html=True,
        )
    with top[2]:
        st.markdown(
            f'<div class="answered-badge"><b>{engine.step}</b> answered</div>',
            unsafe_allow_html=True,
        )

    if engine.done:
        go("human_result")

    i, j = engine.propose_query()
    st.markdown("")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        show_image(i, emb, "A (left)")
        if st.button("Pick A", key=f"pick_a_{engine.step}",
                      type="primary", use_container_width=True):
            record = engine.apply_answer(y=0, status="clean")
            record["choice"] = "A"
            st.session_state.history.append(record)
            st.rerun()
        if st.button("Found it!", key=f"found_a_{engine.step}",
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
        if st.button("Pick B", key=f"pick_b_{engine.step}",
                      type="primary", use_container_width=True):
            record = engine.apply_answer(y=1, status="clean")
            record["choice"] = "B"
            st.session_state.history.append(record)
            st.rerun()
        if st.button("Found it!", key=f"found_b_{engine.step}",
                      use_container_width=True):
            engine.stop_manual("user_found_target")
            st.session_state.found_idx = j
            st.session_state.history.append({
                "step": engine.step, "event": "user_declared_found",
                "i": i, "j": j, "found_side": "B",
            })
            go("human_result")

    st.markdown("")
    _, col_skip, _ = st.columns([2, 2, 2])
    with col_skip:
        if st.button("Skip", key=f"skip_{engine.step}", use_container_width=True):
            record = engine.apply_answer(y=None, status="skip")
            record["choice"] = "skip"
            st.session_state.history.append(record)
            st.rerun()

    render_inline_history(emb)


@st.fragment(run_every="0.05s")
def _auto_timer():
    """Invisible clock for auto-run. Runs 20x/s but draws nothing, so the heavy
    image panel is only re-sent when the state actually changes (about twice
    per step), via a full-app rerun."""
    engine = st.session_state.engine
    if st.session_state.get("auto_showing_answer", False):
        elapsed = time.monotonic() - st.session_state.auto_answer_time
        if elapsed < AUTO_HIGHLIGHT_DELAY:
            return
        record = engine.apply_answer(y=st.session_state.auto_y, status="clean")
        record["choice"] = st.session_state.auto_picked_side
        st.session_state.history.append(record)
        st.session_state.auto_showing_answer = False
        if engine.done:
            st.session_state.screen = "auto_result"
        elif st.session_state.get("auto_playing", False):
            st.session_state.auto_next_time = (
                time.monotonic()
                + SPEED_DELAYS[st.session_state.get("auto_speed", DEFAULT_SPEED)]
            )
        st.rerun(scope="app")
    elif st.session_state.get("auto_playing", False):
        if time.monotonic() < st.session_state.get("auto_next_time", 0.0):
            return
        i, j = engine.propose_query()
        y = engine.oracle_answer(i, j)
        st.session_state.auto_i = i
        st.session_state.auto_j = j
        st.session_state.auto_y = y
        st.session_state.auto_picked_side = "A" if y == 0 else "B"
        st.session_state.auto_showing_answer = True
        st.session_state.auto_answer_time = time.monotonic()
        st.rerun(scope="app")


def screen_auto_search(emb: Embeddings):
    engine = st.session_state.engine
    target_idx = st.session_state.get("auto_target_idx")
    playing = st.session_state.get("auto_playing", False)
    showing_answer = st.session_state.get("auto_showing_answer", False)

    top = st.columns([1, 4, 1])
    with top[0]:
        if st.button("Home", key="auto_home_top"):
            reset_all()
            go("home")
    with top[1]:
        st.markdown(
            f'<div class="query-counter">Query {engine.step + 1} / {AUTO_MAX_QUERIES}</div>',
            unsafe_allow_html=True,
        )
    with top[2]:
        st.markdown(
            f'<div class="answered-badge"><b>{engine.step}</b> answered</div>',
            unsafe_allow_html=True,
        )

    if engine.done:
        go("auto_result")

    ctrl = st.columns([1, 1, 2])

    with ctrl[0]:
        if playing:
            if st.button("Pause", key="auto_pause", use_container_width=True):
                # Freeze where we are: a highlighted pick stays highlighted.
                st.session_state.auto_playing = False
                st.rerun()
        else:
            if st.button("Play", key="auto_play", type="primary", use_container_width=True):
                st.session_state.auto_playing = True
                st.session_state.auto_next_time = time.monotonic()
                if showing_answer:
                    # Show the frozen pick for the usual moment, then move on.
                    st.session_state.auto_answer_time = time.monotonic()
                st.rerun()

    with ctrl[1]:
        step_clicked = st.button(
            "Step",
            key="auto_step",
            use_container_width=True,
            disabled=playing,
        )

    with ctrl[2]:
        speed = st.radio(
            "Speed",
            options=list(SPEED_DELAYS.keys()),
            index=list(SPEED_DELAYS.keys()).index(
                st.session_state.get("auto_speed", DEFAULT_SPEED)
            ),
            horizontal=True,
            key="auto_speed_radio",
            label_visibility="collapsed",
        )
        if speed != st.session_state.get("auto_speed", DEFAULT_SPEED):
            st.session_state.auto_speed = speed
            if playing and not showing_answer:
                st.session_state.auto_next_time = time.monotonic() + SPEED_DELAYS[speed]

    if step_clicked and showing_answer:
        # Paused on a highlighted pick: commit that answer and show the next pair.
        record = engine.apply_answer(y=st.session_state.auto_y, status="clean")
        record["choice"] = st.session_state.auto_picked_side
        st.session_state.history.append(record)
        st.session_state.auto_showing_answer = False
        if engine.done:
            go("auto_result")
        st.rerun()

    if showing_answer:
        i = st.session_state.auto_i
        j = st.session_state.auto_j
        picked_side = st.session_state.auto_picked_side
    else:
        i, j = engine.propose_query()
        picked_side = None

    if step_clicked:
        y_this = engine.oracle_answer(i, j)
        picked_side = "A" if y_this == 0 else "B"
        record = engine.apply_answer(y=y_this, status="clean")
        record["choice"] = picked_side
        st.session_state.history.append(record)
        if engine.done:
            go("auto_result")

    st.markdown("")
    col_a, col_b, col_t = st.columns(3, gap="medium")

    with col_a:
        st.markdown('<div class="col-header">A</div>', unsafe_allow_html=True)
        show_image(
            i, emb, emb.label(i),
            highlight="chosen" if picked_side == "A" else None,
            width=IMG_WIDTH,
        )

    with col_b:
        st.markdown('<div class="col-header">B</div>', unsafe_allow_html=True)
        show_image(
            j, emb, emb.label(j),
            highlight="chosen" if picked_side == "B" else None,
            width=IMG_WIDTH,
        )

    with col_t:
        st.markdown(
            '<div class="col-header col-header-target">TARGET</div>',
            unsafe_allow_html=True,
        )
        show_image(
            target_idx, emb, emb.label(target_idx),
            highlight="target", width=IMG_WIDTH,
        )

    if picked_side is not None:
        st.markdown(f"**Oracle picked: {picked_side}**")

    render_inline_history(emb)

    if playing:
        _auto_timer()


def start_new_human_search(emb: Embeddings):
    for k in ("engine", "history", "found_idx"):
        st.session_state.pop(k, None)
    st.session_state.engine = new_engine(emb, target_idx=None,
                                           max_queries=HUMAN_MAX_QUERIES)
    st.session_state.history = []
    go("human_search")


def start_new_auto_run(emb: Embeddings):
    for k in ("engine", "history", "auto_target_idx", "auto_playing",
              "auto_speed", "auto_next_time", "auto_showing_answer",
              "auto_i", "auto_j", "auto_y", "auto_picked_side",
              "auto_answer_time"):
        st.session_state.pop(k, None)
    rng = np.random.default_rng()
    target = int(rng.integers(0, emb.n))
    st.session_state.engine = new_engine(emb, target_idx=target,
                                           max_queries=AUTO_MAX_QUERIES)
    st.session_state.history = []
    st.session_state.auto_target_idx = target
    st.session_state.auto_playing = False
    st.session_state.auto_speed = DEFAULT_SPEED
    st.session_state.auto_next_time = 0.0
    st.session_state.auto_showing_answer = False
    go("auto_search")


def result_top_bar(emb: Embeddings, new_label: str, start_new, key: str):
    """Home (left) and new-run (right) at the top of a results page, so you
    don't have to scroll past the history to reach them."""
    left, _, right = st.columns([1, 4, 1])
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
    st.caption(algo_subtitle())
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
            show_image(found_idx, emb, f"class: **{emb.label(found_idx)}**")

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


def screen_auto_result(emb: Embeddings):
    engine = st.session_state.engine
    target_idx = st.session_state.get("auto_target_idx")

    result_top_bar(emb, "Run again", start_new_auto_run, "auto_result")
    st.title("Auto-run complete")
    st.caption(algo_subtitle())
    if engine.stop_reason == "in_query":
        st.success(f"Target reached in {engine.step} queries.")
    elif engine.stop_reason == "max_queries":
        st.warning(f"Reached the query cap ({AUTO_MAX_QUERIES}) without hitting the target.")
    else:
        st.info(f"Stopped: {engine.stop_reason}")

    st.markdown("---")

    if target_idx is not None:
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.markdown("### Target was")
            show_image(target_idx, emb, f"class: **{emb.label(target_idx)}**")

    st.markdown("---")
    metrics = st.columns(2)
    metrics[0].metric("queries", engine.step)
    if target_idx is not None:
        metrics[1].metric("target class", emb.label(target_idx))

    st.markdown("---")
    render_history_with_images(emb)

    col1, col2 = st.columns(2)
    if col1.button("Run again", type="primary", use_container_width=True):
        start_new_auto_run(emb)
    if col2.button("Home", use_container_width=True):
        reset_all()
        go("home")


def main():
    st.set_page_config(
        page_title="comparison-search (lite)",
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
    elif screen == "auto_search":
        screen_auto_search(emb)
    elif screen == "human_result":
        screen_human_result(emb)
    elif screen == "auto_result":
        screen_auto_result(emb)
    else:
        st.error(f"unknown screen: {screen}")
        if st.button("Home"):
            go("home")


if __name__ == "__main__":
    main()
