"""Streamlit interface for the LITE version of comparison-based scenery search.

No LLM. The user clicks A, B, or Skip on each pair. Everything else is the
same as the main app's human mode.

Screen flow:
    home           -> start search
    human_search   -> 2 images + A/B/Skip buttons + "Found it!" under each image
                      Collapsible history strip below.
    human_result   -> what you found + full history with images + New / Home
"""
import numpy as np
import streamlit as st

from . import config
from .embeddings import Embeddings
from .search_engine import SearchEngine


IMG_WIDTH = 340
IMG_WIDTH_HISTORY = 180
HUMAN_MAX_QUERIES = 50


COMPACT_CSS = """
<style>
    header[data-testid="stHeader"] { height: 0rem; visibility: hidden; }
    div.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    button[title="View fullscreen"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
</style>
"""


@st.cache_resource(show_spinner=False)
def load_embeddings() -> Embeddings:
    return Embeddings()


def go(screen: str):
    st.session_state.screen = screen
    st.rerun()


def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


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
        choice = rec.get("choice")
        if choice == "A":
            a_hl, b_hl, verdict = "chosen", "rejected", "you picked A"
        elif choice == "B":
            a_hl, b_hl, verdict = "rejected", "chosen", "you picked B"
        elif choice == "skip":
            a_hl, b_hl, verdict = None, None, "you skipped"
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
        "Find a scenery image by comparing pairs. Click whichever image is "
        "closer to what you have in mind; the search narrows down until you "
        "find your target."
    )
    st.markdown("")

    _, col, _ = st.columns([1, 2, 1])
    with col:
        if st.button("Start search", type="primary", use_container_width=True):
            rng = np.random.default_rng()
            st.session_state.engine = SearchEngine(
                X=emb.X,
                sigma_eps=config.SIGMA_EPS,
                target_idx=None,
                seed=int(rng.integers(0, 1_000_000)),
                max_queries=HUMAN_MAX_QUERIES,
            )
            st.session_state.history = []
            go("human_search")

    st.markdown("---")
    st.caption(
        f"[Browse the scenery gallery]({config.DRIVE_GALLERY_URL}) - "
        f"{emb.n} images across {len(emb.classes())} classes"
    )


def screen_human_search(emb: Embeddings):
    engine: SearchEngine = st.session_state.engine

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
        show_image(i, emb, "A (left)", width=IMG_WIDTH)
        col_a1, col_a2 = st.columns(2)
        if col_a1.button("Pick A", key=f"pick_a_{engine.step}",
                          type="primary", use_container_width=True):
            record = engine.apply_answer(y=0, status="clean")
            record["choice"] = "A"
            st.session_state.history.append(record)
            st.rerun()
        if col_a2.button("Found it!", key=f"found_a_{engine.step}",
                          use_container_width=True):
            engine.stop_manual("user_found_target")
            st.session_state.found_idx = i
            st.session_state.history.append({
                "step": engine.step, "event": "user_declared_found",
                "i": i, "j": j, "found_side": "A",
            })
            go("human_result")
    with col_b:
        show_image(j, emb, "B (right)", width=IMG_WIDTH)
        col_b1, col_b2 = st.columns(2)
        if col_b1.button("Pick B", key=f"pick_b_{engine.step}",
                          type="primary", use_container_width=True):
            record = engine.apply_answer(y=1, status="clean")
            record["choice"] = "B"
            st.session_state.history.append(record)
            st.rerun()
        if col_b2.button("Found it!", key=f"found_b_{engine.step}",
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
        if st.button("Skip - neither is close",
                      key=f"skip_{engine.step}",
                      use_container_width=True):
            record = engine.apply_answer(y=None, status="skip")
            record["choice"] = "skip"
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
                        width=IMG_WIDTH + 60)

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
        go("human_search")
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

    emb = load_embeddings()

    screen = st.session_state.screen
    if screen == "home":
        screen_home(emb)
    elif screen == "human_search":
        screen_human_search(emb)
    elif screen == "human_result":
        screen_human_result(emb)
    else:
        st.error(f"unknown screen: {screen}")
        if st.button("Home"):
            go("home")


if __name__ == "__main__":
    main()