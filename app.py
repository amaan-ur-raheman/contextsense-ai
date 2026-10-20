#!/usr/bin/env python3
"""ContextSense AI - Streamlit Web Interface.

Run with: uv run streamlit run app.py

This file is only the entry point. The interface itself lives in
``contextsense_ai.ui``: one module per page, plus shared components, state
and theme.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from contextsense_ai.config import settings
from contextsense_ai.ui import (
    init_state,
    inject_theme,
    render_about,
    render_comparison,
    render_evaluation_page,
    render_playground,
    render_sidebar,
)

settings.ensure_directories()

# set_page_config must be the first Streamlit command of the run.
st.set_page_config(
    page_title="ContextSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()
inject_theme()
render_sidebar()

PAGES = {
    "🎮 Playground": render_playground,
    "📊 Model Comparison": render_comparison,
    "📈 Dataset Evaluation": render_evaluation_page,
    "📚 About & Learn": render_about,
}

page = st.sidebar.radio("Navigation", list(PAGES.keys()))
PAGES[page]()
