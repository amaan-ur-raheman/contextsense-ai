"""Streamlit interface for ContextSense AI.

Split from the single-file ``app.py`` so the entry point stays readable and
each page can be worked on, reviewed and tested on its own.
"""

from contextsense_ai.ui.components import (
    pick_occurrence,
    render_evaluation,
    render_prediction_panel,
    render_sense_chips,
)
from contextsense_ai.ui.pages.about import render_about
from contextsense_ai.ui.pages.comparison import render_comparison
from contextsense_ai.ui.pages.evaluation import render_evaluation_page
from contextsense_ai.ui.pages.playground import render_playground
from contextsense_ai.ui.sidebar import render_sidebar
from contextsense_ai.ui.state import (
    backend_for_variant,
    ensure_backend_loaded,
    get_backend,
    init_state,
    word2vec_installed,
)
from contextsense_ai.ui.theme import inject_theme

__all__ = [
    "backend_for_variant",
    "ensure_backend_loaded",
    "get_backend",
    "init_state",
    "inject_theme",
    "pick_occurrence",
    "render_about",
    "render_comparison",
    "render_evaluation",
    "render_evaluation_page",
    "render_playground",
    "render_prediction_panel",
    "render_sidebar",
    "render_sense_chips",
    "word2vec_installed",
]
