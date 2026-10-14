"""Tests for the Streamlit playground UI.

These drive the real app through ``st.testing.v1.AppTest`` rather than asserting
on source text. Every test here corresponds to a UI defect that shipped in an
earlier revision and passed an HTTP health check while being unusable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _app() -> AppTest:
    # Absolute: AppTest resolves relative paths against the calling file.
    return AppTest.from_file(str(APP_PATH), default_timeout=600)


@pytest.fixture(scope="module")
def loaded_app() -> AppTest:
    app = _app()
    app.run()
    return app


def test_app_starts_without_exceptions(loaded_app: AppTest) -> None:
    """The app must render its first page without raising."""
    assert not loaded_app.exception


def test_example_buttons_populate_the_form(loaded_app: AppTest) -> None:
    """Each example button must load its sentence and target word.

    Assigning to a widget's session state after that widget is instantiated is
    an error in Streamlit, and the buttons are rendered below the inputs they
    drive. An inline assignment therefore raised instead of working, so every
    button was broken while the page still looked healthy.
    """
    for button in loaded_app.button:
        if not button.label.startswith(("Bank:", "Bat:", "Mouse:", "Spring:")):
            continue

        app = _app()
        app.run()
        next(b for b in app.button if b.label == button.label).click().run()

        assert not app.exception, f"{button.label} raised {app.exception[0].value}"
        assert app.text_area[0].value, f"{button.label} left the sentence empty"
        assert app.text_input[0].value, f"{button.label} left the target word empty"


def test_default_sentence_is_populated() -> None:
    """The playground must open with a usable sentence already loaded.

    A text_area cannot be given a default alongside ``key``, so the seed lives
    in session state. If that seeding regressed, the field renders empty and
    the first submission fails validation.
    """
    app = _app()
    app.run()

    assert not app.exception
    assert app.text_area[0].value
    assert app.text_input[0].value
