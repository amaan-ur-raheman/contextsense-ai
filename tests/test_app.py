"""Tests for the Streamlit playground UI.

These drive the real app through ``st.testing.v1.AppTest`` rather than asserting
on source text. Every test here corresponds to a UI defect that shipped in an
earlier revision and passed an HTTP health check while being unusable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from contextsense_ai.config import settings

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


def test_uploaded_csv_is_loaded() -> None:
    """The Upload CSV tab must accept a Streamlit upload.

    ``load_csv_dataset`` called ``Path()`` on its argument, so a real
    ``UploadedFile`` raised ``TypeError`` and the tab only ever failed, while
    the page still rendered normally.
    """
    app = _app()
    app.run()
    app.radio[1].set_value("📈 Dataset Evaluation").run()

    app.file_uploader[0].upload(
        "probe.csv",
        b"sentence,target_word,gold_synset\nShe deposited the cheque at the bank.,bank,bank.n.02\n",
        "text/csv",
    ).run()

    assert not app.exception, f"upload raised {app.exception[0].value}"
    text = [m.value for m in app.markdown]
    assert "Total examples: 1" in text
    assert "Valid: 1" in text


def test_word2vec_toggle_is_honoured_on_submit() -> None:
    """Checking the Word2Vec box must change the backend actually used.

    The submit handler recomputed ``sentence_transformer`` unconditionally,
    so the checkbox only ever changed the info panel. Loading Word2Vec is
    skipped here because it needs a 1.5 GB manual download.
    """
    if any((settings.data_dir / "word2vec").glob("*.bin*")):
        pytest.skip("Word2Vec vectors are installed; loading them is not a test.")

    app = _app()
    app.run()
    # Both deltas must land in one run: a checkbox inside st.form does not
    # commit until submit, so an intermediate run() re-reads it as False.
    next(c for c in app.checkbox if "Word2Vec" in c.label).check()
    next(b for b in app.button if "Disambiguate" in b.label).click().run()

    assert not app.exception, f"submit raised {app.exception[0].value}"
    assert any("word2vec" in e.value for e in app.error), (
        "submission used the sentence-transformer backend despite the checkbox"
    )
