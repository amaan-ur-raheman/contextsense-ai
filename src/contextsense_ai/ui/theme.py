"""Visual design system for the Streamlit interface.

Colour and type live in ``.streamlit/config.toml`` as design tokens, so
Streamlit owns both the light and dark themes instead of CSS overrides
fighting it. This module only holds what a config file cannot express: a
spacing scale, and the interactive states Streamlit leaves half-built.
"""

from __future__ import annotations

import streamlit as st


def inject_theme() -> None:
    """Inject the component stylesheet."""
    st.markdown(
        """
        <style>
        /* Colour lives in .streamlit/config.toml. This block only handles the two
           things a config file cannot express: a spacing/hierarchy scale, and the
           interactive states Streamlit leaves half-built (focus rings especially).
           Every selector here is scoped to a class Streamlit owns, so a rename
           upstream degrades to "unstyled" rather than to something broken. */

        /* ---- scale ---------------------------------------------------------- */
        :root {
            --cs-space-1: 0.25rem;
            --cs-space-2: 0.5rem;
            --cs-space-3: 0.75rem;
            --cs-space-4: 1rem;
            --cs-space-6: 1.5rem;
            --cs-space-8: 2rem;
            /* ease-out curves, not ease-in: motion that starts slow reads as lag at
               exactly the moment the user is watching. */
            --cs-ease-out: cubic-bezier(0.23, 1, 0.32, 1);
            --cs-fast: 140ms;
            --cs-base: 180ms;
        }

        /* Streamlit renders wide containers; pull the measure in so prose stays
           inside the 65-75ch comfortable range instead of running edge to edge. */
        .block-container {
            padding-top: var(--cs-space-8);
            padding-bottom: var(--cs-space-8);
            max-width: 68rem;
        }

        /* ---- headings ------------------------------------------------------- */
        h1, h2, h3 {
            letter-spacing: -0.02em;   /* -0.04em is the floor; tighter looks cramped */
            text-wrap: balance;
        }
        h1 { line-height: 1.2; }
        p, li { text-wrap: pretty; }

        /* The emoji-glyph in a page title is decoration next to a real label.
           Dim it so it reads as a marker, not as the loudest thing on screen. */
        h1 img, h2 img, h3 img {
            opacity: 0.75;
            margin-right: 0.15em;
        }

        /* ---- buttons -------------------------------------------------------- */
        /* Give every button a press response. Without :active scale the UI feels
           like it did not hear the click. 0.97 is small enough to read as
           feedback rather than as a cartoon. */
        .stButton > button,
        .stFormSubmitButton > button {
            transition: transform var(--cs-fast) var(--cs-ease-out),
                        background-color var(--cs-fast) var(--cs-ease-out),
                        border-color var(--cs-fast) var(--cs-ease-out),
                        color var(--cs-fast) var(--cs-ease-out),
                        box-shadow var(--cs-fast) var(--cs-ease-out);
            will-change: transform;
        }
        .stButton > button:active,
        .stFormSubmitButton > button:active {
            transform: scale(0.97);
        }

        /* Hover only where a real pointer exists. On touch, hover fires on tap and
           leaves the element stuck in its hover state after the user is done. */
        @media (hover: hover) and (pointer: fine) {
            .stButton > button:hover,
            .stFormSubmitButton > button:hover {
                transform: translateY(-1px);
            }
        }

        /* Streamlit ships a weak default focus ring. Keyboard users need this to be
           unmistakably visible, so it is drawn on the element's own accent. */
        .stButton > button:focus-visible,
        .stFormSubmitButton > button:focus-visible,
        .stTextInput input:focus-visible,
        .stSelectbox [data-baseweb="select"] > div:focus-within,
        .stTextArea textarea:focus-visible {
            outline: 2px solid var(--primary-color);
            outline-offset: 2px;
        }

        /* ---- sidebar -------------------------------------------------------- */
        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--border-color);
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: var(--cs-space-6);
        }

        /* ---- tabs ----------------------------------------------------------- */
        /* The active tab is identified by an underline, not by a colour shift
           alone, so the current selection survives greyscale and colour blindness. */
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: var(--primary-color);
        }
        .stTabs [data-baseweb="tab"] {
            transition: color var(--cs-base) var(--cs-ease-out),
                        background-color var(--cs-base) var(--cs-ease-out);
            padding-top: var(--cs-space-2);
            padding-bottom: var(--cs-space-2);
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            font-weight: 600;
        }

        /* ---- metrics -------------------------------------------------------- */
        /* Streamlit draws a hairline above each metric. Removing it and using space
           instead stops the row reading as a table. */
        [data-testid="stMetric"] {
            border: none;
            border-top: 2px solid var(--border-color);
            padding-top: var(--cs-space-3);
            background: transparent;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 0.8rem;
            letter-spacing: 0.01em;
            color: var(--text-color);
            opacity: 0.62;
        }
        [data-testid="stMetricValue"] {
            font-variant-numeric: tabular-nums;
            letter-spacing: -0.01em;
        }

        /* ---- expanders ------------------------------------------------------ */
        [data-testid="stExpander"] details {
            transition: background-color var(--cs-base) var(--cs-ease-out);
        }
        [data-testid="stExpander"] summary {
            transition: color var(--cs-fast) var(--cs-ease-out);
        }

        /* ---- app components (defined below in render helpers) --------------- */
        /* A sense chip. Rank 1 is filled with the accent because selection is the
           one thing this UI must make instantly scannable. */
        .sense-chip {
            display: inline-flex;
            align-items: baseline;
            gap: 0.4rem;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 500;
            line-height: 1.5;
            border: 1px solid var(--border-color);
            background: var(--secondary-background-color);
            color: var(--text-color);
            white-space: nowrap;
        }
        .sense-chip[data-selected="true"] {
            background: var(--primary-color);
            border-color: var(--primary-color);
            color: #ffffff;
        }
        /* Dark themes need light ink on the light-ish accent chip, not white. */
        [data-theme="dark"] .sense-chip[data-selected="true"] {
            color: #0d1116;
        }
        .sense-chip .score {
            font-variant-numeric: tabular-nums;
            opacity: 0.7;
            font-weight: 400;
        }
        [data-theme="dark"] .sense-chip[data-selected="true"] .score {
            opacity: 0.75;
        }

        /* The prediction panel. A full border plus a tinted fill -- deliberately no
           coloured side stripe, which reads as a template rather than a design. */
        .prediction-panel {
            border: 1px solid var(--border-color);
            border-radius: 12px;
            background: var(--secondary-background-color);
            padding: var(--cs-space-4) var(--cs-space-6);
            margin-bottom: var(--cs-space-4);
        }
        .prediction-panel .label {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            opacity: 0.55;
            margin-bottom: var(--cs-space-1);
        }
        .prediction-panel .synset {
            font-size: 1.5rem;
            font-weight: 650;
            letter-spacing: -0.02em;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            line-height: 1.25;
        }
        .prediction-panel .gloss {
            margin-top: var(--cs-space-2);
            max-width: 68ch;
        }

        /* Reduced motion: keep the colour transitions that aid comprehension, drop
           everything that moves. Not optional -- vestibular triggers are real. */
        @media (prefers-reduced-motion: reduce) {
            .stButton > button,
            .stFormSubmitButton > button,
            .stTabs [data-baseweb="tab"],
            [data-testid="stExpander"] details,
            [data-testid="stExpander"] summary {
                transition-duration: 1ms !important;
            }
            .stButton > button:hover,
            .stFormSubmitButton > button:hover,
            .stButton > button:active,
            .stFormSubmitButton > button:active {
                transform: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
