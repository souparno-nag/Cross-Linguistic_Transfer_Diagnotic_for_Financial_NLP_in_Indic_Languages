"""Results dashboard: three pages over artefacts that already exist.

    streamlit run dashboard/app.py

Read-only. No model is loaded and nothing is written; every number comes from
a report or data file an earlier phase produced (see `dashboard/data.py`).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from dashboard import data as D  # noqa: E402

st.set_page_config(page_title="Indic transfer diagnostics", page_icon="📊", layout="wide")

with st.sidebar:
    st.radio("Task", list(D.TASK_NAMES), format_func=D.TASK_NAMES.get, key="task")
    st.caption(
        "Models are trained on one language and tested on another. Task 3 sits at its data ceiling "
        "(Phase 4), so expect few clear differences there."
    )

here = Path(__file__).parent / "pages"
st.navigation([
    st.Page(here / "transfer.py", title="Transfer", icon="🌐", default=True),
    st.Page(here / "causes.py", title="Causes", icon="🧩"),
    st.Page(here / "explorer.py", title="Explorer", icon="🔎"),
]).run()
