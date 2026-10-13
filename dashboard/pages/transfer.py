"""Transfer: how much a model loses moving from its training language to another."""

import altair as alt
import streamlit as st

from dashboard import cached as C
from dashboard import data as D

task = st.session_state["task"]
gaps = C.transfer_gaps(task)

st.title("How much is lost between languages")
st.write(
    "Each model is trained on one language and tested on another. A cell shows the drop in macro-F1: "
    "its score in the training language minus its score in the test language. Bigger means worse transfer."
)

measure = st.radio("Show", ["Drop in F1", "F1 in the test language"], horizontal=True)
field, title = ("gap_mean", "F1 drop") if measure == "Drop in F1" else ("target_mean", "Test F1")
top = max(gaps["gap_mean"].max(), gaps["target_mean"].max())


def heatmap(frame):
    base = alt.Chart(frame).encode(
        x=alt.X("target_name:N", sort=D.LANG_ORDER, title="Test language"),
        y=alt.Y("source_name:N", sort=D.LANG_ORDER, title="Training language"),
    )
    tooltip = [
        alt.Tooltip("source_name:N", title="Trained on"),
        alt.Tooltip("target_name:N", title="Tested on"),
        alt.Tooltip("source_mean:Q", title="F1, training language", format=".3f"),
        alt.Tooltip("target_mean:Q", title="F1, test language", format=".3f"),
        alt.Tooltip("gap_mean:Q", title="Drop", format=".3f"),
        alt.Tooltip("gap_std:Q", title="Drop, std over runs", format=".3f"),
        alt.Tooltip("n_seeds:Q", title="Training runs used"),
    ]
    cells = base.mark_rect().encode(
        color=alt.Color(f"{field}:Q", title=title, scale=alt.Scale(domain=[0, top], scheme="orangered"
                                                                   if field == "gap_mean" else "blues")),
        tooltip=tooltip,
    )
    text = base.mark_text(fontSize=15).encode(
        text=alt.Text(f"{field}:Q", format=".2f"),
        color=alt.condition(alt.datum[field] > top * 0.6, alt.value("white"), alt.value("black")),
        tooltip=tooltip,
    )
    return (cells + text).properties(height=260)


for column, (model, frame) in zip(st.columns(2), gaps.groupby("model", sort=False)):
    with column:
        st.subheader(model)
        st.altair_chart(heatmap(frame), width="stretch")

st.caption(
    "Same colour scale on both maps. Test text is written by people, except Malayalam: no native Malayalam "
    "exists, so its test sentences are machine-translated from another language's copy. A blank cell is a "
    "language tested on itself."
)
excluded = gaps[gaps["n_excluded"] > 0]
if not excluded.empty:
    runs = sorted({f"{r.model} trained on {r.source_name}, run {s}" for r in excluded.itertuples() for s in r.excluded_seeds})
    st.caption("Left out because the run never learned its training data: " + "; ".join(runs) + ".")

with st.expander("Numbers"):
    st.dataframe(
        gaps[["model", "source_name", "target_name", "source_mean", "target_mean", "gap_mean", "gap_std", "n_seeds"]]
        .rename(columns={"model": "Model", "source_name": "Trained on", "target_name": "Tested on",
                         "source_mean": "F1, training language", "target_mean": "F1, test language",
                         "gap_mean": "Drop", "gap_std": "Drop std", "n_seeds": "Runs"}),
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.3f")
                       for c in ["F1, training language", "F1, test language", "Drop", "Drop std"]},
    )
    st.caption("Source: `reports/source_comparison.parquet` (T-408).")
