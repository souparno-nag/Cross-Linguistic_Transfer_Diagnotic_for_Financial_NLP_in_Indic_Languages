"""Causes: what each condition's failures are made of (Phase 7 label shares)."""

import altair as alt
import streamlit as st

from dashboard import cached as C
from dashboard import data as D

task = st.session_state["task"]
shares = C.cause_shares(task)

st.title("What the failures are made of")
st.write(
    "A failure is a sentence the model got right in its training language and wrong in the test language. "
    "Each bar splits one setup's failures by the cause they were given. The shares are of failures only, "
    "so they say what the failures look like, not what makes a failure more likely."
)

left, right = st.columns(2)
kind = left.radio("Test text", list(D.KIND_NAMES), format_func=D.KIND_NAMES.get, horizontal=True)
detail = right.radio("Detail", ["4 groups", "9 causes"], horizontal=True)
level, order = ("L1", D.GROUP_ORDER) if detail == "4 groups" else ("L2", D.LABEL_ORDER)

view = shares[(shares["kind"] == kind) & (shares["level"] == level)]
pairs = (view[["source", "target", "pair"]].drop_duplicates()
         .assign(s=lambda f: f["source"].map({"hin": 0, "ben": 1, "tel": 2}),
                 t=lambda f: f["target"].map({"hin": 0, "ben": 1, "tel": 2, "mal": 3}))
         .sort_values(["s", "t"])["pair"].tolist())

palette = {
    "unattributed": "#9e9e9e", "linguistic": "#4c78a8", "data": "#f58518", "model": "#54a24b",
    "translation_drift": "#1f4e79", "tokenizer_fragmentation": "#4c78a8", "morphological_masking": "#7fa7d1",
    "terminology_gap": "#b6cde6", "provenance_specific": "#f58518", "cross_lingual_hard": "#ffbf79",
    "class_collapse": "#2f7d32", "seed_unstable": "#8fd18a",
}
colour = alt.Color("measure:N", title="Cause", sort=order,
                   scale=alt.Scale(domain=order, range=[palette[m] for m in order]))

for column, (model, frame) in zip(st.columns(2), view.groupby("model", sort=False)):
    chart = alt.Chart(frame).mark_bar().encode(
        y=alt.Y("pair:N", sort=pairs, title=None),
        x=alt.X("share:Q", stack="normalize", title="Share of failures", axis=alt.Axis(format="%")),
        color=colour,
        order=alt.Order("order:Q"),
        tooltip=[
            alt.Tooltip("pair:N", title="Setup"),
            alt.Tooltip("measure:N", title="Cause"),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
            alt.Tooltip("ci_low:Q", title="95% CI low", format=".1%"),
            alt.Tooltip("ci_high:Q", title="95% CI high", format=".1%"),
            alt.Tooltip("n_failures:Q", title="Failures"),
            alt.Tooltip("failure_rate:Q", title="Failure rate", format=".1%"),
        ],
    ).properties(height=36 * len(pairs) + 40)
    with column:
        st.subheader(model)
        st.altair_chart(chart, width="stretch")

if kind == "transfer":
    st.caption("Malayalam has no native text, so even here its test sentences are machine-translated.")
st.caption(
    "Every run is counted, including the few that never learned their training data. Hover a bar for its "
    "share, 95% confidence interval and failure count."
)

st.subheader("What the causes mean")
if level == "L1":
    for group in D.GROUP_ORDER:
        st.markdown(f"**{group}**: {D.GROUP_TEXT[group]}")
else:
    for label in D.LABEL_ORDER:
        line = f"**{label}**: {D.LABEL_TEXT[label]}"
        caveat = D.label_caveat(label)
        if caveat:
            line += f" *Limitation: {caveat}.*"
        st.markdown(line)

with st.expander("Numbers"):
    # "model" is also a cause group, so the index columns are renamed first
    named = view.rename(columns={"model": "Model", "pair": "Setup", "n_failures": "Failures",
                                 "failure_rate": "Failure rate"})
    table = (named.pivot_table(index=["Model", "Setup", "Failures", "Failure rate"], columns="measure",
                               values="share")
             .reindex(columns=order).reset_index())
    st.dataframe(
        table,
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="percent") for c in ["Failure rate", *order]},
    )
    st.caption("Source: `reports/phase7/label_shares.parquet` (T-701).")
