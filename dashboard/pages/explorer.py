"""Explorer: one failed sentence at a time, with every run's prediction and diagnosis."""

import pandas as pd
import streamlit as st

from dashboard import cached as C
from dashboard import data as D

task = st.session_state["task"]
conditions = C.conditions(task)
classes = C.class_names(task)

st.title("One failure at a time")
st.write(
    "Pick a setup, then a sentence the model got right in its training language and wrong in the test "
    "language. You see both sentences, every training run's predictions on both models, and the cause each "
    "failure was given."
)

sources = ["hin", "ben", "tel"]
c1, c2, c3 = st.columns(3)
source = c1.selectbox("Trained on", sources, format_func=D.LANG_NAMES.get)
target = c2.selectbox("Tested on", [t for t in ["hin", "ben", "tel", "mal"] if t != source],
                      format_func=D.LANG_NAMES.get)
options = [c for c in conditions if c["source"] == source and c["target"] == target]
condition = c3.selectbox("Test text", options, format_func=D.describe_test_text)

versions = C.corpus_versions(task)
fails = C.failures(task, condition)
preds = C.predictions(task, condition)
untrained = C.untrained_runs()

f1, f2 = st.columns(2)
models = f1.multiselect("Failed on", list(D.ENCODER_NAMES), default=list(D.ENCODER_NAMES),
                        format_func=D.ENCODER_NAMES.get)
causes = f2.multiselect("Cause", D.LABEL_ORDER, placeholder="Any cause")
shown = fails[fails["encoder"].isin(models)]
if causes:
    shown = shown[shown["item_id"].isin(shown.loc[shown["assigned_label"].isin(causes), "item_id"])]
items = D.failed_items(shown, versions, condition) if not shown.empty else pd.DataFrame()

st.caption(f"{len(items)} sentences failed on at least one run. Most consistent failures first; click a row.")
if items.empty:
    st.stop()
picked = st.dataframe(
    items[["IndicBERT-v2 runs failed", "mBERT runs failed", "causes", "text"]],
    hide_index=True, width="stretch", height=260,
    on_select="rerun", selection_mode="single-row", key="items",
    column_config={"text": st.column_config.TextColumn("Test sentence", width="large"),
                   "causes": "Causes"},
)
rows = picked.selection.rows
if not rows:
    st.info("Click a row to see that sentence.")
    st.stop()
item_id = items.iloc[rows[0]]["item_id"]

# --------------------------------------------------------------------------
# The sentence
# --------------------------------------------------------------------------

st.divider()
item_preds = preds[preds["item_id"] == item_id]
gold = int(item_preds["gold"].iloc[0])
st.markdown(f"**Correct label:** {classes[gold]}  ·  item `{item_id}`")

mine = versions[versions["item_id"] == item_id]
train_text = mine[(mine["block_id"] == condition["train_block"]) & (mine["lang"] == source)]["text"].iloc[0]
test_text = mine[(mine["block_id"] == condition["eval_block"]) & (mine["lang"] == target)]["text"].iloc[0]
a, b = st.columns(2)
with a:
    st.markdown(f"**Training language:** {D.LANG_NAMES[source]}, written by people")
    st.info(train_text)
with b:
    st.markdown(f"**Test sentence:** {D.describe_test_text(condition)}")
    st.warning(test_text)

# --------------------------------------------------------------------------
# Every run
# --------------------------------------------------------------------------

item_fails = fails[fails["item_id"] == item_id].set_index(["encoder", "seed"])
table = []
for r in item_preds.sort_values(["encoder", "seed"]).itertuples():
    key = (r.encoder, r.seed)
    if key in item_fails.index:
        outcome = item_fails.loc[key, "assigned_label"]
    elif r.pred_src != r.gold:
        outcome = f"not counted: wrong in {D.LANG_NAMES[source]} too"
    elif r.pred_tgt == r.gold:
        outcome = "right in both"
    else:
        outcome = "—"
    never = r.seed in untrained.get(D.run_prefix(task, source, r.encoder), set())
    table.append({
        "Model": D.ENCODER_NAMES[r.encoder],
        "Run": f"{r.seed}{'  (never learned)' if never else ''}",
        f"In {D.LANG_NAMES[source]}": f"{'✓' if r.pred_src == r.gold else '✗'} {classes[int(r.pred_src)]}",
        f"In {D.LANG_NAMES[target]}": f"{'✓' if r.pred_tgt == r.gold else '✗'} {classes[int(r.pred_tgt)]}",
        "Cause": outcome,
    })
st.subheader("Every training run")
st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")

given = [label for label in D.LABEL_ORDER if label in set(item_fails["assigned_label"])]
for label in given:
    line = f"**{label}**: {D.LABEL_TEXT[label]}"
    caveat = D.label_caveat(label)
    if caveat:
        line += f" *Limitation: {caveat}.*"
    st.markdown(line)

with st.expander("Every check, per failed run"):
    checks = item_fails.reset_index()
    checks["Model"] = checks["encoder"].map(D.ENCODER_NAMES)
    checks = checks.rename(columns={"seed": "Run", "labse_sim": "Similarity", "r_frag": "Token ratio",
                                    **dict(D.CHECKS), "assigned_label": "Cause"})
    st.dataframe(
        checks[["Model", "Run", "Similarity", "Token ratio", *[name for _, name in D.CHECKS], "Cause"]],
        hide_index=True, width="stretch",
        column_config={"Similarity": st.column_config.NumberColumn(format="%.3f"),
                       "Token ratio": st.column_config.NumberColumn(format="%.2f")},
    )
    st.caption(
        "fired: the check found its problem. not_fired: checked, nothing found. not_evaluated: skipped because "
        "the similarity check fired first. not_applicable: the check has nothing to ask of this sentence. "
        "The cause is the first check to fire, in the order shown."
    )

with st.expander("Every version of this item in the corpus"):
    all_versions = mine.assign(
        Language=mine["lang"].map(D.LANG_NAMES),
        Version=[("written by people" if o == "native" else f"translated from {D.LANG_NAMES[s]}")
                 for o, s in zip(mine["origin"], mine["src_lang"])],
        Flags=[", ".join(f) if f is not None else "" for f in mine["flags"]],
    ).sort_values(["lang", "origin"], key=lambda s: s.map({"hin": 0, "ben": 1, "tel": 2, "mal": 3,
                                                          "native": 0, "mt": 1}))
    st.dataframe(all_versions[["Language", "Version", "text", "Flags"]].rename(columns={"text": "Text"}),
                 hide_index=True, width="stretch")
