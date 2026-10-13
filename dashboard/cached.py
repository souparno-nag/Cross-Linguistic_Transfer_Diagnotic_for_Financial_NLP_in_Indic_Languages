"""`dashboard.data`'s loaders, cached for the app. The artefacts do not change
while the app runs, so each is read once per argument."""

import streamlit as st

from dashboard import data as D

class_names = st.cache_data(D.class_names)
untrained_runs = st.cache_data(D.untrained_runs)
transfer_gaps = st.cache_data(D.transfer_gaps)
cause_shares = st.cache_data(D.cause_shares)
conditions = st.cache_data(D.conditions)
corpus_versions = st.cache_data(D.corpus_versions)
failures = st.cache_data(D.failures)
predictions = st.cache_data(D.predictions)
