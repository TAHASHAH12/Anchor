import streamlit as st
import pandas as pd
from anchor_utils import match_links_and_generate_anchors

st.set_page_config(page_title="Smart Anchor Matcher", layout="wide")

st.title("🔗 Anchor Text + Internal Link Matcher for Stake")

st.markdown("Upload two files:")
st.markdown("👉 **1. Opportunities CSV** – The 60–70 URLs/articles to insert links into.")
st.markdown("👉 **2. Stake Internal Links CSV** – Cleaned internal link list with language/topic metadata.")

opp_file = st.file_uploader("Upload Opportunities CSV", type=["csv"], key="opp")
stake_file = st.file_uploader("Upload Stake Internal Links CSV", type=["csv"], key="stake")

if opp_file:
    opportunities_df = pd.read_csv(opp_file)
    st.write("Opportunities CSV Preview:")
    st.dataframe(opportunities_df.head())
    opp_url_col = st.selectbox("Select 'Opportunity URL' column", options=opportunities_df.columns, key="opp_url_col")
    opp_anchor_col = st.selectbox("Select 'Anchor/Keyword' column", options=opportunities_df.columns, key="opp_anchor_col")
    opp_lang_col = st.selectbox("Select 'Language' column (optional)", options=[None] + list(opportunities_df.columns), key="opp_lang_col")

if stake_file:
    stake_df = pd.read_csv(stake_file)
    st.write("Stake Internal Links CSV Preview:")
    st.dataframe(stake_df.head())
    stake_url_col = st.selectbox("Select 'Cleint URL' column", options=stake_df.columns, key="stake_url_col")
    stake_topic_col = st.selectbox("Select 'Topic if not select anchor' column", options=stake_df.columns, key="stake_topic_col")
    stake_lang_col = st.selectbox("Select 'Language' column", options=stake_df.columns, key="stake_lang_col")

process = st.button("🔍 Process Matching and Generate Anchors")

if process:
    if not (opp_file and stake_file):
        st.error("Please upload both CSV files before processing.")
    else:
        results_df = match_links_and_generate_anchors(
            opportunities_df,
            stake_df,
            opp_url_col=opp_url_col,
            opp_anchor_col=opp_anchor_col,
            opp_lang_col=opp_lang_col,
            stake_url_col=stake_url_col,
            stake_topic_col=stake_topic_col,
            stake_lang_col=stake_lang_col,
        )

        st.subheader("🔍 Recommended Internal Links + Anchors")
        st.dataframe(results_df)

        st.download_button(
            label="⬇ Download Results CSV",
            data=results_df.to_csv(index=False),
            file_name="recommended_anchors.csv",
            mime="text/csv"
        )
