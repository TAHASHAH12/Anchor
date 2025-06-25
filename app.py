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

if opp_file and stake_file:
    opportunities_df = pd.read_csv(opp_file)
    stake_df = pd.read_csv(stake_file)

    st.success("✅ Files uploaded. Processing...")

    results_df = match_links_and_generate_anchors(opportunities_df, stake_df)

    st.subheader("🔍 Recommended Internal Links + Anchors")
    st.dataframe(results_df)

    st.download_button(
        label="⬇ Download Results CSV",
        data=results_df.to_csv(index=False),
        file_name="recommended_anchors.csv",
        mime="text/csv"
    )
