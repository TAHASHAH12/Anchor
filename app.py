import streamlit as st
import pandas as pd
from anchor_utils import match_links_and_generate_anchors

st.set_page_config(page_title="Smart Anchor Matcher", layout="wide")

st.title("🔗 Anchor Text + Internal Link Matcher for Stake")

st.markdown("Upload two files:")
st.markdown("👉 **1. Opportunities CSV** – The 60–70 external articles to insert links into.")
st.markdown("👉 **2. Stake Internal Links CSV** – Cleaned internal link list with language/topic metadata.")

opp_file = st.file_uploader("📂 Upload Opportunities CSV", type=["csv"], key="opp")
stake_file = st.file_uploader("📂 Upload Stake Internal Links CSV", type=["csv"], key="stake")

if opp_file and stake_file:
    opportunities_df = pd.read_csv(opp_file)
    stake_df = pd.read_csv(stake_file)

    st.success("✅ Files uploaded. Select columns:")

    with st.expander("🔧 Choose Relevant Columns"):

        opp_anchor_col = st.selectbox("Opportunities: Anchor Column", opportunities_df.columns, index=opportunities_df.columns.get_loc("Anchor") if "Anchor" in opportunities_df.columns else 0)
        opp_url_col = st.selectbox("Opportunities: Live Link Column", opportunities_df.columns, index=opportunities_df.columns.get_loc("Live Link") if "Live Link" in opportunities_df.columns else 0)

        stake_topic_col = st.selectbox("Client: Topic Column can select anchor column as well if no topic column", stake_df.columns, index=stake_df.columns.get_loc("topic") if "topic" in stake_df.columns else 0)
        stake_url_col = st.selectbox("Client: URL Column", stake_df.columns, index=stake_df.columns.get_loc("url") if "url" in stake_df.columns else 0)
        stake_lang_col = st.selectbox("Select: Language Column", stake_df.columns, index=stake_df.columns.get_loc("lang") if "lang" in stake_df.columns else 0)

    st.success("✅ Processing smart matching...")

    results_df = match_links_and_generate_anchors(
        opportunities_df,
        stake_df,
        anchor_col=opp_anchor_col,
        opp_url_col=opp_url_col,
        stake_topic_col=stake_topic_col,
        stake_url_col=stake_url_col,
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
