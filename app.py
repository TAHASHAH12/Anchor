import streamlit as st
import pandas as pd
from anchor_utils import match_links_and_generate_anchors

st.set_page_config(page_title="Smart Anchor Matcher", layout="wide")

st.title("🔗 Smart Anchor Text + Internal Link Matcher")

st.markdown("Upload two CSV files:")
st.markdown("👉 **1. Opportunities CSV** – URLs/articles where you want to insert links.")
st.markdown("👉 **2. Internal Links CSV** – Your site's internal URLs with topic and language metadata.")

opp_file = st.file_uploader("Upload Opportunities CSV", type=["csv"], key="opp")
stake_file = st.file_uploader("Upload Internal Links CSV", type=["csv"], key="stake")

if opp_file and stake_file:
    opp_df = pd.read_csv(opp_file)
    stake_df = pd.read_csv(stake_file)

    st.write("Select the relevant columns for your data:")

    with st.form("column_selector_form"):
        opp_url_col = st.selectbox("Opportunities CSV: URL Column", opp_df.columns, index=opp_df.columns.get_loc("Live Link") if "Live Link" in opp_df.columns else 0)
        anchor_col = st.selectbox("Opportunities CSV: Anchor Text Column", opp_df.columns, index=opp_df.columns.get_loc("Anchor") if "Anchor" in opp_df.columns else 0)

        stake_url_col = st.selectbox("Internal Links CSV: URL Column", stake_df.columns, index=stake_df.columns.get_loc("url") if "url" in stake_df.columns else 0)
        stake_topic_col = st.selectbox("Internal Links CSV: Topic/Keyword Column", stake_df.columns, index=stake_df.columns.get_loc("topic") if "topic" in stake_df.columns else 0)
        stake_lang_col = st.selectbox("Internal Links CSV: Language Column", stake_df.columns, index=stake_df.columns.get_loc("lang") if "lang" in stake_df.columns else 0)

        submitted = st.form_submit_button("Process Matching")

    if submitted:
        with st.spinner("✅ Processing smart matching..."):
            results_df = match_links_and_generate_anchors(
                opp_df,
                stake_df,
                anchor_col=anchor_col,
                opp_url_col=opp_url_col,
                stake_topic_col=stake_topic_col,
                stake_url_col=stake_url_col,
                stake_lang_col=stake_lang_col,
            )

        st.subheader("🔍 Recommended Internal Links + Anchor Texts")
        st.dataframe(results_df)

        csv = results_df.to_csv(index=False)
        st.download_button(
            label="⬇ Download Results CSV",
            data=csv,
            file_name="recommended_anchors.csv",
            mime="text/csv"
        )
