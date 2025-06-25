import streamlit as st
import pandas as pd
import openai
import os
from anchor_utils import match_links_and_generate_anchors

# Load OPENAI_API_KEY from .env or environment variables automatically
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()  # instantiate client for new API usage

def test_openai_connection():
    if not openai.api_key:
        return False, "OPENAI_API_KEY not set"
    try:
        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=5,
        )
        return True, "OpenAI API connected successfully"
    except Exception as e:
        return False, f"OpenAI API connection error: {e}"

st.set_page_config(page_title="Smart Anchor Matcher", layout="wide")
st.title("🔗 Smart Anchor Text + Internal Link Matcher")

connected, msg = test_openai_connection()
if not connected:
    st.error(msg)
    st.stop()
else:
    st.success(msg)

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
        opp_url_col = st.selectbox("Opportunities CSV: URL Column", opp_df.columns, index=0)
        anchor_col = st.selectbox("Opportunities CSV: Anchor Text Column", opp_df.columns, index=0)

        stake_url_col = st.selectbox("Internal Links CSV: URL Column", stake_df.columns, index=0)
        stake_topic_col = st.selectbox("Internal Links CSV: Topic/Keyword Column", stake_df.columns, index=0)
        stake_lang_col = st.selectbox("Internal Links CSV: Language Column", stake_df.columns, index=0)

        submitted = st.form_submit_button("Process Matching")

    if submitted:
        with st.spinner("Processing smart matching..."):
            links_df, anchors_df = match_links_and_generate_anchors(
                opp_df,
                stake_df,
                anchor_col=anchor_col,
                opp_url_col=opp_url_col,
                stake_topic_col=stake_topic_col,
                stake_url_col=stake_url_col,
                stake_lang_col=stake_lang_col,
            )

        st.subheader("🔍 Recommended Internal Links")
        st.dataframe(links_df)

        st.subheader("💡 Suggested Anchor Texts")
        st.dataframe(anchors_df)

        st.download_button(
            label="⬇ Download Recommended Links CSV",
            data=links_df.to_csv(index=False),
            file_name="recommended_internal_links.csv",
            mime="text/csv"
        )
        st.download_button(
            label="⬇ Download Suggested Anchors CSV",
            data=anchors_df.to_csv(index=False),
            file_name="suggested_anchors.csv",
            mime="text/csv"
        )
