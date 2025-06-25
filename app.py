import streamlit as st
import pandas as pd
from anchor_utils import match_links_and_generate_anchors

st.set_page_config(page_title="Smart Anchor & Link Matcher", layout="wide")
st.title("🔗 Smart Anchor Text + Internal Link Matcher")

st.markdown("""
Upload two CSV files:

1. **Opportunities CSV** – URLs/articles where you want to insert links.
2. **Internal Links CSV** – Your internal link targets with metadata (topics, languages, URLs).
""")

# Upload files
opp_file = st.file_uploader("Upload Opportunities CSV", type=["csv"], key="opp")
stake_file = st.file_uploader("Upload Internal Links CSV", type=["csv"], key="stake")

# Load files and show sample columns for column selection
if opp_file:
    opportunities_df = pd.read_csv(opp_file)
    st.write("### Opportunities CSV preview")
    st.dataframe(opportunities_df.head())
    opp_cols = opportunities_df.columns.tolist()
else:
    opportunities_df = None
    opp_cols = []

if stake_file:
    stake_df = pd.read_csv(stake_file)
    st.write("### Internal Links CSV preview")
    st.dataframe(stake_df.head())
    stake_cols = stake_df.columns.tolist()
else:
    stake_df = None
    stake_cols = []

if opportunities_df is not None and stake_df is not None:
    st.markdown("---")
    st.write("### Select relevant columns from your CSVs")

    opp_url_col = st.selectbox("Select Opportunities URL Column", opp_cols, key="opp_url")
    opp_anchor_col = st.selectbox("Select Opportunities Anchor Column", opp_cols, key="opp_anchor")

    stake_url_col = st.selectbox("Select Internal Links URL Column", stake_cols, key="stake_url")
    stake_topic_col = st.selectbox("Select Internal Links Topic/Keyword Column", stake_cols, key="stake_topic")
    stake_lang_col = st.selectbox("Select Internal Links Language Column", stake_cols, key="stake_lang")

    process = st.button("Process & Generate Recommendations")

    if process:
        with st.spinner("Processing matching and generating anchors..."):
            results_df = match_links_and_generate_anchors(
                opportunities_df,
                stake_df,
                anchor_col=opp_anchor_col,
                opp_url_col=opp_url_col,
                stake_topic_col=stake_topic_col,
                stake_url_col=stake_url_col,
                stake_lang_col=stake_lang_col
            )

        st.success("✅ Done!")
        st.subheader("🔍 Recommended Internal Links + Anchor Texts")
        st.dataframe(results_df)

        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="⬇ Download Results CSV",
            data=csv_data,
            file_name="recommended_anchors.csv",
            mime="text/csv"
        )
