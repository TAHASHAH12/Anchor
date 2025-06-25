import streamlit as st
import pandas as pd
from anchor_utils import match_links_and_generate_anchors

st.set_page_config(page_title="Smart Anchor Matcher", layout="wide")

st.title("🔗 Anchor Text + Internal Link Matcher for Stake")

st.markdown("""
Upload two CSV files and select the relevant columns for URLs, anchors, and topics.
Then click **Process** to get recommended internal links and anchor texts.
""")

opp_file = st.file_uploader("Upload Opportunities CSV", type=["csv"], key="opp")
stake_file = st.file_uploader("Upload Stake Internal Links CSV", type=["csv"], key="stake")

def select_column(df, label):
    options = list(df.columns)
    return st.selectbox(label, options)

opportunities_df = None
stake_df = None

if opp_file:
    opportunities_df = pd.read_csv(opp_file)
    st.write("Opportunities CSV preview:")
    st.dataframe(opportunities_df.head())

    opp_url_col = select_column(opportunities_df, "Select Opportunities URL column")
    opp_anchor_col = select_column(opportunities_df, "Select Opportunities Anchor Text column (if any)")
    opp_text_col = select_column(opportunities_df, "Select Opportunities Text/Content column (optional, else leave blank)")

if stake_file:
    stake_df = pd.read_csv(stake_file)
    st.write("Stake Internal Links CSV preview:")
    st.dataframe(stake_df.head())

    stake_url_col = select_column(stake_df, "Select Stake URL column")
    stake_topic_col = select_column(stake_df, "Select Stake Topic column")
    stake_lang_col = select_column(stake_df, "Select Stake Language column (optional, else leave blank)")

process = st.button("🔍 Process Matching and Generate Anchors")

if process:
    if not (opp_file and stake_file):
        st.error("Please upload both CSV files before processing.")
    else:
        st.info("✅ Processing smart matching...")

        # Prepare DataFrames with expected columns
        opp_df = opportunities_df.copy()
        stake_df_cp = stake_df.copy()

        opp_df['Live Link'] = opp_df[opp_url_col].astype(str)
        opp_df['Anchor'] = opp_df[opp_anchor_col].astype(str) if opp_anchor_col else ""
        if opp_text_col and opp_text_col.strip() != "":
            opp_df['Content'] = opp_df[opp_text_col].astype(str)
        else:
            opp_df['Content'] = ""

        stake_df_cp['url'] = stake_df_cp[stake_url_col].astype(str)
        stake_df_cp['topic'] = stake_df_cp[stake_topic_col].astype(str)
        if stake_lang_col and stake_lang_col.strip() != "":
            stake_df_cp['lang'] = stake_df_cp[stake_lang_col].astype(str)
        else:
            stake_df_cp['lang'] = "en"

        results_df = match_links_and_generate_anchors(opp_df, stake_df_cp)

        st.subheader("🔍 Recommended Internal Links + Anchor Texts")
        st.dataframe(results_df)

        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="⬇ Download Results CSV",
            data=csv_data,
            file_name="recommended_anchors.csv",
            mime="text/csv"
        )
