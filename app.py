import os
import streamlit as st
import pandas as pd
from anchor_utils import extract_text_from_url, generate_anchor_suggestions, recommend_internal_link

st.title("Smart Anchor Text Suggestions Tool")

# --- API key input ---
if "OPENAI_API_KEY" not in st.secrets:
    st.warning("Please set your OpenAI API key in Streamlit secrets or env var.")
    openai_key = st.text_input("Enter OpenAI API key (won't be saved):", type="password")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
else:
    openai_key = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = openai_key

# --- Input section ---
input_mode = st.radio("Choose input method:", ["Paste blog text", "Provide blog URL"])

article_text = ""
if input_mode == "Paste blog text":
    article_text = st.text_area("Paste the blog article text here:", height=300)
else:
    blog_url = st.text_input("Enter blog URL:")
    if blog_url:
        with st.spinner("Fetching and parsing article..."):
            article_text = extract_text_from_url(blog_url)
        if not article_text:
            st.error("Failed to extract text from URL or page is empty.")

# --- Anchor text suggestions ---
if article_text:
    st.subheader("Anchor Text Suggestions")
    anchor_suggestions = generate_anchor_suggestions(article_text)
    if anchor_suggestions:
        st.write(anchor_suggestions)
    else:
        st.info("No anchor suggestions could be generated from the text.")

    # --- Upload internal link CSV ---
    st.subheader("Upload Stake Links CSV")
    stake_links_file = st.file_uploader("Upload CSV file with Stake URLs and topics", type=["csv"])
    if stake_links_file:
        df_links = pd.read_csv(stake_links_file)
        st.write("Select columns for anchor topics and URLs:")
        all_cols = df_links.columns.tolist()
        topic_col = st.selectbox("Select Anchor Topic Column", all_cols, index=all_cols.index("Anchor") if "Anchor" in all_cols else 0)
        url_col = st.selectbox("Select URL Column", all_cols, index=all_cols.index("Client URL") if "Client URL" in all_cols else 1)

        if topic_col and url_col:
            stake_links = df_links[[topic_col, url_col]].dropna()
            stake_links.columns = ['topic', 'url']

            # --- Filter out already hyperlinked anchors ---
            # For simplicity, assume all suggested anchors are not linked yet.
            filtered_anchors = [a for a in anchor_suggestions if a.strip()]

            if filtered_anchors:
                st.subheader("Recommended Internal Links")
                recommended = recommend_internal_link(filtered_anchors, stake_links)
                if recommended:
                    df_recommended = pd.DataFrame(recommended, columns=["Anchor Text", "Recommended URL"])
                    st.dataframe(df_recommended)
                else:
                    st.info("No recommendations could be generated.")
            else:
                st.info("No valid anchor texts to recommend links for.")
else:
    st.info("Enter blog text or URL to get anchor text suggestions.")
