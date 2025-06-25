import streamlit as st
from anchor_utils import *
import pandas as pd
import requests

st.set_page_config(page_title="🔗 Smart Anchor Text Tool", layout="wide")
st.title("🔗 Smart Anchor Text Suggestions Tool")

st.markdown("### 📝 Step 1: Provide Blog Content")

# Three content input methods
col1, col2, col3 = st.columns(3)
uploaded_file = col1.file_uploader("📄 Upload Blog File (HTML or TXT)", type=["html", "txt"])
blog_text_input = col2.text_area("✏️ Or Paste Blog Content Here", height=150)
url_input = col3.text_input("🌐 Or Paste Blog URL")

# Upload internal links CSV
stake_links_file = st.file_uploader("📎 Upload Stake Internal Links CSV (any column names)", type=["csv"])

# Load internal links CSV and allow column selection
stake_links = None
topic_col = url_col = None

if stake_links_file:
    stake_links_df = pd.read_csv(stake_links_file)
    st.subheader("🔧 Select Columns for Topics and URLs")
    all_columns = stake_links_df.columns.tolist()

    topic_col = st.selectbox("🧠 Select the Topic Column", all_columns, index=0)
    url_col = st.selectbox("🔗 Select the URL Column", all_columns, index=1 if len(all_columns) > 1 else 0)

    if topic_col and url_col:
        stake_links = stake_links_df[[topic_col, url_col]].dropna()
        stake_links.columns = ["topic", "url"]  # standardize column names

# Extract blog content
def extract_blog_content():
    existing_anchors = []
    if uploaded_file:
        raw_html = uploaded_file.read().decode("utf-8")
        existing_anchors = get_existing_anchors(raw_html)
        return clean_html(raw_html), existing_anchors
    elif blog_text_input.strip():
        return blog_text_input, []
    elif url_input:
        try:
            response = requests.get(url_input, timeout=10)
            html = response.text
            existing_anchors = get_existing_anchors(html)
            return clean_html(html), existing_anchors
        except Exception as e:
            st.error(f"Failed to fetch URL: {e}")
            return "", []
    return "", []

# Main logic
if stake_links is not None and (uploaded_file or blog_text_input.strip() or url_input.strip()):
    raw_text, existing_anchors = extract_blog_content()

    if raw_text.strip() == "":
        st.warning("⚠️ Could not load blog content. Please check the inputs.")
    else:
        st.subheader("🧠 Suggested Anchor Texts")
        anchor_suggestions = generate_anchor_suggestions(raw_text)
        filtered_anchors = [a for a in anchor_suggestions if a not in existing_anchors]

        if filtered_anchors:
            for anchor in filtered_anchors:
                st.markdown(f"- **{anchor}**")
        else:
            st.info("No new anchor suggestions found.")

        st.subheader("🔗 Recommended Internal Links")
        recommended = recommend_internal_link(filtered_anchors, stake_links)
        df = pd.DataFrame(recommended, columns=["Anchor Text", "Suggested Internal URL"])
        st.dataframe(df)

        st.download_button("📥 Download Results as CSV", df.to_csv(index=False), file_name="suggested_anchors.csv")

        st.success("✅ Review and apply suggestions as needed.")
else:
    st.warning("Please provide blog content and upload the internal links CSV to begin.")
