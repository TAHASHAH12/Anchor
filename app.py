import streamlit as st
import pandas as pd
import requests
from anchor_utils import clean_html, get_existing_anchors, generate_anchor_suggestions, recommend_internal_link

st.set_page_config(page_title="Smart Anchor Text Tool", layout="wide")
st.title("🔗 Smart Anchor Text Suggestions Tool")

st.markdown("### Step 1: Provide your blog content")

col1, col2, col3 = st.columns(3)

uploaded_file = col1.file_uploader("Upload Blog HTML/TXT file", type=["html", "txt"])
blog_text = col2.text_area("Or paste blog content here", height=150)
blog_url = col3.text_input("Or enter blog URL")

stake_links_file = st.file_uploader("Upload internal links CSV", type=["csv"])

topic_col = url_col = None
stake_links = None

if stake_links_file:
    df_links = pd.read_csv(stake_links_file)
    st.write("Select the columns for topic and URL:")
    all_cols = df_links.columns.tolist()
    topic_col = st.selectbox("Topic Column", all_cols, index=0)
    url_col = st.selectbox("URL Column", all_cols, index=1 if len(all_cols) > 1 else 0)

    if topic_col and url_col:
        stake_links = df_links[[topic_col, url_col]].dropna()
        stake_links.columns = ['topic', 'url']

def extract_blog_content():
    existing_anchors = []
    if uploaded_file:
        raw_html = uploaded_file.read().decode("utf-8")
        existing_anchors = get_existing_anchors(raw_html)
        return clean_html(raw_html), existing_anchors
    elif blog_text.strip():
        return blog_text.strip(), []
    elif blog_url.strip():
        try:
            response = requests.get(blog_url.strip(), timeout=10)
            html = response.text
            existing_anchors = get_existing_anchors(html)
            return clean_html(html), existing_anchors
        except Exception as e:
            st.error(f"Failed to fetch blog URL: {e}")
            return "", []
    return "", []

if stake_links is not None and (uploaded_file or blog_text.strip() or blog_url.strip()):
    article_text, existing_anchors = extract_blog_content()

    if not article_text:
        st.warning("Please provide valid blog content.")
    else:
        st.subheader("Anchor Text Suggestions")
        anchor_suggestions = generate_anchor_suggestions(article_text)
        filtered_anchors = [a for a in anchor_suggestions if a not in existing_anchors]

        if filtered_anchors:
            for anchor in filtered_anchors:
                st.markdown(f"- **{anchor}**")
        else:
            st.info("No new anchor suggestions found.")

        st.subheader("Recommended Internal Links")
        recommended = recommend_internal_link(filtered_anchors, stake_links)
        df_recommended = pd.DataFrame(recommended, columns=["Anchor Text", "Recommended URL"])
        st.dataframe(df_recommended)

        csv_data = df_recommended.to_csv(index=False)
        st.download_button("Download Suggestions CSV", csv_data, file_name="anchor_suggestions.csv")
else:
    st.info("Upload internal links CSV and provide blog content to see suggestions.")
