import os
import streamlit as st
import pandas as pd
from anchor_utils import generate_anchor_suggestions, recommend_internal_link

st.title("Smart Anchor Text Suggestions Tool")

# Upload CSV of internal links
uploaded_file = st.file_uploader("Upload internal links CSV with columns 'topic' and 'url'", type=["csv"])
internal_links = pd.DataFrame()
if uploaded_file:
    try:
        internal_links = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(internal_links)} internal links")
        # Let user pick columns if needed
        cols = internal_links.columns.tolist()
        topic_col = st.selectbox("Select Topic column", cols, index=cols.index("topic") if "topic" in cols else 0)
        url_col = st.selectbox("Select URL column", cols, index=cols.index("url") if "url" in cols else 1)
        # Rename for consistency
        internal_links = internal_links.rename(columns={topic_col: "topic", url_col: "url"})
        internal_links = internal_links[["topic", "url"]]
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# Input blog/article text
raw_text = st.text_area("Paste blog or article text here")

if st.button("Generate Anchor Suggestions and Recommendations"):
    if not raw_text.strip():
        st.warning("Please enter some blog/article text")
    elif internal_links.empty:
        st.warning("Please upload a valid CSV with internal links")
    else:
        with st.spinner("Generating anchor suggestions..."):
            anchor_suggestions = generate_anchor_suggestions(raw_text)
        
        if anchor_suggestions:
            st.subheader("Anchor Text Suggestions")
            st.write(anchor_suggestions)

            # Filter out anchors that already appear as hyperlinks in the text (simple check)
            filtered_anchors = [a for a in anchor_suggestions if a.lower() not in raw_text.lower()]
            if filtered_anchors:
                with st.spinner("Recommending internal links..."):
                    recommended = recommend_internal_link(filtered_anchors, internal_links)
                if recommended:
                    st.subheader("Recommended Internal Links")
                    df_recommended = pd.DataFrame(recommended, columns=["Anchor Text", "Suggested URL"])
                    st.dataframe(df_recommended)
                else:
                    st.info("No suitable internal links found for the suggested anchors.")
            else:
                st.info("No new anchor suggestions found after filtering already linked text.")
        else:
            st.info("No anchor suggestions generated.")
