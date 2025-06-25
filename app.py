import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
from anchor_utils import generate_anchor_texts, search_blog_links

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def main():
    st.title("SEO Anchor Text & Internal Link Suggestions")

    st.markdown(
        """
        Upload your CSV with live link data. The CSV should contain columns like:
        - 'Anchor' (existing anchor texts)
        - 'Target' (topic or target keywords)
        - 'Client URL' (URLs of internal pages)
        """
    )
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} rows")

        # Show columns and allow user to pick topic column
        st.write("Columns detected:", df.columns.tolist())

        # Select column containing topic (Target or something)
        topic_col = st.selectbox("Select the column with topic/keywords", options=df.columns.tolist(), index=df.columns.get_loc('Target') if 'Target' in df.columns else 0)

        # Select column for anchors
        anchor_col = st.selectbox("Select the column with existing anchors", options=df.columns.tolist(), index=df.columns.get_loc('Anchor') if 'Anchor' in df.columns else 0)

        # Select column for URLs
        url_col = st.selectbox("Select the column with URLs", options=df.columns.tolist(), index=df.columns.get_loc('Client URL') if 'Client URL' in df.columns else 0)

        # Select or enter topic
        unique_topics = df[topic_col].dropna().unique()
        selected_topic = st.selectbox("Select topic from CSV or enter your own:", options=unique_topics)
        custom_topic = st.text_input("Or enter custom topic:", value="")

        topic = custom_topic.strip() if custom_topic.strip() else selected_topic
        st.write(f"Using topic: **{topic}**")

        if st.button("Generate Anchor Text Suggestions"):
            with st.spinner("Generating anchor texts..."):
                anchors = generate_anchor_texts(client, topic)
            if anchors:
                st.subheader("Anchor Text Suggestions")
                for i, a in enumerate(anchors, 1):
                    st.write(f"{i}. {a}")
            else:
                st.info("No anchor suggestions generated.")

        if st.button("Search Blog/Internal URLs for Topic"):
            results = search_blog_links(df, topic, topic_col, url_col, anchor_col)
            if not results.empty:
                st.subheader("Matching Blog/Internal URLs")
                st.dataframe(results)
            else:
                st.info("No matching internal URLs found for this topic.")

if __name__ == "__main__":
    main()
