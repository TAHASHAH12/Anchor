import streamlit as st
import pandas as pd
from anchor_utils import generate_anchor_suggestions, recommend_internal_link, parse_html_from_url

st.title("🔗 Smart Anchor Text Suggestion Tool")

# Upload CSV
uploaded_file = st.file_uploader("Upload stake_links.csv (with 'topic' and 'url' columns)", type=["csv"])
stake_links = None

if uploaded_file:
    stake_links = pd.read_csv(uploaded_file)
    st.success("Stake URLs loaded!")

    # Optional dropdown to let user confirm columns
    col_topic = st.selectbox("Select topic column", stake_links.columns, index=0)
    col_url = st.selectbox("Select URL column", stake_links.columns, index=1)
    stake_links = stake_links.rename(columns={col_topic: "topic", col_url: "url"})

# Input blog
st.subheader("📄 Enter Blog Content or URL")
input_mode = st.radio("Choose input method:", ["Paste blog text", "Fetch from URL"])

raw_text = ""
if input_mode == "Paste blog text":
    raw_text = st.text_area("Paste blog content here")
else:
    input_url = st.text_input("Enter blog URL to scrape HTML")
    if input_url:
        raw_text = parse_html_from_url(input_url)
        st.success("Content extracted from URL!")

# Anchor Suggestion
if raw_text and stake_links is not None:
    with st.spinner("Generating anchor suggestions..."):
        anchor_suggestions = generate_anchor_suggestions(raw_text)

    st.subheader("🔍 Suggested Anchor Texts")
    filtered_anchors = [a for a in anchor_suggestions if a.lower() not in raw_text.lower()]
    st.write(filtered_anchors)

    if not filtered_anchors:
        st.info("No new anchor suggestions found.")
    else:
        st.subheader("🔗 Recommended Internal Links")
        recommended = recommend_internal_link(filtered_anchors, stake_links)
        df_recommended = pd.DataFrame(recommended, columns=["Anchor Text", "Recommended Stake URL"])
        st.dataframe(df_recommended)
