import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from langdetect import detect
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your key is set in env variables

def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"[^\w\s]", "", text.lower())

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def generate_anchor(text_snippet, keyword):
    prompt = (
        f"Based on the following context, suggest 3 short, natural, linkable anchor texts (max 4 words) "
        f"that are relevant to the topic but do NOT just repeat the original anchor '{keyword}'. "
        f"These anchors should be suitable for internal linking and include the idea of '{keyword}' if possible, "
        f"but offer useful variations or expansions. Examples: 'play darts online', 'best darts sites', 'darts tips online'.\n"
        f"Context snippet:\n{text_snippet}\n\n"
        f"List anchors separated by commas:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=50,
        )
        anchors_text = response.choices[0].message.content.strip()
        anchors = [a.strip() for a in anchors_text.split(",") if a.strip()]
        anchors = [a for a in anchors if a.lower() != keyword.lower()]
        if not anchors:
            return [keyword]
        return anchors
    except Exception:
        return [keyword]

def match_links_and_generate_anchors(
    opportunities_df,
    internal_links_df,
    anchor_col,
    opp_url_col,
    stake_topic_col,
    stake_url_col,
    stake_lang_col,
):
    # Prepare clean text columns
    opportunities_df['clean_text'] = (
        opportunities_df[opp_url_col].fillna('').astype(str) + " " +
        opportunities_df[anchor_col].fillna('').astype(str)
    ).apply(clean_text)

    internal_links_df[stake_topic_col] = internal_links_df[stake_topic_col].fillna('').astype(str).apply(clean_text)
    internal_links_df[stake_lang_col] = internal_links_df[stake_lang_col].fillna('en')

    vectorizer = TfidfVectorizer().fit(internal_links_df[stake_topic_col])

    results = []
    all_suggested_anchors = []

    for _, row in opportunities_df.iterrows():
        text = row['clean_text']
        original_anchor = str(row[anchor_col])
        lang = detect_language(text)

        filtered_links = internal_links_df[internal_links_df[stake_lang_col].str.startswith(lang[:2])]
        if filtered_links.empty:
            filtered_links = internal_links_df

        filtered_vectors = vectorizer.transform(filtered_links[stake_topic_col])
        text_vec = vectorizer.transform([text])

        similarities = cosine_similarity(text_vec, filtered_vectors).flatten()
        best_idx = similarities.argmax()

        best_url = filtered_links.iloc[best_idx][stake_url_col]

        snippet = text[:400]

        suggested_anchors_list = generate_anchor(snippet, original_anchor)

        # Save results
        results.append({
            "Opportunity URL": row[opp_url_col],
            "Suggested Internal Link": best_url,
            "Original Anchor": original_anchor,
        })

        # Store suggested anchors with relation to opportunity URL
        for suggested in suggested_anchors_list:
            all_suggested_anchors.append({
                "Opportunity URL": row[opp_url_col],
                "Suggested Anchor Text": suggested
            })

    return pd.DataFrame(results), pd.DataFrame(all_suggested_anchors)
