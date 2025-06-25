import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from langdetect import detect

import os
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI key in environment variables

def clean_text(text):
    if pd.isna(text):
        return ""
    # Lowercase and remove punctuation except spaces
    return re.sub(r"[^\w\s]", "", text.lower())

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def generate_anchor(text_snippet, keyword):
    prompt = (
        f"Suggest a short, natural, linkable anchor text (max 4 words) using '{keyword}' as a base keyword. "
        "Avoid article titles. Examples: 'play darts online', 'online roulette', 'bet on cricket'.\n"
        f"Context snippet:\n{text_snippet}\n\nAnchor text suggestion:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=12,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return keyword  # fallback to original keyword if error

def match_links_and_generate_anchors(
    opportunities_df,
    internal_links_df,
    anchor_col,
    opp_url_col,
    stake_topic_col,
    stake_url_col,
    stake_lang_col,
):
    # Prepare and clean text columns
    opportunities_df['clean_text'] = (
        opportunities_df[opp_url_col].fillna('').astype(str) + " " +
        opportunities_df[anchor_col].fillna('').astype(str)
    ).apply(clean_text)

    internal_links_df[stake_topic_col] = internal_links_df[stake_topic_col].fillna('').astype(str).apply(clean_text)
    internal_links_df[stake_lang_col] = internal_links_df[stake_lang_col].fillna('en')

    # Fit TF-IDF on internal topics
    vectorizer = TfidfVectorizer().fit(internal_links_df[stake_topic_col])

    results = []

    for _, row in opportunities_df.iterrows():
        text = row['clean_text']
        original_anchor = row[anchor_col]  # original anchor from CSV
        lang = detect_language(text)

        # Filter internal links by detected language prefix
        filtered_links = internal_links_df[internal_links_df[stake_lang_col].str.startswith(lang[:2])]
        if filtered_links.empty:
            filtered_links = internal_links_df  # fallback to all if none match lang

        filtered_vectors = vectorizer.transform(filtered_links[stake_topic_col])
        text_vec = vectorizer.transform([text])

        similarities = cosine_similarity(text_vec, filtered_vectors).flatten()
        best_idx = similarities.argmax()

        best_url = filtered_links.iloc[best_idx][stake_url_col]

        snippet = text[:400]

        # Generate suggested anchor with OpenAI GPT
        suggested_anchor = generate_anchor(snippet, original_anchor)

        results.append({
            "Opportunity URL": row[opp_url_col],
            "Suggested Internal Link": best_url,
            "Original Anchor": original_anchor,
            "Suggested Anchor Text": suggested_anchor,
            "Detected Language": lang
        })

    return pd.DataFrame(results)
