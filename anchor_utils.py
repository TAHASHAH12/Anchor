import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import re
from langdetect import detect

openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"[^\w\s]", "", text.lower())

def generate_anchor(text_snippet, keyword):
    prompt = (
        f"Suggest a short anchor text (max 4 words) using '{keyword}' as the base term. "
        "It should be natural, sound like a short anchor text, not an article title. "
        "Examples: 'play poker', 'online darts', 'bet on F1', 'roulette online', etc.\n\n"
        f"Context: {text_snippet}\n\nAnchor:"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=12
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return keyword or "Stake"

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def match_links_and_generate_anchors(opportunities_df, stake_df, anchor_col, opp_url_col, stake_topic_col, stake_url_col, stake_lang_col):
    # Clean and prepare text columns
    opportunities_df['Cleaned Text'] = opportunities_df[opp_url_col].fillna('').astype(str) + " " + opportunities_df[anchor_col].fillna('')
    opportunities_df['Cleaned Text'] = opportunities_df['Cleaned Text'].apply(clean_text)

    stake_df[stake_topic_col] = stake_df[stake_topic_col].fillna('').astype(str).apply(clean_text)
    stake_df[stake_lang_col] = stake_df[stake_lang_col].fillna('en')

    # Fit vectorizer on stake topics
    vectorizer = TfidfVectorizer().fit(stake_df[stake_topic_col])
    results = []

    for _, row in opportunities_df.iterrows():
        ext_text = row['Cleaned Text']
        keyword = row[anchor_col]
        lang = detect_language(ext_text)

        ext_vec = vectorizer.transform([ext_text])
        stake_filtered = stake_df[stake_df[stake_lang_col].str.startswith(lang[:2])]

        if stake_filtered.empty:
            stake_filtered = stake_df  # fallback

        stake_vectors = vectorizer.transform(stake_filtered[stake_topic_col])
        similarities = cosine_similarity(ext_vec, stake_vectors).flatten()
        best_idx = similarities.argmax()

        best_url = stake_filtered.iloc[best_idx][stake_url_col]
        snippet = row['Cleaned Text'][:400]
        anchor = generate_anchor(snippet, keyword)

        results.append({
            "Live Link": row[opp_url_col],
            "Matched Stake URL": best_url,
            "Suggested Anchor Text": anchor,
            "Language": lang,
            "Original Anchor": keyword
        })

    return pd.DataFrame(results)
