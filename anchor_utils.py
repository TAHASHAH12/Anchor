import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from langdetect import detect
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"[^\w\s]", "", text.lower())

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def generate_anchor(text_snippet, original_anchor):
    prompt = (
        f"You are an SEO expert generating anchor text suggestions for internal linking.\n"
        f"Given this article snippet:\n\"\"\"\n{text_snippet}\n\"\"\"\n"
        f"And the original anchor text: '{original_anchor}'.\n"
        "Suggest a short (max 4 words), natural, and relevant anchor text that could be used instead, "
        "incorporating the core meaning or keyword but making it sound natural and clickable.\n"
        "Examples include phrases like 'play darts online', 'online roulette', 'bet on cricket'.\n"
        "Do NOT simply repeat the original anchor.\n"
        "Suggested anchor text:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=15,
        )
        suggested = response.choices[0].message.content.strip()
        if suggested.lower() == original_anchor.lower():
            return f"Play {original_anchor}"
        return suggested
    except Exception:
        return original_anchor

def match_links_and_generate_anchors(
    opportunities_df,
    internal_links_df,
    anchor_col,
    opp_url_col,
    stake_topic_col,
    stake_url_col,
    stake_lang_col,
):
    opportunities_df['clean_text'] = (
        opportunities_df[opp_url_col].fillna('').astype(str) + " " +
        opportunities_df[anchor_col].fillna('').astype(str)
    ).apply(clean_text)

    internal_links_df[stake_topic_col] = internal_links_df[stake_topic_col].fillna('').astype(str).apply(clean_text)
    internal_links_df[stake_lang_col] = internal_links_df[stake_lang_col].fillna('en')

    vectorizer = TfidfVectorizer().fit(internal_links_df[stake_topic_col])

    results = []

    for _, row in opportunities_df.iterrows():
        text = row['clean_text']
        original_anchor = row[anchor_col]
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

        suggested_anchor = generate_anchor(snippet, original_anchor)

        results.append({
            "Opportunity URL": row[opp_url_col],
            "Suggested Internal Link": best_url,
            "Original Anchor": original_anchor,
            "Suggested Anchor Text": suggested_anchor,
            "Detected Language": lang
        })

    return pd.DataFrame(results)
