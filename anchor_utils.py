import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from langdetect import detect
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure your key is set

def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"[^\w\s]", "", text.lower())

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def is_similar(a, b, threshold=0.8):
    # simple similarity check (case insensitive substring or exact match)
    a = a.lower()
    b = b.lower()
    if a == b:
        return True
    if a in b or b in a:
        return True
    # Could extend to fuzzy matching if needed
    return False

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
        anchor_suggestion = response.choices[0].message.content.strip()

        # If the suggestion is too close or identical to original, do fallback tweaks
        if is_similar(anchor_suggestion, keyword):
            # Simple fallback variations to diversify
            variants = [
                f"play {keyword}",
                f"try {keyword}",
                f"{keyword} online",
                f"online {keyword}",
                f"best {keyword}"
            ]
            for variant in variants:
                if not is_similar(variant, keyword):
                    return variant
            # If all variants similar, return original keyword anyway
            return keyword

        return anchor_suggestion

    except Exception as e:
        # On error, fallback to original anchor keyword
        return keyword

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

        suggested_anchor = generate_anchor(snippet, original_anchor)

        results.append({
            "Opportunity URL": row[opp_url_col],
            "Suggested Internal Link": best_url,
            "Original Anchor": original_anchor,
            "Suggested Anchor Text": suggested_anchor,
            "Detected Language": lang
        })

    return pd.DataFrame(results)
