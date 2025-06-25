import pandas as pd
import re
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure your key is set in environment

def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove punctuation, lower case
    return re.sub(r"[^\w\s]", "", str(text).lower())

def detect_language(text):
    try:
        return detect(str(text))
    except:
        return "en"

def generate_anchor(text_snippet, keyword):
    prompt = (
        f"Suggest a short anchor text (max 4 words) using '{keyword}' as a base. "
        "It should be natural and linkable, not an article title. Examples: "
        "'play darts online', 'online roulette', 'bet on cricket', etc. "
        f"Text context:\n{text_snippet}\n\nAnchor:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=12
        )
        anchor = response.choices[0].message.content.strip()
        # Fallback simple check if empty or too long
        if not anchor or len(anchor.split()) > 4:
            return keyword
        return anchor
    except Exception as e:
        # fallback simple anchor
        return keyword

def match_links_and_generate_anchors(
    opportunities_df,
    internal_links_df,
    anchor_col,
    opp_url_col,
    stake_topic_col,
    stake_url_col,
    stake_lang_col
):
    # Clean and prepare columns
    opportunities_df['clean_text'] = (opportunities_df[opp_url_col].fillna('').astype(str) + " " +
                                     opportunities_df[anchor_col].fillna('').astype(str)).apply(clean_text)

    internal_links_df[stake_topic_col] = internal_links_df[stake_topic_col].fillna('').astype(str).apply(clean_text)
    internal_links_df[stake_lang_col] = internal_links_df[stake_lang_col].fillna('en')

    # Vectorize internal link topics for similarity search
    vectorizer = TfidfVectorizer().fit(internal_links_df[stake_topic_col])
    internal_vectors = vectorizer.transform(internal_links_df[stake_topic_col])

    results = []

    for _, row in opportunities_df.iterrows():
        text = row['clean_text']
        keyword = row[anchor_col]
        lang = detect_language(text)

        # Filter internal links by language prefix
        filtered_links = internal_links_df[internal_links_df[stake_lang_col].str.startswith(lang[:2])]
        if filtered_links.empty:
            filtered_links = internal_links_df  # fallback

        filtered_vectors = vectorizer.transform(filtered_links[stake_topic_col])
        text_vec = vectorizer.transform([text])

        similarities = cosine_similarity(text_vec, filtered_vectors).flatten()
        best_idx = similarities.argmax()

        best_url = filtered_links.iloc[best_idx][stake_url_col]

        snippet = text[:400]

        anchor = generate_anchor(snippet, keyword)

        results.append({
            "Opportunity URL": row[opp_url_col],
            "Suggested Internal Link": best_url,
            "Suggested Anchor Text": anchor,
            "Detected Language": lang,
            "Original Anchor": keyword
        })

    return pd.DataFrame(results)
