import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import re
from langdetect import detect

openai.api_key = os.getenv("OPENAI_API_KEY")  # Store securely in environment

def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"[^\w\s]", "", text.lower())

def generate_anchor(text_snippet, keyword):
    prompt = (
        f"Suggest a short anchor text (max 4 words) using '{keyword}' as a base. "
        "It should not sound like an article title. Focus on natural, linkable phrases like "
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
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Stake"

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def match_links_and_generate_anchors(opportunities_df, stake_df):
    opportunities_df['Cleaned Text'] = opportunities_df['Live Link'].fillna('').astype(str) + " " + opportunities_df['Anchor'].fillna('')
    opportunities_df['Cleaned Text'] = opportunities_df['Cleaned Text'].apply(clean_text)

    stake_df['topic'] = stake_df['topic'].fillna('').astype(str).apply(clean_text)
    stake_df['lang'] = stake_df['lang'].fillna('en')

    vectorizer = TfidfVectorizer().fit(stake_df['topic'])
    stake_vectors = vectorizer.transform(stake_df['topic'])

    results = []

    for _, row in opportunities_df.iterrows():
        ext_text = row['Cleaned Text']
        keyword = row['Anchor']
        lang = detect_language(ext_text)

        ext_vec = vectorizer.transform([ext_text])
        stake_df_lang_filtered = stake_df[stake_df['lang'].str.startswith(lang[:2])]

        if stake_df_lang_filtered.empty:
            stake_df_lang_filtered = stake_df  # fallback to all

        stake_vectors_lang = vectorizer.transform(stake_df_lang_filtered['topic'])
        similarities = cosine_similarity(ext_vec, stake_vectors_lang).flatten()
        best_idx = similarities.argmax()

        best_url = stake_df_lang_filtered.iloc[best_idx]['url']
        snippet = row['Cleaned Text'][:400]

        anchor = generate_anchor(snippet, keyword)

        results.append({
            "Live Link": row['Live Link'],
            "Matched Stake URL": best_url,
            "Suggested Anchor Text": anchor,
            "Language": lang,
            "Original Anchor": keyword
        })

    return pd.DataFrame(results)
