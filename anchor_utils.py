import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Initialize OpenAI client (make sure OPENAI_API_KEY is set in env variables)
client = OpenAI()

# Generic anchors to exclude
GENERIC_ANCHORS = {"click here", "bonus", "read more", "here", "link"}

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def generate_anchor_suggestions(text, max_suggestions=10):
    """
    Generate anchor text suggestions based on input text using OpenAI.
    """
    # Basic prompt to get anchor suggestions, avoiding essay-like text
    prompt = (
        "Suggest a list of short, natural, varied anchor text phrases "
        "related to the main topic of this text. Avoid generic anchors "
        "like 'click here' or 'bonus'. Each suggestion should be concise, "
        "no longer than 4 words:\n\n"
        f"{text}\n\nSuggestions:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
            n=1,
        )
        suggestions_text = response.choices[0].message.content.strip()
        # Extract suggestions as list - assume one per line
        suggestions = [line.strip("- ").strip() for line in suggestions_text.split("\n") if line.strip()]
        # Filter out generic anchors and duplicates
        filtered = []
        for s in suggestions:
            s_clean = clean_text(s)
            if s_clean and s_clean not in GENERIC_ANCHORS and s_clean not in filtered:
                filtered.append(s)
            if len(filtered) >= max_suggestions:
                break
        return filtered
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return []

def recommend_internal_link(anchors, internal_links):
    """
    Recommend the best internal link for each anchor text based on
    TF-IDF cosine similarity between anchor text and internal link topics.

    internal_links: pd.DataFrame with columns ['topic', 'url']
    anchors: list of strings (anchor texts)
    """
    if not anchors or internal_links.empty:
        return []

    # Clean internal_links
    internal_links = internal_links.dropna(subset=['topic', 'url'])
    internal_links = internal_links[
        (internal_links['topic'].astype(str).str.strip() != '') &
        (internal_links['url'].astype(str).str.strip() != '')
    ]
    if internal_links.empty:
        return []

    # Clean anchors
    anchors = [str(a).strip() for a in anchors if str(a).strip()]
    if not anchors:
        return []

    corpus = internal_links['topic'].astype(str).tolist() + anchors

    vectorizer = TfidfVectorizer().fit(corpus)

    topic_vecs = vectorizer.transform(internal_links['topic'].astype(str).tolist())
    anchor_vecs = vectorizer.transform(anchors)

    results = []
    for i, anchor_vec in enumerate(anchor_vecs):
        similarities = cosine_similarity(anchor_vec, topic_vecs).flatten()
        best_idx = similarities.argmax()
        best_url = internal_links.iloc[best_idx]['url']
        results.append((anchors[i], best_url))

    return results
