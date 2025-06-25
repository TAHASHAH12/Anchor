import os
import re
import openai
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    for script in soup(["script", "style", "head", "meta", "title"]):
        script.extract()
    text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_existing_anchors(html):
    soup = BeautifulSoup(html, "html.parser")
    anchors = [a.get_text().strip() for a in soup.find_all('a') if a.get_text()]
    return anchors

def generate_anchor_suggestions(article_text):
    # Generic example anchors (demonstrate style, not topic-specific)
    example_anchors = [
        "learn more",
        "online games",
        "top strategies",
        "best practices",
        "free bonus",
        "quick guide"
    ]

    few_shot_examples = "\n".join(f"- {a}" for a in example_anchors)

    prompt = f"""
You are an SEO expert who creates natural and varied anchor text suggestions based on an article's main topic.

Here are some examples of natural anchors to emulate:
{few_shot_examples}

Given the following article content:

\"\"\"{article_text[:2000]}\"\"\"

Generate 8-12 short anchor text phrases (2 to 4 words each) that:
- Are relevant to the article topic,
- Avoid full article titles, paragraphs, or generic phrases like "click here" or "bonus",
- Sound natural as anchors that a reader might click on,
- Provide variety in phrasing around the main topic.

Output ONLY a JSON array of anchor phrases, like:
["learn more", "online games", "quick guide"]

Do not add any explanation or extra text.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
            n=1
        )
        raw_text = response.choices[0].message.content.strip()
        anchors = json.loads(raw_text)

        # Clean anchors: unique, length check, exclude generic unwanted
        filtered = []
        seen = set()
        for a in anchors:
            a_clean = a.strip()
            if 2 <= len(a_clean.split()) <= 4 and a_clean.lower() not in ['click here', 'bonus', 'here'] and a_clean not in seen:
                filtered.append(a_clean)
                seen.add(a_clean)
        return filtered

    except Exception as e:
        print(f"OpenAI error: {e}")
        return []

def recommend_internal_link(anchors, internal_links):
    internal_links = internal_links.dropna(subset=['topic', 'url'])
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
