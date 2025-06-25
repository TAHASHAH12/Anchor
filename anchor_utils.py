import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_html_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except:
        return ""

def generate_anchor_suggestions(blog_text):
    prompt = f"""
You are a link building assistant. Your task is to read the blog content and return a list of concise, keyword-rich anchor texts (2 to 5 words). These anchors should sound natural, reflect the page's theme, and resemble examples like:

"cricket betting", "online roulette", "casino game odds", "bet on darts", "NFL betting odds", "slot gratis", "Stake", "blackjack", "casino RNG games"

Avoid:

- Generic terms like "click here"
- Full sentence or title-style phrases
- Repeating same keyword
- Irrelevant or overused terms

Only return a Python list of 10 anchor text strings.

Blog content:
\"\"\"
{blog_text[:4000]}
\"\"\"
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    output = response.choices[0].message.content.strip()
    try:
        suggestions = eval(output)
        return [s.strip() for s in suggestions if isinstance(s, str)]
    except:
        return []

def recommend_internal_link(anchors, internal_links):
    internal_links = internal_links.dropna(subset=["topic", "url"]).copy()
    corpus = internal_links["topic"].astype(str).tolist() + anchors
    if not corpus:
        return []

    vectorizer = TfidfVectorizer().fit(corpus)
    topic_vecs = vectorizer.transform(internal_links["topic"].astype(str))
    anchor_vecs = vectorizer.transform(anchors)

    results = []
    for i, anchor_vec in enumerate(anchor_vecs):
        sim_scores = cosine_similarity(anchor_vec, topic_vecs).flatten()
        best_idx = sim_scores.argmax()
        best_url = internal_links.iloc[best_idx]["url"]
        results.append((anchors[i], best_url))
    return results
