import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract main article text; simple version: all <p>
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        return text
    except Exception as e:
        return ""

def generate_anchor_suggestions(text, max_tokens=256):
    if not text or len(text.strip()) < 10:
        return []

    prompt = (
        "Suggest 10 concise, varied, and natural anchor texts for linking within this article text."
        " Avoid generic phrases like 'click here' or 'bonus'."
        " The suggestions should be semantically relevant, not long paragraphs."
        " Provide only the anchor texts separated by commas.\n\n"
        f"Article text:\n{text[:2000]}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            n=1,
            stop=None,
        )
        suggestions_text = response.choices[0].message.content.strip()
        # Split by commas or new lines to get list of suggestions
        suggestions = []
        for part in suggestions_text.replace("\n", ",").split(","):
            candidate = part.strip()
            if candidate and len(candidate) <= 60:  # limit length for anchors
                suggestions.append(candidate)
        # Deduplicate
        suggestions = list(dict.fromkeys(suggestions))
        return suggestions[:10]
    except Exception as e:
        return []

def recommend_internal_link(anchors, internal_links):
    # internal_links: DataFrame with ['topic', 'url'] columns
    if not anchors or internal_links.empty:
        return []

    internal_links = internal_links.dropna(subset=['topic', 'url'])
    if internal_links.empty:
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
