import os
import re
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from dotenv import load_dotenv

load_dotenv()
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def extract_sentences(text):
    return sent_tokenize(text)

def get_existing_anchors(html):
    soup = BeautifulSoup(html, "html.parser")
    return [a.get_text() for a in soup.find_all("a")]

def generate_anchor_suggestions(text, max_suggestions=5):
    prompt = (
        f"Suggest {max_suggestions} natural, semantically rich anchor text phrases "
        "related to the topic of this blog:\n\n"
        f"{text[:1500]}\n\n"
        "Avoid generic words like 'click here' or 'bonus'."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    content = response.choices[0].message.content
    return [line.strip("-• ").strip() for line in content.split("\n") if line.strip()]

def recommend_internal_link(anchors, internal_links):
    # Clean the data
    internal_links = internal_links.dropna(subset=["topic", "url"]).copy()
    internal_links = internal_links[internal_links["topic"].apply(lambda x: isinstance(x, str))]

    results = []
    vectorizer = TfidfVectorizer().fit([*internal_links["topic"].tolist(), *anchors])

    anchor_vectors = vectorizer.transform(anchors)
    topic_vectors = vectorizer.transform(internal_links["topic"])

    for i, anchor_vec in enumerate(anchor_vectors):
        sims = cosine_similarity(anchor_vec, topic_vectors)[0]
        best_match_index = sims.argmax()
        best_url = internal_links.iloc[best_match_index]["url"]
        results.append((anchors[i], best_url))

    return results
