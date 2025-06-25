import json
import pandas as pd

def generate_anchor_texts(client, topic: str):
    prompt = f"""
You are an SEO anchor text expert.

Given a main keyword/topic, generate a list of 10 natural, concise anchor texts for internal linking.

Constraints:
- Each anchor text must be 2 to 4 words max.
- Each anchor text must contain the main keyword or its close variation.
- Use varied phrasing that looks natural in an article.
- Avoid article titles, long phrases, or generic terms like "click here".
- Examples for the keyword "darts": "play darts online", "online darts", "bet on darts", "darts on Stake".

Output the results as a JSON array of strings ONLY.

Main keyword/topic:
"{topic}"
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150,
    )
    content = response.choices[0].message.content.strip()
    try:
        anchors = json.loads(content)
        # Filter anchors length 2-4 words
        anchors = [a.strip() for a in anchors if 2 <= len(a.split()) <= 4]
        return anchors
    except Exception as e:
        print(f"Error parsing GPT output: {e}")
        return []

def search_blog_links(df, topic, topic_col, url_col, anchor_col):
    # Simple substring match on topic column or URLs for the topic keyword
    mask = df[topic_col].str.contains(topic, case=False, na=False) | df[url_col].str.contains(topic, case=False, na=False)
    results = df.loc[mask, [url_col, anchor_col]].drop_duplicates()
    results.columns = ["URL", "Anchor"]
    return results
