import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import openai
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

client = openai.OpenAI()

def clean_text(text):
    """Clean and normalize text for processing"""
    if pd.isna(text):
        return ""
    return re.sub(r"[^\w\s]", "", text.lower())

def detect_language_enhanced(text, fallback='en'):
    """
    Enhanced language detection with multiple attempts and confidence checking
    """
    if not text or len(text.strip()) < 10:
        return fallback
    
    try:
        # First attempt with original text
        lang = detect(text)
        return lang
    except LangDetectException:
        try:
            # Second attempt with cleaned text
            cleaned = re.sub(r'[^\w\s]', ' ', text)
            if len(cleaned.strip()) > 10:
                lang = detect(cleaned)
                return lang
        except LangDetectException:
            pass
    
    # Language detection based on common words
    language_keywords = {
        'en': ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'],
        'es': ['el', 'la', 'y', 'o', 'pero', 'en', 'con', 'por', 'para', 'de', 'que', 'se'],
        'fr': ['le', 'la', 'et', 'ou', 'mais', 'dans', 'sur', 'avec', 'par', 'pour', 'de', 'que'],
        'de': ['der', 'die', 'das', 'und', 'oder', 'aber', 'in', 'auf', 'mit', 'von', 'zu', 'für'],
        'it': ['il', 'la', 'e', 'o', 'ma', 'in', 'su', 'con', 'per', 'di', 'che', 'si'],
        'pt': ['o', 'a', 'e', 'ou', 'mas', 'em', 'com', 'por', 'para', 'de', 'que', 'se']
    }
    
    words = text.lower().split()
    lang_scores = {}
    
    for lang, keywords in language_keywords.items():
        score = sum(1 for word in words if word in keywords)
        if score > 0:
            lang_scores[lang] = score / len(words)
    
    if lang_scores:
        return max(lang_scores, key=lang_scores.get)
    
    return fallback

def extract_keywords_from_text(text, top_n=10):
    """
    Extract top keywords from text using TF-IDF and frequency analysis
    """
    if not text or len(text.strip()) < 10:
        return []
    
    try:
        # Get stopwords for detected language
        lang = detect_language_enhanced(text)
        try:
            stop_words = set(stopwords.words('english'))  # Default to English
            if lang in ['spanish', 'es']:
                stop_words = set(stopwords.words('spanish'))
            elif lang in ['french', 'fr']:
                stop_words = set(stopwords.words('french'))
            elif lang in ['german', 'de']:
                stop_words = set(stopwords.words('german'))
            elif lang in ['portuguese', 'pt']:
                stop_words = set(stopwords.words('portuguese'))
        except:
            stop_words = set(stopwords.words('english'))
        
        # Tokenize and clean
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in string.punctuation and token not in stop_words]
        tokens = [token for token in tokens if len(token) > 2]
        
        # Get frequency distribution
        freq_dist = Counter(tokens)
        
        # Also use TF-IDF for better keyword extraction
        vectorizer = TfidfVectorizer(max_features=top_n, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            tfidf_keywords = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
            tfidf_keywords.sort(key=lambda x: x[1], reverse=True)
        except:
            tfidf_keywords = []
        
        # Combine frequency and TF-IDF results
        combined_keywords = []
        
        # Add top frequency keywords
        for word, freq in freq_dist.most_common(top_n):
            combined_keywords.append(word)
        
        # Add TF-IDF keywords
        for word, score in tfidf_keywords:
            if word not in combined_keywords:
                combined_keywords.append(word)
        
        return combined_keywords[:top_n]
    
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        return []

def generate_anchor_enhanced(text_snippet, original_keyword, extracted_keywords, lang_code='en'):
    """
    Generate anchor text suggestions using both original keyword and extracted keywords
    """
    # Combine original keyword with top extracted keywords
    all_keywords = [original_keyword] + extracted_keywords[:5]
    keywords_text = ", ".join(all_keywords)
    
    language_names = {
        'en': 'English',
        'es': 'Spanish', 
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese'
    }
    
    lang_name = language_names.get(lang_code, 'English')
    
    prompt = f"""
    Based on this content and keywords, suggest 5 natural anchor texts for internal linking.
    
    Content snippet: {text_snippet[:300]}
    
    Main keyword: {original_keyword}
    Related keywords: {keywords_text}
    Language: {lang_name}
    
    Requirements:
    - Each anchor should be 2-4 words maximum
    - Should sound natural and clickable
    - Avoid repeating the exact main keyword
    - Use related keywords and synonyms
    - Make them contextually relevant
    
    Provide only the anchor texts, separated by commas:
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100,
        )
        
        anchors_text = response.choices[0].message.content.strip()
        anchors = [a.strip().strip('"').strip("'") for a in anchors_text.split(",") if a.strip()]
        
        # Filter out exact matches and very similar anchors
        filtered_anchors = []
        for anchor in anchors:
            if (anchor.lower() != original_keyword.lower() and 
                len(anchor.split()) <= 4 and 
                len(anchor) > 2):
                filtered_anchors.append(anchor)
        
        # If no good anchors generated, create some based on extracted keywords
        if not filtered_anchors and extracted_keywords:
            for keyword in extracted_keywords[:3]:
                if keyword.lower() != original_keyword.lower():
                    filtered_anchors.append(keyword.title())
        
        return filtered_anchors[:5] if filtered_anchors else [original_keyword]
    
    except Exception as e:
        print(f"OpenAI error: {e}")
        # Fallback to extracted keywords
        fallback_anchors = []
        for keyword in extracted_keywords[:3]:
            if keyword.lower() != original_keyword.lower():
                fallback_anchors.append(keyword.title())
        return fallback_anchors if fallback_anchors else [original_keyword]

def match_links_and_generate_anchors(
    opportunities_df,
    internal_links_df,
    anchor_col,
    opp_url_col,
    stake_topic_col,
    stake_url_col,
    stake_lang_col,
    progress_callback=None
):
    """
    Enhanced matching with better language detection and keyword-based anchor generation
    """
    print("Starting enhanced matching process...")
    
    # Prepare data
    opportunities_df['clean_text'] = (
        opportunities_df[opp_url_col].fillna('') + " " +
        opportunities_df[anchor_col].fillna('')
    ).apply(clean_text)

    internal_links_df[stake_topic_col] = internal_links_df[stake_topic_col].fillna('').apply(clean_text)
    internal_links_df[stake_lang_col] = internal_links_df[stake_lang_col].fillna('en')

    # Create vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    try:
        vectorizer.fit(internal_links_df[stake_topic_col])
    except:
        print("Warning: TF-IDF vectorizer fitting failed, using basic matching")
        vectorizer = None

    suggested_links = []
    suggested_anchors = []
    
    total_rows = len(opportunities_df)
    
    for idx, row in opportunities_df.iterrows():
        if progress_callback:
            progress_callback(idx + 1, total_rows)
        
        text = row['clean_text']
        original_text = str(row[anchor_col])
        full_text = str(row[opp_url_col]) + " " + original_text
        
        # Enhanced language detection
        detected_lang = detect_language_enhanced(full_text)
        
        # Extract keywords from the content
        extracted_keywords = extract_keywords_from_text(full_text)
        
        # Filter internal links by language
        lang_prefix = detected_lang[:2] if detected_lang else 'en'
        filtered_links = internal_links_df[
            internal_links_df[stake_lang_col].str.startswith(lang_prefix, na=False)
        ]
        
        if filtered_links.empty:
            filtered_links = internal_links_df
        
        # Find best matching internal link
        if vectorizer and not filtered_links.empty:
            try:
                filtered_vectors = vectorizer.transform(filtered_links[stake_topic_col])
                text_vec = vectorizer.transform([text])
                similarities = cosine_similarity(text_vec, filtered_vectors).flatten()
                best_idx = similarities.argmax()
                similarity_score = similarities[best_idx]
            except:
                best_idx = 0
                similarity_score = 0.0
        else:
            best_idx = 0
            similarity_score = 0.0
        
        best_url = filtered_links.iloc[best_idx][stake_url_col]
        
        # Generate enhanced anchor suggestions
        anchor_variants = generate_anchor_enhanced(
            full_text, original_text, extracted_keywords, detected_lang
        )
        
        # Store results
        suggested_links.append({
            "Opportunity URL": row[opp_url_col],
            "Suggested Internal Link": best_url,
            "Original Anchor": original_text,
            "Detected Language": detected_lang,
            "Similarity Score": round(similarity_score, 3),
            "Top Keywords": ", ".join(extracted_keywords[:5])
        })
        
        for anchor in anchor_variants:
            suggested_anchors.append({
                "Opportunity URL": row[opp_url_col],
                "Original Anchor": original_text,
                "Suggested Anchor Text": anchor,
                "Detected Language": detected_lang,
                "Source": "AI + Keywords"
            })
    
    print("Matching process completed!")
    return pd.DataFrame(suggested_links), pd.DataFrame(suggested_anchors)