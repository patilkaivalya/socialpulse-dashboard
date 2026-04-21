import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from typing import List, Tuple, Dict, Any
from utils.data_processing import clean_text, extract_hashtags

def extract_top_keywords(texts: pd.Series, n: int = 20) -> List[Tuple[str, int]]:
    """Return top n keywords from cleaned text."""
    from collections import Counter
    all_words = []
    for text in texts.dropna():
        cleaned = clean_text(text)
        words = [w for w in cleaned.split() if len(w) > 2]
        all_words.extend(words)
    return Counter(all_words).most_common(n)

def extract_top_hashtags(df: pd.DataFrame, text_col: str = 'text') -> List[Tuple[str, int]]:
    """Extract and count hashtags from text column."""
    hashtag_counter = Counter()
    for text in df[text_col].dropna():
        hashtags = extract_hashtags(text)
        hashtag_counter.update(hashtags)
    return hashtag_counter.most_common(20)

def cluster_posts(texts: pd.Series, n_clusters: int = 5) -> Tuple[List[int], Dict[int, List[str]]]:
    """
    Perform KMeans clustering on TF-IDF vectors.
    Returns cluster labels and top terms per cluster.
    """
    cleaned = texts.dropna().apply(clean_text)
    if len(cleaned) < n_clusters:
        return [], {}
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(cleaned)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Get top terms per cluster
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    cluster_top_terms = {}
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        cluster_top_terms[i] = top_terms
    return labels.tolist(), cluster_top_terms

def detect_emerging_topics(df: pd.DataFrame, text_col: str = 'text', date_col: str = 'date',
                           recent_days: int = 7) -> List[Tuple[str, float]]:
    """
    Identify emerging topics by comparing recent vs older keyword frequencies.
    Returns list of (keyword, growth_score).
    """
    if date_col not in df.columns or df[date_col].isna().all():
        return []
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    cutoff = df['date'].max() - pd.Timedelta(days=recent_days)
    recent = df[df['date'] >= cutoff][text_col]
    older = df[df['date'] < cutoff][text_col]
    
    if len(recent) == 0 or len(older) == 0:
        return []
    
    recent_words = []
    for text in recent.dropna():
        recent_words.extend(clean_text(text).split())
    older_words = []
    for text in older.dropna():
        older_words.extend(clean_text(text).split())
    
    recent_counter = Counter([w for w in recent_words if len(w) > 2])
    older_counter = Counter([w for w in older_words if len(w) > 2])
    
    growth_scores = {}
    for word, count in recent_counter.items():
        older_count = older_counter.get(word, 0)
        if older_count > 0:
            growth = (count - older_count) / older_count
        else:
            growth = count  # new word
        growth_scores[word] = growth
    
    return sorted(growth_scores.items(), key=lambda x: x[1], reverse=True)[:10]