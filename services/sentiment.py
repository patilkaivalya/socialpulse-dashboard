import pandas as pd
import numpy as np
from typing import List, Dict, Any

def analyze_sentiment(texts: pd.Series) -> pd.Series:
    """
    Assign sentiment using TextBlob; fallback to rule-based if TextBlob unavailable.
    Returns Series with values: 'positive', 'negative', 'neutral'.
    """
    try:
        from textblob import TextBlob
        def get_sentiment(text):
            if not isinstance(text, str) or not text.strip():
                return 'neutral'
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.1:
                return 'positive'
            elif polarity < -0.1:
                return 'negative'
            else:
                return 'neutral'
        return texts.apply(get_sentiment)
    except ImportError:
        # Fallback: simple keyword-based sentiment
        positive_words = {'good', 'great', 'awesome', 'excellent', 'love', 'amazing', 'happy', 'best', 'fantastic', 'wonderful'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'poor', 'worst', 'disappointing', 'sad', 'angry', 'horrible'}
        def fallback_sentiment(text):
            if not isinstance(text, str):
                return 'neutral'
            text_lower = text.lower()
            pos_count = sum(1 for w in positive_words if w in text_lower)
            neg_count = sum(1 for w in negative_words if w in text_lower)
            if pos_count > neg_count:
                return 'positive'
            elif neg_count > pos_count:
                return 'negative'
            else:
                return 'neutral'
        return texts.apply(fallback_sentiment)

def get_sentiment_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Return sentiment counts and percentages."""
    if 'sentiment' not in df.columns:
        return {}
    counts = df['sentiment'].value_counts()
    total = len(df)
    return {
        'counts': counts.to_dict(),
        'percentages': (counts / total * 100).round(1).to_dict(),
        'total': total
    }