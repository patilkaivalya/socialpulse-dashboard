import pandas as pd
import numpy as np
import re
from typing import List, Optional, Tuple

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove URLs, mentions, special chars."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)  # remove mentions
    text = re.sub(r'#(\w+)', r'\1', text)  # keep hashtag text without #
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def validate_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Ensure required columns exist; add missing ones with default values."""
    required = ['text']
    optional = ['date', 'username', 'likes', 'platform']
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required column(s): {missing_required}")
    
    # Add missing optional columns with defaults
    for col in optional:
        if col not in df.columns:
            if col == 'date':
                df[col] = pd.NaT
            elif col == 'likes':
                df[col] = 0
            else:
                df[col] = 'unknown'
    
    # Convert date column to datetime if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    return df, [col for col in optional if col in df.columns]

def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from raw text."""
    if not isinstance(text, str):
        return []
    return re.findall(r'#(\w+)', text)

def get_top_keywords(texts: pd.Series, n: int = 20) -> List[Tuple[str, int]]:
    """Extract top keywords using simple frequency after cleaning."""
    from collections import Counter
    all_words = []
    for text in texts.dropna():
        cleaned = clean_text(text)
        words = [w for w in cleaned.split() if len(w) > 2]
        all_words.extend(words)
    counter = Counter(all_words)
    return counter.most_common(n)