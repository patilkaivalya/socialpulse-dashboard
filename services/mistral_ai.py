import os
import requests
import json
from typing import Dict, Any, Optional

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def _call_mistral(prompt: str, max_tokens: int = 500) -> Optional[str]:
    """Internal helper to call Mistral API with error handling."""
    if not MISTRAL_API_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-small-latest",  # or "mistral-medium-latest"
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

def generate_summary(stats: Dict[str, Any], top_keywords: list) -> str:
    """Generate a conversational summary of social conversation."""
    prompt = f"""
You are a social media analyst. Given the following data about social posts, write a concise paragraph (3-4 sentences) summarizing the overall conversation sentiment and main topics.

Sentiment counts: {stats.get('counts', {})}
Top keywords: {', '.join([w for w, _ in top_keywords[:10]])}

Provide a natural language summary suitable for a brand manager.
"""
    response = _call_mistral(prompt, max_tokens=200)
    if response:
        return response
    # Fallback summary
    total = stats.get('total', 0)
    pos_pct = stats.get('percentages', {}).get('positive', 0)
    neg_pct = stats.get('percentages', {}).get('negative', 0)
    keywords_str = ', '.join([w for w, _ in top_keywords[:5]])
    return (f"Analysis of {total} posts shows {pos_pct:.1f}% positive and {neg_pct:.1f}% negative sentiment. "
            f"Key topics include: {keywords_str}.")

def explain_trend(topic: str, growth: float, context_stats: Dict[str, Any]) -> str:
    """Explain why a topic is trending."""
    prompt = f"""
A social media topic "{topic}" is emerging with a growth score of {growth:.2f}. Based on typical social media patterns, write one sentence explaining why this might be happening (e.g., recent event, viral content, user interest).
"""
    response = _call_mistral(prompt, max_tokens=100)
    if response:
        return response
    return f"The topic '{topic}' is gaining traction, possibly due to recent discussions or events."

def generate_recommendations(stats: Dict[str, Any], trends: list) -> str:
    """Generate 3-5 actionable brand recommendations."""
    prompt = f"""
You are a social media strategist. Based on the following insights, provide 3-5 actionable recommendations for a brand to engage with their audience.

Sentiment: {stats.get('percentages', {})}
Emerging topics: {trends[:5]}

Return the recommendations as a numbered list, each item one sentence.
"""
    response = _call_mistral(prompt, max_tokens=300)
    if response:
        return response
    # Fallback recommendations
    return ("1. Engage with positive sentiment by sharing user-generated content.\n"
            "2. Monitor negative topics and respond with helpful information.\n"
            "3. Create content around emerging keywords to capture interest.")

def generate_executive_summary(all_stats: Dict[str, Any], top_keywords: list, trends: list) -> str:
    """Generate a concise executive summary for presentation."""
    prompt = f"""
You are a data analyst. Write a short executive summary (2-3 paragraphs) covering key findings from social media analysis.

- Total posts: {all_stats.get('total', 0)}
- Sentiment distribution: {all_stats.get('percentages', {})}
- Top keywords: {', '.join([w for w, _ in top_keywords[:5]])}
- Emerging topics: {', '.join([t[0] for t in trends[:3]])}

Focus on actionable insights and overall tone.
"""
    response = _call_mistral(prompt, max_tokens=400)
    if response:
        return response
    return ("Executive Summary:\n"
            f"Out of {all_stats.get('total', 0)} posts, sentiment is predominantly "
            f"{max(all_stats.get('percentages', {}), key=all_stats.get('percentages', {}).get)}.\n"
            f"Key conversations revolve around {', '.join([w for w, _ in top_keywords[:3]])}. "
            f"Emerging topics include {', '.join([t[0] for t in trends[:2]])}. "
            "Brands should consider leveraging positive sentiment and addressing any negative themes.")