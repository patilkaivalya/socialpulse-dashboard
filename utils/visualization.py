import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

def plot_sentiment_pie(sentiment_counts: pd.Series) -> go.Figure:
    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                 title="Sentiment Distribution", color_discrete_sequence=px.colors.qualitative.Set2)
    return fig

def plot_sentiment_over_time(df: pd.DataFrame) -> go.Figure:
    if 'date' not in df.columns or df['date'].isna().all():
        return go.Figure().add_annotation(text="No date data available", showarrow=False)
    daily = df.groupby([pd.Grouper(key='date', freq='D'), 'sentiment']).size().reset_index(name='count')
    fig = px.line(daily, x='date', y='count', color='sentiment',
                  title="Sentiment Trend Over Time")
    return fig

def plot_top_keywords(keywords: list) -> go.Figure:
    words, counts = zip(*keywords)
    fig = px.bar(x=words, y=counts, title="Top Keywords",
                 labels={'x': 'Keyword', 'y': 'Frequency'})
    return fig

def generate_wordcloud(text_series: pd.Series) -> str:
    """Return base64 encoded image of wordcloud."""
    text = ' '.join(text_series.dropna().astype(str))
    if not text.strip():
        return ""
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    img = io.BytesIO()
    wc.to_image().save(img, format='PNG')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()