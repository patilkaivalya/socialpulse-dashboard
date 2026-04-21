import streamlit as st
import pandas as pd
import os
from io import BytesIO
from datetime import datetime

from utils.data_processing import validate_columns, get_top_keywords
from utils.visualization import (
    plot_sentiment_pie, plot_sentiment_over_time, plot_top_keywords,
    generate_wordcloud
)
from services.sentiment import analyze_sentiment, get_sentiment_stats
from services.trends import (
    extract_top_keywords, extract_top_hashtags, cluster_posts,
    detect_emerging_topics
)
from services.mistral_ai import (
    generate_summary, explain_trend, generate_recommendations,
    generate_executive_summary
)

st.set_page_config(page_title="SocialPulse", layout="wide")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

def process_data(df):
    """Run sentiment analysis and additional processing."""
    df = df.copy()
    if 'text' in df.columns:
        df['cleaned_text'] = df['text'].astype(str).apply(lambda x: x[:500])  # limit length
        with st.spinner("Analyzing sentiment..."):
            df['sentiment'] = analyze_sentiment(df['text'])
        st.session_state.df = df
        st.session_state.processed = True
        return df
    return df

# Sidebar navigation
st.sidebar.title("SocialPulse 📊")
st.sidebar.markdown("Multi-Agent Social Listening Dashboard")
page = st.sidebar.radio("Navigate", [
    "🏠 Home", "📂 Upload & Preview", "😊 Sentiment Analysis",
    "📈 Trend Detection", "🤖 AI Insights", "📊 Dashboard", "💾 Export"
])

# Home page
if page == "🏠 Home":
    st.title("SocialPulse: Multi-Agent Social Listening & Trend Prediction")
    st.markdown("""
    ### Overview
    SocialPulse ingests social media text data, performs **sentiment analysis**, detects **trending topics**, 
    and uses **Mistral AI** to generate executive summaries and actionable insights.
    
    **Key Features:**
    - CSV upload with automatic column validation
    - Sentiment classification (positive/negative/neutral)
    - Keyword & hashtag extraction
    - Topic clustering and emerging trend detection
    - AI-powered summaries and recommendations
    - Interactive visualisations and export options
    
    **Use Cases:**
    - Brands monitoring campaign sentiment
    - Creators tracking audience conversations
    - Analysts identifying emerging topics
    
    👉 Get started by uploading your CSV in the **Upload & Preview** section.
    """)
    
    # Sample metrics (placeholder)
    col1, col2, col3 = st.columns(3)
    col1.metric("Sample Posts", "1,245", "+12%")
    col2.metric("Avg. Sentiment", "Positive", "👍")
    col3.metric("Top Keyword", "launch", "#trending")

# Upload page
elif page == "📂 Upload & Preview":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Validate columns
            try:
                df, optional_present = validate_columns(df)
                st.info(f"Required columns present. Optional columns found: {', '.join(optional_present)}")
            except ValueError as e:
                st.error(str(e))
                st.stop()
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            if st.button("Process Data for Analysis"):
                with st.spinner("Processing..."):
                    process_data(df)
                st.success("Data processed! Navigate to other sections.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.markdown("""
        **Expected CSV format** (minimum `text` column):
        | text | date | username | likes | platform |
        |------|------|----------|-------|----------|
        | "Loving the new update!" | 2024-01-15 | user123 | 45 | Twitter |
        | "This feature is confusing" | 2024-01-15 | user456 | 12 | Instagram |
        
        *Missing optional columns will be filled with defaults.*
        """)
        
        # Option to load sample data
        if st.button("Load Sample Dataset"):
            sample_data = pd.DataFrame({
                'text': [
                    "Absolutely love the new interface! So intuitive. #design",
                    "Having trouble logging in, keeps saying error.",
                    "The customer support was super helpful today.",
                    "Not sure how I feel about the pricing changes...",
                    "Just saw the announcement! Excited for what's coming. #launch",
                    "Why is this app so slow? Frustrating.",
                    "Great tutorial, learned a lot! Thanks!",
                    "Meh, it's okay I guess.",
                    "Wow, this is a game changer! #innovation",
                    "Can someone explain how to use the new feature?"
                ],
                'date': pd.date_range(end=datetime.today(), periods=10).tolist(),
                'username': [f"user_{i}" for i in range(10)],
                'likes': [34, 5, 22, 8, 67, 3, 45, 2, 89, 11],
                'platform': ['Twitter', 'Instagram', 'Facebook', 'Twitter', 'Instagram',
                             'Facebook', 'Twitter', 'Instagram', 'Twitter', 'Facebook']
            })
            process_data(sample_data)
            st.success("Sample data loaded and processed!")

# Sentiment Analysis page
elif page == "😊 Sentiment Analysis":
    st.header("Sentiment Analysis")
    if st.session_state.df is None or not st.session_state.processed:
        st.warning("Please upload and process data first.")
    else:
        df = st.session_state.df
        stats = get_sentiment_stats(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution")
            if stats:
                fig = plot_sentiment_pie(pd.Series(stats['counts']))
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"Total posts: {stats['total']}")
                st.write("Percentages:", stats['percentages'])
            else:
                st.error("Sentiment data not available.")
        
        with col2:
            st.subheader("Sentiment Over Time")
            if 'date' in df.columns and not df['date'].isna().all():
                fig = plot_sentiment_over_time(df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Date column missing or empty. Cannot plot over time.")

# Trend Detection page
elif page == "📈 Trend Detection":
    st.header("Trend Detection & Topic Clustering")
    if st.session_state.df is None or not st.session_state.processed:
        st.warning("Please upload and process data first.")
    else:
        df = st.session_state.df
        tab1, tab2, tab3 = st.tabs(["Keywords & Hashtags", "Clusters", "Emerging Topics"])
        
        with tab1:
            st.subheader("Top Keywords")
            keywords = extract_top_keywords(df['text'])
            fig_kw = plot_top_keywords(keywords[:15])
            st.plotly_chart(fig_kw, use_container_width=True)
            
            st.subheader("Top Hashtags")
            hashtags = extract_top_hashtags(df)
            if hashtags:
                h_df = pd.DataFrame(hashtags, columns=['Hashtag', 'Count'])
                st.bar_chart(h_df.set_index('Hashtag'))
            else:
                st.info("No hashtags found.")
        
        with tab2:
            st.subheader("Post Clusters (K-Means)")
            n_clusters = st.slider("Number of clusters", 2, 8, 5)
            if st.button("Run Clustering"):
                with st.spinner("Clustering posts..."):
                    labels, top_terms = cluster_posts(df['text'], n_clusters=n_clusters)
                if labels:
                    df['cluster'] = labels[:len(df)]  # align lengths
                    cluster_counts = pd.Series(labels).value_counts().sort_index()
                    st.write("Cluster sizes:", cluster_counts.to_dict())
                    for cid, terms in top_terms.items():
                        st.write(f"**Cluster {cid}:** {', '.join(terms)}")
                else:
                    st.warning("Not enough data for clustering.")
        
        with tab3:
            st.subheader("Emerging Topics (last 7 days vs earlier)")
            emerging = detect_emerging_topics(df)
            if emerging:
                emerging_df = pd.DataFrame(emerging, columns=['Keyword', 'Growth Score'])
                st.dataframe(emerging_df.head(10))
            else:
                st.info("Insufficient date data to detect emerging topics.")

# AI Insights page
elif page == "🤖 AI Insights":
    st.header("AI-Powered Insights (Mistral)")
    if st.session_state.df is None or not st.session_state.processed:
        st.warning("Please upload and process data first.")
    else:
        df = st.session_state.df
        stats = get_sentiment_stats(df)
        keywords = extract_top_keywords(df['text'])
        emerging = detect_emerging_topics(df)
        
        if not os.environ.get("MISTRAL_API_KEY"):
            st.warning("MISTRAL_API_KEY not set. Using fallback local summaries.")
        
        with st.spinner("Generating insights..."):
            summary = generate_summary(stats, keywords)
            st.subheader("Conversation Summary")
            st.write(summary)
            
            st.subheader("Trend Explanation")
            if emerging:
                top_trend = emerging[0]
                explanation = explain_trend(top_trend[0], top_trend[1], stats)
                st.write(f"**{top_trend[0]}** (growth {top_trend[1]:.2f}): {explanation}")
            else:
                st.info("No emerging trends detected.")
            
            st.subheader("Brand Recommendations")
            recs = generate_recommendations(stats, emerging)
            st.markdown(recs)
            
            st.subheader("Executive Summary")
            exec_sum = generate_executive_summary(stats, keywords, emerging)
            st.text_area("Copy for presentation", exec_sum, height=200)

# Dashboard Visualizations page
elif page == "📊 Dashboard":
    st.header("Visual Dashboard")
    if st.session_state.df is None or not st.session_state.processed:
        st.warning("Please upload and process data first.")
    else:
        df = st.session_state.df
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Pie")
            stats = get_sentiment_stats(df)
            if stats:
                fig = plot_sentiment_pie(pd.Series(stats['counts']))
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Top Keywords")
            keywords = extract_top_keywords(df['text'], n=10)
            fig = plot_top_keywords(keywords)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Word Cloud")
        try:
            wordcloud_img = generate_wordcloud(df['text'])
            if wordcloud_img:
                st.image(f"data:image/png;base64,{wordcloud_img}", use_column_width=True)
            else:
                st.info("Not enough text for word cloud.")
        except Exception as e:
            st.error(f"Word cloud generation failed: {e}")
        
        st.subheader("Top Authors by Engagement")
        if 'username' in df.columns and 'likes' in df.columns:
            top_authors = df.groupby('username')['likes'].sum().nlargest(5).reset_index()
            st.bar_chart(top_authors.set_index('username'))
        else:
            st.info("Username or likes column missing.")

# Export page
elif page == "💾 Export":
    st.header("Export Data & Report")
    if st.session_state.df is None or not st.session_state.processed:
        st.warning("No processed data to export.")
    else:
        df = st.session_state.df
        # Cleaned data CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Data (CSV)",
            data=csv,
            file_name="socialpulse_cleaned.csv",
            mime="text/csv"
        )
        
        # Analysis summary TXT
        stats = get_sentiment_stats(df)
        keywords = extract_top_keywords(df['text'])
        emerging = detect_emerging_topics(df)
        summary = generate_summary(stats, keywords)
        exec_sum = generate_executive_summary(stats, keywords, emerging)
        
        report = f"""SocialPulse Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Posts: {stats.get('total', 0)}

Sentiment Distribution:
{stats.get('counts', {})}
Percentages: {stats.get('percentages', {})}

Top Keywords:
{', '.join([f"{w} ({c})" for w, c in keywords[:10]])}

Emerging Topics:
{', '.join([f"{t[0]} (growth {t[1]:.2f})" for t in emerging[:5]])}

AI Summary:
{summary}

Executive Summary:
{exec_sum}
"""
        st.download_button(
            label="Download Analysis Report (TXT)",
            data=report,
            file_name="socialpulse_report.txt",
            mime="text/plain"
        )
        st.text_area("Report Preview", report, height=300)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit & Mistral AI")