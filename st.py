# lightweight_app.py - For displaying pre-processed results

import streamlit as st
import pandas as pd
import json
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Comment Analysis Results", 
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #b8daff;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 Comment Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for file upload options
st.sidebar.header("📁 Load Analysis Results")
st.sidebar.markdown("---")

# Option 1: Upload JSON file with results
uploaded_json = st.sidebar.file_uploader(
    "Upload Analysis JSON", 
    type="json",
    help="Upload the JSON file generated from Google Colab processing"
)

# Option 2: Upload CSV file with results
uploaded_csv = st.sidebar.file_uploader(
    "Upload Results CSV", 
    type="csv", 
    help="Upload the CSV file with sentiment analysis results"
)

# Function to load and display results
def display_analysis_results(results_data, df=None):
    """Display comprehensive analysis results"""
    
    # Extract data if from JSON format
    if isinstance(results_data, dict):
        if 'data' in results_data:
            df = pd.DataFrame(results_data['data'])
        metadata = results_data.get('metadata', {})
        sentiment_analysis = results_data.get('sentiment_analysis', {})
        visualizations = results_data.get('visualizations', {})
    else:
        metadata = {}
        sentiment_analysis = {}
        visualizations = {}
    
    # Key Metrics Dashboard
    st.markdown("## 🎯 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_comments = metadata.get('total_comments', len(df) if df is not None else 0)
    avg_confidence = sentiment_analysis.get('average_confidence', 0)
    positive_pct = sentiment_analysis.get('positive_percentage', 0)
    negative_pct = sentiment_analysis.get('negative_percentage', 0)
    
    with col1:
        st.metric("📝 Total Comments", f"{total_comments:,}")
    with col2:
        st.metric("🎯 Avg Confidence", f"{avg_confidence:.3f}")
    with col3:
        st.metric("😊 Positive %", f"{positive_pct:.1f}%")
    with col4:
        st.metric("😞 Negative %", f"{negative_pct:.1f}%")
    
    # Analysis Information
    if metadata:
        st.markdown("## ℹ️ Analysis Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            if 'analysis_date' in metadata:
                analysis_date = datetime.fromisoformat(metadata['analysis_date'].replace('Z', '+00:00'))
                st.info(f"**Analysis Date:** {analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")
            
        with info_col2:
            if 'processing_time_minutes' in metadata:
                processing_time = metadata['processing_time_minutes']
                st.info(f"**Processing Time:** {processing_time:.2f} minutes")
    
    # Data Display
    if df is not None and not df.empty:
        st.markdown("## 📋 Analysis Results")
        
        # Data filtering options
        with st.expander("🔍 Filter Options"):
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                if 'sentiment_label' in df.columns:
                    sentiment_filter = st.multiselect(
                        "Filter by Sentiment", 
                        options=df['sentiment_label'].unique(),
                        default=df['sentiment_label'].unique()
                    )
                else:
                    sentiment_filter = None
            
            with filter_col2:
                if 'sentiment_score' in df.columns:
                    min_confidence = st.slider(
                        "Minimum Confidence Score",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.1
                    )
                else:
                    min_confidence = 0.0
        
        # Apply filters
        filtered_df = df.copy()
        if sentiment_filter and 'sentiment_label' in df.columns:
            filtered_df = filtered_df[filtered_df['sentiment_label'].isin(sentiment_filter)]
        if 'sentiment_score' in df.columns:
            filtered_df = filtered_df[filtered_df['sentiment_score'] >= min_confidence]
        
        st.write(f"Showing {len(filtered_df)} of {len(df)} comments")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Visualizations
        st.markdown("## 📈 Data Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Sentiment Distribution", "📉 Confidence Scores", "☁️ Word Cloud", "🔍 Data Explorer"])
        
        with tab1:
            if 'sentiment_label' in filtered_df.columns:
                sentiment_counts = filtered_df['sentiment_label'].value_counts()
                
                # Interactive bar chart
                fig = px.bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    title="Sentiment Distribution",
                    color=sentiment_counts.values,
                    color_continuous_scale="viridis",
                    labels={'x': 'Sentiment', 'y': 'Count'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Pie chart
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Proportion"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("No sentiment data available for visualization")
        
        with tab2:
            if 'sentiment_score' in filtered_df.columns:
                # Histogram of confidence scores
                fig = px.histogram(
                    filtered_df,
                    x='sentiment_score',
                    nbins=20,
                    title="Distribution of Confidence Scores",
                    labels={'sentiment_score': 'Confidence Score', 'count': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Box plot by sentiment
                if 'sentiment_label' in filtered_df.columns:
                    fig_box = px.box(
                        filtered_df,
                        x='sentiment_label',
                        y='sentiment_score',
                        title="Confidence Scores by Sentiment",
                        labels={'sentiment_label': 'Sentiment', 'sentiment_score': 'Confidence Score'}
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning("No confidence score data available")
        
        with tab3:
            # Display word cloud if available in visualizations
            if visualizations and 'wordcloud' in visualizations:
                try:
                    wordcloud_data = base64.b64decode(visualizations['wordcloud'])
                    st.image(wordcloud_data, caption="Word Cloud of All Comments", use_column_width=True)
                except Exception as e:
                    st.error(f"Could not display word cloud: {str(e)}")
            else:
                # Generate simple word frequency if comment data is available
                if 'comment' in filtered_df.columns:
                    st.markdown("### 🔤 Most Common Words")
                    
                    # Simple word frequency analysis
                    all_text = " ".join(filtered_df['comment'].astype(str))
                    words = all_text.lower().split()
                    
                    # Remove common stop words
                    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'between'}
                    words = [word for word in words if word not in stop_words and len(word) > 2]
                    
                    word_freq = pd.Series(words).value_counts().head(20)
                    
                    if not word_freq.empty:
                        fig = px.bar(
                            x=word_freq.values,
                            y=word_freq.index,
                            orientation='h',
                            title="Top 20 Most Common Words",
                            labels={'x': 'Frequency', 'y': 'Words'}
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No word frequency data available")
                else:
                    st.warning("No comment text available for word analysis")
        
        with tab4:
            st.markdown("### 🔍 Data Explorer")
            
            # Summary statistics
            if 'sentiment_score' in filtered_df.columns:
                st.markdown("#### Confidence Score Statistics")
                score_stats = filtered_df['sentiment_score'].describe()
                
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                with stats_col1:
                    st.metric("Mean", f"{score_stats['mean']:.3f}")
                    st.metric("Min", f"{score_stats['min']:.3f}")
                with stats_col2:
                    st.metric("Median", f"{score_stats['50%']:.3f}")
                    st.metric("Max", f"{score_stats['max']:.3f}")
                with stats_col3:
                    st.metric("Std Dev", f"{score_stats['std']:.3f}")
                    st.metric("Count", f"{int(score_stats['count'])}")
            
            # Sample comments by sentiment
            if 'sentiment_label' in filtered_df.columns and 'comment' in filtered_df.columns:
                st.markdown("#### Sample Comments by Sentiment")
                
                for sentiment in filtered_df['sentiment_label'].unique():
                    with st.expander(f"Sample {sentiment} Comments"):
                        sentiment_comments = filtered_df[filtered_df['sentiment_label'] == sentiment]['comment'].head(5)
                        for i, comment in enumerate(sentiment_comments, 1):
                            st.write(f"**{i}.** {comment}")
        
        # Download Options
        st.markdown("## 💾 Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download filtered results
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Results (CSV)",
                data=csv_data,
                file_name=f"filtered_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Download full results
            full_csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=full_csv_data,
                file_name=f"full_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# Main app logic
def main():
    """Main application logic"""
    
    if uploaded_json is not None:
        try:
            # Load JSON results
            json_data = json.load(uploaded_json)
            st.success("✅ JSON analysis results loaded successfully!")
            display_analysis_results(json_data)
            
        except Exception as e:
            st.error(f"❌ Error loading JSON file: {str(e)}")
    
    elif uploaded_csv is not None:
        try:
            # Load CSV results
            df = pd.read_csv(uploaded_csv)
            st.success(f"✅ CSV results loaded successfully! ({len(df)} rows)")
            
            # Create basic metadata
            metadata = {
                'total_comments': len(df),
                'analysis_date': datetime.now().isoformat()
            }
            
            # Calculate sentiment analysis summary if available
            sentiment_analysis = {}
            if 'sentiment_label' in df.columns:
                sentiment_counts = df['sentiment_label'].value_counts()
                total = len(df)
                sentiment_analysis = {
                    'distribution': sentiment_counts.to_dict(),
                    'average_confidence': df['sentiment_score'].mean() if 'sentiment_score' in df.columns else 0,
                    'positive_percentage': (sentiment_counts.get('POSITIVE', sentiment_counts.get('positive', 0)) / total) * 100,
                    'negative_percentage': (sentiment_counts.get('NEGATIVE', sentiment_counts.get('negative', 0)) / total) * 100
                }
            
            results_data = {
                'metadata': metadata,
                'sentiment_analysis': sentiment_analysis,
                'data': df.to_dict(orient='records')
            }
            
            display_analysis_results(results_data, df)
            
        except Exception as e:
            st.error(f"❌ Error loading CSV file: {str(e)}")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to the Comment Analysis Dashboard!
        
        This dashboard displays results from sentiment analysis processed in Google Colab.
        
        ### 📋 How to Use:
        
        1. **Process your data in Google Colab** using the provided notebook
        2. **Download the results** (JSON or CSV format)  
        3. **Upload the results** using the sidebar file uploader
        4. **Explore your analysis** with interactive visualizations
        
        ### 📁 Supported File Formats:
        
        - **JSON**: Complete analysis report with visualizations (recommended)
        - **CSV**: Processed data with sentiment scores and summaries
        
        ### 🎯 What You'll Get:
        
        - 📊 **Interactive dashboards** with key metrics
        - 📈 **Dynamic visualizations** (charts, graphs, word clouds)
        - 🔍 **Data filtering** and exploration tools  
        - 💾 **Download options** for filtered results
        - 📋 **Detailed statistics** and sample data
        
        ### 🔄 Colab + Streamlit Pipeline Benefits:
        
        ✅ **Fast processing** - Heavy ML work done in Colab with GPU support  
        ✅ **Quick deployment** - Lightweight Streamlit app loads instantly  
        ✅ **No timeouts** - Pre-processed data means no waiting  
        ✅ **Easy sharing** - Share results without re-processing  
        ✅ **Cost effective** - Free Colab processing + free Streamlit hosting  
        
        ---
        
        **Ready to get started?** Upload your analysis results using the sidebar! 👈
        """)
        
        # Sample data format
        st.markdown("### 📄 Expected Data Format")
        
        sample_data = pd.DataFrame({
            'comment': [
                'This product is amazing! Highly recommend it.',
                'Not satisfied with the quality. Could be better.',  
                'Great customer service and fast delivery.'
            ],
            'sentiment_label': ['POSITIVE', 'NEGATIVE', 'POSITIVE'],
            'sentiment_score': [0.995, 0.892, 0.967],
            'summary': [
                'Product is amazing and recommended.',
                'Quality not satisfactory, needs improvement.',
                'Great service and fast delivery.'
            ]
        })
        
        st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main()
