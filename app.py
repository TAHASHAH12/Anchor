import streamlit as st
import pandas as pd
import openai
import os
from anchor_utils import match_links_and_generate_anchors

# Configure page
st.set_page_config(
    page_title="Smart Anchor Matcher", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

def test_openai_connection():
    """Test OpenAI API connection"""
    if not openai.api_key:
        return False, "OPENAI_API_KEY environment variable not set"
    try:
        client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5,
        )
        return True, "✅ OpenAI API connected successfully"
    except Exception as e:
        return False, f"❌ OpenAI API error: {str(e)[:100]}..."

def display_csv_preview(df, title, max_rows=3):
    """Display a preview of CSV data"""
    st.write(f"**{title}** ({len(df)} rows, {len(df.columns)} columns)")
    st.dataframe(df.head(max_rows), use_container_width=True)
    
    with st.expander(f"View all columns in {title}"):
        cols = st.columns(3)
        for i, col in enumerate(df.columns):
            with cols[i % 3]:
                st.write(f"• **{col}**")
                sample_val = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "No data"
                st.caption(f"Sample: {sample_val[:50]}...")

def show_help_section():
    """Display help information"""
    with st.sidebar:
        st.markdown("### 📖 How to Use")
        
        with st.expander("📁 File Requirements"):
            st.markdown("""
            **Opportunities CSV should contain:**
            - URL column (pages to link from)
            - Anchor text/keyword column
            
            **Internal Links CSV should contain:**
            - URL column (pages to link to)
            - Topic/keyword column (what the page is about)
            - Language column (en, es, fr, etc.)
            """)
        
        with st.expander("🔍 What This Tool Does"):
            st.markdown("""
            1. **Detects language** of your content automatically
            2. **Extracts keywords** from your text using AI
            3. **Matches opportunities** with relevant internal pages
            4. **Suggests natural anchor texts** based on context
            5. **Provides download links** for results
            """)
        
        with st.expander("💡 Tips for Best Results"):
            st.markdown("""
            - Use descriptive column names
            - Ensure language codes are consistent (en, es, fr)
            - Include relevant keywords in your opportunity text
            - Review suggestions before implementing
            """)

# Main app
st.title("🔗 Smart Anchor Text + Internal Link Matcher")
st.markdown("*Enhanced with AI-powered keyword extraction and improved language detection*")

# Show help section
show_help_section()

# Test OpenAI connection
connected, msg = test_openai_connection()
if not connected:
    st.error(msg)
    st.info("💡 Make sure to set your OPENAI_API_KEY environment variable")
    st.stop()
else:
    st.success(msg)

# Step 1: File Upload
st.markdown("## 📁 Step 1: Upload Your CSV Files")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Opportunities CSV")
    st.info("📄 Contains pages you want to add internal links to")
    opp_file = st.file_uploader(
        "Choose opportunities file", 
        type=["csv"],
        help="CSV with URLs and anchor text/keywords"
    )

with col2:
    st.markdown("### Internal Links CSV") 
    st.info("🔗 Contains your internal pages that can be linked to")
    stake_file = st.file_uploader(
        "Choose internal links file",
        type=["csv"], 
        help="CSV with URLs, topics, and language information"
    )

# Load and preview files
if opp_file and stake_file:
    try:
        opp_df = pd.read_csv(opp_file)
        stake_df = pd.read_csv(stake_file)
        
        st.success("✅ Files uploaded successfully!")
        
        # Display previews
        st.markdown("## 👀 File Previews")
        
        col1, col2 = st.columns(2)
        with col1:
            display_csv_preview(opp_df, "Opportunities CSV")
        with col2:
            display_csv_preview(stake_df, "Internal Links CSV")
        
        # Step 2: Column Mapping
        st.markdown("## 🗂️ Step 2: Map Your CSV Columns")
        st.info("👇 Tell us which columns contain which information")
        
        with st.form("column_mapping_form"):
            st.markdown("### Opportunities CSV Columns")
            col1, col2 = st.columns(2)
            
            with col1:
                opp_url_col = st.selectbox(
                    "🌐 URL Column",
                    opp_df.columns,
                    help="Column containing the URLs of pages to link from"
                )
                
            with col2:
                anchor_col = st.selectbox(
                    "⚓ Anchor Text/Keyword Column", 
                    opp_df.columns,
                    help="Column with anchor text or target keywords"
                )
            
            st.markdown("### Internal Links CSV Columns")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                stake_url_col = st.selectbox(
                    "🔗 URL Column",
                    stake_df.columns,
                    help="Column containing internal page URLs"
                )
                
            with col2:
                stake_topic_col = st.selectbox(
                    "📋 Topic/Keyword Column",
                    stake_df.columns, 
                    help="Column describing what each page is about"
                )
                
            with col3:
                stake_lang_col = st.selectbox(
                    "🌍 Language Column",
                    stake_df.columns,
                    help="Column with language codes (en, es, fr, etc.)"
                )
            
            # Preview selected mappings
            st.markdown("### 📋 Selected Column Mapping Preview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Opportunities:**")
                st.write(f"URL: `{opp_url_col}`")
                st.write(f"Anchor: `{anchor_col}`")
                if not opp_df[opp_url_col].dropna().empty:
                    st.caption(f"Sample URL: {opp_df[opp_url_col].dropna().iloc[0]}")
                if not opp_df[anchor_col].dropna().empty:
                    st.caption(f"Sample Anchor: {opp_df[anchor_col].dropna().iloc[0]}")
            
            with col2:
                st.markdown("**Internal Links:**")
                st.write(f"URL: `{stake_url_col}`")
                st.write(f"Topic: `{stake_topic_col}`") 
                st.write(f"Language: `{stake_lang_col}`")
                if not stake_df[stake_url_col].dropna().empty:
                    st.caption(f"Sample URL: {stake_df[stake_url_col].dropna().iloc[0]}")
                if not stake_df[stake_topic_col].dropna().empty:
                    st.caption(f"Sample Topic: {stake_df[stake_topic_col].dropna().iloc[0]}")
            
            st.markdown("---")
            submitted = st.form_submit_button(
                "🚀 Start Processing", 
                use_container_width=True,
                type="primary"
            )

        # Step 3: Processing
        if submitted:
            st.markdown("## ⚙️ Processing Your Data")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Processing {current}/{total} opportunities...")
            
            try:
                with st.spinner("🔍 Analyzing content and generating suggestions..."):
                    links_df, anchors_df = match_links_and_generate_anchors(
                        opp_df,
                        stake_df,
                        anchor_col=anchor_col,
                        opp_url_col=opp_url_col,
                        stake_topic_col=stake_topic_col,
                        stake_url_col=stake_url_col,
                        stake_lang_col=stake_lang_col,
                        progress_callback=update_progress
                    )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                st.success("🎉 Analysis complete! Here are your results:")
                
                # Step 4: Results
                st.markdown("## 📊 Results")
                
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["🔗 Suggested Links", "💬 Anchor Texts", "📈 Summary"])
                
                with tab1:
                    st.markdown("### 🔗 Suggested Internal Links")
                    st.info("These are the recommended internal pages to link to from each opportunity")
                    
                    # Add filters
                    col1, col2 = st.columns(2)
                    with col1:
                        lang_filter = st.multiselect(
                            "Filter by Language:",
                            options=links_df['Detected Language'].unique(),
                            default=links_df['Detected Language'].unique()
                        )
                    with col2:
                        min_similarity = st.slider(
                            "Minimum Similarity Score:",
                            0.0, 1.0, 0.0, 0.1
                        )
                    
                    # Filter data
                    filtered_links = links_df[
                        (links_df['Detected Language'].isin(lang_filter)) &
                        (links_df['Similarity Score'] >= min_similarity)
                    ]
                    
                    st.dataframe(filtered_links, use_container_width=True)
                    
                    # Download button
                    csv_links = filtered_links.to_csv(index=False)
                    st.download_button(
                        "📥 Download Internal Links CSV",
                        csv_links,
                        "suggested_internal_links.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with tab2:
                    st.markdown("### 💬 Suggested Anchor Texts")
                    st.info("AI-generated anchor text suggestions based on content analysis")
                    
                    # Filter options
                    lang_filter_anchors = st.multiselect(
                        "Filter by Language:",
                        options=anchors_df['Detected Language'].unique(),
                        default=anchors_df['Detected Language'].unique(),
                        key="anchor_lang_filter"
                    )
                    
                    filtered_anchors = anchors_df[
                        anchors_df['Detected Language'].isin(lang_filter_anchors)
                    ]
                    
                    st.dataframe(filtered_anchors, use_container_width=True)
                    
                    # Download button
                    csv_anchors = filtered_anchors.to_csv(index=False)
                    st.download_button(
                        "📥 Download Anchor Texts CSV",
                        csv_anchors,
                        "suggested_anchor_texts.csv", 
                        "text/csv",
                        use_container_width=True
                    )
                
                with tab3:
                    st.markdown("### 📈 Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Opportunities", len(links_df))
                    with col2:
                        st.metric("Languages Detected", len(links_df['Detected Language'].unique()))
                    with col3:
                        st.metric("Anchor Suggestions", len(anchors_df))
                    with col4:
                        avg_similarity = links_df['Similarity Score'].mean()
                        st.metric("Avg Similarity", f"{avg_similarity:.2f}")
                    
                    # Language distribution
                    st.markdown("#### Language Distribution")
                    lang_counts = links_df['Detected Language'].value_counts()
                    st.bar_chart(lang_counts)
                    
                    # Top keywords
                    st.markdown("#### Most Common Keywords")
                    all_keywords = []
                    for keywords_str in links_df['Top Keywords'].dropna():
                        all_keywords.extend([k.strip() for k in keywords_str.split(',')])
                    
                    if all_keywords:
                        from collections import Counter
                        keyword_counts = Counter(all_keywords)
                        top_keywords = dict(keyword_counts.most_common(10))
                        st.bar_chart(top_keywords)
                
            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")
                st.info("Please check your file formats and column mappings, then try again.")
    
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        st.info("Please make sure your files are valid CSV format and try again.")

else:
    if not opp_file:
        st.info("📤 Please upload your Opportunities CSV file to get started")
    if not stake_file:
        st.info("📤 Please upload your Internal Links CSV file to get started")

# Footer
st.markdown("---")
st.markdown(
    "🤖 *Developed by Taha Shah*"
)