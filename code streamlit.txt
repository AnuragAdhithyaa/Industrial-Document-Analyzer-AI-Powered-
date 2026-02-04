import streamlit as st
from transformers import pipeline
from keybert import KeyBERT
import PyPDF2
import os
import time
import matplotlib.pyplot as plt
import io
import base64

# ======================
# Function to convert image to base64 (for background)
# ======================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your front page image (save it in the same folder as app.py)
WATERFALL_IMAGE = "art.webp"  # ← Your uploaded image

# Convert image to base64
waterfall_b64 = get_base64_of_bin_file(WATERFALL_IMAGE)

# ======================
# Custom CSS for full-screen background
# ======================
page_bg_img = f'''
<style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{waterfall_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .header-section {{
        background: rgba(0, 0, 0, 0.7);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px auto;
        max-width: 900px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    }}
    .chart-container {{
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# ======================
# Load models
# ======================
@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model="t5-base")

@st.cache_resource(show_spinner=False)
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource(show_spinner=False)
def load_keyword_extractor():
    return KeyBERT(model="distilbert-base-nli-mean-tokens")

# ======================
# Text extraction
# ======================
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ======================
# Function to generate sentiment pie chart
# ======================
def generate_sentiment_chart(label, score):
    labels = [label, 'Other']
    sizes = [score * 100, 100 - (score * 100)]
    colors = ['#4CAF50' if label == 'POSITIVE' else '#F44336' if label == 'NEGATIVE' else '#2196F3', '#E0E0E0']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
    ax.axis('equal')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

# ======================
# Function to generate keyword bar chart
# ======================
def generate_keyword_chart(keywords):
    words = [kw[0] for kw in keywords]
    scores = [kw[1] for kw in keywords]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(words, scores, color='#2196F3')
    ax.set_xlabel('Relevance Score')
    ax.set_title('Top Keywords')
    ax.invert_yaxis()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

# ======================
# Main App
# ======================
st.markdown("""
    <div class="header-section">
        <h1 style="color:#00D4FF; font-size:50px; margin:0;">Industrial Document Analyzer</h1>
        <p style="color:#A0E7FF; font-size:22px;">AI-Powered Insights for Industrial Reports | Intel FICE Program</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("### Upload your industrial document (PDF/TXT)")

uploaded_file = st.file_uploader("", type=["pdf", "txt"])

if uploaded_file is not None:
    file_path = os.path.join(".", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    st.success("Document uploaded & processed!")

    # Extracted Text
    with st.expander("Extracted Text (Preview)", expanded=False):
        st.write(text[:2000] + ("..." if len(text) > 2000 else ""))

    # Sentiment Analysis with Pie Chart
    st.markdown("### Sentiment Analysis")
    with st.spinner("Analyzing sentiment..."):
        sentiment_analyzer = load_sentiment_analyzer()
        result = sentiment_analyzer(text[:512])[0]
        label = result['label']
        score = result['score']
        remarks = f"The document leans towards a {label.lower()} tone with {score:.2%} confidence. This suggests {'optimistic or informative content' if label == 'POSITIVE' else 'critical or warning-focused content' if label == 'NEGATIVE' else 'balanced and objective content'}."
        
        # Display metric
        st.metric("Overall Sentiment", label.upper(), f"{score:.2%} confidence")
        
        # Display pie chart
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        pie_buf = generate_sentiment_chart(label, score)
        st.image(pie_buf, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display remarks
        st.markdown("**Remarks:** " + remarks)

    # Keywords with Bar Chart
    st.markdown("### Top Keywords")
    with st.spinner("Extracting keywords..."):
        kw_model = load_keyword_extractor()
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=10)
        keyword_text = "\n".join([f"• **{k[0]}** ({k[1]:.3f})" for k in keywords])
        keyword_remarks = f"These keywords highlight the core themes of the document, with the highest score indicating the most prominent topic: '{keywords[0][0]}' ({keywords[0][1]:.3f})."
        
        # Display list
        st.markdown(keyword_text)
        
        # Display bar chart
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        bar_buf = generate_keyword_chart(keywords)
        st.image(bar_buf, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display remarks
        st.markdown("**Remarks:** " + keyword_remarks)

    # Summary with Remarks
    st.markdown("### Document Summary")
    with st.spinner("Generating summary..."):
        summarizer = load_summarizer()
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        progress_bar = st.progress(0)
        summary = ""
        for i, chunk in enumerate(chunks):
            piece = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            summary += "• " + piece + "\n\n"
            progress_bar.progress((i + 1) / len(chunks))
        summary_remarks = f"This summary captures the essence of the {len(chunks)} key sections in the document. Overall, it appears to focus on [key theme from keywords], with potential implications for industrial efficiency."
        progress_bar.empty()
        st.markdown(summary)
        
        # Display remarks
        st.markdown("**Remarks:** " + summary_remarks)

    # Cleanup
    if os.path.exists(file_path):
        os.remove(file_path)