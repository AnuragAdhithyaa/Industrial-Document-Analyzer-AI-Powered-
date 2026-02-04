import io, re
import pdfplumber
from transformers import pipeline
import yake

def extract_text_from_pdf_bytes(pdf_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_txt_bytes(txt_bytes, encoding='utf-8'):
    return txt_bytes.decode(encoding)

def chunk_text_by_sentences(text, max_chars=1000, overlap_chars=200):
    sentences = re.split(r'(?<=[\.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
            else:
                prev = overlapped[-1]
                overlap = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
                overlapped.append(overlap + " " + ch)
        chunks = overlapped
    return chunks

_summarizer = None
def get_summarizer(model_name="sshleifer/distilbart-cnn-12-6", device=-1):
    global _summarizer
    if _summarizer is None:
        print("Loading summarization model...")
        _summarizer = pipeline("summarization", model=model_name, device=device)
    return _summarizer

def summarize_long_text(text, model_name="sshleifer/distilbart-cnn-12-6"):
    summarizer = get_summarizer(model_name)
    if len(text) < 1200:
        out = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return out[0]['summary_text']
    chunks = chunk_text_by_sentences(text, max_chars=1000, overlap_chars=200)
    partial_summaries = []
    for chunk in chunks:
        out = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        partial_summaries.append(out[0]['summary_text'])
    merged = " ".join(partial_summaries)
    final = summarizer(merged, max_length=180, min_length=50, do_sample=False)
    return final[0]['summary_text']

def extract_keywords(text, max_keywords=10):
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=max_keywords, features=None)
    return kw_extractor.extract_keywords(text)