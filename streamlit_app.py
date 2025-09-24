import streamlit as st
import PyPDF2
import re
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Extract text from PDF
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    texts = [p.extract_text() for p in reader.pages if p.extract_text()]
    full_text = "\n".join(texts)
    clean_text = re.sub(r"[^A-Za-z0-9.,;:()\-_\n ]+", "", full_text)
    clean_text = re.sub(r"\n+", "\n", clean_text)
    return clean_text.strip()

# Compute cosine similarity
def cosine_sim(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    if np.linalg.norm(v1)==0 or np.linalg.norm(v2)==0:
        return 0
    return float(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

st.title("AI Resume Analyzer")

resume_file = st.file_uploader("Upload Resume (PDF only)", type="pdf")
job_desc = st.text_area("Paste Job Description", height=150)

if st.button("Analyze"):
    if not resume_file or not job_desc.strip():
        st.error("Please upload a PDF and paste a job description.")
    else:
        resume_text = extract_text(resume_file)
        job_text = re.sub(r"[^A-Za-z0-9.,;:()\-_\n ]+", "", job_desc)
        job_text = re.sub(r"\n+", "\n", job_text.strip())
        
        # Get embeddings
        resume_vec = model.encode(resume_text)
        job_vec = model.encode(job_text)
        
        # Compute similarity
        score = cosine_sim(resume_vec, job_vec) * 100
        
        # Generate simple AI feedback
        feedback = "Excellent match!" if score > 70 else "You can improve your resume further."
        
        st.subheader(f"Similarity Score: {score:.2f}%")
        st.subheader("Feedback:")
        st.write(feedback)