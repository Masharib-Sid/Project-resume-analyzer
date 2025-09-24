import streamlit as st
import PyPDF2, re, numpy as np
from sentence_transformers import SentenceTransformer
# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Keywords and suggestions
TECH_KEYS = {
    "python","flask","rest api","rest apis","rest","sql","postgresql","mysql",
    "docker","kubernetes","k8s","aws","azure","gcp","cloud","microservices",
    "scalable web applications","scalable","version control","git",
    "machine learning","tensorflow","pytorch","nlp","computer vision","react",
    "node.js","javascript"
}

SUGGESTIONS = {
    "rest api": "REST APIs (design & consumption)",
    "rest apis": "REST APIs (design & consumption)",
    "rest": "REST APIs (design & consumption)",
    "version control": "Version control (Git) — mention branches/PRs",
    "git": "Version control (Git) — mention branches/PRs",
    "cloud": "Cloud platforms (AWS / GCP / Azure) — mention services used",
    "aws": "AWS (EC2, S3, Lambda, EKS) — mention specific services",
    "kubernetes": "Kubernetes (cluster orchestration) / Docker",
    "docker": "Docker (containerization) — mention images & compose",
    "scalable web applications": "Deploying scalable web apps (Docker, Kubernetes, load balancing)",
    "scalable": "Scalable web apps (microservices, autoscaling examples)",
    "sql": "SQL (Postgres/MySQL) — mention schemas, queries, optimization",
    "machine learning": "Machine Learning (models, evaluation, deployment)",
    "tensorflow": "TensorFlow (model building/serving)",
    "pytorch": "PyTorch (model building/serving)",
    "nlp": "NLP (transformers, tokenization, evaluation)",
    "microservices": "Microservices (API design, service interaction)",
    "react": "React (frontend components, hooks, state management)",
    "node.js": "Node.js (backend services, express)",
    "javascript": "JavaScript (ES6+, async patterns)"
}
# Helper functions
def extract_text_from_pdf(stream):
    try:
        r = PyPDF2.PdfReader(stream)
        pages = [p.extract_text() for p in r.pages if p.extract_text()]
        text = "\n".join(pages)
        return re.sub(r"[^\x00-\x7F]+", " ", text).strip()
    except Exception:
        return ""

def norm(s):
    return re.sub(r'[_\-\.\+]', ' ', (s or "").lower())

def find_keys(text):
    t = norm(text)
    found = set()
    sorted_keys = sorted(TECH_KEYS, key=lambda x: -len(x))
    for key in sorted_keys:
        if key.lower() in t:
            found.add(key.lower())
    return found

def doc_similarity(a, b):
    if not a or not b:
        return 0.0
    va = model.encode(a, convert_to_numpy=True, show_progress_bar=False)
    vb = model.encode(b, convert_to_numpy=True, show_progress_bar=False)
    sim = float(np.dot(va, vb) / (np.linalg.norm(va)*np.linalg.norm(vb)+1e-12))
    return round(sim * 100, 2)

def build_recommendations(missing_keys, top_n=6):
    recs = []
    for k in missing_keys:
        rec = SUGGESTIONS.get(k, k.title() if len(k.split())<=3 else k)
        if rec not in recs:
            recs.append(rec)
        if len(recs) >= top_n:
            break
    return recs

# Streamlit UI
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("AI Resume Analyzer")

resume_file = st.file_uploader("Upload Resume (PDF only)", type="pdf")
job_desc = st.text_area("Paste Job Description", height=150)

if st.button("Analyze"):
    if not resume_file or not job_desc.strip():
        st.error("Please upload a PDF and paste a job description.")
    else:
        resume_file.seek(0)
        resume_text = extract_text_from_pdf(resume_file)
        job_text = re.sub(r"[^\x00-\x7F]+", " ", job_desc).strip()

        job_keys = find_keys(job_text)
        resume_keys = find_keys(resume_text)
        missing = sorted(job_keys - resume_keys)

        # fallback if job_keys empty
        if not job_keys:
            words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-\+]{3,}\b', job_text.lower())
            job_keys = set(words[:20])
            missing = sorted(job_keys - resume_keys)

        # build recommendations
        if not resume_text:
            recs = build_recommendations(list(job_keys)[:8])
        else:
            recs = build_recommendations(missing) if missing else []

        # if nothing missing but low similarity, suggest key job terms
        score = doc_similarity(resume_text, job_text)
        if not recs:
            if score < 70 and job_keys:
                recs = build_recommendations(list(job_keys)[:8])
            else:
                recs = ["No major tech-skill gaps detected."]

        # Display output
        st.subheader(f"Similarity Score: {score:.2f}%")
        st.subheader("Feedback:")
        st.write("Great match!" if score >= 70 else "You can improve your resume further.")
        st.subheader("AI Recommendations:")
        st.write(", ".join(recs))
