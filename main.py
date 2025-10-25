from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import numpy as np
import re
import pickle

# ---------------- CONFIG ----------------
STUDENT_CREDENTIALS = {
    "malaravanee.cs24@bitsathy.ac.in": "123456",
    "rishithav.cs24@bitsathy.ac.in": "123456",
    "amirthae.cs24@bitsathy.ac.in": "123456",
    "kavinkumart.cs24@bitsathy.ac.in": "123456"
}
ADMIN_EMAIL = "nataraj@bitsathy.ac.in"
ADMIN_PASSWORD = "123456"

UPLOAD_DIR = "static/uploads"
VECTOR_DIR = "vector_store"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ---------------- FASTAPI SETUP ----------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------- MODELS SETUP ----------------
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
qa_model = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# ---------------- FAISS SETUP ----------------
def load_faiss_index():
    index_path = os.path.join(VECTOR_DIR, "vector_store.index")
    meta_path = os.path.join(VECTOR_DIR, "metadata.pkl")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        print("[INFO] Existing FAISS index loaded.")
    else:
        dimension = 768
        index = faiss.IndexFlatL2(dimension)
        metadata = []
        print("[INFO] New FAISS index created.")
    return index, metadata

index, metadata = load_faiss_index()

def save_faiss_index():
    faiss.write_index(index, os.path.join(VECTOR_DIR, "vector_store.index"))
    with open(os.path.join(VECTOR_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    print("[INFO] FAISS index and metadata saved successfully.")


# ---------------- PDF FUNCTIONS ----------------
def pdf_to_chunks(pdf_path, chunk_size=400, overlap=100):
    chunks = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                words = text.split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = ' '.join(words[i:i + chunk_size])
                    chunks.append(chunk)
    return chunks

def embed_pdf(pdf_path):
    chunks = pdf_to_chunks(pdf_path)
    if not chunks:
        print(f"[WARN] No text found in {pdf_path}")
        return

    vectors = embedding_model.encode(chunks, convert_to_numpy=True)
    
    # Ensure vectors are 2D (num_chunks x embedding_dim)
    if len(vectors.shape) == 1:
        vectors = np.expand_dims(vectors, axis=0)
    
    index.add(vectors)
    metadata.extend([(pdf_path, c) for c in chunks])
    save_faiss_index()
    print(f"[INFO] Embedded {len(chunks)} chunks from {pdf_path}")

# ---------------- SEARCH + QA ----------------
def semantic_search(query, top_k=5):
    if len(metadata) == 0:
        return "", 9999
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    results = [metadata[idx][1] for idx in I[0] if idx < len(metadata)]
    avg_distance = np.mean(D)
    return " ".join(results), avg_distance

def ask_model(context, question):
    prompt = f"""
Answer this question using the context below. Keep your answer concise.

Context: {context}

Question: {question}

Answer:
"""
    result = qa_model(prompt, max_new_tokens=150, do_sample=False, temperature=0.1)
    answer = result[0]['generated_text']
    return re.sub(r'(?<=[.!?])\s+', '\n', answer.strip())

# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Ask user to choose Admin or Student
    return templates.TemplateResponse("index.html", {"request": request})

# --- Admin Login ---
@app.get("/login_admin", response_class=HTMLResponse)
def admin_login_page(request: Request):
    return templates.TemplateResponse("login_admin.html", {"request": request, "error": ""})

@app.post("/admin_login")
def admin_login(request: Request, email: str = Form(...), password: str = Form(...)):
    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        return RedirectResponse(url="/admin_dashboard", status_code=302)
    return templates.TemplateResponse("login_admin.html", {"request": request, "error": "Invalid credentials"})

@app.get("/admin_dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    files = os.listdir(UPLOAD_DIR)
    return templates.TemplateResponse("admin_dashboard.html", {"request": request, "files": files})

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    embed_pdf(save_path)
    return RedirectResponse(url="/admin_dashboard", status_code=302)

# --- Student Login ---
@app.get("/login_student", response_class=HTMLResponse)
def student_login_page(request: Request):
    return templates.TemplateResponse("login_student.html", {"request": request, "error": ""})

@app.post("/student_login")
def student_login(request: Request, email: str = Form(...), password: str = Form(...)):
    if email in STUDENT_CREDENTIALS and STUDENT_CREDENTIALS[email] == password:
        return RedirectResponse(url="/voicebot", status_code=302)
    return templates.TemplateResponse("login_student.html", {"request": request, "error": "Invalid credentials"})

# --- Student Voicebot ---
@app.get("/voicebot", response_class=HTMLResponse)
def voicebot_page(request: Request):
    return templates.TemplateResponse("voicebot.html", {"request": request})

@app.post("/ask")
def ask_question(query: str = Form(...)):
    context, distance = semantic_search(query)
    threshold = 1.5
    if distance > threshold:
        return JSONResponse({'answer': "The information you requested is not available..Better call +91 9342236331 for further assistance."})
    return JSONResponse({'answer': ask_model(context, query)})

# ---------------- RUN ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)