from fastapi import FastAPI, Request, Form, UploadFile, File, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, re, pickle
import PyPDF2
import numpy as np
import pandas as pd
import faiss
from datetime import datetime, date, timezone
from collections import defaultdict
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from database import get_db
from models import StudentLogin
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------- CONFIG ----------------
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

# ---------------- DEVICE SETUP ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ---------------- MODELS ----------------
# Embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

# LLaMA 3.1 for answer generation
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3-7b-hf",
    use_auth_token=True
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=True
)

qa_model = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    device=0 if device=="cuda" else -1)

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

# ---------------- PDF / EXCEL FUNCTIONS ----------------
def pdf_to_chunks(pdf_path, chunk_size=400, overlap=100):
    chunks = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                words = text.split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = " ".join(words[i:i + chunk_size])
                    chunks.append(chunk)
    return chunks

def embed_chunks(chunks, source_file):
    global index, metadata
    if not chunks:
        print(f"[WARN] No text found in {source_file}")
        return
    vectors = embedding_model.encode(chunks, convert_to_numpy=True)
    if len(vectors.shape) == 1:
        vectors = np.expand_dims(vectors, axis=0)
    index.add(vectors)
    metadata.extend([(source_file, c) for c in chunks])
    save_faiss_index()
    print(f"[INFO] Embedded {len(chunks)} chunks from {source_file}")

# ---------------- DELETE EMBEDDINGS ----------------
def delete_embeddings_for(filename):
    global index, metadata
    remaining = [m for m in metadata if m[0] != filename]
    removed_count = len(metadata) - len(remaining)
    if removed_count == 0:
        print(f"[INFO] No embeddings found for {filename} (nothing to delete).")
        return

    print(f"[INFO] Rebuilding FAISS index after removing {removed_count} chunks for {filename} ...")
    dimension = 768
    new_index = faiss.IndexFlatL2(dimension)
    remaining_chunks = [m[1] for m in remaining]
    if remaining_chunks:
        batch_size = 256
        all_vectors = []
        for i in range(0, len(remaining_chunks), batch_size):
            batch = remaining_chunks[i:i+batch_size]
            vecs = embedding_model.encode(batch, convert_to_numpy=True)
            if len(vecs.shape) == 1:
                vecs = np.expand_dims(vecs, axis=0)
            all_vectors.append(vecs)
        all_vectors = np.vstack(all_vectors)
        new_index.add(all_vectors)
    index = new_index
    metadata = remaining
    save_faiss_index()
    print(f"[INFO] Rebuilt FAISS index. Removed {removed_count} chunks from metadata.")

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
    result = qa_model(prompt, max_new_tokens=250, do_sample=True, temperature=0.7)
    answer = result[0]["generated_text"]
    return answer.strip()

# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------- ADMIN LOGIN ----------------
@app.get("/login_admin", response_class=HTMLResponse)
def admin_login_page(request: Request):
    return templates.TemplateResponse("login_admin.html", {"request": request, "error": ""})

@app.post("/admin_login")
def admin_login(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        return RedirectResponse(url="/admin_dashboard", status_code=302)
    return templates.TemplateResponse("login_admin.html", {"request": request, "error": "Invalid credentials"})

# ---------------- ADMIN DASHBOARD ----------------
@app.get("/admin_dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    files = os.listdir(UPLOAD_DIR)
    logins = db.query(StudentLogin).all()
    today = date.today()
    login_counts = defaultdict(lambda: {"count": 0, "last_login": "N/A"})
    for login in logins:
        ts = login.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.now().astimezone().tzinfo)
        if ts.date() == today:
            login_counts[login.email]["count"] += 1
            login_counts[login.email]["last_login"] = ts.astimezone(timezone.utc).isoformat()
    return templates.TemplateResponse(
        "admin_dashboard.html",
        {"request": request, "files": files, "login_counts": login_counts, "year": datetime.now().year}
    )

# ---------------- FILE UPLOAD ----------------
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    if file.filename.lower().endswith(".pdf"):
        chunks = pdf_to_chunks(save_path)
        embed_chunks(chunks, save_path)
    elif file.filename.lower().endswith(".xlsx"):
        df = pd.read_excel(save_path)
        chunks = [str(cell) for row in df.values for cell in row if str(cell).strip()]
        embed_chunks(chunks, save_path)
    else:
        return JSONResponse({"error": "Unsupported file type. Only PDF or Excel allowed."})

    return RedirectResponse(url="/admin_dashboard", status_code=302)

# ---------------- DELETE FILE ----------------
@app.post("/delete_file")
def delete_file(request: Request, filename: str = Form(...)):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            delete_embeddings_for(file_path)
        except Exception as e:
            return JSONResponse({"error": f"Failed to delete: {str(e)}"}, status_code=500)
    return RedirectResponse(url="/admin_dashboard", status_code=302)

# ---------------- STUDENT LOGIN ----------------
@app.get("/login_student", response_class=HTMLResponse)
def student_login_page(request: Request):
    return templates.TemplateResponse("login_student.html", {"request": request, "error": ""})

@app.post("/student_login")
def student_login(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if email.endswith("@bitsathy.ac.in"):
        login_entry = StudentLogin(email=email)
        db.add(login_entry)
        db.commit()
        return RedirectResponse(url="/voicebot", status_code=302)
    return templates.TemplateResponse("login_student.html", {"request": request, "error": "Only @bitsathy.ac.in emails allowed"})

# ---------------- VOICEBOT ----------------
@app.get("/voicebot", response_class=HTMLResponse)
def voicebot_page(request: Request):
    return templates.TemplateResponse("voicebot.html", {"request": request})

@app.post("/ask")
def ask_question(query: str = Form(...)):
    context, distance = semantic_search(query)
    threshold = 1.5
    if distance > threshold:
        return JSONResponse({"answer": "The information you requested is not available. Please contact +91 9342236331 for assistance."})
    return JSONResponse({"answer": ask_model(context, query)})

# ---------------- RUN ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
