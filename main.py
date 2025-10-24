from fastapi import FastAPI, Request, Form, UploadFile, File, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import Base, engine, get_db
from models import User, File as FileModel, Embedding
from utils.auth_utils import hash_password, verify_password
from utils.pdf_utils import extract_text_from_pdf
from utils.embeddings_utils import create_embeddings, search_in_embeddings
import numpy as np
import os, json

app = FastAPI()

# ---------- Setup ----------
Base.metadata.create_all(bind=engine)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- Home ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------- Admin ----------
@app.get("/admin", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    return templates.TemplateResponse("login_admin.html", {"request": request})

@app.post("/admin/login")
async def admin_login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email, User.role == "admin").first()
    if user and verify_password(password, user.password):
        return templates.TemplateResponse("admin_dashboard.html", {"request": {}, "admin": user.name})
    else:
        return {"error": "Invalid admin credentials"}

@app.get("/admin_dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})

# ---------- Student ----------
@app.get("/student", response_class=HTMLResponse)
async def student_login_page(request: Request):
    return templates.TemplateResponse("login_student.html", {"request": request})

@app.post("/student/login")
async def student_login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email, User.role == "student").first()
    if user and verify_password(password, user.password):
        return templates.TemplateResponse("student_dashboard.html", {"request": {}, "student": user.name})
    else:
        return {"error": "Invalid student credentials"}

@app.get("/student_dashboard", response_class=HTMLResponse)
async def student_dashboard(request: Request):
    return templates.TemplateResponse("student_dashboard.html", {"request": request})

# ---------- Upload PDF ----------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    db_file = FileModel(filename=file.filename, filepath=file_path, uploaded_by="admin")
    db.add(db_file)
    db.commit()

    text = extract_text_from_pdf(file_path)
    chunks, vectors = create_embeddings(text)

    for i, c in enumerate(chunks):
        e = Embedding(file_id=db_file.id, text_chunk=c, vector=json.dumps(vectors[i].tolist()))
        db.add(e)
    db.commit()

    return {"message": f"{file.filename} uploaded and processed successfully."}

# ---------- Student Query ----------
@app.post("/query")
async def query_db(query: str = Form(...), db: Session = Depends(get_db)):
    embeddings = db.query(Embedding).all()
    if not embeddings:
        return {"answer": "No documents found in the database. Please ask admin to upload one."}

    chunks = [e.text_chunk for e in embeddings]
    vectors = np.array([json.loads(e.vector) for e in embeddings])
    answer = search_in_embeddings(query, chunks, vectors)
    return {"answer": answer}
