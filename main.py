from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sqlite3, os, shutil

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create static/uploads if not exists
os.makedirs("static/uploads", exist_ok=True)

# ---------------- DB SETUP ---------------- #
def init_db():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, password TEXT)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS pdfs (filename TEXT)''')
    # Default student for testing
    cur.execute("INSERT OR IGNORE INTO users VALUES (?, ?)", ("student@gmail.com", "1234"))
    conn.commit()
    conn.close()
init_db()

# ---------------- ROUTES ---------------- #

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/student_login")
async def student_login(request: Request, email: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
    user = cur.fetchone()
    conn.close()
    if user:
        response = RedirectResponse(url="/student_dashboard", status_code=303)
        return response
    else:
        return HTMLResponse("<h3>Invalid credentials! <a href='/login'>Try again</a></h3>")

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("SELECT filename FROM pdfs")
    pdfs = [row[0] for row in cur.fetchall()]
    conn.close()
    return templates.TemplateResponse("admin_dashboard.html", {"request": request, "pdfs": pdfs})

@app.post("/upload_pdf")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    save_path = os.path.join("static/uploads", pdf_file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(pdf_file.file, f)

    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO pdfs VALUES (?)", (pdf_file.filename,))
    conn.commit()
    conn.close()

    return RedirectResponse(url="/admin", status_code=303)

@app.get("/student_dashboard", response_class=HTMLResponse)
async def student_dashboard(request: Request):
    return templates.TemplateResponse("student_dashboard.html", {"request": request})

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    query_text = data.get("query", "").lower()

    # Mock AI responses
    if "timetable" in query_text:
        answer = "Today's timetable is available in the student handbook."
    elif "handbook" in query_text:
        answer = "The student handbook can be found in your dashboard PDFs."
    else:
        answer = "Sorry, I couldn’t find relevant information."

    return JSONResponse({"answer": answer})
