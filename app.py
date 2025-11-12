# app.py
import os
import re
import io
import sqlite3
import imaplib
import email
from email.header import decode_header
from datetime import datetime
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, send_from_directory, redirect, url_for, request, session, flash,jsonify
from transformers import pipeline
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import docx
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import cv2
from langdetect import detect
from googletrans import Translator
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import spacy

# ---------------- CONFIG ----------------
IMAP_SERVER = "imap.gmail.com"
EMAIL_ACCOUNT = os.getenv("EMAIL_ACCOUNT", "sdgirija2@gmail.com")   # replace or set env
APP_PASSWORD = os.getenv("APP_PASSWORD", "xbdi fphq wowe rydv")       # replace or set env

RAW_DIR = "raw_docs"
DB_FILE = "db.sqlite"

os.makedirs(RAW_DIR, exist_ok=True)

# ----------------- NLP / Tools init -----------------
# Translator (googletrans)
translator = Translator()

# Spacy English model (ensure downloaded)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # If not installed, user must run: python -m spacy download en_core_web_sm
    nlp = None

# Summarization model (smaller for prototype)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # compact

# Zero-shot classification (lightweight)
classifier = pipeline("zero-shot-classification",
                      model="typeform/distilbert-base-uncased-mnli")

# ----------------- Database helpers -----------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        source TEXT,
        subject TEXT,
        raw_path TEXT,
        extracted_text TEXT,
        summary TEXT,
        domain TEXT,
        domain_score REAL,
        urgency TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_document(meta: dict):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    INSERT INTO documents (filename, source, subject, raw_path, extracted_text, summary, domain, domain_score, urgency, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        meta.get("filename"),
        meta.get("source"),
        meta.get("subject"),
        meta.get("raw_path"),
        meta.get("extracted_text"),
        meta.get("summary"),
        meta.get("domain"),
        meta.get("domain_score"),
        meta.get("urgency"),
        meta.get("created_at")
    ))
    conn.commit()
    conn.close()

# ----------------- Text extraction -----------------
def extract_text_from_pdf(path):
    text = ""
    try:
        doc = fitz.open(path)
        for page in doc:
            page_text = page.get_text()
            if page_text and page_text.strip():
                text += page_text + "\n"
            else:
                # Rasterize page -> OCR
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img, lang='eng+mal') + "\n"
        return text
    except Exception as e:
        # fallback: try pdf2image (not shown here to keep compact)
        print("PDF extraction fallback:", e)
        return ""

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_image(path):
    img = Image.open(path)
    # basic preprocessing
    gray = img.convert("L")
    arr = np.array(gray)
    _, thresh = cv2.threshold(arr, 150, 255, cv2.THRESH_BINARY)
    img2 = Image.fromarray(thresh)
    return pytesseract.image_to_string(img2, lang='eng+mal')

def clean_text(text):
    # basic cleaning
    text = re.sub(r'\|+', ' ', text)
    text = re.sub(r'-{2,}', ' ', text)
    text = re.sub(r'(From|To|Subject|Date):.*', '', text)
    text = re.sub(r'(Regards|Sincerely|Best|Thanks).*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z0-9\s.,?!:()-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------- Translation -----------------
def translate_if_malayalam(text):
    try:
        lang = detect(text)
    except:
        lang = "en"
    if lang == "ml" or lang == "malayalam":
        try:
            return translator.translate(text, src='ml', dest='en').text
        except Exception as e:
            print("Translation error:", e)
            return text
    return text

# ----------------- Summarization (hybrid) -----------------
def extractive_summary(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer_ex = TextRankSummarizer()
    summary = summarizer_ex(parser.document, sentence_count)
    return " ".join(str(sent) for sent in summary)

def abstractive_summary(text, max_len=120, min_len=40):
    # summarizer pipeline is defined globally
    # keep input length moderate
    text = text[:2000]
    try:
        out = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return out[0]['summary_text']
    except Exception as e:
        print("Abstractive summarization failed:", e)
        # fallback to extractive short text
        return extractive_summary(text, sentence_count=3)

def hybrid_summarize(text):
    ext = extractive_summary(text, sentence_count=6)
    return abstractive_summary(ext, max_len=120, min_len=40)

# ----------------- Classification (zero-shot + rules) -----------------
DOMAIN_LABELS = [
    "Engineering",
    "Maintenance",
    "Incident Report",
    "Finance / Vendor Invoice",
    "Purchase Order",
    "Regulatory Directive",
    "Environmental Impact",
    "Safety Circular",
    "HR Policy",
    "Legal Opinion",
    "Board Meeting Minutes"
]

URGENCY_LABELS = ["Critical - Requires immediate action", "Normal"]

def refine_urgency(domain, text, ml_urgency):
    t = text.lower()
    if "finance" in domain.lower() or "invoice" in domain.lower():
        if any(k in t for k in ["overdue", "urgent", "immediate", "penalty", "due", "due date"]):
            return "Critical - Requires immediate action"
        return "Normal"
    if "incident" in domain.lower() or "safety" in domain.lower():
        if any(k in t for k in ["crack", "fire", "accident", "injury", "fatal", "collapse", "urgent"]):
            return "Critical - Requires immediate action"
        return ml_urgency
    if "regulatory" in domain.lower():
        if any(k in t for k in ["deadline", "submit", "compliance", "penalty"]):
            return "Critical - Requires immediate action"
        return ml_urgency
    return ml_urgency

def classify_text(text):
    # domain
    d_res = classifier(text, candidate_labels=DOMAIN_LABELS, multi_class=False)
    domain = d_res['labels'][0]
    domain_score = d_res['scores'][0]
    # urgency via ML
    u_res = classifier(text, candidate_labels=URGENCY_LABELS, multi_class=False)
    ml_urgency = u_res['labels'][0]
    # refine
    urgency = refine_urgency(domain, text, ml_urgency)
    return domain, domain_score, urgency

# ----------------- Email fetcher and orchestrator -----------------
def process_file_and_store(path, source, subject):
    # extract text by extension
    ext = path.lower().split('.')[-1]
    if ext == 'pdf':
        raw_text = extract_text_from_pdf(path)
    elif ext in ('docx', 'doc'):
        raw_text = extract_text_from_docx(path)
    elif ext in ('png', 'jpg', 'jpeg', 'tiff', 'bmp'):
        raw_text = extract_text_from_image(path)
    elif ext == 'txt':
        with open(path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    else:
        raw_text = ""

    if not raw_text.strip():
        raw_text = ""  # ensure string

    # translate if Malayalam
    translated = translate_if_malayalam(raw_text)

    # clean
    cleaned = clean_text(translated)

    # optional spacy tokenization (keeps text similar)
    if nlp:
        doc = nlp(cleaned)
        processed = " ".join([t.text for t in doc])
    else:
        processed = cleaned

    # summarize
    summary = hybrid_summarize(processed) if processed.strip() else ""

    # classify
    domain, domain_score, urgency = classify_text(processed if processed else summary)

    meta = {
        "filename": os.path.basename(path),
        "source": source,
        "subject": subject,
        "raw_path": path,
        "extracted_text": processed,
        "summary": summary,
        "domain": domain,
        "domain_score": domain_score,
        "urgency": urgency,
        "created_at": datetime.utcnow().isoformat()
    }
    save_document(meta)
    if urgency == "Critical - Requires immediate action":
        send_alert_email(domain, urgency, subject, summary)

    print(f"[Saved] {path} -> domain:{domain} urgency:{urgency}")
    return meta

def fetch_emails_and_process(limit=10):
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ACCOUNT, APP_PASSWORD)
    mail.select("inbox")
    status, messages = mail.search(None, "UNSEEN")  # only unread
    if status != "OK":
        mail.logout()
        return 0  # no new messages

    email_ids = messages[0].split()
    new_docs_count = 0

    # process last `limit` emails
    for eid in email_ids[-limit:]:
        res, msg_data = mail.fetch(eid, "(RFC822)")
        if res != "OK":
            continue
        for part in msg_data:
            if isinstance(part, tuple):
                msg = email.message_from_bytes(part[1])
                subj_raw = msg.get("Subject", "")
                try:
                    subject, enc = decode_header(subj_raw)[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(enc if enc else "utf-8")
                except:
                    subject = subj_raw

                for mpart in msg.walk():
                    content_disposition = str(mpart.get("Content-Disposition") or "")
                    ctype = mpart.get_content_type()
                    if ctype == "text/plain" and "attachment" not in content_disposition:
                        body = mpart.get_payload(decode=True)
                        if body:
                            text = body.decode(errors="ignore")
                            fname = f"email_body_{eid.decode()}.txt"
                            fpath = os.path.join(RAW_DIR, fname)
                            with open(fpath, "w", encoding="utf-8") as fw:
                                fw.write(text)
                            process_file_and_store(fpath, source="email", subject=subject)
                            new_docs_count += 1
                    elif "attachment" in content_disposition:
                        filename = mpart.get_filename()
                        if filename:
                            fname, enc = decode_header(filename)[0]
                            if isinstance(fname, bytes):
                                fname = fname.decode(enc if enc else "utf-8")
                            safe_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{fname}"
                            fpath = os.path.join(RAW_DIR, safe_name)
                            with open(fpath, "wb") as fw:
                                fw.write(mpart.get_payload(decode=True))
                            process_file_and_store(fpath, source="email", subject=subject)
                            new_docs_count += 1
        # mark as seen
        mail.store(eid, '+FLAGS', '\\Seen')
    mail.logout()
    return new_docs_count

# ----------------- Flask app -----------------
app = Flask(__name__, template_folder="templates")

app = Flask(__name__)
app.secret_key = "supersecretkey"  # required for sessions
DB_USERS = "users.sqlite"

# Initialize user table
def init_users_db():
    conn = sqlite3.connect(DB_USERS)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)
    conn.commit()
    conn.close()

init_users_db()

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        username = request.form["username"]
        password = request.form["password"]
        role = request.form["role"]
        hashed_pw = generate_password_hash(password)

        conn = sqlite3.connect(DB_USERS)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (name, username, password, role) VALUES (?, ?, ?, ?)",
                      (name, username, hashed_pw, role))
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        role = request.form["role"]

        conn = sqlite3.connect(DB_USERS)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND role=?", (username, role))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session["user"] = {"id": user[0], "name": user[1], "role": user[4]}
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials or role.", "danger")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    user_role = session["user"]["role"]  # User's domain

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Fetch docs related to this domain with summary and urgency
    c.execute("""
    SELECT id, filename, subject, domain, urgency, summary, created_at
    FROM documents
    WHERE domain LIKE ?
    ORDER BY created_at DESC
""", (f"%{user_role}%",))


    docs = c.fetchall()
    conn.close()

    return render_template("dashboard.html", docs=docs, role=user_role)


@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, filename, subject, domain, urgency, created_at FROM documents ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return render_template("index.html", rows=rows)

@app.route("/view/<int:doc_id>")
def view(doc_id):
    # 1. Ensure user is logged in
    if "user" not in session:
        return redirect(url_for("login"))

    # 2. Fetch from DB
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT id, filename, subject, raw_path, summary, domain, urgency, created_at
        FROM documents
        WHERE id=?
    """, (doc_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return "Document not found", 404

    # 3. Read raw file content
    raw_content = ""
    if row[3]:  # raw_path exists
        try:
            with open(row[3], "r", encoding="utf-8") as f:
                raw_content = f.read()
        except Exception as e:
            raw_content = f"[Error reading raw file: {e}]"

    # 4. Map row to dictionary
    doc = {
        "id": row[0],
        "filename": row[1],
        "subject": row[2],
        "raw_content": raw_content,  # raw email text
        "final_summary": row[4],     # final summary
        "domain": row[5],
        "urgency": row[6],
        "created_at": row[7]
    }

    return render_template("view.html", doc=doc)



@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(RAW_DIR, filename, as_attachment=True)

@app.route("/fetch_emails", methods=["POST"])
def trigger_fetch():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Not logged in"})

    new_docs = fetch_emails_and_process(limit=20)

    if new_docs > 0:
        return jsonify({"status": "success", "new_docs": new_docs})
    else:
        return jsonify({"status": "empty", "message": "No new unread emails"})

    return redirect(url_for("dashboard"))



@app.route("/manual_upload", methods=["POST"])
def manual_upload():
    if "user" not in session:
        return redirect(url_for("login"))
    
    file = request.files["file"]
    urgency = request.form["urgency"]
    domain = session["user"]["role"]

    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    file.save(file_path)

    # --- Extract text based on file type ---
    ext = filename.lower().split('.')[-1]
    if ext == 'pdf':
        raw_text = extract_text_from_pdf(file_path)
    elif ext in ('docx', 'doc'):
        raw_text = extract_text_from_docx(file_path)
    elif ext in ('png', 'jpg', 'jpeg', 'tiff', 'bmp'):
        raw_text = extract_text_from_image(file_path)
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    else:
        raw_text = ""

    # Clean & translate
    cleaned = clean_text(translate_if_malayalam(raw_text))

    # Summarize
    summarized_text = hybrid_summarize(cleaned) if cleaned else "No summary available"

    # Save to DB
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO documents (filename, subject, domain, urgency, summary, created_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
    """, (filename, filename, domain, urgency, summarized_text))
    conn.commit()
    conn.close()

    flash("File uploaded and summarized!", "success")
    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ----------------- Domain Heads Mapping -----------------
DOMAIN_HEADS = {
    "Finance / Vendor Invoice": "dgirija651@gmail.com",
    # Add others here later, e.g.
    # "Engineering": "eng_head@example.com",
    # "HR Policy": "hr_head@example.com",
}

def send_alert_email(domain, urgency, subject, summary):
    head_email = DOMAIN_HEADS.get(domain)
    if not head_email:
        return  # no mapping found
    
    sender_email = 'sdgirija2@gmail.com'
    password = 'xbdi fphq wowe rydv'  # same app password you used for fetching emails

    # Create email
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = head_email
    msg["Subject"] = f"URGENT ACTION REQUIRED: {domain}"

    body = f"""
    Dear {domain} Head,

    A new document has been categorized as **URGENT**.

    Subject: {subject}
    Urgency: {urgency}
    Domain: {domain}

    Summary:
    {summary}

    Please take immediate action.

    Regards,
    Automated Doc System
    """
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, head_email, msg.as_string())
        server.quit()
        print(f"[ALERT] Email sent to {head_email} for urgent {domain} document.")
    except Exception as e:
        print(f"Failed to send alert email: {e}")

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
