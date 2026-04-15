###### Packages Used ######
import streamlit as st # core package used in this project
import pandas as pd
import base64, random
import time,datetime
#import pymysql
import os
import socket
import platform
import geocoder
import secrets
import io,random
import plotly.express as px # to create visualisations at the admin session
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
# libraries used to parse the pdf files
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
from streamlit_tags import st_tags
from PIL import Image
# pre stored data for prediction purposes
#from Courses import ds_course,web_course,android_course,ios_course,uiux_course,resume_videos,interview_videos
import nltk
from typer import style
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from collections import Counter
import sqlite3
import hashlib
from fpdf import FPDF


###### Preprocessing functions ######


# Generates a link allowing the data in a given panda dataframe to be downloaded in csv format 
def get_csv_download_link(df,filename,text):
    csv = df.to_csv(index=False)
    ## bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()      
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Reads Pdf file and check_extractable
def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    ## close open handles
    converter.close()
    fake_file_handle.close()
    return text


# show uploaded file path to view pdf_display
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


###### Live Market Scraper ######

def scrape_market_skills(job_role):
    """
    Scrapes live Indian job portals for in-demand skills for the given role.
    Tries Naukri → TimesJobs → Shine in order.
    Uses keyword frequency counting on raw HTML — reliable across site updates.
    Returns: (top_skills list, skill_counts dict, source_label string)
    """
    known_skills = [
        "python","java","javascript","typescript","sql","r","scala","kotlin","swift",
        "c++","c#","go","rust","php","ruby","react","angular","vue","node","express",
        "django","flask","spring","fastapi","nextjs","redux","mongodb","postgresql",
        "mysql","sqlite","redis","elasticsearch","cassandra","dynamodb","docker",
        "kubernetes","aws","azure","gcp","git","jenkins","ci/cd","terraform","ansible",
        "linux","tensorflow","pytorch","scikit-learn","pandas","numpy","keras","opencv",
        "machine learning","deep learning","nlp","computer vision","data science",
        "html","css","rest api","graphql","microservices","system design","agile","scrum",
        "tableau","power bi","excel","hadoop","spark","kafka","airflow","figma",
        "adobe xd","android","ios","flutter","react native","data visualization",
        "statistics","data analysis","data engineering","cloud computing","devops",
        "blockchain","cybersecurity","networking","fastapi","selenium","junit",
        "postman","jira","bitbucket","github","azure devops"
    ]

    role_slug_map = {
        "Data Analyst":              "data-analyst",
        "Backend Developer":         "backend-developer",
        "Frontend Developer":        "frontend-developer",
        "Machine Learning Engineer": "machine-learning-engineer",
        "Software Developer":        "software-developer",
    }
    role_plus_map = {
        "Data Analyst":              "data+analyst",
        "Backend Developer":         "backend+developer",
        "Frontend Developer":        "frontend+developer",
        "Machine Learning Engineer": "machine+learning+engineer",
        "Software Developer":        "software+developer",
    }

    slug  = role_slug_map.get(job_role, job_role.lower().replace(" ", "-"))
    plus  = role_plus_map.get(job_role, job_role.lower().replace(" ", "+"))

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    raw_text     = ""
    source_label = ""

    sources = [
        (f"https://www.naukri.com/{slug}-jobs",                              "Naukri.com"),
        (f"https://www.timesjobs.com/candidate/job-search.html"
         f"?searchType=personalizedSearch&from=submit"
         f"&txtKeywords={plus}&txtLocation=",                                 "TimesJobs.com"),
        (f"https://www.shine.com/job-search/{slug}-jobs",                    "Shine.com"),
    ]

    for url, label in sources:
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code == 200 and len(resp.text) > 3000:
                raw_text     = resp.text.lower()
                source_label = label
                break
        except Exception:
            continue

    if not raw_text:
        return None, None, "Could not reach any job portal. Check your internet connection."

    skill_counts = {}
    for skill in known_skills:
        count = raw_text.count(skill.lower())
        if count > 0:
            skill_counts[skill] = count

    if not skill_counts:
        return None, None, f"Connected to {source_label} but no recognisable skills found."

    sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
    top_skills    = [s for s, _ in sorted_skills[:12]]
    top_counts    = dict(sorted_skills[:12])

    return top_skills, top_counts, source_label


###### SQLite User Database ######

DB_PATH = "./resumeiq_users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            email       TEXT    UNIQUE NOT NULL,
            mobile      TEXT    NOT NULL,
            password    TEXT    NOT NULL,
            created_at  TEXT    NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS resume_scans (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email          TEXT    NOT NULL,
            pdf_name            TEXT    NOT NULL,
            resume_score        INTEGER NOT NULL,
            reco_field          TEXT,
            cand_level          TEXT,
            skills              TEXT,
            recommended_skills  TEXT,
            timestamp           TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(name, email, mobile, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        ts   = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute(
            "INSERT INTO users (name, email, mobile, password, created_at) VALUES (?,?,?,?,?)",
            (name, email, mobile, hash_password(password), ts)
        )
        conn.commit()
        conn.close()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    except Exception as e:
        return False, str(e)

def login_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        "SELECT id, name, email, mobile, created_at FROM users WHERE email=? AND password=?",
        (email, hash_password(password))
    )
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "name": row[1], "email": row[2],
                "mobile": row[3], "created_at": row[4]}
    return None

def save_scan(user_email, pdf_name, resume_score, reco_field,
              cand_level, skills, recommended_skills):
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        ts   = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute(
            """INSERT INTO resume_scans
               (user_email, pdf_name, resume_score, reco_field,
                cand_level, skills, recommended_skills, timestamp)
               VALUES (?,?,?,?,?,?,?,?)""",
            (user_email, pdf_name, resume_score, reco_field,
             cand_level, str(skills), str(recommended_skills), ts)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

def get_user_scans(user_email):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        """SELECT pdf_name, resume_score, reco_field, cand_level,
                  skills, recommended_skills, timestamp
           FROM resume_scans WHERE user_email=?
           ORDER BY timestamp DESC""",
        (user_email,)
    )
    rows = c.fetchall()
    conn.close()
    return rows

def generate_analysis_report(user, scan):
    """
    Generates a clean receipt-style PDF for a resume scan.
    Returns the PDF as bytes for download.
    """
    pdf_name, res_score, reco_field, cand_level, skills, rec_skills, ts = scan

    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    # ── Header ──────────────────────────────────────────────────────────────
    pdf.set_font("Courier", "B", 16)
    pdf.cell(0, 8, "RESUMEIQ", ln=True, align="C")
    pdf.set_font("Courier", "", 9)
    pdf.cell(0, 5, "Smart Resume Analysis Report", ln=True, align="C")
    pdf.ln(3)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.5)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(5)

    # ── Report Meta ──────────────────────────────────────────────────────────
    pdf.set_font("Courier", "", 8)
    pdf.cell(0, 5, f"Report Generated  :  {ts}", ln=True)
    pdf.cell(0, 5, f"Report For        :  {user['name']}", ln=True)
    pdf.cell(0, 5, f"Email             :  {user['email']}", ln=True)
    pdf.cell(0, 5, f"Mobile            :  {user['mobile']}", ln=True)
    pdf.ln(3)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(5)

    # ── Resume Info ──────────────────────────────────────────────────────────
    pdf.set_font("Courier", "B", 9)
    pdf.cell(0, 5, "RESUME DETAILS", ln=True)
    pdf.ln(2)
    pdf.set_font("Courier", "", 8)
    pdf.cell(0, 5, f"File Name         :  {pdf_name}", ln=True)
    pdf.cell(0, 5, f"Scan Date         :  {ts[:10]}", ln=True)
    pdf.ln(3)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(5)

    # ── Analysis Results ─────────────────────────────────────────────────────
    pdf.set_font("Courier", "B", 9)
    pdf.cell(0, 5, "ANALYSIS RESULTS", ln=True)
    pdf.ln(2)
    pdf.set_font("Courier", "", 8)
    pdf.cell(95, 6, f"Resume Score      :  {res_score} / 100")
    pdf.cell(0,  6, f"Experience Level  :  {cand_level or 'N/A'}", ln=True)
    pdf.cell(0,  6, f"Detected Field    :  {reco_field or 'N/A'}", ln=True)
    pdf.ln(3)

    # Score bar (ASCII style)
    bar_filled = int((res_score / 100) * 30)
    bar_empty  = 30 - bar_filled
    bar        = "[" + "#" * bar_filled + "-" * bar_empty + "]"
    pdf.set_font("Courier", "B", 8)
    pdf.cell(0, 5, f"Score : {res_score}%", ln=True)
    pdf.ln(3)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(5)

    # ── Skills Detected ──────────────────────────────────────────────────────
    pdf.set_font("Courier", "B", 9)
    pdf.cell(0, 5, "SKILLS DETECTED IN RESUME", ln=True)
    pdf.ln(2)
    pdf.set_font("Courier", "", 8)
    skills_list = []
    if isinstance(skills, str):
        skills_list = [s.strip().strip("'[]\"") for s in skills.split(",") if s.strip()]
    elif isinstance(skills, list):
        skills_list = skills

    if skills_list:
        # Print 3 per row
        row_items = []
        for i, skill in enumerate(skills_list):
            row_items.append(skill)
            if len(row_items) == 3 or i == len(skills_list) - 1:
                pdf.cell(0, 5, "  " + "   |   ".join(row_items), ln=True)
                row_items = []
    else:
        pdf.cell(0, 5, "  No skills detected.", ln=True)

    pdf.ln(3)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(5)

    # ── Recommended Skills ───────────────────────────────────────────────────
    pdf.set_font("Courier", "B", 9)
    pdf.cell(0, 5, "RECOMMENDED SKILLS TO ADD", ln=True)
    pdf.ln(2)
    pdf.set_font("Courier", "", 8)
    rec_list = []
    if isinstance(rec_skills, str):
        rec_list = [s.strip().strip("'[]\"") for s in rec_skills.split(",") if s.strip()]
    elif isinstance(rec_skills, list):
        rec_list = rec_skills

    if rec_list:
        row_items = []
        for i, skill in enumerate(rec_list):
            row_items.append(skill)
            if len(row_items) == 3 or i == len(rec_list) - 1:
                pdf.cell(0, 5, "  + " + "   + ".join(row_items), ln=True)
                row_items = []
    else:
        pdf.cell(0, 5, "  No recommendations available.", ln=True)

    pdf.ln(3)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(5)

    # ── Footer ───────────────────────────────────────────────────────────────
    pdf.set_font("Courier", "I", 7)
    pdf.cell(0, 5, "This report was auto-generated by ResumeIQ.", ln=True, align="C")
    pdf.cell(0, 5, "Scores are based on resume content analysis and live market data.", ln=True, align="C")

    return bytes(pdf.output())


# Initialise DB on every app start
init_db()


###### Database Stuffs ######


# sql connector
#connection = pymysql.connect(host='localhost',user='root',password='root@MySQL4admin',db='cv')
#cursor = connection.cursor()


# inserting miscellaneous data, fetched results, prediction and recommendation into user_data table
def insert_data(sec_token,host_name,dev_user,os_name_ver,latlong,city,state,country,act_name,act_mail,act_mob,name,email,res_score,timestamp,no_of_pages,reco_field,cand_level,skills,recommended_skills,courses,pdf_name):
    DB_table_name = 'user_data'
    insert_sql = "insert into " + DB_table_name + """
    values (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    rec_values = (str(sec_token),host_name,dev_user,os_name_ver,str(latlong),city,state,country,act_name,act_mail,act_mob,name,email,str(res_score),timestamp,str(no_of_pages),reco_field,cand_level,skills,recommended_skills,courses,pdf_name)
    #cursor.execute(insert_sql, rec_values)
    #connection.commit()


# inserting feedback data into user_feedback table
def insertf_data(feed_name,feed_email,feed_score,comments,Timestamp):
    DBf_table_name = 'user_feedback'
    insertfeed_sql = "insert into " + DBf_table_name + """
    values (0,%s,%s,%s,%s,%s)"""
    rec_values = (feed_name, feed_email, feed_score, comments, Timestamp)
    cursor.execute(insertfeed_sql, rec_values)
    connection.commit()


###### Setting Page Configuration (favicon, Logo, Title) ######


st.set_page_config(
   page_title="ResumeIQ - Smart Resume Analyzer",
   page_icon='./Logo/recommend.png',
)


###### Main function run() ######


def run():
    
    # (Logo, Heading, Sidebar etc)
    from PIL import Image
    import base64
    from io import BytesIO

    img = Image.open('./Logo/resumeIQ_logo.jpeg')

    # Convert image to base64 so HTML can render it reliably
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
        f"""
        <style>
        .logo-container {{
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }}

        .logo-container img {{
            border-radius: 16px;
            box-shadow: 0 0 10px rgba(74, 144, 226, 0.4);
            transition: transform 0.35s ease, box-shadow 0.35s ease;
            max-width: 400px;
            height: auto;
        }}

        .logo-container img:hover {{
            transform: scale(1.05);
            box-shadow: 0 0 22px rgba(74, 144, 226, 0.7);
        }}
        </style>

        <div class="logo-container">
            <img src="data:image/jpeg;base64,{img_base64}" />
        </div>
        """,
        unsafe_allow_html=True
    ) 
   # ── Session state bootstrap ──────────────────────────────────────────────────
    if "page" not in st.session_state:
        st.session_state.page = "landing"
    if "show_job_section" not in st.session_state:
        st.session_state.show_job_section = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "show_profile" not in st.session_state:
        st.session_state.show_profile = False
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    # ── Shared CSS (landing cards + recruiter login) ─────────────────────────────
    
    st.markdown("""
    <style>
    /* ── Global font ── */
    html, body, [class*="css"], .stMarkdown, .stText,
    .stTextInput, .stSelectbox, .stFileUploader,
    button, input, textarea, select {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                     "SF Pro Text", "Segoe UI", Roboto, Oxygen, Ubuntu,
                     Cantarell, "Helvetica Neue", sans-serif !important;
        -webkit-font-smoothing: antialiased;
        letter-spacing: -0.01em;
    }

    /* ── Landing screen ── */
    
    .landing-wrapper {
    
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem 1rem 2rem;
        gap: 0.5rem;
    }
    .landing-heading {
        font-size: 1.75rem;
        font-weight: 700;
        color: #e2e6f0;
        text-align: center;
        margin-bottom: 0.25rem;
        letter-spacing: -0.01em;
    }
    .landing-sub {
        font-size: 1rem;
        color: #8b92a5;
        text-align: center;
        margin-bottom: 2.25rem;
    }

    /* ── Recruiter login card ── */
    .rec-card {
        background: #1c1f26;
        border: 1px solid #2c3240;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        max-width: 420px;
        margin: 2rem auto 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    }
    .rec-card-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #e2e6f0;
        text-align: center;
        margin-bottom: 0.35rem;
    }
    .rec-card-sub {
        font-size: 0.9rem;
        color: #8b92a5;
        text-align: center;
        margin-bottom: 1.75rem;
    }

    /* ── Shared button overrides ── */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 1.6rem;
        transition: background-color 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(74, 144, 226, 0.35);
    }
    }

    /* ── Back button (subtle) ── */
    .back-btn > button {
        background: transparent !important;
        color: #8b92a5 !important;
        border: 1px solid #2c3240 !important;
        font-size: 0.85rem !important;
        padding: 0.35rem 1rem !important;
    }
    .back-btn > button:hover {
        color: #e2e6f0 !important;
        border-color: #4A90E2 !important;
        box-shadow: none !important;
        transform: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    

    # ── LANDING PAGE ─────────────────────────────────────────────────────────────
    if st.session_state.page == "landing":
        st.markdown("""
    <style>

    /* 🌊 MAIN BACKGROUND CONTAINER */
    [data-testid="stAppViewContainer"] {
        background: #0f1117;
        position: relative;
        overflow: hidden;
    }

    /* 🌊 WAVE LAYER 1 */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: -20%;
        left: -20%;
        width: 140%;
        height: 140%;
        background: radial-gradient(circle at 20% 30%, rgba(74,144,226,0.25), transparent 50%),
                    radial-gradient(circle at 80% 70%, rgba(0,119,255,0.18), transparent 55%);
        filter: blur(120px);
        animation: waveMove1 10s ease-in-out infinite alternate;
        z-index: 0;
    }

    /* 🌊 WAVE LAYER 2 */
    [data-testid="stAppViewContainer"]::after {
        content: "";
        position: fixed;
        top: -20%;
        left: -20%;
        width: 140%;
        height: 140%;
        background: radial-gradient(circle at 70% 30%, rgba(30,60,150,0.22), transparent 50%),
                    radial-gradient(circle at 30% 80%, rgba(10,30,80,0.20), transparent 55%);
        filter: blur(140px);
        animation: waveMove2 14s ease-in-out infinite alternate;
        z-index: 0;
    }

    /* 🌊 ANIMATIONS (MORE VISIBLE NOW) */
    @keyframes waveMove1 {
        0%   { transform: translate(0%, 0%) scale(1); }
        100% { transform: translate(8%, 10%) scale(1.1); }
    }

    @keyframes waveMove2 {
        0%   { transform: translate(0%, 0%) scale(1); }
        100% { transform: translate(-10%, -6%) scale(1.15); }
    }

    /* 🔥 MAKE CONTENT ABOVE WAVES */
    .block-container {
        position: relative;
        z-index: 1;
    }

    /* CLEAN HEADER */
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background: transparent !important;
    }

    </style>
    """, unsafe_allow_html=True)

        st.markdown("""
        <style>
        .landing-heading {
            font-size: 1.6rem;
            font-weight: 700;
            color: #e2e6f0;
            text-align: center;
            margin-bottom: 0.4rem;
        }
        .landing-sub {
            font-size: 0.92rem;
            color: #8b92a5;
            text-align: center;
            margin-bottom: 2.5rem;
        }
        .landing-btns .stButton > button {
            background: rgba(28, 31, 38, 0.55) !important;
            border: 1px solid #3a3f50 !important;
            border-radius: 16px !important;
            padding: 2rem 1.5rem !important;
            width: 100% !important;
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            color: #e2e6f0 !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease,
                        border-color 0.2s ease !important;
            min-height: 100px !important;
        }
        .landing-btns .stButton > button:hover {
            transform: translateY(-4px) !important;
            border-color: rgba(74, 144, 226, 0.6) !important;
            box-shadow: 0 8px 24px rgba(74, 144, 226, 0.2) !important;
            color: #ffffff !important;
            background: rgba(74, 144, 226, 0.08) !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(
            "<div class='landing-heading'>How would you like to continue?</div>"
            "<div class='landing-sub'>Select the option that best describes you</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='landing-btns'>", unsafe_allow_html=True)
        _, col_js, col_gap, col_rec, _ = st.columns([2, 1.6, 0.4, 1.6, 2])
        with col_js:
            if st.button("🎯  Aspirant", key="landing_js"):
                st.session_state.page = "js_auth"
                st.rerun()
         
        with col_rec:
            if st.button("🏢Recruiter", key="landing_rec"):
                st.session_state.page = "recruiter"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.stop()

    # ── RECRUITER LOGIN (placeholder) ────────────────────────────────────────────
    if st.session_state.page == "recruiter":

        # Back button
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back", key="rec_back"):
            st.session_state.page = "landing"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Login gate ───────────────────────────────────────────────────────

        st.markdown("""
        <div class="rec-card">
            <div class="rec-card-title">🏢 Recruiter Login</div>
            <div class="rec-card-sub">Access the recruiter dashboard</div>
        </div>
        """, unsafe_allow_html=True)

       # ── Login gate ───────────────────────────────────────────────────────
        if "rec_logged_in" not in st.session_state:
            st.session_state.rec_logged_in = False

        if not st.session_state.rec_logged_in:
            _, col_form, _ = st.columns([1, 2, 1])
            with col_form:
                st.markdown("<br>", unsafe_allow_html=True)
                ad_user     = st.text_input("Admin Username", key="rec_user")
                ad_password = st.text_input("Admin Password", type="password", key="rec_pass")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Login"):
                    if ad_user == "admin" and ad_password == "123456":
                        st.session_state.rec_logged_in = True
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
        
            st.stop()

        st.markdown("<div style='margin-top:2.5rem;'></div>", unsafe_allow_html=True)



    # ── JOB SEEKER AUTH (Login / Register) ───────────────────────────────────
    if st.session_state.page == "js_auth":

        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back", key="auth_back"):
            st.session_state.page = "landing"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align:center;padding:1.5rem 0 0.5rem;'>"
            "<div style='font-size:1.4rem;font-weight:700;color:#e2e6f0;margin-bottom:0.25rem;'>"
            "Welcome to ResumeIQ</div>"
            "<div style='font-size:0.88rem;color:#8b92a5;'>Sign in or create an account to continue</div>"
            "</div>",
            unsafe_allow_html=True
        )

        # Toggle tabs
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if st.button("🔑  Login",
                         key="tab_login",
                         use_container_width=True if hasattr(st.button, '__kwdefaults__') else False):
                st.session_state.auth_mode = "login"
                st.rerun()
        with col_t2:
            if st.button("✏️  Register",
                         key="tab_register",
                         use_container_width=True if hasattr(st.button, '__kwdefaults__') else False):
                st.session_state.auth_mode = "register"
                st.rerun()

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

        _, col_form, _ = st.columns([1, 2, 1])

        if st.session_state.auth_mode == "login":
            with col_form:
                st.markdown(
                    "<div style='font-size:1.05rem;font-weight:600;color:#4A90E2;"
                    "margin-bottom:1rem;'>Login to your account</div>",
                    unsafe_allow_html=True
                )
                login_email = st.text_input("Email", key="login_email")
                login_pass  = st.text_input("Password", type="password", key="login_pass")
                st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
                if st.button("Login", key="do_login"):
                    if login_email.strip() and login_pass.strip():
                        user = login_user(login_email.strip(), login_pass.strip())
                        if user:
                            st.session_state.current_user = user
                            st.session_state.page = "job_seeker"
                            st.rerun()
                        else:
                            st.error("Invalid email or password.")
                    else:
                        st.warning("Please fill in both fields.")

        else:
            with col_form:
                st.markdown(
                    "<div style='font-size:1.05rem;font-weight:600;color:#4A90E2;"
                    "margin-bottom:1rem;'>Create your account</div>",
                    unsafe_allow_html=True
                )
                reg_name   = st.text_input("Full Name",     key="reg_name")
                reg_email  = st.text_input("Email",         key="reg_email")
                reg_mobile = st.text_input("Mobile Number", key="reg_mobile")
                reg_pass   = st.text_input("Password", type="password", key="reg_pass")
                reg_pass2  = st.text_input("Confirm Password", type="password", key="reg_pass2")
                st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
                if st.button("Create Account", key="do_register"):
                    if not all([reg_name.strip(), reg_email.strip(),
                                reg_mobile.strip(), reg_pass.strip()]):
                        st.warning("Please fill in all fields.")
                    elif reg_pass != reg_pass2:
                        st.error("Passwords do not match.")
                    elif not reg_mobile.strip().isdigit() or len(reg_mobile.strip()) != 10:
                        st.error("Mobile number must be exactly 10 digits.")
                    else:
                        ok, msg = register_user(
                            reg_name.strip(), reg_email.strip(),
                            reg_mobile.strip(), reg_pass.strip()
                        )
                        if ok:
                            user = login_user(reg_email.strip(), reg_pass.strip())
                            st.session_state.current_user = user
                            st.session_state.page = "job_seeker"
                            st.rerun()
                        else:
                            st.error(msg)

        st.stop()


    # ── PROFILE PAGE ─────────────────────────────────────────────────────────
    if st.session_state.show_profile and st.session_state.current_user:
        user = st.session_state.current_user

        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back", key="profile_back"):
            st.session_state.show_profile = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        initials = "".join([w[0].upper() for w in user['name'].split()[:2]])

        st.markdown(
            f"<div style='display:flex;align-items:center;gap:1.25rem;"
            f"padding:1.5rem 0 0.5rem;'>"
            f"<div style='width:64px;height:64px;border-radius:50%;"
            f"background:linear-gradient(135deg,#4A90E2,#2a6db5);"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-size:1.4rem;font-weight:700;color:#fff;flex-shrink:0;'>"
            f"{initials}</div>"
            f"<div>"
            f"<div style='font-size:1.3rem;font-weight:700;color:#e2e6f0;'>{user['name']}</div>"
            f"<div style='font-size:0.85rem;color:#8b92a5;'>{user['email']}</div>"
            f"</div></div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<hr style='border:none;border-top:1px solid #2c3240;margin:1rem 0;'>",
            unsafe_allow_html=True
        )

        # Account details
        st.markdown(
            "<div style='font-size:1.05rem;font-weight:600;color:#4A90E2;"
            "margin-bottom:0.75rem;'>Account Details</div>",
            unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div style='font-size:0.68rem;color:#5a6278;text-transform:uppercase;"
                        "letter-spacing:0.07em;margin-bottom:0.2rem;'>Name</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.92rem;color:#a8b2c8;'>{user['name']}</div>",
                        unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='font-size:0.68rem;color:#5a6278;text-transform:uppercase;"
                        "letter-spacing:0.07em;margin-bottom:0.2rem;'>Mobile</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.92rem;color:#a8b2c8;'>{user['mobile']}</div>",
                        unsafe_allow_html=True)
        with col3:
            st.markdown("<div style='font-size:0.68rem;color:#5a6278;text-transform:uppercase;"
                        "letter-spacing:0.07em;margin-bottom:0.2rem;'>Member Since</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.92rem;color:#a8b2c8;'>{user['created_at'][:10]}</div>",
                        unsafe_allow_html=True)

        st.markdown(
            "<hr style='border:none;border-top:1px solid #2c3240;margin:1.25rem 0;'>",
            unsafe_allow_html=True
        )

        # Resume history
        st.markdown(
            "<div style='font-size:1.05rem;font-weight:600;color:#4A90E2;"
            "margin-bottom:0.75rem;'>Resume History</div>",
            unsafe_allow_html=True
        )

        scans = get_user_scans(user['email'])

        if not scans:
            st.markdown(
                "<div style='font-size:0.88rem;color:#5a6278;padding:0.75rem;"
                "background:#161920;border:1px solid #2c3240;border-radius:10px;'>"
                "No resume scans yet. Upload your resume to get started.</div>",
                unsafe_allow_html=True
            )
        else:
            # Score trend chart
            if len(scans) > 1:
                st.markdown(
                    "<div style='font-size:0.75rem;color:#5a6278;text-transform:uppercase;"
                    "letter-spacing:0.07em;margin-bottom:0.5rem;'>Resume Score Over Time</div>",
                    unsafe_allow_html=True
                )
                chart_data = pd.DataFrame({
                    "Date":  [s[6][:10] for s in scans][::-1],
                    "Score": [s[1]      for s in scans][::-1]
                })
                st.line_chart(chart_data.set_index("Date"))

            # Individual scan cards
            for idx, scan in enumerate(scans):
                pdf_name, res_score, reco_field, cand_level, skills, rec_skills, ts = scan
                score_color = (
                    "#1ed760" if res_score >= 70 else
                    "#fba171" if res_score >= 40 else
                    "#d73b5c"
                )
                level_color = {
                    "Fresher": "#fba171", "Intermediate": "#1ed760",
                    "Experienced": "#4A90E2", "NA": "#d73b5c"
                }.get(cand_level, "#8b92a5")

                st.markdown(
                    f"<div style='background:#161920;border:1px solid #2c3240;"
                    f"border-radius:12px;padding:1rem 1.25rem;margin-bottom:0.5rem;'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.6rem;'>"
                    f"<div style='font-size:0.92rem;font-weight:600;color:#e2e6f0;'>"
                    f"&#128196;&nbsp;{pdf_name}</div>"
                    f"<div style='font-size:0.72rem;color:#5a6278;'>{ts}</div>"
                    f"</div>"
                    f"<div style='display:flex;gap:1.5rem;flex-wrap:wrap;'>"
                    f"<div><div style='font-size:0.65rem;color:#5a6278;text-transform:uppercase;"
                    f"letter-spacing:0.07em;margin-bottom:0.15rem;'>Score</div>"
                    f"<div style='font-size:1.1rem;font-weight:800;color:{score_color};'>"
                    f"{res_score}/100</div></div>"
                    f"<div><div style='font-size:0.65rem;color:#5a6278;text-transform:uppercase;"
                    f"letter-spacing:0.07em;margin-bottom:0.15rem;'>Level</div>"
                    f"<div style='font-size:0.88rem;font-weight:600;color:{level_color};'>"
                    f"{cand_level or 'N/A'}</div></div>"
                    f"<div><div style='font-size:0.65rem;color:#5a6278;text-transform:uppercase;"
                    f"letter-spacing:0.07em;margin-bottom:0.15rem;'>Field</div>"
                    f"<div style='font-size:0.88rem;color:#a8b2c8;'>{reco_field or 'N/A'}</div>"
                    f"</div></div></div>",
                    unsafe_allow_html=True
                )

                # Download report button
                try:
                    report_bytes = generate_analysis_report(user, scan)
                    report_filename = f"ResumeIQ_Report_{pdf_name.replace('.pdf','')}_"  \
                                      f"{ts[:10]}.pdf"
                    st.download_button(
                        label       = "⬇ Download Report",
                        data        = report_bytes,
                        file_name   = report_filename,
                        mime        = "application/pdf",
                        key         = f"dl_report_{idx}",
                        use_container_width = False,
                    )
                except Exception as e:
                    st.markdown(
                        f"<div style='font-size:0.78rem;color:#5a6278;'>Report unavailable</div>",
                        unsafe_allow_html=True
                    )

                st.markdown("<div style='margin-bottom:0.5rem;'></div>",
                            unsafe_allow_html=True)               
                    
        # Logout button
        st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
        if st.button("Logout", key="profile_logout"):
            st.session_state.current_user  = None
            st.session_state.show_profile  = False
            st.session_state.page          = "landing"
            st.session_state.show_job_section = False
            st.session_state.analyse_clicked  = False
            st.rerun()

        st.stop()


    ###### Creating Database and Table ######

    # ── RECRUITER DASHBOARD (only renders when logged in) ────────────────────
    if st.session_state.page == "recruiter" and st.session_state.get("rec_logged_in"):

        # General scoring helper — reuses same keyword logic as job seeker flow
        def compute_resume_score(text):
            score = 0
            sections = {
                'OBJECTIVE':      4,  'Objective':      4,
                'DECLARATION':    2,  'Declaration':    2,
                'HOBBIES':        4,  'Hobbies':        4,
                'INTERESTS':      5,  'Interests':      5,
                'ACHIEVEMENTS':  13,  'Achievements':  13,
                'CERTIFICATIONS':12,  'Certifications':12,  'Certification': 12,
                'PROJECTS':      19,  'PROJECT':       19,  'Projects':      19,  'Project': 19,
                'EXPERIENCE':    16,  'Experience':    16,
                'INTERNSHIPS':    6,  'INTERNSHIP':     6,  'Internships':    6,  'Internship': 6,
                'SKILLS':         7,  'SKILL':          7,  'Skills':         7,  'Skill':      7,
            }
            counted = set()
            for keyword, points in sections.items():
                norm = keyword.upper()
                if norm not in counted and keyword in text:
                    score += points
                    counted.add(norm)
            return min(score, 100)

        # Role-based skill matching
        role_skills = {
            "Data Analyst":               ["python", "sql", "pandas", "power bi", "excel", "statistics", "tableau", "data visualization"],
            "Backend Developer":          ["node", "express", "mongodb", "api", "sql", "docker", "rest", "python", "java", "git"],
            "Frontend Developer":         ["html", "css", "javascript", "react", "angular", "vue", "responsive design", "typescript", "git"],
            "Machine Learning Engineer":  ["python", "tensorflow", "sklearn", "pandas", "pytorch", "numpy", "nlp", "deep learning", "mlops"],
            "Software Developer":         ["java", "python", "dsa", "sql", "git", "oop", "algorithms", "system design", "c++"],
        }

        def compute_match_score(candidate_skills, role):
            required   = role_skills.get(role, [])
            if not required:
                return 0, [], required
            c_skills_lower = [s.lower() for s in candidate_skills]
            matched  = [r for r in required if any(r in cs or cs in r for cs in c_skills_lower)]
            missing  = [r for r in required if r not in matched]
            score    = round((len(matched) / len(required)) * 100)
            return score, matched, missing

        def compute_jd_match(resume_text, jd_text):
            if not jd_text.strip() or not resume_text.strip():
                return 0
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                vectors    = vectorizer.fit_transform([resume_text, jd_text])
                score      = cosine_similarity(vectors[0], vectors[1])[0][0]
                return round(score * 100, 1)
            except:
                return 0

        # ── Dashboard header ─────────────────────────────────────────────────
        st.markdown(
            """
            <div style='margin-bottom:1.25rem;'>
                <div style='font-size:1.3rem; font-weight:700; color:#e2e6f0;
                            margin-bottom:0.25rem; margin-top:0.5rem;'>
                    Recruiter Dashboard
                </div>
                <div style='font-size:0.88rem; color:#8b92a5;'>
                    Upload multiple resumes and rank candidates by role fit.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── Role selector ────────────────────────────────────────────────────
        selected_role = st.selectbox(
            "Select Job Role for Evaluation",
            list(role_skills.keys()),
            key="rec_role"
        )

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

        # ── Job Description Input ─────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:1.05rem;font-weight:600;color:#4A90E2;"
            "margin-bottom:0.25rem;'>Job Description Match</div>"
            "<div style='font-size:0.82rem;color:#8b92a5;margin-bottom:0.75rem;'>"
            "Paste the actual job description below. Resumes will be ranked by how "
            "closely they match this JD using TF-IDF semantic scoring.</div>",
            unsafe_allow_html=True
        )

        jd_text = st.text_area(
            "Paste Job Description here",
            height=180,
            placeholder="e.g. We are looking for a Backend Developer with experience in "
                        "Node.js, REST APIs, MongoDB, Docker and CI/CD pipelines...",
            key="jd_input"
        )

        if jd_text.strip():
            st.markdown(
                "<div style='font-size:0.82rem;color:#1ed760;margin:0.3rem 0 0.75rem;'>"
                "✓ &nbsp;Job description received — candidates will be ranked by JD match.</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='font-size:0.82rem;color:#8b92a5;margin:0.3rem 0 0.75rem;'>"
                "ℹ &nbsp;No JD provided — candidates will be ranked by role skill match instead.</div>",
                unsafe_allow_html=True
            )

        st.markdown("<hr style='border:none;border-top:1px solid #2c3240;margin:1rem 0;'>",
                    unsafe_allow_html=True)

        # ── File uploader ────────────────────────────────────────────────────
        uploaded_resumes = st.file_uploader(
            "Upload Resumes (PDF only)",
            type=["pdf"],
            accept_multiple_files=True,
            key="recruiter_uploads"
        )

        if uploaded_resumes:
            st.markdown(
                f"<div style='color:#1ed760; font-size:0.9rem; margin:0.5rem 0 1rem;'>"
                f"✓ &nbsp; {len(uploaded_resumes)} resume(s) uploaded.</div>",
                unsafe_allow_html=True
            )

            _, col_analyze, _ = st.columns([2, 1.5, 2])
            with col_analyze:
                analyze_clicked = st.button("Analyze Candidates")

            if analyze_clicked:
                candidates = []

                with st.spinner("Analyzing resumes..."):
                    for uploaded_file in uploaded_resumes:
                        try:
                            save_path = f"./Uploaded_Resumes/rec_{uploaded_file.name}"
                            with open(save_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            resume_data  = ResumeParser(save_path).get_extracted_data()
                            resume_text  = pdf_reader(save_path)

                            name         = resume_data.get('name',   'Unknown') if resume_data else 'Unknown'
                            email        = resume_data.get('email',  'N/A')     if resume_data else 'N/A'
                            skills       = resume_data.get('skills', [])        if resume_data else []
                            base_score   = compute_resume_score(resume_text)
                            match_score, matched_skills, missing_skills = compute_match_score(skills, selected_role)
                            jd_score     = compute_jd_match(resume_text, jd_text)

                            candidates.append({
                                "name":            name,
                                "email":           email,
                                "skills":          skills,
                                "base_score":      base_score,
                                "match_score":     match_score,
                                "matched_skills":  matched_skills,
                                "missing_skills":  missing_skills,
                                "jd_score":        jd_score,
                                "resume_text":     resume_text,
                                "filename":        uploaded_file.name,
                            })
                        except Exception as e:
                            st.warning(f"Could not parse {uploaded_file.name}: {e}")

                ## Sort by JD score if JD provided, otherwise by role match score
                if jd_text.strip():
                    candidates.sort(key=lambda x: x["jd_score"], reverse=True)
                else:
                    candidates.sort(key=lambda x: x["match_score"], reverse=True)

                # ── Results header ───────────────────────────────────────────
                # ── Results header ───────────────────────────────────────────
                ranking_basis = "JD Match Score" if jd_text.strip() else "Role Skill Match"
                st.markdown(
                    f"<div style='margin:1.75rem 0 1rem;'>"
                    f"<div style='font-size:0.78rem;color:#5a6278;text-transform:uppercase;"
                    f"letter-spacing:0.06em;margin-bottom:0.3rem;'>Evaluation Results</div>"
                    f"<div style='font-size:1.05rem;font-weight:600;color:#e2e6f0;'>"
                    f"Evaluating Candidates for:&nbsp;"
                    f"<span style='color:#4A90E2;'>{selected_role}</span>"
                    f"</div>"
                    f"<div style='font-size:0.8rem;color:#8b92a5;margin-top:0.25rem;'>"
                    f"Ranked by: <span style='color:#fba171;'>{ranking_basis}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

                # ── Best candidate banner ────────────────────────────────────
                best = candidates[0]
                st.markdown(
                    f"<div style='background:#0f1f12;border:1px solid #1ed76044;"
                    f"border-radius:12px;padding:0.85rem 1.25rem;margin-bottom:1.25rem;"
                    f"display:flex;align-items:center;gap:0.75rem;'>"
                    f"<span style='font-size:1.3rem;'>&#127942;</span>"
                    f"<div>"
                    f"<div style='font-size:0.7rem;color:#5a6278;text-transform:uppercase;"
                    f"letter-spacing:0.07em;margin-bottom:0.15rem;'>Best Fit</div>"
                    f"<div style='font-size:0.98rem;font-weight:700;color:#e2e6f0;'>"
                    f"{best['name']}"
                    f"<span style='font-weight:400;color:#8b92a5;'> for {selected_role} — </span>"
                    f"<span style='color:#1ed760;'>{best['match_score']}% match</span>"
                    f"</div>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                st.markdown("""
                <style>
                .candidate-card {
                    border: 1px solid rgba(74,144,226,0.25);
                    border-radius: 16px;
                    padding: 1.5rem 1.75rem;
                    background: #1c1f26;
                    margin-bottom: 1.25rem;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
                    transition: all 0.25s ease;
                }
                .candidate-card:hover {
                    transform: translateY(-3px);
                    box-shadow: 0 8px 28px rgba(74,144,226,0.2);
                    border-color: rgba(74,144,226,0.6);
                }
                </style>
                """, unsafe_allow_html=True)

                rank_icons = {1: "🥇", 2: "🥈", 3: "🥉"}

                for rank, c in enumerate(candidates, start=1):
                    rank_icon   = rank_icons.get(rank, f"#{rank}")
                    match_color = (
                        "#1ed760" if c["match_score"] >= 70 else
                        "#fba171" if c["match_score"] >= 40 else
                        "#d73b5c"
                    )

                    matched_pills = "".join(
                        f"<span style='display:inline-block;background:#0d2218;"
                        f"border:1px solid #1ed76044;border-radius:20px;padding:4px 12px;"
                        f"margin:3px 5px 3px 0;font-size:0.8rem;color:#1ed760;"
                        f"font-weight:500;'>&#10004; {s.title()}</span>"
                        for s in c["matched_skills"]
                    ) or "<span style='color:#5a6278;font-size:0.85rem;'>None detected</span>"

                    missing_pills = "".join(
                        f"<span style='display:inline-block;background:#220d0d;"
                        f"border:1px solid #d73b5c44;border-radius:20px;padding:4px 12px;"
                        f"margin:3px 5px 3px 0;font-size:0.8rem;color:#d73b5c;"
                        f"font-weight:500;'>&#10008; {s.title()}</span>"
                        for s in c["missing_skills"]
                    ) or "<span style='color:#5a6278;font-size:0.85rem;'>None — full match!</span>"

                    st.markdown(
                        f"<div class='candidate-card'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.9rem;'>"
                        f"<div>"
                        f"<div style='font-size:1.15rem;font-weight:700;color:#e2e6f0;margin-bottom:0.2rem;'>{rank_icon}&nbsp;{c['name']}</div>"
                        f"<div style='font-size:0.78rem;color:#5a6278;'>Evaluated for: {selected_role}</div>"
                        f"</div>"
                        f"<div style='display:flex;gap:1.5rem;'>"
                        f"<div style='text-align:center;'>"
                        f"<div style='font-size:0.68rem;color:#5a6278;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.2rem;'>Match Score</div>"
                        f"<div style='font-size:1.4rem;font-weight:800;color:{match_color};'>{c['match_score']}%</div>"
                        f"</div>"
                        f"<div style='text-align:center;'>"
                        f"<div style='font-size:0.68rem;color:#5a6278;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.2rem;'>Resume Score</div>"
                        f"<div style='font-size:1.4rem;font-weight:700;color:#8b92a5;'>{c['base_score']}<span style='font-size:0.85rem;color:#5a6278;'>/100</span></div>"
                        f"</div>"
                        f"{'<div style=text-align:center;><div style=font-size:0.68rem;color:#5a6278;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.2rem;>JD Match</div><div style=font-size:1.4rem;font-weight:800;color:#fba171;>' + str(c['jd_score']) + '%</div></div>' if jd_text.strip() else ''}"
                        f"</div>"
                        f"</div>"
                        f"<hr style='border:none;border-top:1px solid #2c3240;margin:0.75rem 0;'>"
                        f"<div style='display:flex;gap:2.5rem;flex-wrap:wrap;margin-bottom:0.9rem;'>"
                        f"<div>"
                        f"<div style='font-size:0.68rem;color:#5a6278;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.25rem;'>Email</div>"
                        f"<div style='font-size:0.92rem;color:#a8b2c8;'>{c['email']}</div>"
                        f"</div>"
                        f"<div>"
                        f"<div style='font-size:0.68rem;color:#5a6278;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.25rem;'>File</div>"
                        f"<div style='font-size:0.92rem;color:#a8b2c8;'>{c['filename']}</div>"
                        f"</div>"
                        f"</div>"
                        f"<hr style='border:none;border-top:1px solid #2c3240;margin:0.75rem 0;'>"
                        f"<div style='display:flex;gap:2rem;flex-wrap:wrap;'>"
                        f"<div style='flex:1;min-width:180px;'>"
                        f"<div style='font-size:0.68rem;color:#5a6278;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem;'>&#10004;&nbsp;Matching Skills</div>"
                        f"{matched_pills}"
                        f"</div>"
                        f"<div style='flex:1;min-width:180px;'>"
                        f"<div style='font-size:0.68rem;color:#5a6278;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem;'>&#10008;&nbsp;Missing Skills</div>"
                        f"{missing_pills}"
                        f"</div>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        st.stop()   # nothing below renders on the recruiter dashboard

    # Create the DB
    db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""


    # Create table user_data and user_feedback
    DB_table_name = 'user_data'
    table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                    sec_token varchar(20) NOT NULL,
                    
                    host_name varchar(50) NULL,
                    dev_user varchar(50) NULL,
                    os_name_ver varchar(50) NULL,
                    latlong varchar(50) NULL,
                    city varchar(50) NULL,
                    state varchar(50) NULL,
                    country varchar(50) NULL,
                    act_name varchar(50) NOT NULL,
                    act_mail varchar(50) NOT NULL,
                    act_mob varchar(20) NOT NULL,
                    Name varchar(500) NOT NULL,
                    Email_ID VARCHAR(500) NOT NULL,
                    resume_score VARCHAR(8) NOT NULL,
                    Timestamp VARCHAR(50) NOT NULL,
                    Page_no VARCHAR(5) NOT NULL,
                    Predicted_Field BLOB NOT NULL,
                    User_level BLOB NOT NULL,
                    Actual_skills BLOB NOT NULL,
                    Recommended_skills BLOB NOT NULL,
                    Recommended_courses BLOB NOT NULL,
                    pdf_name varchar(50) NOT NULL,
                    PRIMARY KEY (ID)
                    );
                """
    #cursor.execute(table_sql)


    DBf_table_name = 'user_feedback'
    tablef_sql = "CREATE TABLE IF NOT EXISTS " + DBf_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                        feed_name varchar(50) NOT NULL,
                        feed_email VARCHAR(50) NOT NULL,
                        feed_score VARCHAR(5) NOT NULL,
                        comments VARCHAR(100) NULL,
                        Timestamp VARCHAR(50) NOT NULL,
                        PRIMARY KEY (ID)
                    );
                """
    #cursor.execute(tablef_sql)


    ###### CODE FOR CLIENT SIDE (USER) ######

        # ── JOB SEEKER FLOW ──────────────────────────────────────────────────────────
    # (reached only when st.session_state.page == "job_seeker")

    # Back button at the top of the job-seeker flow
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Back", key="js_back"):
        st.session_state.page    = "landing"
        st.session_state.show_job_section = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


    if st.session_state.page == "job_seeker":
        # Collecting Miscellaneous Information
        # (show_job_section already initialised in session-state bootstrap above)
        import re

        def section_title(text):
            st.markdown(
                f"<div style='font-size:1.05rem; font-weight:600; color:#4A90E2; "
                f"letter-spacing:0.02em; margin-top:1.5rem; margin-bottom:0.5rem;'>"
                f"{text}</div>",
                unsafe_allow_html=True
            )

        def section_divider():
            st.markdown(
                "<hr style='border:none; border-top:1px solid #2c3240; margin:1.25rem 0;'>",
                unsafe_allow_html=True
            )
        # ── Progress Indicator ───────────────────────────────────────────────
        if not st.session_state.show_job_section:
            current_step = 1
        elif 'pdf_file' not in st.session_state or st.session_state.get('pdf_file') is None:
            current_step = 2
        else:
            current_step = 4

        steps = ["Personal Details", "Role Selection", "Resume Upload", "Analysis"]

        step_html = ""
        for i, label in enumerate(steps, start=1):
            if i < current_step:
                # completed
                circle_bg   = "#2a6db5"
                circle_color = "#ffffff"
                label_color  = "#5a6278"
                border       = "2px solid #2a6db5"
            elif i == current_step:
                # active
                circle_bg    = "#4A90E2"
                circle_color = "#ffffff"
                label_color  = "#e2e6f0"
                border       = "2px solid #4A90E2"
            else:
                # upcoming
                circle_bg    = "transparent"
                circle_color = "#5a6278"
                label_color  = "#5a6278"
                border       = "2px solid #2c3240"

            connector = (
                f"<div style='flex:1; height:1px; background:"
                f"{'#2a6db5' if i < current_step else '#2c3240'}; "
                f"margin: 0 6px; align-self:center;'></div>"
                if i < len(steps) else ""
            )

            step_html += f"""
            <div style='display:flex; flex-direction:column; align-items:center; gap:6px;'>
                <div style='
                    width:30px; height:30px; border-radius:50%;
                    background:{circle_bg}; border:{border};
                    display:flex; align-items:center; justify-content:center;
                    font-size:0.78rem; font-weight:700; color:{circle_color};
                '>{i}</div>
                <div style='font-size:0.72rem; color:{label_color};
                            text-align:center; white-space:nowrap;'>
                    {label}
                </div>
            </div>
            {connector}
            """

        st.markdown(
            f"""
            <div style='
                display:flex; align-items:flex-start;
                justify-content:center;
                padding: 1rem 1rem 1.5rem;
                max-width: 520px;
                margin: 0 auto;
            '>
                {step_html}
            </div>
            """,
            unsafe_allow_html=True
        )
        # ── End Progress Indicator ───────────────────────────────────────────
        # Pre-fill from logged-in user
        if st.session_state.current_user:
            user = st.session_state.current_user
            act_name = user['name']
            act_mail = user['email']
            act_mob  = user['mobile']

            # Profile circle — top right
            initials = "".join([w[0].upper() for w in act_name.split()[:2]])
            st.markdown(
                f"<div style='position:fixed;top:14px;right:18px;z-index:9999;'>"
                f"<div onclick=\"window.location.href='?'\" "
                f"title='View Profile' "
                f"style='width:40px;height:40px;border-radius:50%;"
                f"background:linear-gradient(135deg,#4A90E2,#2a6db5);"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-size:0.9rem;font-weight:700;color:#fff;"
                f"cursor:pointer;box-shadow:0 2px 10px rgba(74,144,226,0.4);'>"
                f"{initials}</div></div>",
                unsafe_allow_html=True
            )
            if st.button(f"👤 {act_name}", key="open_profile"):
                st.session_state.show_profile = True
                st.rerun()

            # Show account info card
            st.markdown(
                f"<div style='background:#161920;border:1px solid #2c3240;"
                f"border-radius:12px;padding:0.85rem 1.25rem;margin-bottom:0.5rem;"
                f"display:flex;align-items:center;gap:1rem;'>"
                f"<div style='width:42px;height:42px;border-radius:50%;"
                f"background:linear-gradient(135deg,#4A90E2,#2a6db5);"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-size:1rem;font-weight:700;color:#fff;flex-shrink:0;'>"
                f"{initials}</div>"
                f"<div>"
                f"<div style='font-size:0.95rem;font-weight:600;color:#e2e6f0;'>{act_name}</div>"
                f"<div style='font-size:0.8rem;color:#8b92a5;'>{act_mail} · {act_mob}</div>"
                f"</div></div>",
                unsafe_allow_html=True
            )

            if not st.session_state.show_job_section:
                if st.button("Next →", key="next_btn"):
                    st.session_state.show_job_section = True
                    st.rerun()

        else:
            # Not logged in — redirect to auth
            st.session_state.page = "js_auth"
            st.rerun()


        sec_token = secrets.token_urlsafe(12)
        host_name = socket.gethostname()
        dev_user = os.getlogin()
        os_name_ver = platform.system() + " " + platform.release()
        g = geocoder.ip('me')
        latlong = g.latlng
        geolocator = Nominatim(user_agent="http")
        location = geolocator.reverse(latlong, language='en')
        address = location.raw['address']
        cityy = address.get('city', '')
        statee = address.get('state', '')
        countryy = address.get('country', '')  
        city = cityy
        state = statee
        country = countryy

        st.markdown(
    """
    <style>
        /* Base font and dark theme background for form area */
        .stTextInput, .stSelectbox, .stFileUploader {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
                         Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
            margin-bottom: 1.1rem;
        }

        /* Labels: slightly larger, clean, professional */
        .stTextInput label, .stSelectbox label, .stFileUploader label {
            font-size: 1.15rem;
            font-weight: 600;
            color: #e2e6f0;
            letter-spacing: 0.01em;
        }

        .stFileUploader label {
            font-size: 1.15rem;
        }

        /* Text input field */
        .stTextInput > div > div > input {
            background-color: #1c1f26;
            border-radius: 10px;
            border: 1px solid #2c3240;
            color: #f5f7fb;
            padding: 0.55rem 0.75rem;
            font-size: 0.95rem;
            transition: border-color 0.25s ease, box-shadow 0.25s ease, background-color 0.25s ease;
        }
        .stTextInput > div > div > input:focus {
            outline: none;
            border-color: #4A90E2;
            box-shadow: 0 0 0 1px rgba(74, 144, 226, 0.65);
            background-color: #20232b;
        }

        /* Selectbox container */
        .stSelectbox > div > div {
            background-color: #1c1f26;
            border-radius: 10px;
            border: 1px solid #2c3240;
            transition: border-color 0.25s ease, box-shadow 0.25s ease, background-color 0.25s ease;
        }
        .stSelectbox > div > div:hover {
            border-color: #3b4254;
        }
        .stSelectbox > div > div:focus-within {
            border-color: #4A90E2;
            box-shadow: 0 0 0 1px rgba(74, 144, 226, 0.65);
            background-color: #20232b;
        }
        .stSelectbox > div > div > div {
            color: #f5f7fb;
        }

        /* File uploader dropzone */
        .stFileUploader > div > div {
            background-color: #1c1f26;
            border-radius: 10px;
            border: 1px dashed #2c3240;
            transition: border-color 0.25s ease, box-shadow 0.25s ease, background-color 0.25s ease;
        }
        .stFileUploader > div > div:hover {
            border-color: #3b4254;
        }
        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            padding: 0.75rem 0.9rem;
        }
        .stFileUploader [data-testid="stFileUploaderDropzone"] div {
            color: #d6d9e3;
        }
        .stFileUploader > div > div:focus-within {
            border-color: #4A90E2;
            box-shadow: 0 0 0 1px rgba(74, 144, 226, 0.65);
            background-color: #20232b;
        }

        /* Make buttons in uploader / selectbox align with dark theme */
        .stFileUploader button, .stSelectbox button {
            border-radius: 8px !important;
        }

        .step-two-fade {
            animation: fadeInSection 0.4s ease-out;
        }

        @keyframes fadeInSection {
            from { opacity: 0; transform: translateY(6px); }
            to   { opacity: 1; transform: translateY(0); }
        }
    </style>
    """,
    unsafe_allow_html=True,
        )

        if not st.session_state.show_job_section:
            st.stop()

        st.markdown(
            """
            <div class="step-two-fade">
            """,
            unsafe_allow_html=True,
        )

        # Upload Resume
        st.markdown(
            """
            <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
                <div style='
                    font-size: 1.35rem;
                    font-weight: 600;
                    color: #e2e6f0;
                    letter-spacing: 0.01em;
                    margin-bottom: 0.5rem;
                '>
                    Upload Your Resume &amp; Get Smart Recommendations
                </div>
                <div style='
                    width: 48px;
                    height: 2px;
                    background: #4A90E2;
                    margin: 0 auto 0.75rem;
                    border-radius: 2px;
                '></div>
                <div style='
                    font-size: 0.88rem;
                    color: #8b92a5;
                    max-width: 480px;
                    margin: 0 auto;
                    line-height: 1.6;
                '>
                    Our system analyses your resume and provides personalised
                    insights to help you improve your career profile.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        ## file upload in pdf format
        section_divider()
        section_title("Select Job Role")

        job_role = st.selectbox(
            "Choose the job role you are applying for:",
            ["Data Analyst", "Backend Developer", "Frontend Developer", "Machine Learning Engineer", "Software Developer"]
        )

        role_info = {
            "Data Analyst": {
                "description": "Data Analysts work with structured data to identify trends, generate insights, and support business decision-making.",
                "skills": ["Python", "SQL", "Data Visualization", "Statistics", "Excel", "Power BI / Tableau"]
            },
            "Backend Developer": {
                "description": "Backend Developers build and maintain the server-side logic, databases, and APIs that power applications.",
                "skills": ["Python / Node.js / Java", "REST APIs", "SQL & NoSQL Databases", "Authentication", "Docker", "Git"]
            },
            "Frontend Developer": {
                "description": "Frontend Developers create the visual and interactive elements of web applications that users directly engage with.",
                "skills": ["HTML & CSS", "JavaScript", "React / Angular / Vue", "Responsive Design", "Git", "REST API Integration"]
            },
            "Machine Learning Engineer": {
                "description": "ML Engineers design, build, and deploy machine learning models and pipelines that power intelligent applications.",
                "skills": ["Python", "TensorFlow / PyTorch", "Scikit-learn", "Data Preprocessing", "Model Deployment", "MLOps"]
            },
            "Software Developer": {
                "description": "Software Developers design and build software solutions across platforms, translating requirements into reliable, scalable code.",
                "skills": ["Data Structures & Algorithms", "OOP", "Version Control (Git)", "System Design", "Testing", "CI/CD"]
            },
        }

        if job_role in role_info:
            info = role_info[job_role]
            skills_html = "".join(
                f"<span style='display:inline-block; background:#1e2330; border:1px solid #2c3240; "
                f"border-radius:6px; padding:3px 10px; margin:3px 4px 3px 0; "
                f"font-size:0.82rem; color:#a8b2c8;'>{s}</span>"
                for s in info["skills"]
            )
            st.markdown(
                f"""
                <div style='
                    background:#161920;
                    border:1px solid #2c3240;
                    border-radius:12px;
                    padding:1.1rem 1.4rem;
                    margin-top:0.75rem;
                    margin-bottom:0.5rem;
                '>
                    <div style='font-size:0.78rem; color:#5a6278; letter-spacing:0.06em;
                                text-transform:uppercase; margin-bottom:0.3rem;'>
                        Selected Role
                    </div>
                    <div style='font-size:1.1rem; font-weight:700; color:#e2e6f0;
                                margin-bottom:0.55rem;'>
                        {job_role}
                    </div>
                    <div style='font-size:0.9rem; color:#8b92a5; line-height:1.55;
                                margin-bottom:0.85rem;'>
                        {info["description"]}
                    </div>
                    <div style='font-size:0.78rem; color:#5a6278; letter-spacing:0.06em;
                                text-transform:uppercase; margin-bottom:0.45rem;'>
                        Typical Skills
                    </div>
                    <div>{skills_html}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.write("")
        st.write("")
        section_divider()
        section_title("Select Experience Level")
        st.markdown("<div style='margin-top:0.25rem;'></div>", unsafe_allow_html=True)
        experience_level = st.selectbox(
            "Select Your Experience Level",
            ["Fresher / Student", "0–2 Years", "2–5 Years", "5+ Years"],
            key="experience_level"
        )
        role_expectations = {
            "Data Analyst": {
                "Fresher / Student":    ["Assist in cleaning and organising datasets", "Create basic reports and dashboards", "Learn SQL and spreadsheet tools on the job"],
                "0–2 Years":            ["Write SQL queries to extract and analyse data", "Build dashboards using Power BI or Tableau", "Identify trends and present findings to stakeholders"],
                "2–5 Years":            ["Own end-to-end analysis pipelines for business units", "Design and maintain reporting infrastructure", "Translate complex data into actionable business insights"],
                "5+ Years":             ["Define the data strategy and analytics roadmap", "Lead a team of analysts and review their outputs", "Partner with leadership to drive data-informed decisions"],
            },
            "Backend Developer": {
                "Fresher / Student":    ["Build simple REST APIs under senior guidance", "Write and run basic unit tests", "Learn version control workflows using Git"],
                "0–2 Years":            ["Develop and maintain API endpoints", "Integrate third-party services and databases", "Debug issues and participate in code reviews"],
                "2–5 Years":            ["Design scalable backend architecture for new features", "Optimise database queries and system performance", "Mentor junior developers and lead sprint tasks"],
                "5+ Years":             ["Architect distributed systems and microservices", "Set coding standards and review practices for the team", "Drive technical decisions aligned with business goals"],
            },
            "Frontend Developer": {
                "Fresher / Student":    ["Build static web pages using HTML, CSS, JavaScript", "Implement UI components from design mockups", "Fix minor bugs and improve responsiveness"],
                "0–2 Years":            ["Develop reusable components in React or Vue", "Collaborate with designers to implement pixel-perfect UIs", "Integrate frontend with backend APIs"],
                "2–5 Years":            ["Lead frontend architecture for mid-size features", "Optimise web performance and accessibility", "Review code and guide junior frontend developers"],
                "5+ Years":             ["Define frontend tech stack and best practices", "Drive cross-team alignment on UI/UX standards", "Lead development of large-scale frontend systems"],
            },
            "Machine Learning Engineer": {
                "Fresher / Student":    ["Implement ML models from research papers or tutorials", "Preprocess datasets and perform exploratory analysis", "Experiment with Scikit-learn or TensorFlow on sample problems"],
                "0–2 Years":            ["Train and evaluate supervised learning models", "Build data pipelines for model training", "Document experiments and report model performance"],
                "2–5 Years":            ["Deploy ML models to production environments", "Design feature engineering strategies for business use cases", "Collaborate with data and product teams on ML integrations"],
                "5+ Years":             ["Architect end-to-end ML platforms and MLOps pipelines", "Define model governance and evaluation standards", "Lead research initiatives and mentor ML engineers"],
            },
            "Software Developer": {
                "Fresher / Student":    ["Write clean code for assigned modules under supervision", "Fix bugs and write unit tests", "Learn the codebase and development workflows"],
                "0–2 Years":            ["Implement features across the application stack", "Participate in code reviews and agile ceremonies", "Debug and resolve production issues independently"],
                "2–5 Years":            ["Design and implement application features end-to-end", "Collaborate with product and engineering teams", "Review code and mentor junior developers"],
                "5+ Years":             ["Lead technical design and architecture discussions", "Set engineering standards and review processes", "Drive delivery across multiple teams or workstreams"],
            },
        }

        if job_role in role_expectations and experience_level in role_expectations[job_role]:
            responsibilities = role_expectations[job_role][experience_level]
            bullets_html = "".join(
                f"<div style='display:flex; gap:0.6rem; margin-bottom:0.45rem;'>"
                f"<span style='color:#4A90E2; margin-top:1px;'>▸</span>"
                f"<span style='font-size:0.88rem; color:#a8b2c8; line-height:1.5;'>{r}</span>"
                f"</div>"
                for r in responsibilities
            )
            st.markdown(
                f"""
                <div style='
                    background:#161920;
                    border:1px solid #2c3240;
                    border-radius:12px;
                    padding:1.1rem 1.4rem;
                    margin-top:0.75rem;
                    margin-bottom:0.5rem;
                '>
                    <div style='font-size:0.78rem; color:#5a6278; letter-spacing:0.06em;
                                text-transform:uppercase; margin-bottom:0.3rem;'>
                        Role Expectations
                    </div>
                    <div style='font-size:0.92rem; color:#e2e6f0; font-weight:600;
                                margin-bottom:0.75rem;'>
                        Typical responsibilities for a {experience_level} {job_role}
                    </div>
                    {bullets_html}
                </div>
                """,
                unsafe_allow_html=True
            )
        st.write("")#space here
        section_divider()
        section_title("Upload Your Resume 👇🏻")
        pdf_file = st.file_uploader("",type=["pdf"])
        st.markdown(
            """
            </div>
            """,
            unsafe_allow_html=True,
        )
        if pdf_file is not None:
            if "last_uploaded" not in st.session_state:
                st.session_state.last_uploaded = None
            if st.session_state.last_uploaded != pdf_file.name:
                st.session_state.analyse_clicked = False
                st.session_state.last_uploaded = pdf_file.name

            st.markdown(
                """
                <div style='
                    text-align: center;
                    color: #1ed760;
                    font-size: 0.92rem;
                    font-weight: 500;
                    margin: 0.5rem 0 1rem;
                '>
                    ✓ &nbsp; Resume uploaded successfully.
                </div>
                """,
                unsafe_allow_html=True
            )

            if "analyse_clicked" not in st.session_state:
                st.session_state.analyse_clicked = False

            _, col_btn, _ = st.columns([2, 1.5, 2])
            with col_btn:
                if st.button("View Detailed Analysis"):
                    st.session_state.analyse_clicked = True

            if st.session_state.analyse_clicked:
                with st.spinner('Analyzing your resume and generating insights...'):
                    st.success("Analysis complete")
                    time.sleep(4)

                ### saving the uploaded resume to folder
                    save_image_path = './Uploaded_Resumes/'+pdf_file.name
                pdf_name = pdf_file.name
                with open(save_image_path, "wb") as f:
                    f.write(pdf_file.getbuffer())
                show_pdf(save_image_path)

                ### parsing and extracting whole resume 
                resume_data = ResumeParser(save_image_path).get_extracted_data()
                if resume_data:
                    
                    ## Get the whole resume data into resume_text
                    resume_text = pdf_reader(save_image_path)

                    ## Showing Analyzed data from (resume_data)
                    ## Showing Analyzed data from (resume_data)
                    st.markdown("<hr style='border:none;border-top:1px solid #2c3240;margin:1.5rem 0;'>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-size:0.72rem;color:#5a6278;text-transform:uppercase;"
                        "letter-spacing:0.08em;margin-bottom:0.3rem;'>Analysis Results</div>"
                        f"<div style='font-size:1.4rem;font-weight:700;color:#e2e6f0;'>"
                        f"Hello, {resume_data['name']} 👋</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

                    try:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown("<div style='font-size:0.7rem;color:#5a6278;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.2rem;'>Email</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-size:0.9rem;color:#a8b2c8;'>{resume_data['email']}</div>", unsafe_allow_html=True)
                        with col_b:
                            st.markdown("<div style='font-size:0.7rem;color:#5a6278;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.2rem;'>Contact</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-size:0.9rem;color:#a8b2c8;'>{resume_data['mobile_number']}</div>", unsafe_allow_html=True)
                        with col_c:
                            st.markdown("<div style='font-size:0.7rem;color:#5a6278;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.2rem;'>Pages</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-size:0.9rem;color:#a8b2c8;'>{resume_data['no_of_pages']}</div>", unsafe_allow_html=True)
                    except:
                        pass
                    ## Predicting Candidate Experience Level 

                    ### Trying with different possibilities
                    cand_level = ''
                    if resume_data['no_of_pages'] < 1:
                        cand_level = "NA"
                    elif 'INTERNSHIP' in resume_text or 'INTERNSHIPS' in resume_text or 'Internship' in resume_text or 'Internships' in resume_text:
                        cand_level = "Intermediate"
                    elif 'EXPERIENCE' in resume_text or 'WORK EXPERIENCE' in resume_text or 'Experience' in resume_text or 'Work Experience' in resume_text:
                        cand_level = "Experienced"
                    else:
                        cand_level = "Fresher"

                    level_color = {"Fresher": "#fba171", "Intermediate": "#1ed760", "Experienced": "#4A90E2", "NA": "#d73b5c"}
                    lc = level_color.get(cand_level, "#8b92a5")
                    st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='display:inline-block;background:#161920;border:1px solid {lc}44;"
                        f"border-radius:10px;padding:0.5rem 1.1rem;'>"
                        f"<span style='font-size:0.72rem;color:#5a6278;text-transform:uppercase;"
                        f"letter-spacing:0.07em;'>Experience Level &nbsp;</span>"
                        f"<span style='font-size:0.95rem;font-weight:700;color:{lc};'>{cand_level}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )


                    ## Skills Analyzing and Recommendation
                    ## Skills Analyzing and Recommendation
                    st.markdown("<hr style='border:none;border-top:1px solid #2c3240;margin:1.5rem 0 1rem;'>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-size:1.05rem;font-weight:600;color:#4A90E2;"
                        "margin-bottom:0.75rem;'>Skills Detected</div>",
                        unsafe_allow_html=True
                    )

                    ### Store detected skills — displayed after rec skills are known
                    ### Store detected skills — displayed after rec skills are known
                    detected_skills = resume_data['skills'] or []
                    keywords = detected_skills

                    ### Safe defaults — prevents UnboundLocalError if no skill matches
                    recommended_skills = []
                    reco_field         = ''
                    rec_course         = ''

                    ### Keywords for Recommendations — mapped to actual role options

                    ### Keywords for Recommendations — mapped to actual role options
                    da_keyword  = [
                        'pandas','numpy','sql','excel','tableau','power bi','data analysis',
                        'data visualization','statistics','matplotlib','seaborn','mysql',
                        'postgresql','google analytics','looker','spreadsheet','etl',
                        'business intelligence','data cleaning','pivot','vlookup','dax'
                    ]
                    be_keyword  = [
                        'node','node js','express','django','flask','spring','fastapi',
                        'rest api','api','mongodb','postgresql','mysql','redis','docker',
                        'kubernetes','microservices','graphql','java','python','git',
                        'ci/cd','jenkins','aws','azure','linux','backend','server','database'
                    ]
                    fe_keyword  = [
                        'react','angular','vue','javascript','typescript','html','css',
                        'bootstrap','tailwind','sass','redux','webpack','next js','gatsby',
                        'responsive design','ui','frontend','figma','jquery','ajax',
                        'rest api','json','vercel','netlify'
                    ]
                    ml_keyword  = [
                        'machine learning','deep learning','tensorflow','pytorch','keras',
                        'scikit-learn','sklearn','nlp','computer vision','neural network',
                        'pandas','numpy','matplotlib','data science','model training',
                        'huggingface','transformers','mlops','feature engineering',
                        'regression','classification','clustering','opencv','jupyter'
                    ]
                    sd_keyword  = [
                        'java','python','c++','c#','algorithms','data structures','oop',
                        'system design','git','agile','scrum','unit testing','junit',
                        'software development','design patterns','solid principles',
                        'multithreading','rest api','microservices','docker','linux',
                        'debugging','code review','jira','bitbucket'
                    ]
                    n_any = [
                        'english','communication','writing','microsoft office',
                        'leadership','customer management','social media','ms word',
                        'powerpoint','teamwork','presentation'
                    ]
                    ### condition starts to check skills from keywords and predict field
                    for i in resume_data['skills']:

                        #### Data Analyst
                        if i.lower() in da_keyword:
                            reco_field = 'Data Analyst'
                            recommended_skills = [
                                'SQL','Python','Pandas','NumPy','Tableau','Power BI',
                                'Excel','Data Visualization','Statistics','ETL',
                                'Business Intelligence','Google Analytics','Looker'
                            ]
                            #rec_course = course_recommender(ds_course)
                            break

                        #### Backend Developer
                        elif i.lower() in be_keyword:
                            reco_field = 'Backend Developer'
                            recommended_skills = [
                                'Node.js','Express','Django','REST API','MongoDB',
                                'PostgreSQL','Docker','Redis','Microservices',
                                'CI/CD','AWS','Linux','GraphQL','Kubernetes'
                            ]
                            #rec_course = course_recommender(web_course)
                            break

                        #### Frontend Developer
                        elif i.lower() in fe_keyword:
                            reco_field = 'Frontend Developer'
                            recommended_skills = [
                                'React','TypeScript','Next.js','Tailwind CSS','Redux',
                                'HTML5','CSS3','Webpack','Figma','REST API',
                                'Responsive Design','Vue.js','Angular'
                            ]
                            #rec_course = course_recommender(web_course)
                            break

                        #### Machine Learning Engineer
                        elif i.lower() in ml_keyword:
                            reco_field = 'Machine Learning Engineer'
                            recommended_skills = [
                                'TensorFlow','PyTorch','Scikit-learn','Pandas','NumPy',
                                'Deep Learning','NLP','Computer Vision','MLOps',
                                'HuggingFace','OpenCV','Feature Engineering',
                                'Model Deployment','Jupyter'
                            ]
                            #rec_course = course_recommender(ds_course)
                            break

                        #### Software Developer
                        elif i.lower() in sd_keyword:
                            reco_field = 'Software Developer'
                            recommended_skills = [
                                'Data Structures','Algorithms','System Design','OOP',
                                'Design Patterns','Git','Docker','REST API',
                                'Unit Testing','Agile','CI/CD','Linux','Code Review'
                            ]
                            #rec_course = course_recommender(web_course)
                            break

                        #### No match
                        elif i.lower() in n_any:
                            reco_field = 'NA'
                            recommended_skills = []
                            #rec_course = "Sorry! Not Available for this Field"
                            break
                    # ── Side-by-side skills display ──────────────────────────
                    col_det, col_rec = st.columns(2)

                    with col_det:
                        st.markdown(
                            "<div style='font-size:0.78rem;color:#5a6278;text-transform:uppercase;"
                            "letter-spacing:0.07em;margin-bottom:0.6rem;'>"
                            "&#10003;&nbsp; Your Current Skills</div>",
                            unsafe_allow_html=True
                        )
                        if detected_skills:
                            for skill in detected_skills:
                                st.markdown(
                                    f"<span style='display:inline-block;background:#1a1e2e;"
                                    f"border:1px solid #4A90E244;border-radius:20px;"
                                    f"padding:5px 14px;margin:3px 5px 3px 0;"
                                    f"font-size:0.82rem;color:#a8b2c8;font-weight:500;'>"
                                    f"&#10003; {skill}</span>",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.markdown(
                                "<span style='color:#5a6278;font-size:0.85rem;'>"
                                "No skills detected.</span>",
                                unsafe_allow_html=True
                            )

                    with col_rec:
                        st.markdown(
                            "<div style='font-size:0.78rem;color:#5a6278;text-transform:uppercase;"
                            "letter-spacing:0.07em;margin-bottom:0.6rem;'>"
                            f"+&nbsp; Recommended Skills to Add for {job_role}</div>",
                            unsafe_allow_html=True
                        )
                        if recommended_skills:
                            for skill in recommended_skills:
                                st.markdown(
                                    f"<span style='display:inline-block;background:#1f1a0e;"
                                    f"border:1px solid #fba17144;border-radius:20px;"
                                    f"padding:5px 14px;margin:3px 5px 3px 0;"
                                    f"font-size:0.82rem;color:#fba171;font-weight:500;'>"
                                    f"+ {skill}</span>",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.markdown(
                                "<span style='color:#5a6278;font-size:0.85rem;'>"
                                "No recommendations available.</span>",
                                unsafe_allow_html=True
                            )


                    ## ── Live Market Intelligence ──────────────────────────────
                    st.markdown(
                        "<hr style='border:none;border-top:1px solid #2c3240;margin:1.5rem 0 1rem;'>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        "<div style='font-size:1.05rem;font-weight:600;color:#4A90E2;"
                        "margin-bottom:0.2rem;'>📡 Live Market Intelligence</div>"
                        "<div style='font-size:0.82rem;color:#8b92a5;margin-bottom:0.9rem;'>"
                        "Skills actively demanded in live job postings for your selected role — "
                        "fetched right now from Indian job portals.</div>",
                        unsafe_allow_html=True
                    )

                    with st.spinner("Fetching live job market data..."):
                        market_skills, skill_counts, source_info = scrape_market_skills(job_role)

                    if market_skills:

                        # Source badge
                        st.markdown(
                            f"<div style='font-size:0.75rem;color:#5a6278;margin-bottom:1rem;'>"
                            f"&#128200; Live data sourced from "
                            f"<span style='color:#4A90E2;font-weight:600;'>{source_info}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        # Compare user skills vs market skills
                        user_skills_lower = [s.lower() for s in (resume_data['skills'] or [])]
                        market_have    = []
                        market_missing = []
                        for skill in market_skills:
                            found = any(skill in us or us in skill for us in user_skills_lower)
                            if found:
                                market_have.append(skill)
                            else:
                                market_missing.append(skill)

                        # Market alignment score
                        alignment   = round((len(market_have) / len(market_skills)) * 100) if market_skills else 0
                        align_color = (
                            "#1ed760" if alignment >= 70 else
                            "#fba171" if alignment >= 40 else
                            "#d73b5c"
                        )

                        # Alignment score card
                        col_align, col_desc = st.columns([1, 3])
                        with col_align:
                            st.markdown(
                                f"<div style='background:#161920;border:1px solid {align_color}44;"
                                f"border-radius:14px;padding:1rem;text-align:center;'>"
                                f"<div style='font-size:0.68rem;color:#5a6278;text-transform:uppercase;"
                                f"letter-spacing:0.07em;margin-bottom:0.3rem;'>Market Fit</div>"
                                f"<div style='font-size:2rem;font-weight:800;color:{align_color};'>"
                                f"{alignment}%</div>"
                                f"<div style='font-size:0.72rem;color:#5a6278;'>of top skills</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        with col_desc:
                            st.markdown(
                                f"<div style='font-size:0.85rem;color:#8b92a5;line-height:1.7;"
                                f"padding-top:0.4rem;'>"
                                f"Your resume matches <b style='color:{align_color};'>"
                                f"{len(market_have)} out of {len(market_skills)}</b> skills "
                                f"actively demanded in live <b style='color:#e2e6f0;'>"
                                f"{job_role}</b> job postings right now. "
                                f"Focus on the missing skills below to increase your shortlisting chances."
                                f"</div>",
                                unsafe_allow_html=True
                            )

                        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

                        # Skills columns
                        col_have, col_miss = st.columns(2)

                        with col_have:
                            st.markdown(
                                "<div style='font-size:0.68rem;color:#5a6278;text-transform:uppercase;"
                                "letter-spacing:0.07em;margin-bottom:0.5rem;'>&#10004;&nbsp;You Have</div>",
                                unsafe_allow_html=True
                            )
                            if market_have:
                                for s in market_have:
                                    st.markdown(
                                        f"<span style='display:inline-block;background:#0d2218;"
                                        f"border:1px solid #1ed76044;border-radius:20px;"
                                        f"padding:4px 12px;margin:3px 5px 3px 0;"
                                        f"font-size:0.8rem;color:#1ed760;font-weight:500;'>"
                                        f"&#10004; {s.title()}</span>",
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.markdown(
                                    "<span style='color:#5a6278;font-size:0.85rem;'>"
                                    "None of the top market skills detected in your resume.</span>",
                                    unsafe_allow_html=True
                                )

                        with col_miss:
                            st.markdown(
                                "<div style='font-size:0.68rem;color:#5a6278;text-transform:uppercase;"
                                "letter-spacing:0.07em;margin-bottom:0.5rem;'>&#10008;&nbsp;In Demand — You're Missing</div>",
                                unsafe_allow_html=True
                            )
                            if market_missing:
                                for s in market_missing:
                                    freq = skill_counts.get(s, 0)
                                    st.markdown(
                                        f"<span style='display:inline-block;background:#220d0d;"
                                        f"border:1px solid #d73b5c44;border-radius:20px;"
                                        f"padding:4px 12px;margin:3px 5px 3px 0;"
                                        f"font-size:0.8rem;color:#d73b5c;font-weight:500;'>"
                                        f"&#10008; {s.title()}"
                                        f"<span style='font-size:0.7rem;color:#5a6278;margin-left:6px;'>"
                                        f"({freq} mentions)</span></span>",
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.markdown(
                                    "<span style='color:#1ed760;font-size:0.85rem;'>"
                                    "You have all top in-demand skills! 🎉</span>",
                                    unsafe_allow_html=True
                                )

                        # Bar chart of skill frequency
                        st.markdown(
                            "<div style='font-size:0.68rem;color:#5a6278;text-transform:uppercase;"
                            "letter-spacing:0.07em;margin:1.25rem 0 0.5rem;'>"
                            "Skill Demand — Mention Frequency in Live Postings</div>",
                            unsafe_allow_html=True
                        )
                        chart_df = pd.DataFrame({
                            "Skill":     [s.title() for s in market_skills],
                            "Mentions":  [skill_counts[s] for s in market_skills]
                        })
                        st.bar_chart(chart_df.set_index("Skill"))

                    else:
                        st.markdown(
                            f"<div style='font-size:0.85rem;color:#8b92a5;padding:0.75rem;"
                            f"background:#161920;border:1px solid #2c3240;border-radius:10px;'>"
                            f"&#9888;&nbsp;Live market data unavailable — {source_info}. "
                            f"Showing standard analysis only.</div>",
                            unsafe_allow_html=True
                        )

                    ## Resume Scorer & Resume Writing Tips
                    ## Resume Scorer & Resume Writing Tips
                    st.markdown("<hr style='border:none;border-top:1px solid #2c3240;margin:1.5rem 0 1rem;'>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-size:1.05rem;font-weight:600;color:#4A90E2;"
                        "margin-bottom:0.75rem;'>Resume Tips & Checklist</div>",
                        unsafe_allow_html=True
                    )
                    resume_score = 0

                    ### Predicting Whether these key points are added to the resume
                    def tip(found, label_ok, label_missing):
                        if found:
                            st.markdown(f"<div style='font-size:0.88rem;color:#1ed760;padding:3px 0;'>✔ &nbsp;{label_ok}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='font-size:0.88rem;color:#8b92a5;padding:3px 0;'>✖ &nbsp;{label_missing}</div>", unsafe_allow_html=True)

                    ### Predicting Whether these key points are added to the resume
                    has_objective = any(k in resume_text for k in ['Objective','OBJECTIVE','Summary','SUMMARY','Career Objective','Professional Summary'])
                    if has_objective:
                        resume_score = resume_score + 6
                    tip(has_objective, "Objective / Summary detected", "Add a career objective or summary")

                    has_education = any(k in resume_text for k in ['Education','EDUCATION','School','College','University','Degree','Bachelor','Master','B.Tech','M.Tech','B.Sc','M.Sc'])
                    if has_education:
                        resume_score = resume_score + 12
                    tip(has_education, "Education details detected", "Add your education details")
                    has_exp = 'EXPERIENCE' in resume_text or 'Experience' in resume_text
                    if has_exp:
                        resume_score = resume_score + 16
                    tip(has_exp, "Experience detected", "Add your work experience")

                    has_intern = any(k in resume_text for k in ['INTERNSHIPS','INTERNSHIP','Internships','Internship'])
                    if has_intern:
                        resume_score = resume_score + 6
                    tip(has_intern, "Internships detected", "Add internships to stand out")

                    has_skills = any(k in resume_text for k in ['SKILLS','SKILL','Skills','Skill'])
                    if has_skills:
                        resume_score = resume_score + 7
                    tip(has_skills, "Skills section detected", "Add a dedicated skills section")

                    has_hobbies = 'HOBBIES' in resume_text or 'Hobbies' in resume_text
                    if has_hobbies:
                        resume_score = resume_score + 4
                    tip(has_hobbies, "Hobbies detected", "Add hobbies to show personality")

                    has_interests = 'INTERESTS' in resume_text or 'Interests' in resume_text
                    if has_interests:
                        resume_score = resume_score + 5
                    tip(has_interests, "Interests detected", "Add interests beyond the job")

                    has_achievements = 'ACHIEVEMENTS' in resume_text or 'Achievements' in resume_text
                    if has_achievements:
                        resume_score = resume_score + 13
                    tip(has_achievements, "Achievements detected", "Add achievements to show impact")

                    has_certs = any(k in resume_text for k in ['CERTIFICATIONS','Certifications','Certification'])
                    if has_certs:
                        resume_score = resume_score + 12
                    tip(has_certs, "Certifications detected", "Add certifications for specialization")

                    has_projects = any(k in resume_text for k in ['PROJECTS','PROJECT','Projects','Project'])
                    if has_projects:
                        resume_score = resume_score + 19
                    tip(has_projects, "Projects detected", "Add projects to show hands-on work")

                    st.markdown("<hr style='border:none;border-top:1px solid #2c3240;margin:1.5rem 0 1rem;'>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-size:1.05rem;font-weight:600;color:#4A90E2;"
                        "margin-bottom:1rem;'>Resume Score</div>",
                        unsafe_allow_html=True
                    )

                    score_color = "#1ed760" if resume_score >= 70 else "#fba171" if resume_score >= 40 else "#d73b5c"

                    col_score, col_note = st.columns([1, 3])
                    with col_score:
                        st.markdown(
                            f"<div style='background:#161920;border:1px solid {score_color}44;"
                            f"border-radius:14px;padding:1rem;text-align:center;'>"
                            f"<div style='font-size:0.7rem;color:#5a6278;text-transform:uppercase;"
                            f"letter-spacing:0.07em;margin-bottom:0.3rem;'>Score</div>"
                            f"<div style='font-size:2rem;font-weight:800;color:{score_color};'>"
                            f"{resume_score}</div>"
                            f"<div style='font-size:0.75rem;color:#5a6278;'>/100</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    with col_note:
                        st.markdown(
                            "<div style='font-size:0.85rem;color:#8b92a5;line-height:1.7;"
                            "padding-top:0.5rem;'>"
                            "This score reflects the completeness of your resume based on "
                            "key sections recruiters look for — experience, skills, projects, "
                            "certifications, and more. Aim for 70+ for best results."
                            "</div>",
                            unsafe_allow_html=True
                        )

                    st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
                    st.progress(resume_score)
                    score = resume_score

                    # ── Save scan to DB ───────────────────────────────────────
                    if st.session_state.current_user:
                        save_scan(
                            user_email         = st.session_state.current_user['email'],
                            pdf_name           = pdf_name,
                            resume_score       = resume_score,
                            reco_field         = reco_field,
                            cand_level         = cand_level,
                            skills             = resume_data.get('skills', []),
                            recommended_skills = recommended_skills
                        )    

                    # print(str(sec_token), str(ip_add), (host_name), (dev_user), (os_name_ver), (latlong), (city), (state), (country), (act_name), (act_mail), (act_mob), resume_data['name'], resume_data['email'], str(resume_score), timestamp, str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']), str(recommended_skills), str(rec_course), pdf_name)

                    
                    ### Getting Current Date and Time
                    ts = time.time()
                    cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    timestamp = str(cur_date+'_'+cur_time)


                    ## Calling insert_data to add all the data into user_data                
                    insert_data(str(sec_token), (host_name), (dev_user), (os_name_ver), (latlong), (city), (state), (country), (act_name), (act_mail), (act_mob), resume_data['name'], resume_data['email'], str(resume_score), timestamp, str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']), str(recommended_skills), str(rec_course), pdf_name)

                    ## Recommending Resume Writing Video
                    st.header("**Bonus Video for Resume Writing Tips💡**")
                    resume_vid = random.choice(resume_videos)
                    st.video(resume_vid)

                    ## Recommending Interview Preparation Video
                    st.header("**Bonus Video for Interview Tips💡**")
                    interview_vid = random.choice(interview_videos)
                    st.video(interview_vid)

                    ## On Successful Result 
                    st.balloons()

            else:
                    st.error('Something went wrong..')                


run()
   


    