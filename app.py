import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Placement Predictor AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
# NOTE: Primary colors, text colors, and slider fill are controlled via
#       .streamlit/config.toml — DO NOT override those here.
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap"
      rel="stylesheet">

<style>
/* ── Font ──────────────────────────────────────────────── */
html, body, * {
    font-family: 'Inter', sans-serif !important;
}

/* ── Layout ────────────────────────────────────────────── */
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
footer, #MainMenu         { visibility: hidden !important; }
.block-container          { padding: 2rem 3.5rem !important; max-width: 1280px !important; }

::-webkit-scrollbar       { width: 6px; }
::-webkit-scrollbar-track { background: #f0f4ff; }
::-webkit-scrollbar-thumb { background: #a5b4fc; border-radius: 3px; }

/* ── Labels ────────────────────────────────────────────── */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] > div {
    font-size: 13.5px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
}

/* Slider: NO custom CSS here — all slider theming via config.toml */

/* ── Select Box ─────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    border-radius: 11px !important;
    border-width: 1.5px !important;
}
[data-baseweb="popover"] ul,
[data-baseweb="menu"]    ul {
    background: #ffffff !important;
    border: 1.5px solid #e0e7ff !important;
    border-radius: 13px !important;
    box-shadow: 0 10px 40px rgba(79,70,229,0.13) !important;
}

/* ── Number Input ───────────────────────────────────────── */
[data-testid="stNumberInput"] input {
    border-radius: 11px !important;
    border-width: 1.5px !important;
    font-weight: 500 !important;
}
[data-testid="stNumberInput"] button {
    border-radius: 8px !important;
}

/* ═══════════════════════════════════════════════════════════
   CUSTOM COMPONENTS
   ═══════════════════════════════════════════════════════════ */

/* Hero */
.hero {
    background: linear-gradient(135deg, #eef2ff 0%, #ede9fe 50%, #e0f2fe 100%);
    border: 1.5px solid rgba(99,102,241,0.22);
    border-radius: 28px;
    padding: 56px 48px 52px;
    margin-bottom: 36px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(99,102,241,0.10);
}
.hero::before {
    content: ''; position: absolute; border-radius: 50%;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(99,102,241,0.13) 0%, transparent 70%);
    top: -140px; right: -100px; pointer-events: none;
}
.hero::after {
    content: ''; position: absolute; border-radius: 50%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(56,189,248,0.10) 0%, transparent 70%);
    bottom: -80px; left: -80px; pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.13);
    border: 1px solid rgba(99,102,241,0.38);
    color: #4f46e5;
    font-size: 11px; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    padding: 6px 20px; border-radius: 999px;
    margin-bottom: 22px;
}
.hero-title {
    font-size: 54px; font-weight: 900; line-height: 1.12;
    background: linear-gradient(135deg, #3730a3 0%, #4f46e5 45%, #0ea5e9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 16px;
}
.hero-sub {
    font-size: 17px; color: #6b7280; font-weight: 400;
    max-width: 540px; margin: 0 auto; line-height: 1.75;
}

/* Section Cards */
.card {
    background: #ffffff;
    border: 1.5px solid rgba(99,102,241,0.16);
    border-radius: 22px;
    padding: 26px 28px 10px;
    margin-bottom: 6px;
    box-shadow: 0 4px 24px rgba(99,102,241,0.07), 0 1px 4px rgba(0,0,0,0.03);
    transition: box-shadow 0.25s, border-color 0.25s;
}
.card:hover {
    box-shadow: 0 8px 40px rgba(99,102,241,0.12);
    border-color: rgba(99,102,241,0.30);
}
.card-header {
    display: flex; align-items: center; gap: 12px;
    padding-bottom: 18px; border-bottom: 1.5px solid #f1f5f9;
    margin-bottom: 4px;
}
.card-icon {
    width: 40px; height: 40px; border-radius: 12px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    display: flex; align-items: center; justify-content: center;
    font-size: 19px; line-height: 1;
    box-shadow: 0 4px 12px rgba(79,70,229,0.28);
}
.card-title { font-size: 16px; font-weight: 700; color: #1e1b4b; }

/* Divider */
.divider {
    height: 1.5px;
    background: linear-gradient(90deg, transparent, #c7d2fe, transparent);
    margin: 32px 0;
}

/* Predict Button */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #6d28d9 55%, #2563eb 100%) !important;
    color: #ffffff !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    padding: 16px 40px !important;
    border: none !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(79,70,229,0.35) !important;
    transition: all 0.28s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 14px 44px rgba(79,70,229,0.50) !important;
    filter: brightness(1.07) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Result Cards */
.result-placed {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 55%, #e0f2fe 100%);
    border: 1.5px solid rgba(16,185,129,0.30);
    border-radius: 24px; padding: 46px 40px;
    text-align: center;
    box-shadow: 0 10px 44px rgba(16,185,129,0.13);
}
.result-not-placed {
    background: linear-gradient(135deg, #fff1f2 0%, #fecdd3 55%, #fff7ed 100%);
    border: 1.5px solid rgba(239,68,68,0.28);
    border-radius: 24px; padding: 46px 40px;
    text-align: center;
    box-shadow: 0 10px 44px rgba(239,68,68,0.10);
}
.result-emoji  { font-size: 56px; margin-bottom: 10px; }
.result-badge  {
    display: inline-block; border-radius: 999px;
    font-size: 11px; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; padding: 5px 16px; margin-bottom: 16px;
}
.badge-green { background: rgba(16,185,129,0.15); color: #065f46; }
.badge-red   { background: rgba(239,68,68,0.12);  color: #991b1b; }
.result-salary {
    font-size: 70px; font-weight: 900; line-height: 1; margin-bottom: 8px;
    background: linear-gradient(135deg, #065f46, #0d9488, #0369a1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.result-unit     { font-size: 15px; font-weight: 600; color: #047857; }
.result-not-msg  { font-size: 30px; font-weight: 800; color: #991b1b; margin-bottom: 6px; }
.result-not-sub  { font-size: 15px; color: #b91c1c; }
.tip-box {
    background: #eef2ff; border: 1.5px solid rgba(99,102,241,0.22);
    border-radius: 14px; padding: 16px 20px; margin-top: 18px;
    font-size: 14px; color: #3730a3; line-height: 1.65;
    display: flex; align-items: flex-start; gap: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODELS ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    pm = joblib.load('placement_model.pkl')
    sm = joblib.load('salary_model.pkl')
    sc = joblib.load('scaler.pkl')
    return pm, sm, sc

placement_model, salary_model, scaler = load_models()


# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI Powered &nbsp;·&nbsp; ML Prediction</div>
    <div class="hero-title">Placement Predictor AI</div>
    <div class="hero-sub">
        Fill in your academic profile and activities to get an instant AI prediction
        of your placement outcome and expected salary package.
    </div>
</div>
""", unsafe_allow_html=True)


# ─── INPUTS ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">📘</div>
            <div class="card-title">Academic Profile</div>
        </div>
    </div>""", unsafe_allow_html=True)

    gender  = st.selectbox("Gender",  ["Male", "Female"])
    branch  = st.selectbox("Branch",  ["CSE", "IT", "ECE"])
    st.write("")
    cgpa    = st.slider("CGPA",    0.0, 10.0,  7.5, 0.1,  help="Cumulative GPA out of 10.0")
    tenth   = st.slider("10th %",  40.0, 100.0, 75.0, 0.5)
    twelfth = st.slider("12th %",  40.0, 100.0, 72.0, 0.5)
    st.write("")
    backlogs      = st.number_input("Active Backlogs",       0, 10, 0)
    coding        = st.slider("Coding Skill",        1, 10, 6, help="Self-rated 1–10")
    communication = st.slider("Communication Skill", 1, 10, 6)
    aptitude      = st.slider("Aptitude Skill",      1, 10, 6)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">💼</div>
            <div class="card-title">Activities &amp; Lifestyle</div>
        </div>
    </div>""", unsafe_allow_html=True)

    study_hours = st.slider("Study Hours / Day", 0.0, 12.0, 4.0, 0.5)
    attendance  = st.slider("Attendance %",      50.0, 100.0, 80.0, 0.5)
    sleep       = st.slider("Sleep Hours / Day", 3.0, 12.0,  7.0, 0.5)
    stress      = st.slider("Stress Level",      1, 10, 4, help="1 = Minimal · 10 = Extreme")
    st.write("")
    projects       = st.number_input("Projects Completed", 0, 20,  2)
    internships    = st.number_input("Internships",        0, 10,  1)
    hackathons     = st.number_input("Hackathons",         0, 10,  0)
    certifications = st.number_input("Certifications",     0, 20,  1)
    st.write("")
    part_time = st.selectbox("Part-time Job",       ["No", "Yes"])
    income    = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
    city      = st.selectbox("City Tier",           ["Tier 3", "Tier 2", "Tier 1"])
    internet  = st.selectbox("Internet Access",     ["No", "Yes"])
    extra     = st.selectbox("Extracurricular",     ["Low", "Medium", "High"])


# ─── ENCODE ────────────────────────────────────────────────────────────────────
gender_enc    = 1 if gender    == "Male"   else 0
branch_enc    = {"CSE": 2, "IT": 1, "ECE": 3}[branch]
part_time_enc = 1 if part_time == "Yes"    else 0
internet_enc  = 1 if internet  == "Yes"    else 0
income_enc    = {"Low": 1, "Medium": 2, "High": 3}[income]
city_enc      = {"Tier 3": 0, "Tier 2": 1, "Tier 1": 2}[city]
extra_enc     = {"Low": 0, "Medium": 1, "High": 2}[extra]


# ─── PREDICT ───────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("🚀  Predict My Placement", use_container_width=True)

if predict:
    input_dict = {
        'Student_ID':                  0,
        'gender':                      gender_enc,
        'branch':                      branch_enc,
        'cgpa':                        cgpa,
        'tenth_percentage':            tenth,
        'twelfth_percentage':          twelfth,
        'backlogs':                    backlogs,
        'study_hours_per_day':         study_hours,
        'attendance_percentage':       attendance,
        'projects_completed':          projects,
        'internships_completed':       internships,
        'coding_skill_rating':         coding,
        'communication_skill_rating':  communication,
        'aptitude_skill_rating':       aptitude,
        'hackathons_participated':     hackathons,
        'certifications_count':        certifications,
        'sleep_hours':                 sleep,
        'stress_level':                stress,
        'part_time_job':               part_time_enc,
        'family_income_level':         income_enc,
        'city_tier':                   city_enc,
        'internet_access':             internet_enc,
        'extracurricular_involvement': extra_enc,
    }

    input_df     = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    placement    = placement_model.predict(input_scaled)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    _, res_col, _ = st.columns([1, 2, 1])

    with res_col:
        if placement[0] == 1:
            salary = np.expm1(salary_model.predict(input_scaled))[0]
            st.markdown(f"""
            <div class="result-placed">
                <div class="result-emoji">🎉</div>
                <div class="result-badge badge-green">Placement Predicted ✓</div>
                <div class="result-salary">{salary:.2f}</div>
                <div class="result-unit">LPA &nbsp;·&nbsp; Expected Starting Package</div>
            </div>
            <div class="tip-box">
                💡 <span><strong>Pro tip:</strong> Aim for a CGPA above
                <strong>{min(cgpa + 0.5, 10.0):.1f}</strong> and secure one more internship
                to potentially boost your package by 10–15%.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-not-placed">
                <div class="result-emoji">⚠️</div>
                <div class="result-badge badge-red">Low Placement Probability</div>
                <div class="result-not-msg">Keep Improving!</div>
                <div class="result-not-sub">Your profile needs a little more work — see the tip below.</div>
            </div>
            <div class="tip-box">
                💡 <span><strong>Focus areas:</strong> Clear active backlogs, improve CGPA,
                complete at least one internship, and sharpen your coding &amp;
                communication skills.</span>
            </div>
            """, unsafe_allow_html=True)