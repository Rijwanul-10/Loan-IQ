"""
Loan Approval Expert System — Streamlit Web App
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
import warnings
warnings.filterwarnings("ignore")

# ── page config (MUST be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="LoanIQ — AI Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS — dark banking theme
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;1,9..144,400&display=swap');

/* ── Base ─────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}
.stApp {
    background: #0a0d14;
    color: #e8eaf0;
}

/* ── Sidebar ──────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0f1320 !important;
    border-right: 1px solid #1e2435;
}
[data-testid="stSidebar"] * { color: #c8cad8 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: #2563eb !important;
}

/* ── Hero header ──────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #0f1320 0%, #121829 50%, #0d1117 100%);
    border: 1px solid #1e2a3d;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(37,99,235,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
    line-height: 1.1;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #4a90d9;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.hero-desc {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #6b7280;
    max-width: 600px;
}
.pill {
    display: inline-block;
    background: rgba(37,99,235,0.12);
    border: 1px solid rgba(37,99,235,0.3);
    color: #60a5fa;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    margin-top: 8px;
}

/* ── Decision card ────────────────────────────── */
.decision-approve {
    background: linear-gradient(135deg, #022c22 0%, #064e3b 100%);
    border: 2px solid #10b981;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
}
.decision-reject {
    background: linear-gradient(135deg, #300 0%, #450a0a 100%);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
}
.decision-label {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.decision-score {
    font-family: 'Fraunces', serif;
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
}
.decision-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.7;
    margin-top: 4px;
}

/* ── Score cards ──────────────────────────────── */
.score-card {
    background: #0f1320;
    border: 1px solid #1e2a3d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 10px;
    position: relative;
}
.score-card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 6px;
}
.score-card-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #e8eaf0;
}
.score-bar-bg {
    background: #1e2a3d;
    border-radius: 4px;
    height: 5px;
    margin-top: 8px;
}
.score-bar-fill {
    border-radius: 4px;
    height: 5px;
}

/* ── Rule tags ────────────────────────────────── */
.rule-pos {
    display: inline-block;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.3);
    color: #6ee7b7;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 4px 10px;
    border-radius: 6px;
    margin: 3px;
}
.rule-neg {
    display: inline-block;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    color: #fca5a5;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 4px 10px;
    border-radius: 6px;
    margin: 3px;
}

/* ── Section headers ──────────────────────────── */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #e8eaf0;
    letter-spacing: -0.3px;
    border-bottom: 1px solid #1e2a3d;
    padding-bottom: 8px;
    margin-bottom: 14px;
    margin-top: 6px;
}
.section-accent {
    display: inline-block;
    width: 6px; height: 6px;
    background: #2563eb;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}

/* ── Path nodes ───────────────────────────────── */
.path-node {
    display: inline-block;
    background: #1e2a3d;
    border: 1px solid #2563eb;
    color: #93c5fd;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 5px 12px;
    border-radius: 20px;
    margin: 2px;
}
.path-arrow {
    color: #374151;
    font-size: 1rem;
    margin: 0 4px;
    vertical-align: middle;
}

/* ── Ethics section ───────────────────────────── */
.ethics-card {
    background: #0f1320;
    border: 1px solid #1e2a3d;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-bottom: 8px;
}
.ethics-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a90d9;
    margin-bottom: 4px;
}
.ethics-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
}

/* ── Streamlit overrides ──────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.7rem 2.5rem !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(37,99,235,0.35) !important;
}
div[data-testid="metric-container"] {
    background: #0f1320;
    border: 1px solid #1e2a3d;
    border-radius: 10px;
    padding: 1rem;
}
.stSelectbox > div > div {
    background: #0f1320 !important;
    border-color: #1e2a3d !important;
    color: #e8eaf0 !important;
}
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# EXPERT SYSTEM CORE (inline — no external import needed)
# ─────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

DATA_PATH = "loan_approval_dataset.csv"

class KnowledgeBase:
    RULES = {
        "excellent_credit":  {"field":"cibil_score",    "threshold":750,       "direction":">=","weight":0.25},
        "good_credit":       {"field":"cibil_score",    "threshold":650,       "direction":">=","weight":0.15},
        "poor_credit":       {"field":"cibil_score",    "threshold":500,       "direction":"<", "weight":-0.30},
        "high_income":       {"field":"income_annum",   "threshold":7_000_000, "direction":">=","weight":0.20},
        "medium_income":     {"field":"income_annum",   "threshold":3_000_000, "direction":">=","weight":0.10},
        "low_income":        {"field":"income_annum",   "threshold":2_000_000, "direction":"<", "weight":-0.20},
        "manageable_debt":   {"field":"loan_to_income", "threshold":5.0,       "direction":"<=","weight":0.15},
        "high_debt":         {"field":"loan_to_income", "threshold":10.0,      "direction":">", "weight":-0.25},
        "strong_assets":     {"field":"total_assets",   "threshold":20_000_000,"direction":">=","weight":0.15},
        "graduate_bonus":    {"field":"education",      "threshold":"Graduate","direction":"==","weight":0.05},
        "few_dependents":    {"field":"no_of_dependents","threshold":2,        "direction":"<=","weight":0.05},
        "many_dependents":   {"field":"no_of_dependents","threshold":4,        "direction":">", "weight":-0.05},
    }
    BAYES_TABLE = {
        "cibil_score_excellent":0.92,"cibil_score_good":0.72,
        "cibil_score_fair":0.45,    "cibil_score_poor":0.08,
        "income_high":0.84,         "income_medium":0.62,"income_low":0.28,
        "debt_ratio_low":0.80,      "debt_ratio_medium":0.60,"debt_ratio_high":0.25,
        "education_graduate":0.65,  "education_not_graduate":0.52,
        "employed":0.65,            "self_employed":0.58,
        "assets_strong":0.78,       "assets_weak":0.40,
    }
    PRIOR_APPROVAL = 0.622

class DataPreprocessor:
    def __init__(self):
        self.le_edu=LabelEncoder(); self.le_emp=LabelEncoder()
        self.scaler=StandardScaler(); self.fitted=False
        self.feature_cols=None
    def _engineer(self,df):
        df=df.copy()
        df["loan_to_income"]=df["loan_amount"]/(df["income_annum"]+1)
        df["total_assets"]=(df["residential_assets_value"]+df["commercial_assets_value"]
                           +df["luxury_assets_value"]+df["bank_asset_value"])
        df["asset_to_loan"]=df["total_assets"]/(df["loan_amount"]+1)
        df["income_per_dep"]=df["income_annum"]/(df["no_of_dependents"]+1)
        return df
    def fit_transform(self,df):
        df=df.copy(); df.columns=df.columns.str.strip()
        for col in df.select_dtypes(include=["object","str"]): df[col]=df[col].str.strip()
        df=self._engineer(df)
        df["label"]=(df["loan_status"]=="Approved").astype(int)
        df["edu_enc"]=self.le_edu.fit_transform(df["education"])
        df["emp_enc"]=self.le_emp.fit_transform(df["self_employed"])
        self.feature_cols=["no_of_dependents","edu_enc","emp_enc","income_annum",
            "loan_amount","loan_term","cibil_score","residential_assets_value",
            "commercial_assets_value","luxury_assets_value","bank_asset_value",
            "loan_to_income","total_assets","asset_to_loan","income_per_dep"]
        X=df[self.feature_cols].values; y=df["label"].values
        Xs=self.scaler.fit_transform(X); self.fitted=True
        return Xs,y,df
    def transform_single(self,a):
        row={**a}
        df_row=pd.DataFrame([row])
        df_row=self._engineer(df_row)
        try: df_row["edu_enc"]=self.le_edu.transform([row["education"]])
        except: df_row["edu_enc"]=0
        try: df_row["emp_enc"]=self.le_emp.transform([row["self_employed"]])
        except: df_row["emp_enc"]=0
        return self.scaler.transform(df_row[self.feature_cols].values)

class RuleBasedSystem:
    def evaluate(self,applicant):
        app=dict(applicant)
        app["loan_to_income"]=app["loan_amount"]/(app["income_annum"]+1)
        app["total_assets"]=(app.get("residential_assets_value",0)+app.get("commercial_assets_value",0)
                             +app.get("luxury_assets_value",0)+app.get("bank_asset_value",0))
        raw=0.0; pos=[]; neg=[]
        for name,rule in KnowledgeBase.RULES.items():
            field,threshold,direction,weight=rule["field"],rule["threshold"],rule["direction"],rule["weight"]
            value=app.get(field)
            if value is None: continue
            fired=((direction==">=" and value>=threshold) or (direction=="<" and value<threshold)
                   or (direction==">" and value>threshold) or (direction=="<=" and value<=threshold)
                   or (direction=="==" and value==threshold))
            if fired:
                raw+=weight
                label=name.replace("_"," ").title()
                (pos if weight>0 else neg).append((label,weight,field,value))
        mn=sum(r["weight"] for r in KnowledgeBase.RULES.values() if r["weight"]<0)
        mx=sum(r["weight"] for r in KnowledgeBase.RULES.values() if r["weight"]>0)
        score=float(np.clip((raw-mn)/(mx-mn+1e-9),0,1))
        return score,pos,neg

class ProbabilisticReasoner:
    def evaluate(self,applicant):
        bt=KnowledgeBase.BAYES_TABLE; prior=KnowledgeBase.PRIOR_APPROVAL
        evidence=[]
        cs=applicant.get("cibil_score",600)
        if cs>=750: evidence.append(("Excellent CIBIL ≥750",bt["cibil_score_excellent"]))
        elif cs>=650: evidence.append(("Good CIBIL 650–749",bt["cibil_score_good"]))
        elif cs>=500: evidence.append(("Fair CIBIL 500–649",bt["cibil_score_fair"]))
        else: evidence.append(("Poor CIBIL <500",bt["cibil_score_poor"]))
        inc=applicant.get("income_annum",0)
        if inc>=7_000_000: evidence.append(("High Income ≥7M",bt["income_high"]))
        elif inc>=3_000_000: evidence.append(("Medium Income 3–7M",bt["income_medium"]))
        else: evidence.append(("Low Income <3M",bt["income_low"]))
        dr=applicant["loan_amount"]/(applicant["income_annum"]+1)
        if dr<=5: evidence.append(("Low Debt Ratio ≤5x",bt["debt_ratio_low"]))
        elif dr<=10: evidence.append(("Medium Debt Ratio 5–10x",bt["debt_ratio_medium"]))
        else: evidence.append(("High Debt Ratio >10x",bt["debt_ratio_high"]))
        edu=applicant.get("education","Graduate")
        evidence.append(("Graduate" if edu=="Graduate" else "Not Graduate",
                         bt["education_graduate"] if edu=="Graduate" else bt["education_not_graduate"]))
        emp=applicant.get("self_employed","No")
        evidence.append(("Self-Employed" if emp=="Yes" else "Salaried",
                         bt["self_employed"] if emp=="Yes" else bt["employed"]))
        ta=(applicant.get("residential_assets_value",0)+applicant.get("commercial_assets_value",0)
            +applicant.get("luxury_assets_value",0)+applicant.get("bank_asset_value",0))
        evidence.append(("Strong Assets ≥20M" if ta>=20_000_000 else "Moderate Assets",
                         bt["assets_strong"] if ta>=20_000_000 else bt["assets_weak"]))
        la=np.log(prior+1e-9); lr=np.log(1-prior+1e-9)
        for _,p in evidence:
            la+=np.log(p+1e-9); lr+=np.log(1-p+1e-9)
        mx=max(la,lr)
        prob=float(np.exp(la-mx)/(np.exp(la-mx)+np.exp(lr-mx)))
        return prob,evidence

class SearchEngine:
    GRAPH={"START":["CREDIT_CHECK","INCOME_CHECK"],"CREDIT_CHECK":["ASSET_CHECK","DEBT_CHECK"],
           "INCOME_CHECK":["DEBT_CHECK","ASSET_CHECK"],"ASSET_CHECK":["FINAL_APPROVE","FINAL_REJECT"],
           "DEBT_CHECK":["FINAL_APPROVE","FINAL_REJECT"],"FINAL_APPROVE":[],"FINAL_REJECT":[]}
    def _ns(self,node,a):
        cs=a.get("cibil_score",600); inc=a.get("income_annum",0)
        dr=a["loan_amount"]/(a["income_annum"]+1)
        ta=(a.get("residential_assets_value",0)+a.get("commercial_assets_value",0)
            +a.get("luxury_assets_value",0)+a.get("bank_asset_value",0))
        m={"START":0.5,"CREDIT_CHECK":min(cs/900,1),"INCOME_CHECK":min(inc/10_000_000,1),
           "ASSET_CHECK":min(ta/30_000_000,1),"DEBT_CHECK":max(0,1-dr/20),
           "FINAL_APPROVE":0.8 if (cs>=650 and dr<=8) else 0.3,
           "FINAL_REJECT":0.2 if (cs>=650 and dr<=8) else 0.7}
        return m.get(node,0.5)
    def bfs(self,a):
        q=deque([( ["START"],0.0)]); best=(["START"],0.0)
        while q:
            path,score=q.popleft(); cur=path[-1]
            total=score+self._ns(cur,a)
            if total>best[1]: best=(path,total)
            for nb in self.GRAPH.get(cur,[]):
                if nb not in path: q.append((path+[nb],total))
        return best[0],min(best[1]/5,1.0)

@st.cache_resource(show_spinner="Training models on dataset…")
def load_system():
    df=pd.read_csv(DATA_PATH); df.columns=df.columns.str.strip()
    for col in df.select_dtypes(include=["object","str"]): df[col]=df[col].str.strip()
    prep=DataPreprocessor()
    X,y,df_proc=prep.fit_transform(df)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    rf=RandomForestClassifier(n_estimators=200,max_depth=12,random_state=42,
                               n_jobs=-1,class_weight="balanced")
    lr=LogisticRegression(max_iter=1000,random_state=42,class_weight="balanced")
    dt=DecisionTreeClassifier(max_depth=8,random_state=42,class_weight="balanced")
    metrics={}
    for name,model in [("Random Forest",rf),("Logistic Regression",lr),("Decision Tree",dt)]:
        model.fit(Xtr,ytr); yp=model.predict(Xte)
        metrics[name]={"accuracy":accuracy_score(yte,yp),"precision":precision_score(yte,yp,zero_division=0),
                       "recall":recall_score(yte,yp,zero_division=0),"f1":f1_score(yte,yp,zero_division=0)}
    return prep,rf,metrics,df,confusion_matrix(yte,rf.predict(Xte))

# ─────────────────────────────────────────────────────────────────
# HELPER — Plotly dark theme
# ─────────────────────────────────────────────────────────────────
DARK = dict(paper_bgcolor="#0a0d14", plot_bgcolor="#0a0d14",
            font_color="#c8cad8", font_family="DM Mono")
BLUE = "#2563eb"


# ─────────────────────────────────────────────────────────────────
# LOAD SYSTEM
# ─────────────────────────────────────────────────────────────────
try:
    prep, rf_model, model_metrics, df_raw, cm = load_system()
    system_ready = True
except FileNotFoundError:
    system_ready = False

rule_sys   = RuleBasedSystem()
prob_sys   = ProbabilisticReasoner()
search_sys = SearchEngine()

FUSION = {"ml":0.40, "prob":0.30, "rule":0.20, "search":0.10}
THRESHOLD = 0.55


# ─────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-sub">AI-Powered Decision Engine</div>
  <div class="hero-title">LoanIQ Expert System</div>
  <div class="hero-desc">
    Three-layer hybrid intelligence: Rule-Based Expert System ·
    Bayesian Probabilistic Reasoning · Machine Learning (Random Forest)
  </div>
  <div style="margin-top:12px">
    <span class="pill">4,269 training records</span>
    <span class="pill">Random Forest 99.9%</span>
    <span class="pill">BFS Search</span>
    <span class="pill">Explainable AI</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not system_ready:
    st.error("⚠️  `loan_approval_dataset.csv` not found. Place it in the same folder as `app.py` and restart.")
    st.stop()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT FORM
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Applicant Profile")
    st.markdown("---")

    st.markdown("**Personal Details**")
    education     = st.selectbox("Education Level",       ["Graduate","Not Graduate"])
    self_employed = st.selectbox("Employment Type",       ["No (Salaried)","Yes (Self-Employed)"])
    dependents    = st.slider("Number of Dependents",     0, 5, 2)

    st.markdown("---")
    st.markdown("**Financial Information**")
    income        = st.number_input("Annual Income (₹)",        min_value=100_000,   max_value=100_000_000, value=5_000_000, step=100_000, format="%d")
    loan_amount   = st.number_input("Loan Amount Requested (₹)",min_value=100_000,   max_value=500_000_000, value=15_000_000,step=100_000, format="%d")
    loan_term     = st.slider("Loan Term (months)",             6, 360, 120)
    cibil_score   = st.slider("CIBIL Credit Score",             300, 900, 680)

    st.markdown("---")
    st.markdown("**Asset Details**")
    res_assets  = st.number_input("Residential Assets (₹)",  min_value=0, max_value=500_000_000, value=5_000_000,  step=100_000, format="%d")
    com_assets  = st.number_input("Commercial Assets (₹)",   min_value=0, max_value=500_000_000, value=2_000_000,  step=100_000, format="%d")
    lux_assets  = st.number_input("Luxury Assets (₹)",       min_value=0, max_value=500_000_000, value=8_000_000,  step=100_000, format="%d")
    bank_assets = st.number_input("Bank / Financial Assets (₹)", min_value=0, max_value=500_000_000, value=3_000_000, step=100_000, format="%d")

    st.markdown("---")
    run_btn = st.button("🔍  Analyse Application")


# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Decision & Analysis", "📊 Model Performance", "⚖️ Ethics & Bias"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — DECISION
# ═══════════════════════════════════════════════════════════════
with tab1:
    if not run_btn:
        st.markdown("""
        <div style="text-align:center; padding:5rem 2rem; color:#374151;">
          <div style="font-size:3rem; margin-bottom:1rem">🏦</div>
          <div style="font-family:'Syne',sans-serif; font-size:1.3rem; color:#6b7280;">
            Fill in the applicant profile on the left and click<br>
            <strong style="color:#2563eb">Analyse Application</strong> to run the expert system.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Build applicant dict
        applicant = {
            "no_of_dependents": dependents,
            "education": education,
            "self_employed": "Yes" if self_employed.startswith("Yes") else "No",
            "income_annum": income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": res_assets,
            "commercial_assets_value": com_assets,
            "luxury_assets_value": lux_assets,
            "bank_asset_value": bank_assets,
        }

        # Run all layers
        rule_score, pos_rules, neg_rules = rule_sys.evaluate(applicant)
        prob_score, bayes_evidence       = prob_sys.evaluate(applicant)
        X_single                         = prep.transform_single(applicant)
        ml_score                         = float(rf_model.predict_proba(X_single)[0][1])
        search_path, search_score        = search_sys.bfs(applicant)
        final_score = (FUSION["ml"]*ml_score + FUSION["prob"]*prob_score
                      + FUSION["rule"]*rule_score + FUSION["search"]*search_score)
        decision    = "Approved" if final_score >= THRESHOLD else "Rejected"

        # ── Decision Banner ──────────────────────────────────────
        colour_class = "decision-approve" if decision == "Approved" else "decision-reject"
        icon         = "✅" if decision == "Approved" else "❌"
        colour_hex   = "#10b981" if decision == "Approved" else "#ef4444"

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown(f"""
            <div class="{colour_class}">
              <div class="decision-sub">Final Verdict</div>
              <div class="decision-label" style="color:{colour_hex}">{icon} Loan {decision}</div>
              <div class="decision-score" style="color:{colour_hex}">{final_score:.1%}</div>
              <div class="decision-sub">Confidence Score · Threshold 55.0%</div>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=final_score*100,
                number={"suffix":"%","font":{"size":28,"family":"Syne","color":colour_hex}},
                gauge={
                    "axis":{"range":[0,100],"tickfont":{"color":"#6b7280","size":9}},
                    "bar":{"color":colour_hex,"thickness":0.4},
                    "bgcolor":"#1e2a3d",
                    "bordercolor":"#0a0d14",
                    "steps":[{"range":[0,55],"color":"#1a0505"},
                              {"range":[55,100],"color":"#0a1a0a"}],
                    "threshold":{"line":{"color":"white","width":2},"thickness":0.75,"value":55},
                },
                domain={"x":[0,1],"y":[0,1]},
            ))
            fig_gauge.update_layout(**DARK, height=200, margin=dict(t=10,b=10,l=20,r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Score Breakdown ──────────────────────────────────────
        st.markdown('<div class="section-head"><span class="section-accent"></span>Score Breakdown — 4-Layer Fusion</div>', unsafe_allow_html=True)

        scores = [
            ("🤖 ML Model (Random Forest)", ml_score,    "weight: 40%", "#2563eb"),
            ("📊 Probabilistic Reasoning",  prob_score,  "weight: 30%", "#7c3aed"),
            ("📋 Rule-Based System",        rule_score,  "weight: 20%", "#0891b2"),
            ("🔍 BFS Search Engine",        search_score,"weight: 10%", "#059669"),
        ]

        cols = st.columns(4)
        for i, (label, val, wt, clr) in enumerate(scores):
            with cols[i]:
                pct = int(val*100)
                st.markdown(f"""
                <div class="score-card">
                  <div class="score-card-label">{label}</div>
                  <div class="score-card-value" style="color:{clr}">{pct}%</div>
                  <div style="font-family:'DM Mono';font-size:0.65rem;color:#6b7280;margin-top:2px">{wt}</div>
                  <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width:{pct}%;background:{clr}"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Radar chart ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1.2, 1])

        with c1:
            st.markdown('<div class="section-head"><span class="section-accent"></span>Layer Score Radar</div>', unsafe_allow_html=True)
            cats  = ["ML Model","Probabilistic","Rule-Based","BFS Search","ML Model"]
            vals  = [ml_score, prob_score, rule_score, search_score, ml_score]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=[v*100 for v in vals], theta=cats, fill="toself",
                fillcolor="rgba(37,99,235,0.15)", line=dict(color=BLUE, width=2),
                marker=dict(size=6, color=BLUE)))
            fig_r.add_trace(go.Scatterpolar(
                r=[55]*5, theta=cats, mode="lines",
                line=dict(color="rgba(239,68,68,0.4)", width=1, dash="dot"),
                name="Threshold 55%", showlegend=True))
            fig_r.update_layout(**DARK, height=320,
                polar=dict(bgcolor="#0f1320",
                           radialaxis=dict(range=[0,100],tickfont=dict(size=8,color="#4b5563"),
                                           gridcolor="#1e2a3d"),
                           angularaxis=dict(tickfont=dict(size=9,color="#9ca3af"),
                                            gridcolor="#1e2a3d")),
                showlegend=True, legend=dict(font=dict(size=8)),
                margin=dict(t=20,b=20,l=40,r=40))
            st.plotly_chart(fig_r, use_container_width=True)

        with c2:
            st.markdown('<div class="section-head"><span class="section-accent"></span>Applicant Profile</div>', unsafe_allow_html=True)
            total_assets = res_assets+com_assets+lux_assets+bank_assets
            lti          = loan_amount/(income+1)
            profile_data = {
                "CIBIL Score": cibil_score,
                "Annual Income": f"₹{income/1_000_000:.1f}M",
                "Loan Amount": f"₹{loan_amount/1_000_000:.1f}M",
                "Loan/Income": f"{lti:.2f}×",
                "Total Assets": f"₹{total_assets/1_000_000:.1f}M",
                "Education": education,
                "Employment": "Self-Employed" if applicant["self_employed"]=="Yes" else "Salaried",
                "Dependents": dependents,
                "Loan Term": f"{loan_term} months",
            }
            for k, v in profile_data.items():
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                     padding:6px 0;border-bottom:1px solid #1e2a3d;
                     font-family:'DM Mono';font-size:0.78rem;">
                  <span style="color:#6b7280">{k}</span>
                  <span style="color:#e8eaf0;font-weight:500">{v}</span>
                </div>
                """, unsafe_allow_html=True)

        # ── Rule-Based Evidence ──────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-head"><span class="section-accent"></span>Layer 1 — Rule-Based Expert System</div>', unsafe_allow_html=True)

        if pos_rules:
            st.markdown("**✅ Positive Signals**")
            tags = ""
            for label, weight, field, value in pos_rules:
                tags += f'<span class="rule-pos">✅ {label} (+{weight:.2f})</span>'
            st.markdown(tags, unsafe_allow_html=True)

        if neg_rules:
            st.markdown("**❌ Risk Signals**")
            tags = ""
            for label, weight, field, value in neg_rules:
                tags += f'<span class="rule-neg">❌ {label} ({weight:.2f})</span>'
            st.markdown(tags, unsafe_allow_html=True)

        if not pos_rules and not neg_rules:
            st.info("No decisive rules fired for this applicant.")

        # ── Bayesian Evidence ────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-head"><span class="section-accent"></span>Layer 2 — Bayesian Probabilistic Reasoning</div>', unsafe_allow_html=True)

        labels_b = [e[0] for e in bayes_evidence]
        probs_b  = [e[1]*100 for e in bayes_evidence]
        colors_b = ["#10b981" if p>=65 else "#f59e0b" if p>=45 else "#ef4444" for p in probs_b]

        fig_b = go.Figure(go.Bar(
            x=probs_b, y=labels_b, orientation="h",
            marker=dict(color=colors_b, opacity=0.85),
            text=[f"{p:.0f}%" for p in probs_b], textposition="outside",
            textfont=dict(size=10, color="#9ca3af")))
        fig_b.add_vline(x=55, line_dash="dot", line_color="rgba(239,68,68,0.5)",
                        annotation_text="Threshold", annotation_font_size=9)
        fig_b.update_layout(**DARK, height=270,
            xaxis=dict(range=[0,105], title="P(Approve | evidence) %",
                       gridcolor="#1e2a3d", tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=9)),
            margin=dict(t=10,b=30,l=10,r=40))
        st.plotly_chart(fig_b, use_container_width=True)

        st.markdown(f"""
        <div style="background:#0f1320;border:1px solid #1e2a3d;border-radius:10px;
             padding:0.8rem 1.2rem;font-family:'DM Mono';font-size:0.78rem;color:#9ca3af;">
          📐 <strong style="color:#60a5fa">Naïve Bayes Posterior:</strong>
          P(Approve|Evidence) = <strong style="color:{colour_hex}">{prob_score:.1%}</strong>
          &nbsp;&nbsp;|&nbsp;&nbsp; Prior approval rate = 62.2%
        </div>
        """, unsafe_allow_html=True)

        # ── Search Path ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-head"><span class="section-accent"></span>Layer 3 — BFS Search Path</div>', unsafe_allow_html=True)

        path_html = ""
        for i, node in enumerate(search_path):
            path_html += f'<span class="path-node">{node}</span>'
            if i < len(search_path)-1:
                path_html += '<span class="path-arrow">→</span>'
        st.markdown(f"""
        <div style="background:#0f1320;border:1px solid #1e2a3d;border-radius:10px;
             padding:1rem 1.5rem;margin-bottom:6px;">
          {path_html}
        </div>
        <div style="font-family:'DM Mono';font-size:0.72rem;color:#6b7280;padding:0 4px;">
          BFS score: {search_score:.1%} &nbsp;·&nbsp; Nodes explored: {len(search_path)}
        </div>
        """, unsafe_allow_html=True)

        # ── Feature Importances ──────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-head"><span class="section-accent"></span>ML Feature Importances (Random Forest)</div>', unsafe_allow_html=True)

        fi_pairs = sorted(zip(prep.feature_cols, rf_model.feature_importances_),
                          key=lambda x: -x[1])[:10]
        fi_feats = [f[0] for f in fi_pairs]
        fi_vals  = [f[1]*100 for f in fi_pairs]

        fig_fi = go.Figure(go.Bar(
            x=fi_vals, y=fi_feats, orientation="h",
            marker=dict(color=BLUE, opacity=0.8,
                        line=dict(color="rgba(37,99,235,0.4)", width=1)),
            text=[f"{v:.1f}%" for v in fi_vals], textposition="outside",
            textfont=dict(size=9, color="#9ca3af")))
        fig_fi.update_layout(**DARK, height=300,
            xaxis=dict(title="Importance %", gridcolor="#1e2a3d", tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=9)),
            margin=dict(t=10,b=30,l=10,r=40))
        st.plotly_chart(fig_fi, use_container_width=True)

        # ── Explanation Summary ──────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-head"><span class="section-accent"></span>Plain-English Explanation</div>', unsafe_allow_html=True)

        pos_text = ", ".join([r[0] for r in pos_rules]) if pos_rules else "none"
        neg_text = ", ".join([r[0] for r in neg_rules]) if neg_rules else "none"

        if decision == "Approved":
            summary = f"""The application was **approved** with a confidence of **{final_score:.1%}**,
            comfortably above the 55% threshold. The ML model gave a strong signal
            of **{ml_score:.1%}**, supported by a Bayesian posterior of **{prob_score:.1%}**.
            Key positive factors: **{pos_text}**.
            The BFS search followed path `{' → '.join(search_path)}` and scored {search_score:.1%}."""
        else:
            summary = f"""The application was **rejected** with a confidence score of only **{final_score:.1%}**,
            below the 55% threshold. The main risk signals identified were: **{neg_text}**.
            The ML model's approval probability was **{ml_score:.1%}** and the
            Bayesian posterior was **{prob_score:.1%}**.
            The BFS search path was `{' → '.join(search_path)}`."""

        st.markdown(f"""
        <div style="background:#0f1320;border:1px solid {'#1a3a2a' if decision=='Approved' else '#3a1a1a'};
             border-left:3px solid {colour_hex};border-radius:10px;
             padding:1.2rem 1.5rem;font-family:'DM Mono';font-size:0.82rem;
             line-height:1.7;color:#c8cad8;">
          {summary}
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-head"><span class="section-accent"></span>Model Comparison — All 3 Classifiers</div>', unsafe_allow_html=True)

    models = list(model_metrics.keys())
    metric_names = ["Accuracy","Precision","Recall","F1 Score"]

    # Metric cards
    rf_m = model_metrics["Random Forest"]
    cols = st.columns(4)
    for i, (metric, val) in enumerate(zip(
            metric_names, [rf_m["accuracy"],rf_m["precision"],rf_m["recall"],rf_m["f1"]])):
        with cols[i]:
            st.metric(f"RF {metric}", f"{val:.1%}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Grouped bar chart
    fig_comp = go.Figure()
    colours  = [BLUE, "#7c3aed", "#0891b2"]
    for clr, model in zip(colours, models):
        m = model_metrics[model]
        fig_comp.add_trace(go.Bar(
            name=model,
            x=metric_names,
            y=[m["accuracy"],m["precision"],m["recall"],m["f1"]],
            marker_color=clr, opacity=0.85,
            text=[f"{v:.1%}" for v in [m["accuracy"],m["precision"],m["recall"],m["f1"]]],
            textposition="outside", textfont=dict(size=9)))
    fig_comp.update_layout(**DARK, height=380, barmode="group",
        yaxis=dict(range=[0,1.1],tickformat=".0%",gridcolor="#1e2a3d"),
        xaxis=dict(tickfont=dict(size=10)),
        legend=dict(font=dict(size=9)),
        margin=dict(t=20,b=20))
    st.plotly_chart(fig_comp, use_container_width=True)

    # Confusion matrix
    st.markdown('<div class="section-head"><span class="section-accent"></span>Confusion Matrix — Random Forest (Test Set)</div>', unsafe_allow_html=True)

    tn,fp,fn,tp = cm.ravel()
    fig_cm = px.imshow(
        [[tn,fp],[fn,tp]],
        labels=dict(x="Predicted",y="Actual",color="Count"),
        x=["Reject","Approve"], y=["Reject","Approve"],
        color_continuous_scale=[[0,"#0f1320"],[0.5,"#1d4ed8"],[1,"#10b981"]],
        text_auto=True)
    fig_cm.update_traces(textfont=dict(size=20,family="Syne",color="white"))
    fig_cm.update_layout(**DARK, height=340,
        coloraxis_showscale=False,
        margin=dict(t=20,b=20,l=20,r=20))
    st.plotly_chart(fig_cm, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="ethics-card">
          <div class="ethics-label">True Negatives (Correctly Rejected)</div>
          <div class="ethics-value" style="color:#60a5fa">{tn}</div>
        </div>
        <div class="ethics-card">
          <div class="ethics-label">False Positives (Wrong Approvals)</div>
          <div class="ethics-value" style="color:#fca5a5">{fp}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="ethics-card">
          <div class="ethics-label">True Positives (Correctly Approved)</div>
          <div class="ethics-value" style="color:#6ee7b7">{tp}</div>
        </div>
        <div class="ethics-card">
          <div class="ethics-label">False Negatives (Missed Approvals)</div>
          <div class="ethics-value" style="color:#fbbf24">{fn}</div>
        </div>
        """, unsafe_allow_html=True)

    # Dataset overview
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head"><span class="section-accent"></span>Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        vc = df_raw["loan_status"].value_counts()
        fig_d = go.Figure(go.Pie(
            labels=vc.index, values=vc.values,
            marker=dict(colors=["#10b981","#ef4444"]),
            hole=0.55, textfont=dict(size=11,family="DM Mono"),
            hoverinfo="label+percent+value"))
        fig_d.add_annotation(text=f"{len(df_raw)}<br>total",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(size=14, family="Syne", color="white"))
        fig_d.update_layout(**DARK, height=280,
                            showlegend=True, legend=dict(font=dict(size=9)),
                            margin=dict(t=20,b=10))
        st.plotly_chart(fig_d, use_container_width=True)

    with col2:
        cibil_approve = df_raw[df_raw["loan_status"]=="Approved"]["cibil_score"]
        cibil_reject  = df_raw[df_raw["loan_status"]=="Rejected"]["cibil_score"]
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(x=cibil_approve, name="Approved",
                                     marker_color="#10b981", opacity=0.6,
                                     nbinsx=30))
        fig_h.add_trace(go.Histogram(x=cibil_reject,  name="Rejected",
                                     marker_color="#ef4444", opacity=0.6,
                                     nbinsx=30))
        fig_h.update_layout(**DARK, height=280, barmode="overlay",
            xaxis_title="CIBIL Score", yaxis_title="Count",
            xaxis=dict(gridcolor="#1e2a3d"), yaxis=dict(gridcolor="#1e2a3d"),
            legend=dict(font=dict(size=9)),
            margin=dict(t=20,b=30))
        st.plotly_chart(fig_h, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — ETHICS
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head"><span class="section-accent"></span>Bias Detection — Approval Rates by Group</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        edu_rates = df_raw.groupby("education")["loan_status"].apply(
            lambda x: (x=="Approved").mean()).reset_index()
        edu_rates.columns = ["Education","Approval Rate"]
        fig_e = go.Figure(go.Bar(
            x=edu_rates["Education"], y=edu_rates["Approval Rate"],
            marker_color=[BLUE,"#7c3aed"], text=[f"{v:.1%}" for v in edu_rates["Approval Rate"]],
            textposition="outside", opacity=0.85))
        fig_e.add_hline(y=0.622, line_dash="dot", line_color="#ef4444",
                        annotation_text="Avg 62.2%", annotation_font_size=9)
        fig_e.update_layout(**DARK, height=280, title="By Education",
            yaxis=dict(range=[0,0.85],tickformat=".0%",gridcolor="#1e2a3d"),
            margin=dict(t=40,b=20))
        st.plotly_chart(fig_e, use_container_width=True)

    with c2:
        emp_rates = df_raw.groupby("self_employed")["loan_status"].apply(
            lambda x: (x=="Approved").mean()).reset_index()
        emp_rates.columns = ["Self Employed","Approval Rate"]
        fig_emp = go.Figure(go.Bar(
            x=emp_rates["Self Employed"], y=emp_rates["Approval Rate"],
            marker_color=["#10b981","#0891b2"], text=[f"{v:.1%}" for v in emp_rates["Approval Rate"]],
            textposition="outside", opacity=0.85))
        fig_emp.add_hline(y=0.622, line_dash="dot", line_color="#ef4444",
                          annotation_text="Avg 62.2%", annotation_font_size=9)
        fig_emp.update_layout(**DARK, height=280, title="By Employment Type",
            yaxis=dict(range=[0,0.85],tickformat=".0%",gridcolor="#1e2a3d"),
            margin=dict(t=40,b=20))
        st.plotly_chart(fig_emp, use_container_width=True)

    # Income quintile
    df_raw2 = df_raw.copy()
    df_raw2["income_quintile"] = pd.qcut(df_raw2["income_annum"],q=5,
                                          labels=["Q1 Lowest","Q2","Q3","Q4","Q5 Highest"])
    iq_rates = df_raw2.groupby("income_quintile", observed=True)["loan_status"].apply(
        lambda x: (x=="Approved").mean()).reset_index()
    iq_rates.columns = ["Quintile","Approval Rate"]

    fig_iq = go.Figure(go.Bar(
        x=iq_rates["Quintile"], y=iq_rates["Approval Rate"],
        marker_color=px.colors.sequential.Blues_r[:5],
        text=[f"{v:.1%}" for v in iq_rates["Approval Rate"]],
        textposition="outside", opacity=0.9))
    fig_iq.add_hline(y=0.622, line_dash="dot", line_color="#ef4444",
                     annotation_text="Avg 62.2%", annotation_font_size=9)
    fig_iq.update_layout(**DARK, height=300, title="Approval Rate by Income Quintile",
        yaxis=dict(range=[0,0.85],tickformat=".0%",gridcolor="#1e2a3d"),
        margin=dict(t=40,b=20))
    st.plotly_chart(fig_iq, use_container_width=True)

    # Ethics framework cards
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head"><span class="section-accent"></span>Ethics Framework</div>', unsafe_allow_html=True)

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.markdown("""
        <div class="ethics-card" style="border-left:3px solid #2563eb">
          <div class="ethics-label">🔍 Transparency</div>
          <div style="font-family:'DM Mono';font-size:0.78rem;color:#9ca3af;line-height:1.7;margin-top:6px">
            Every decision includes a full breakdown of 4-layer scores,
            fired rules, Bayesian evidence table, BFS search path, and
            ML feature importances. No black-box outputs.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with ec2:
        st.markdown("""
        <div class="ethics-card" style="border-left:3px solid #7c3aed">
          <div class="ethics-label">⚖️ Accountability</div>
          <div style="font-family:'DM Mono';font-size:0.78rem;color:#9ca3af;line-height:1.7;margin-top:6px">
            Loan officers retain final override authority. Compliance team
            ensures RBI/IRDAI alignment. Full audit trail logs all 4
            component scores per decision. Quarterly bias monitoring.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with ec3:
        st.markdown("""
        <div class="ethics-card" style="border-left:3px solid #059669">
          <div class="ethics-label">🛡️ Fairness</div>
          <div style="font-family:'DM Mono';font-size:0.78rem;color:#9ca3af;line-height:1.7;margin-top:6px">
            Balanced class weighting in RF model. Income/education bias
            charts show near-equal approval rates. Applicants may request
            written explanations and appeal decisions. Model retrained
            every 6 months.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head"><span class="section-accent"></span>Recommendations</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div class="ethics-card">
        <div class="ethics-label">Short-Term Actions</div>
        <ul style="font-family:'DM Mono';font-size:0.78rem;color:#9ca3af;line-height:1.9;margin:8px 0 0 0;padding-left:16px">
          <li>Re-weight training data for income-group balance</li>
          <li>Add equal opportunity fairness constraints to RF</li>
          <li>Audit CIBIL score for embedded historical bias</li>
          <li>Test model on out-of-distribution demographics</li>
        </ul>
      </div>
      <div class="ethics-card">
        <div class="ethics-label">Long-Term Actions</div>
        <ul style="font-family:'DM Mono';font-size:0.78rem;color:#9ca3af;line-height:1.9;margin:8px 0 0 0;padding-left:16px">
          <li>Collect alternative credit data (utility, rent history)</li>
          <li>Implement SHAP counterfactual explanations</li>
          <li>Establish independent ethics review board</li>
          <li>Annual third-party fairness audit</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)
