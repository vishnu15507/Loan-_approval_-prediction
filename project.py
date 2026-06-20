import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f172a; color: #e2e8f0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #1a2744);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card h3 { color: #94a3b8; font-size: 13px; margin: 0 0 8px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { color: #38bdf8; font-size: 32px; margin: 0; font-weight: 700; }

    /* Algorithm cards */
    .algo-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 10px;
        transition: border-color 0.2s;
    }
    .algo-card:hover { border-color: #38bdf8; }
    .algo-card h4 { color: #38bdf8; margin: 0 0 4px; }
    .algo-card p { color: #94a3b8; font-size: 13px; margin: 0; }

    /* Result boxes */
    .result-high {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: #fca5a5;
        font-size: 20px;
        font-weight: 600;
    }
    .result-low {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 1px solid #22c55e;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: #86efac;
        font-size: 20px;
        font-weight: 600;
    }

    /* Page title */
    .page-title {
        font-size: 28px;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 6px;
    }
    .page-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 28px;
    }

    /* Divider */
    hr { border-color: #334155; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #3b82f6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 14px;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Accuracy badge */
    .acc-badge {
        display: inline-block;
        background: #0c4a6e;
        color: #38bdf8;
        border: 1px solid #0284c7;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 13px;
        font-weight: 600;
    }

    /* Input labels */
    .stTextInput label, .stNumberInput label, .stSelectbox label { color: #cbd5e1 !important; font-size: 13px !important; }

    /* Nav pills in sidebar */
    .nav-item {
        padding: 10px 16px;
        border-radius: 8px;
        margin-bottom: 4px;
        cursor: pointer;
        color: #94a3b8;
        font-size: 14px;
    }
    .nav-item.active {
        background: #1d4ed8;
        color: white;
        font-weight: 600;
    }

    /* Login container */
    .login-box {
        max-width: 400px;
        margin: 80px auto;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for key, val in {
    "logged_in": False,
    "page": "login",
    "trained_models": {},
    "model_accuracies": {},
    "feature_cols": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────
# HELPER: SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
def sidebar_nav():
    with st.sidebar:
        st.markdown("### 🏦 Loan Predict")
        st.markdown("---")

        pages = [
            ("📊", "Dashboard",   "dashboard"),
            ("🤖", "Train Model", "train"),
            ("📝", "User Input",  "predict"),
        ]
        for icon, label, key in pages:
            active_style = "background:#1d4ed8;color:white;font-weight:600;" if st.session_state.page == key else ""
            if st.button(f"{icon}  {label}", key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()

        st.markdown("---")
        if st.button("🚪  Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.session_state.trained_models = {}
            st.session_state.model_accuracies = {}
            st.rerun()


# ─────────────────────────────────────────────
# PAGE 1 — LOGIN
# ─────────────────────────────────────────────
def login_page():
    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown("""
        <div style='text-align:center; padding-top:60px;'>
            <div style='font-size:52px;'>🏦</div>
            <h1 style='color:#f1f5f9; margin:8px 0 4px;'>Loan Default</h1>
            <p style='color:#64748b; margin-bottom:36px;'>Prediction System — Admin Portal</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='background:#1e293b;border:1px solid #334155;border-radius:16px;padding:36px;'>", unsafe_allow_html=True)
        st.markdown("**Username**")
        username = st.text_input("Username", label_visibility="collapsed", placeholder="Enter username")
        st.markdown("**Password**")
        password = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Enter password")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔐  Login", use_container_width=True):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.session_state.page = "dashboard"
                st.rerun()
            else:
                st.error("❌  Invalid credentials. Try admin / admin123")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<p style='text-align:center;color:#475569;margin-top:24px;font-size:12px;'>Default: admin / admin123</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 2 — DASHBOARD
# ─────────────────────────────────────────────
def dashboard_page():
    st.markdown('<div class="page-title">📊 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Overview of the loan dataset and class distribution</div>', unsafe_allow_html=True)

    try:
        df = pd.read_csv("loan_default.csv")

        # ── Metrics row ──
        defaults = int(df["Loan_Default_Risk"].sum()) if "Loan_Default_Risk" in df.columns else 0
        safe     = len(df) - defaults
        rate     = round(defaults / len(df) * 100, 1) if len(df) else 0

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, color in zip(
            [c1, c2, c3, c4],
            ["Total Records", "Default Cases", "Safe Cases", "Default Rate"],
            [len(df), defaults, safe, f"{rate}%"],
            ["#38bdf8", "#f87171", "#4ade80", "#fb923c"]
        ):
            col.markdown(f"""
            <div class="metric-card">
                <h3>{label}</h3>
                <h2 style='color:{color};'>{val}</h2>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Charts ──
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("#### Class Distribution")
            fig, ax = plt.subplots(figsize=(5, 3), facecolor="#1e293b")
            ax.set_facecolor("#1e293b")
            counts = df["Loan_Default_Risk"].value_counts()
            bars = ax.bar(["Safe (0)", "Default (1)"], counts.values,
                          color=["#22c55e", "#ef4444"], width=0.5, edgecolor="none")
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                        str(int(bar.get_height())), ha="center", color="#e2e8f0", fontsize=11)
            ax.tick_params(colors="#94a3b8")
            ax.spines[["top","right","left","bottom"]].set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_title("Default vs Safe Loans", color="#e2e8f0", pad=10)
            plt.tight_layout()
            st.pyplot(fig)

        with ch2:
            if "Annual_Income" in df.columns:
                st.markdown("#### Income Distribution by Risk")
                fig2, ax2 = plt.subplots(figsize=(5, 3), facecolor="#1e293b")
                ax2.set_facecolor("#1e293b")
                for risk, color, label in [(0,"#22c55e","Safe"), (1,"#ef4444","Default")]:
                    subset = df[df["Loan_Default_Risk"] == risk]["Annual_Income"]
                    ax2.hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor="none")
                ax2.tick_params(colors="#94a3b8")
                ax2.spines[["top","right"]].set_visible(False)
                ax2.spines[["left","bottom"]].set_color("#334155")
                ax2.legend(facecolor="#1e293b", labelcolor="#e2e8f0")
                ax2.set_title("Annual Income by Risk Level", color="#e2e8f0", pad=10)
                plt.tight_layout()
                st.pyplot(fig2)

        # ── Data preview ──
        st.markdown("---")
        st.markdown("#### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown(f"<p style='color:#64748b;font-size:13px;'>Showing 10 of {len(df)} rows · {df.shape[1]} columns</p>", unsafe_allow_html=True)

    except FileNotFoundError:
        st.warning("⚠️  `loan_default.csv` not found. Place the dataset file in the same directory.")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")


# ─────────────────────────────────────────────
# PAGE 3 — TRAIN MODEL
# ─────────────────────────────────────────────
ALGORITHMS = {
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=100, random_state=42),
        "icon": "🌲",
        "desc": "Ensemble of decision trees; robust and accurate",
        "file": "model_rf.pkl",
    },
    "SVM": {
        "model": SVC(kernel="rbf", probability=True, random_state=42),
        "icon": "📐",
        "desc": "Finds optimal hyperplane with kernel trick",
        "file": "model_svm.pkl",
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(max_depth=10, random_state=42),
        "icon": "🌿",
        "desc": "Interpretable tree-based splits",
        "file": "model_dt.pkl",
    },
    "XGBoost": {
        "model": XGBClassifier(n_estimators=100, use_label_encoder=False,
                               eval_metric="logloss", random_state=42),
        "icon": "⚡",
        "desc": "Gradient boosting — often top performer",
        "file": "model_xgb.pkl",
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "icon": "📊",
        "desc": "Probabilistic; fast and lightweight",
        "file": "model_nb.pkl",
    },
}


def train_model_page():
    st.markdown('<div class="page-title">🤖 Train Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Select one or more algorithms and train them on the loan dataset</div>', unsafe_allow_html=True)

    # ── Algorithm selector ──
    st.markdown("#### Choose Algorithms")
    algo_cols = st.columns(5)
    selected = []
    for i, (name, info) in enumerate(ALGORITHMS.items()):
        with algo_cols[i]:
            checked = st.checkbox(f"{info['icon']} {name}", value=True, key=f"chk_{name}")
            if checked:
                selected.append(name)
            st.markdown(f"<p style='color:#64748b;font-size:11px;margin-top:-8px;'>{info['desc']}</p>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Train button ──
    if st.button("🚀  Train Selected Models", use_container_width=False):
        if not selected:
            st.warning("Select at least one algorithm.")
            return

        try:
            df = pd.read_csv("loan_default.csv")
        except FileNotFoundError:
            st.error("Dataset `loan_default.csv` not found.")
            return

        le = LabelEncoder()
        for col in df.select_dtypes(include="object").columns:
            df[col] = le.fit_transform(df[col])

        drop_cols = [c for c in ["Applicant_ID", "Loan_Default_Risk"] if c in df.columns]
        X = df.drop(columns=drop_cols)
        y = df["Loan_Default_Risk"]
        st.session_state.feature_cols = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = {}
        progress = st.progress(0, text="Training…")

        for idx, name in enumerate(selected):
            info = ALGORITHMS[name]
            with st.spinner(f"Training {name}…"):
                model = info["model"]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc    = accuracy_score(y_test, y_pred)

                joblib.dump(model, info["file"])
                st.session_state.trained_models[name] = model
                st.session_state.model_accuracies[name] = acc
                results[name] = {"acc": acc, "y_test": y_test, "y_pred": y_pred, "info": info}

            progress.progress((idx + 1) / len(selected), text=f"Done: {name}")

        progress.empty()
        st.success(f"✅  {len(selected)} model(s) trained successfully!")

        # ── Results table ──
        st.markdown("#### Accuracy Comparison")
        res_df = pd.DataFrame([
            {"Algorithm": n, "Accuracy": f"{v['acc']*100:.2f}%", "Score": v['acc']}
            for n, v in results.items()
        ]).sort_values("Score", ascending=False).drop(columns="Score")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        # ── Bar chart ──
        fig, ax = plt.subplots(figsize=(8, 3), facecolor="#1e293b")
        ax.set_facecolor("#1e293b")
        names = list(results.keys())
        accs  = [results[n]["acc"] * 100 for n in names]
        colors = ["#38bdf8" if a == max(accs) else "#475569" for a in accs]
        bars  = ax.barh(names, accs, color=colors, edgecolor="none")
        for bar, acc in zip(bars, accs):
            ax.text(acc - 1, bar.get_y() + bar.get_height()/2,
                    f"{acc:.2f}%", va="center", ha="right", color="white", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 105)
        ax.tick_params(colors="#94a3b8")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_title("Model Accuracy (%)", color="#e2e8f0", pad=10)
        plt.tight_layout()
        st.pyplot(fig)

        # ── Confusion matrices ──
        st.markdown("---")
        st.markdown("#### Confusion Matrices")
        cm_cols = st.columns(len(results))
        for i, (name, data) in enumerate(results.items()):
            with cm_cols[i]:
                cm = confusion_matrix(data["y_test"], data["y_pred"])
                fig_cm, ax_cm = plt.subplots(figsize=(3, 2.5), facecolor="#1e293b")
                ax_cm.set_facecolor("#1e293b")
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            ax=ax_cm, cbar=False,
                            xticklabels=["Safe","Default"],
                            yticklabels=["Safe","Default"])
                ax_cm.set_title(f"{data['info']['icon']} {name}", color="#e2e8f0", fontsize=10)
                ax_cm.tick_params(colors="#94a3b8", labelsize=8)
                plt.tight_layout()
                st.pyplot(fig_cm)

    # ── Already-trained summary ──
    if st.session_state.model_accuracies:
        st.markdown("---")
        st.markdown("#### Previously Trained Models")
        for name, acc in st.session_state.model_accuracies.items():
            info = ALGORITHMS[name]
            st.markdown(f"""
            <div class="algo-card">
                <h4>{info['icon']} {name}</h4>
                <p>Accuracy: <span class="acc-badge">{acc*100:.2f}%</span></p>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 4 — USER INPUT / PREDICT
# ─────────────────────────────────────────────
def prediction_page():
    st.markdown('<div class="page-title">📝 User Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Fill in the applicant details and predict default risk</div>', unsafe_allow_html=True)

    # ── Model selector ──
    available = list(st.session_state.trained_models.keys()) or list(ALGORITHMS.keys())
    chosen_model_name = st.selectbox("🤖  Select Model for Prediction", available)

    st.markdown("---")

    # ── Input fields ──
    st.markdown("#### Applicant Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        income      = st.number_input("💰 Annual Income (₹)", min_value=0, value=500000, step=10000)
        age         = st.number_input("🎂 Applicant Age", min_value=18, max_value=80, value=30)
        experience  = st.number_input("💼 Work Experience (years)", min_value=0, max_value=50, value=5)
        years_emp   = st.number_input("🏢 Years in Current Employment", min_value=0, max_value=50, value=3)

    with col2:
        years_res   = st.number_input("🏠 Years in Current Residence", min_value=0, max_value=50, value=4)
        marital     = st.selectbox("💍 Marital Status",  ["Single", "Married"])
        house       = st.selectbox("🏡 House Ownership", ["Owned", "Rented", "Norent_Noown"])
        vehicle     = st.selectbox("🚗 Vehicle Ownership", ["Yes", "No"])

    with col3:
        occupation  = st.selectbox("👔 Occupation", [
            "Software_Developer", "Mechanical_engineer", "Graphic_Designer",
            "Financial_Analyst", "Sales_Executive", "Librarian", "Technician",
            "Lawyer", "Doctor", "Teacher", "Accountant", "Other"
        ])
        city        = st.text_input("🏙 Residence City",  "Coimbatore")
        state       = st.text_input("📍 Residence State", "Tamil Nadu")

    st.markdown("---")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("🔮  Predict Risk", use_container_width=True)

    if predict_clicked:
        marital_enc = 1 if marital  == "Married" else 0
        house_enc   = {"Owned": 0, "Rented": 1, "Norent_Noown": 2}.get(house, 0)
        vehicle_enc = 1 if vehicle  == "Yes" else 0
        occ_enc     = hash(occupation) % 50
        city_enc    = hash(city)       % 100
        state_enc   = hash(state)      % 30

        sample = pd.DataFrame({
            "Annual_Income":              [income],
            "Applicant_Age":              [age],
            "Work_Experience":            [experience],
            "Marital_Status":             [marital_enc],
            "House_Ownership":            [house_enc],
            "Vehicle_Ownership(car)":     [vehicle_enc],
            "Occupation":                 [occ_enc],
            "Residence_City":             [city_enc],
            "Residence_State":            [state_enc],
            "Years_in_Current_Employment":[years_emp],
            "Years_in_Current_Residence": [years_res],
        })

        # Align columns with training features if available
        if st.session_state.feature_cols:
            for col in st.session_state.feature_cols:
                if col not in sample.columns:
                    sample[col] = 0
            sample = sample[st.session_state.feature_cols]

        # Load model
        if chosen_model_name in st.session_state.trained_models:
            model = st.session_state.trained_models[chosen_model_name]
        else:
            try:
                model = joblib.load(ALGORITHMS[chosen_model_name]["file"])
            except FileNotFoundError:
                st.error(f"⚠️  Model not trained yet. Go to **Train Model** page first.")
                return

        pred = model.predict(sample)
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(sample)[0]

        # ── Rule-based override on Income & House Ownership ──
        # Priority: Income rule is checked first, then house ownership
        if income <= 50000:
            rule_risk   = "HIGH"
            rule_reason = f"Annual Income ₹{income:,} is ≤ ₹50,000 — income too low to service the loan."
        elif income > 50000 and house in ("Rented", "Norent_Noown"):
            rule_risk   = "HIGH"
            rule_reason = f"Annual Income ₹{income:,} is > ₹50,000, but House Ownership is '{house}' — no owned asset as collateral."
        elif income > 50000 and house == "Owned":
            rule_risk   = "LOW"
            rule_reason = f"Annual Income ₹{income:,} is > ₹50,000 and House Ownership is 'Owned' — stable financial profile."
        else:
            # Fallback to ML model prediction
            rule_risk   = "HIGH" if pred[0] == 1 else "LOW"
            rule_reason = "Determined by ML model prediction."

        st.markdown("---")
        st.markdown("#### Prediction Result")

        # ── Rule trigger info box ──
        st.markdown(f"""
        <div style='background:#1e293b;border:1px solid #334155;border-radius:10px;
                    padding:14px 18px;margin-bottom:16px;'>
            <span style='color:#94a3b8;font-size:12px;text-transform:uppercase;letter-spacing:1px;'>
                Rule Applied
            </span><br>
            <span style='color:#e2e8f0;font-size:14px;'>{rule_reason}</span>
        </div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns([2, 1])
        with r1:
            if rule_risk == "HIGH":
                st.markdown("""
                <div class="result-high">
                    ⚠️ HIGH LOAN DEFAULT RISK<br>
                    <span style='font-size:14px;font-weight:400;'>This applicant is likely to default on the loan.</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-low">
                    ✅ LOW LOAN DEFAULT RISK<br>
                    <span style='font-size:14px;font-weight:400;'>This applicant is unlikely to default on the loan.</span>
                </div>""", unsafe_allow_html=True)

        with r2:
            info = ALGORITHMS[chosen_model_name]
            acc  = st.session_state.model_accuracies.get(chosen_model_name, None)
            st.markdown(f"""
            <div class="algo-card" style='margin-top:0;'>
                <h4>{info['icon']} {chosen_model_name}</h4>
                {"<p>Model Accuracy: <span class='acc-badge'>" + f"{acc*100:.2f}%" + "</span></p>" if acc else ""}
            </div>""", unsafe_allow_html=True)

            if prob is not None:
                st.markdown("**Confidence**")
                fig_p, ax_p = plt.subplots(figsize=(3, 1.8), facecolor="#1e293b")
                ax_p.set_facecolor("#1e293b")
                ax_p.barh(["Safe", "Default"], [prob[0]*100, prob[1]*100],
                          color=["#22c55e","#ef4444"], edgecolor="none")
                for i, v in enumerate([prob[0]*100, prob[1]*100]):
                    ax_p.text(v+1, i, f"{v:.1f}%", va="center", color="#e2e8f0", fontsize=9)
                ax_p.set_xlim(0, 115)
                ax_p.tick_params(colors="#94a3b8", labelsize=8)
                ax_p.spines[["top","right","left","bottom"]].set_visible(False)
                ax_p.xaxis.set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_p)

        # ── Input summary ──
        st.markdown("---")
        with st.expander("📋 View Applicant Summary"):
            summary = {
                "Annual Income": f"₹{income:,}",
                "Age": age,
                "Work Experience": f"{experience} yrs",
                "Years Employed (current)": f"{years_emp} yrs",
                "Years at Residence": f"{years_res} yrs",
                "Marital Status": marital,
                "House Ownership": house,
                "Vehicle": vehicle,
                "Occupation": occupation,
                "City": city,
                "State": state,
            }
            s_df = pd.DataFrame(summary.items(), columns=["Field", "Value"])
            st.dataframe(s_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────
if not st.session_state.logged_in:
    login_page()
else:
    sidebar_nav()

    if st.session_state.page == "dashboard":
        dashboard_page()
    elif st.session_state.page == "train":
        train_model_page()
    elif st.session_state.page == "predict":
        prediction_page()
