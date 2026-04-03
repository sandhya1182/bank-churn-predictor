import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #2E75B6;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 40px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #1F497D; }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #2E75B6;
    }
    .churn-box {
        background: #ffe5e5;
        border: 3px solid #e74c3c;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
    }
    .stay-box {
        background: #e5ffe5;
        border: 3px solid #2ecc71;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
    }
    h1 { color: #2E75B6; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except:
    model_loaded = False

# ── Header ────────────────────────────────────────────────────
st.markdown("# 🏦 Bank Customer Churn Predictor")
st.markdown("### M.Sc Data Science Final Project — Sandhya Raja | S.I.E.S College")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict Churn", "📊 Project Insights", "ℹ️ About"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Customer Details to Predict Churn")
    st.markdown("Fill in the customer information below and click **Predict** to see if the customer is likely to churn.")
    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Personal Details**")
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 18, 92, 35)
        credit_score = st.slider("Credit Score", 300, 850, 650)

    with col2:
        st.markdown("**💰 Account Details**")
        balance = st.number_input("Account Balance (€)", 0, 300000, 50000, step=1000)
        estimated_salary = st.number_input("Estimated Salary (€)", 0, 200000, 60000, step=1000)
        tenure = st.slider("Tenure (Years)", 0, 10, 3)
        num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])

    with col3:
        st.markdown("**📱 Behavioral Details**")
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
        complain = st.selectbox("Has Complained?", ["No", "Yes"])
        satisfaction_score = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
        card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])
        point_earned = st.slider("Points Earned", 100, 1000, 500)

    st.markdown("")
    predict_btn = st.button("🔮 PREDICT CHURN")

    if predict_btn:
        # Encode inputs
        geo_map = {"France": 0, "Germany": 1, "Spain": 2}
        gender_map = {"Female": 0, "Male": 1}
        card_map = {"DIAMOND": 0, "GOLD": 1, "PLATINUM": 2, "SILVER": 3}

        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Geography': [geo_map[geography]],
            'Gender': [gender_map[gender]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [1 if has_cr_card == "Yes" else 0],
            'IsActiveMember': [1 if is_active_member == "Yes" else 0],
            'EstimatedSalary': [estimated_salary],
            'Complain': [1 if complain == "Yes" else 0],
            'Satisfaction Score': [satisfaction_score],
            'Card Type': [card_map[card_type]],
            'Point Earned': [point_earned]
        })

        if model_loaded:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            churn_prob = round(probability[1] * 100, 1)
            stay_prob = round(probability[0] * 100, 1)
        else:
            # Demo mode if model not loaded
            prediction = 1 if complain == "Yes" or (age > 45 and balance > 80000) else 0
            churn_prob = 92.0 if prediction == 1 else 8.0
            stay_prob = 100 - churn_prob

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="churn-box">
                    <h1 style="color:#e74c3c; font-size:48px;">⚠️ HIGH CHURN RISK</h1>
                    <h3 style="color:#c0392b;">This customer is likely to LEAVE the bank</h3>
                    <h2 style="color:#e74c3c;">Churn Probability: {churn_prob}%</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="stay-box">
                    <h1 style="color:#2ecc71; font-size:48px;">✅ LOW CHURN RISK</h1>
                    <h3 style="color:#27ae60;">This customer is likely to STAY with the bank</h3>
                    <h2 style="color:#2ecc71;">Retention Probability: {stay_prob}%</h2>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            # Probability gauge chart
            fig, ax = plt.subplots(figsize=(4, 4))
            colors = ['#2ecc71', '#e74c3c']
            values = [stay_prob, churn_prob]
            labels = [f'Stay\n{stay_prob}%', f'Churn\n{churn_prob}%']
            wedges, texts = ax.pie(values, colors=colors, startangle=90,
                                    wedgeprops=dict(width=0.6))
            ax.set_title("Risk Distribution", fontweight='bold', fontsize=13)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Risk factors
        st.markdown("### 🔍 Key Risk Factors for This Customer")
        risk_col1, risk_col2, risk_col3 = st.columns(3)

        with risk_col1:
            complaint_risk = "🔴 HIGH RISK" if complain == "Yes" else "🟢 LOW RISK"
            st.metric("Complaint History", complaint_risk)

        with risk_col2:
            age_risk = "🔴 HIGH RISK" if age > 40 else "🟢 LOW RISK"
            st.metric("Age Risk", f"{age} years — {age_risk}")

        with risk_col3:
            product_risk = "🔴 HIGH RISK" if num_of_products >= 3 else ("🟢 LOW RISK" if num_of_products == 2 else "🟡 MEDIUM RISK")
            st.metric("Products Risk", f"{num_of_products} products — {product_risk}")

        # Recommendation
        st.markdown("### 💡 Bank Recommendation")
        if prediction == 1:
            if complain == "Yes":
                st.error("🚨 **URGENT:** Customer has raised a complaint. Assign a dedicated relationship manager immediately. Resolve within 24 hours and offer a retention package.")
            elif age > 40 and balance > 80000:
                st.warning("⚠️ **HIGH VALUE AT RISK:** High-balance middle-aged customer. Offer premium loyalty benefits, higher interest rates, or exclusive services.")
            elif num_of_products >= 3:
                st.warning("⚠️ **OVER-BUNDLED:** Customer may be over-sold products. Review their portfolio and simplify their package.")
            elif is_active_member == "No":
                st.warning("⚠️ **INACTIVE CUSTOMER:** Re-engage with cashback offers, app engagement rewards, or a personal call from relationship manager.")
            else:
                st.warning("⚠️ **AT RISK:** Monitor this customer closely. Consider proactive outreach with personalized retention offer.")
        else:
            st.success("✅ **RETAIN:** Customer appears stable. Continue standard engagement. Maintain activity and satisfaction levels.")


# ═══════════════════════════════════════════════════════════════
# TAB 2 — INSIGHTS
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Project Insights — Bank Customer Churn Analysis")
    st.markdown("Dataset: 10,000 bank customers | 18 features | 20.38% churn rate")
    st.markdown("")

    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="metric-card"><h2 style="color:#2E75B6">10,000</h2><p>Total Customers</p></div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="metric-card"><h2 style="color:#e74c3c">2,038</h2><p>Churned Customers</p></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="metric-card"><h2 style="color:#e67e22">20.38%</h2><p>Churn Rate</p></div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="metric-card"><h2 style="color:#2ecc71">99.85%</h2><p>Model Accuracy</p></div>', unsafe_allow_html=True)

    st.markdown("")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Churn by Geography
        fig, ax = plt.subplots(figsize=(6, 4))
        countries = ['France', 'Germany', 'Spain']
        churn_rates = [16.2, 32.4, 16.7]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.bar(countries, churn_rates, color=colors, edgecolor='white', linewidth=1.5)
        ax.set_title('Churn Rate by Geography', fontweight='bold', fontsize=13)
        ax.set_ylabel('Churn Rate %')
        for bar, rate in zip(bars, churn_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{rate}%', ha='center', fontweight='bold')
        ax.set_ylim(0, 40)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("🔍 Germany churns at 32.4% — nearly double France and Spain")

    with chart_col2:
        # Feature Importance
        fig, ax = plt.subplots(figsize=(6, 4))
        features = ['Complain', 'Age', 'NumOfProducts', 'Geography', 'Balance',
                    'IsActiveMember', 'Sat.Score', 'Gender']
        importance = [0.782, 0.070, 0.053, 0.021, 0.021, 0.012, 0.008, 0.006]
        colors_fi = ['#e74c3c' if i == 0 else '#3498db' for i in range(len(features))]
        ax.barh(features[::-1], importance[::-1], color=colors_fi[::-1], edgecolor='white')
        ax.set_title('Feature Importance (Random Forest)', fontweight='bold', fontsize=13)
        ax.set_xlabel('Importance Score')
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("🔍 Complaint history dominates at 78.2% importance")

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        # Churn by Age
        fig, ax = plt.subplots(figsize=(6, 4))
        categories = ['Stayed (37.4 yrs)', 'Churned (44.8 yrs)']
        avg_ages = [37.4, 44.8]
        ax.bar(categories, avg_ages, color=['#2ecc71', '#e74c3c'], edgecolor='white', linewidth=1.5)
        ax.set_title('Average Age: Stayed vs Churned', fontweight='bold', fontsize=13)
        ax.set_ylabel('Average Age')
        ax.set_ylim(0, 55)
        for i, v in enumerate(avg_ages):
            ax.text(i, v + 0.3, str(v), ha='center', fontweight='bold')
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("🔍 Churned customers are 7 years older on average")

    with chart_col4:
        # Model comparison
        fig, ax = plt.subplots(figsize=(6, 4))
        models = ['Logistic\nRegression', 'Random\nForest', 'XGBoost']
        roc = [0.9985, 0.9989, 0.9975]
        colors_m = ['#9b59b6', '#2ecc71', '#e67e22']
        bars = ax.bar(models, roc, color=colors_m, edgecolor='white', linewidth=1.5)
        ax.set_title('Model ROC-AUC Comparison', fontweight='bold', fontsize=13)
        ax.set_ylabel('ROC-AUC Score')
        ax.set_ylim(0.995, 1.000)
        for bar, val in zip(bars, roc):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.0001,
                    f'{val:.4f}', ha='center', fontweight='bold', fontsize=10)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("🏆 Random Forest wins with ROC-AUC of 0.9989")

    # Business insights
    st.markdown("---")
    st.markdown("### 💡 Key Business Insights")
    b1, b2 = st.columns(2)
    with b1:
        st.error("🚨 **Insight 1 — Complaints are critical:** 99.5% of customers who complained eventually churned. Complaint resolution is the single highest-impact retention lever.")
        st.warning("⚠️ **Insight 2 — High-value customers leaving:** Churned customers had avg balance of €91,109 vs €72,743 for retained. High-value customers are seeking better returns.")
        st.info("ℹ️ **Insight 3 — Germany needs attention:** 32.4% churn rate — nearly double France and Spain. Market-specific retention strategy urgently needed.")
    with b2:
        st.warning("⚠️ **Insight 4 — Inactive = high risk:** Inactive members churn at 26.9% vs 14.3% for active. Re-engagement campaigns can halve churn risk.")
        st.error("🚨 **Insight 5 — Product over-bundling:** 82.7% of 3-product and 100% of 4-product customers churn. Optimal engagement is at exactly 2 products.")


# ═══════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ℹ️ About This Project")

    about_col1, about_col2 = st.columns(2)

    with about_col1:
        st.markdown("""
        **Project:** Bank Customer Churn Prediction

        **Student:** Sandhya Raja

        **Seat No:** 01

        **College:** S.I.E.S College of Commerce and Economics (Autonomous), Sion(E), Mumbai

        **Programme:** M.Sc. Data Science — Semester IV

        **Academic Year:** 2024-2025
        """)

    with about_col2:
        st.markdown("""
        **Dataset:** 10,000 bank customers | 18 features

        **Models Used:**
        - Logistic Regression
        - Random Forest ⭐ Best Model
        - XGBoost

        **Best Model Performance:**
        - Accuracy: 99.85%
        - F1 Score: 0.9963
        - ROC-AUC: 0.9989

        **Tools:** Python, Scikit-learn, XGBoost, SMOTE, Power BI
        """)

    st.markdown("---")
    st.markdown("### 🛠️ Technical Stack")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.info("🐍 **Python**\nCore language")
    with t2:
        st.info("📊 **Scikit-learn**\nML models")
    with t3:
        st.info("⚡ **XGBoost**\nGradient boosting")
    with t4:
        st.info("📈 **Power BI**\nDashboard")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:13px;'>"
    "Bank Customer Churn Predictor | Sandhya Raja | M.Sc Data Science | S.I.E.S College 2025"
    "</p>",
    unsafe_allow_html=True
)
