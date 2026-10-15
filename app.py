import streamlit as st
import numpy as np
import joblib
import warnings
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Weight Predictor ✨",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS (Pinterest-inspired aesthetic) ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* ── Global ── */
.stApp {
    background: linear-gradient(160deg, #FFF5F7 0%, #FEF0F5 25%, #F5F0FF 50%, #F0F7FF 75%, #FFF5F7 100%);
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 760px;}

/* ── Hero Section ── */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    animation: fadeInDown 0.8s ease-out;
}
@keyframes fadeInDown {
    from {opacity: 0; transform: translateY(-20px);}
    to {opacity: 1; transform: translateY(0);}
}
.hero-emoji {
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
    display: inline-block;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% {transform: translateY(0);}
    50% {transform: translateY(-10px);}
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    background: linear-gradient(135deg, #E8507B 0%, #B24BDB 50%, #6C63FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    line-height: 1.2;
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 1.05rem;
    color: #8E8EA0;
    letter-spacing: 0.02em;
}

/* ── Glass Card ── */
.glass-card {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.8);
    box-shadow:
        0 8px 32px rgba(232, 80, 123, 0.08),
        0 2px 8px rgba(108, 99, 255, 0.06);
    padding: 2rem 2.2rem;
    margin: 1rem 0;
    transition: all 0.3s ease;
}
.glass-card:hover {
    box-shadow:
        0 12px 40px rgba(232, 80, 123, 0.12),
        0 4px 12px rgba(108, 99, 255, 0.08);
    transform: translateY(-2px);
}

/* ── Section Labels ── */
.section-label {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #B24BDB;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(178, 75, 219, 0.3), transparent);
}

/* ── Input Styling ── */
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    border-radius: 16px !important;
    border: 2px solid rgba(178, 75, 219, 0.15) !important;
    padding: 0.7rem 1rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    background: rgba(255, 255, 255, 0.8) !important;
}
.stNumberInput > div > div > input:focus {
    border-color: rgba(178, 75, 219, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(178, 75, 219, 0.1) !important;
}

/* ── Streamlit Labels ── */
.stNumberInput label, .stSelectbox label, .stSlider label {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #4A4A5A !important;
    font-size: 0.92rem !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #E8507B 0%, #B24BDB 50%, #6C63FF 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.08rem !important;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(178, 75, 219, 0.3) !important;
    text-transform: none;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 30px rgba(178, 75, 219, 0.4) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(0.99) !important;
}

/* ── Result Card ── */
.result-card {
    background: linear-gradient(135deg, rgba(232, 80, 123, 0.05) 0%, rgba(178, 75, 219, 0.08) 50%, rgba(108, 99, 255, 0.05) 100%);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    border: 1.5px solid rgba(178, 75, 219, 0.2);
    padding: 2rem;
    margin: 1.5rem 0;
    text-align: center;
    animation: scaleIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}
@keyframes scaleIn {
    from {opacity: 0; transform: scale(0.95);}
    to {opacity: 1; transform: scale(1);}
}
.result-value {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 3.5rem;
    background: linear-gradient(135deg, #E8507B, #B24BDB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.result-unit {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 1.1rem;
    color: #8E8EA0;
    margin-top: 0.3rem;
}
.result-secondary {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 0.95rem;
    color: #B24BDB;
    margin-top: 0.8rem;
    padding-top: 0.8rem;
    border-top: 1px solid rgba(178, 75, 219, 0.15);
}

/* ── Stat Pills ── */
.stat-row {
    display: flex;
    gap: 0.8rem;
    margin-top: 1rem;
    flex-wrap: wrap;
    justify-content: center;
}
.stat-pill {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 14px;
    padding: 0.7rem 1.2rem;
    border: 1px solid rgba(178, 75, 219, 0.12);
    flex: 1;
    min-width: 140px;
    text-align: center;
    transition: all 0.3s ease;
}
.stat-pill:hover {
    background: rgba(255, 255, 255, 0.9);
    transform: translateY(-2px);
}
.stat-pill-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8E8EA0;
    margin-bottom: 0.2rem;
}
.stat-pill-value {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: #4A4A5A;
}

/* ── BMI Badge ── */
.bmi-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    margin-top: 0.5rem;
}
.bmi-underweight {background: rgba(96, 165, 250, 0.15); color: #3B82F6; border: 1px solid rgba(96, 165, 250, 0.3);}
.bmi-normal {background: rgba(74, 222, 128, 0.15); color: #16A34A; border: 1px solid rgba(74, 222, 128, 0.3);}
.bmi-overweight {background: rgba(251, 191, 36, 0.15); color: #D97706; border: 1px solid rgba(251, 191, 36, 0.3);}
.bmi-obese {background: rgba(248, 113, 113, 0.15); color: #DC2626; border: 1px solid rgba(248, 113, 113, 0.3);}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: #B0B0C0;
}

/* ── Plotly Chart Container ── */
.stPlotlyChart {
    border-radius: 20px;
    overflow: hidden;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: linear-gradient(135deg, #E8507B, #B24BDB) !important;
}

/* ── Conversion Info ── */
.conversion-info {
    background: rgba(108, 99, 255, 0.06);
    border-radius: 14px;
    padding: 0.8rem 1.2rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #6C63FF;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-top: 0.8rem;
    border: 1px solid rgba(108, 99, 255, 0.12);
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("Best_model_Weight_Height.pkl")


model = load_model()


# ─── Helper Functions ────────────────────────────────────────────────────
def cm_to_inches(cm):
    return cm / 2.54


def feet_inches_to_inches(feet, inches):
    return feet * 12 + inches


def inches_to_ft_in(total_inches):
    feet = int(total_inches // 12)
    inches = round(total_inches % 12, 1)
    return feet, inches


def predict_weight(height_inches):
    """Predict weight using the loaded model. Index is set to 0."""
    prediction = model.predict(np.array([[0, height_inches]]))[0]
    return max(prediction, 0)


def get_bmi(weight_lbs, height_inches):
    """Calculate BMI from weight in lbs and height in inches."""
    if height_inches <= 0:
        return 0
    return (weight_lbs * 703) / (height_inches ** 2)


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "underweight"
    elif bmi < 25:
        return "Normal", "normal"
    elif bmi < 30:
        return "Overweight", "overweight"
    else:
        return "Obese", "obese"


# ─── Hero Section ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-emoji">⚖️</div>
    <div class="hero-title">Weight Predictor</div>
    <div class="hero-subtitle">Enter your height and let our ML model predict your ideal weight</div>
</div>
""", unsafe_allow_html=True)

# ─── Input Card ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📏 Enter Your Height</div>', unsafe_allow_html=True)

# Height input mode
input_mode = st.selectbox(
    "How would you like to enter your height?",
    ["Feet & Inches", "Centimeters", "Inches"],
    label_visibility="visible",
)

height_inches = 0.0

if input_mode == "Feet & Inches":
    col1, col2 = st.columns(2)
    with col1:
        feet = st.number_input("Feet", min_value=3, max_value=8, value=5, step=1)
    with col2:
        inches = st.number_input("Inches", min_value=0, max_value=11, value=7, step=1)
    height_inches = feet_inches_to_inches(feet, inches)

    ft, inc = inches_to_ft_in(height_inches)
    st.markdown(f"""
    <div class="conversion-info">
        📐 <strong>{ft} ft {int(inc)} in</strong> = <strong>{height_inches:.1f} inches</strong>
        ({height_inches * 2.54:.1f} cm)
    </div>
    """, unsafe_allow_html=True)

elif input_mode == "Centimeters":
    cm = st.number_input("Height in Centimeters", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
    height_inches = cm_to_inches(cm)

    ft, inc = inches_to_ft_in(height_inches)
    st.markdown(f"""
    <div class="conversion-info">
        📐 <strong>{cm:.1f} cm</strong> = <strong>{height_inches:.1f} inches</strong>
        ({ft} ft {inc:.0f} in)
    </div>
    """, unsafe_allow_html=True)

else:
    height_inches = st.number_input("Height in Inches", min_value=40.0, max_value=96.0, value=67.0, step=0.5)

    ft, inc = inches_to_ft_in(height_inches)
    st.markdown(f"""
    <div class="conversion-info">
        📐 <strong>{height_inches:.1f} inches</strong> = <strong>{ft} ft {inc:.0f} in</strong>
        ({height_inches * 2.54:.1f} cm)
    </div>
    """, unsafe_allow_html=True)

# ─── Predict Button ─────────────────────────────────────────────────────
predict_clicked = st.button("✨  Predict My Weight", use_container_width=True)

# ─── Results ─────────────────────────────────────────────────────────────
if predict_clicked:
    if height_inches < 40 or height_inches > 96:
        st.error("⚠️ Please enter a valid height (between 40 and 96 inches).")
    else:
        weight_lbs = predict_weight(height_inches)
        weight_kg = weight_lbs * 0.453592
        bmi = get_bmi(weight_lbs, height_inches)
        bmi_cat, bmi_cls = bmi_category(bmi)
        ft, inc = inches_to_ft_in(height_inches)

        # ── Result Card ──
        st.markdown(f"""
        <div class="result-card">
            <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #8E8EA0; margin-bottom: 0.5rem;">
                Predicted Weight for {ft}′{int(inc)}″ ({height_inches:.1f}″)
            </div>
            <div class="result-value">{weight_lbs:.1f}</div>
            <div class="result-unit">pounds</div>
            <div class="result-secondary">
                ≈ <strong>{weight_kg:.1f} kg</strong> &nbsp;·&nbsp;
                BMI: <strong>{bmi:.1f}</strong>
                <br>
                <span class="bmi-badge bmi-{bmi_cls}">{bmi_cat}</span>
            </div>
            <div class="stat-row">
                <div class="stat-pill">
                    <div class="stat-pill-label">Height</div>
                    <div class="stat-pill-value">{height_inches:.1f}″</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">In CM</div>
                    <div class="stat-pill-value">{height_inches * 2.54:.1f}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">Weight (kg)</div>
                    <div class="stat-pill-value">{weight_kg:.1f}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">BMI</div>
                    <div class="stat-pill-value">{bmi:.1f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Plotly Chart: Weight across heights ──
        st.markdown('<div class="section-label">📊 Weight vs Height Curve</div>', unsafe_allow_html=True)

        heights = np.arange(55, 80, 0.5)
        weights = [predict_weight(h) for h in heights]

        fig = go.Figure()

        # Main curve
        fig.add_trace(go.Scatter(
            x=heights, y=weights,
            mode='lines',
            name='Predicted Weight',
            line=dict(
                color='rgba(178, 75, 219, 0.8)',
                width=3,
                shape='spline',
            ),
            fill='tozeroy',
            fillcolor='rgba(178, 75, 219, 0.06)',
            hovertemplate='<b>Height:</b> %{x:.1f}″<br><b>Weight:</b> %{y:.1f} lbs<extra></extra>',
        ))

        # User's point
        fig.add_trace(go.Scatter(
            x=[height_inches], y=[weight_lbs],
            mode='markers+text',
            name='You',
            marker=dict(
                size=16,
                color='#E8507B',
                line=dict(width=3, color='white'),
                symbol='circle',
            ),
            text=[f'{weight_lbs:.1f} lbs'],
            textposition='top center',
            textfont=dict(
                family='Outfit',
                size=13,
                color='#E8507B',
            ),
            hovertemplate='<b>Your Height:</b> %{x:.1f}″<br><b>Predicted Weight:</b> %{y:.1f} lbs<extra></extra>',
        ))

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#4A4A5A'),
            margin=dict(l=20, r=20, t=30, b=20),
            height=340,
            showlegend=False,
            xaxis=dict(
                title='Height (inches)',
                gridcolor='rgba(178, 75, 219, 0.08)',
                zerolinecolor='rgba(178, 75, 219, 0.08)',
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title='Weight (lbs)',
                gridcolor='rgba(178, 75, 219, 0.08)',
                zerolinecolor='rgba(178, 75, 219, 0.08)',
                tickfont=dict(size=11),
            ),
            hovermode='x unified',
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # ── Comparison Table ──
        st.markdown('<div class="section-label">📋 Quick Reference Table</div>', unsafe_allow_html=True)

        ref_heights = [60, 62, 64, 66, 68, 70, 72, 74, 76]
        table_html = """
        <table style="width: 100%; border-collapse: separate; border-spacing: 0 6px; font-family: 'Inter', sans-serif; font-size: 0.88rem;">
            <thead>
                <tr style="color: #8E8EA0; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.1em;">
                    <th style="text-align: left; padding: 0.5rem 1rem; font-weight: 500;">Height</th>
                    <th style="text-align: center; padding: 0.5rem 1rem; font-weight: 500;">Inches</th>
                    <th style="text-align: center; padding: 0.5rem 1rem; font-weight: 500;">Weight (lbs)</th>
                    <th style="text-align: center; padding: 0.5rem 1rem; font-weight: 500;">Weight (kg)</th>
                </tr>
            </thead>
            <tbody>
        """
        for h in ref_heights:
            w = predict_weight(h)
            w_kg = w * 0.453592
            f, i = inches_to_ft_in(h)
            is_user = abs(h - height_inches) < 1
            row_bg = "rgba(232, 80, 123, 0.08)" if is_user else "rgba(255,255,255,0.5)"
            row_border = "2px solid rgba(232, 80, 123, 0.3)" if is_user else "1px solid rgba(178, 75, 219, 0.08)"
            table_html += f"""
                <tr style="background: {row_bg}; border-radius: 12px;">
                    <td style="padding: 0.6rem 1rem; border-left: {row_border}; border-top: {row_border}; border-bottom: {row_border}; border-radius: 12px 0 0 12px; font-weight: 500; color: #4A4A5A;">
                        {f}′{int(i)}″ {'<span style="color: #E8507B; font-weight: 600;">← You</span>' if is_user else ''}
                    </td>
                    <td style="padding: 0.6rem 1rem; border-top: {row_border}; border-bottom: {row_border}; text-align: center; color: #6C63FF;">{h}″</td>
                    <td style="padding: 0.6rem 1rem; border-top: {row_border}; border-bottom: {row_border}; text-align: center; font-weight: 600; color: #B24BDB;">{w:.1f}</td>
                    <td style="padding: 0.6rem 1rem; border-right: {row_border}; border-top: {row_border}; border-bottom: {row_border}; border-radius: 0 12px 12px 0; text-align: center; color: #8E8EA0;">{w_kg:.1f}</td>
                </tr>
            """
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Powered by Machine Learning · Linear Regression Model<br>
    <span style="font-size: 0.7rem;">Predictions are estimates based on training data. Actual weight varies by body composition.</span>
</div>
""", unsafe_allow_html=True)
