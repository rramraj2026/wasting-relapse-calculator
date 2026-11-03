import os

import joblib
import plotly.graph_objects as go
import streamlit as st

from inference import predict_relapse_risk

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model_bundle.joblib")

CLASS_COLORS = {"No relapse": "#22C55E", "Relapse to MAM": "#FB923C", "Relapse to SAM": "#F87171"}
CARD_BG = "#111827"
SECTION_ACCENTS = {"admission": "#3B82F6", "discharge": "#8B5CF6"}


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def section_card_open(accent):
    st.markdown(
        f"""
        <div class="section-card" style="border-left: 3px solid {accent}; background: rgba(255,255,255,0.03);
            border-radius: 0 14px 14px 0; padding: 16px 18px 10px 18px; margin-bottom: 10px;
            transition: all 0.3s ease;">
        """,
        unsafe_allow_html=True,
    )


def section_card_close(filled, total, accent):
    done = filled == total
    badge_color = "#22C55E" if done else "#94A3B8"
    icon = "✅" if done else "🟡"
    bar_color = "#22C55E" if done else accent
    pct = int(100 * filled / total)
    st.markdown(
        f"""
            <div style="margin-top:6px;">
                <div style="background:rgba(255,255,255,0.08); border-radius:999px; height:6px; overflow:hidden;">
                    <div style="width:{pct}%; background:{bar_color}; height:100%; border-radius:999px;
                        transition: width 0.4s ease;"></div>
                </div>
                <p style="text-align:right; font-size:12px; color:{badge_color}; margin:4px 0 0 0;">
                    {icon} {filled}/{total} filled
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def whz_tag(value):
    if value is None:
        return
    if value < -3:
        st.caption("🔴 Severe wasting range (WHZ < -3)")
    elif value < -2:
        st.caption("🟠 Moderate wasting range (-3 to -2)")
    else:
        st.caption("🟢 Within normal range (≥ -2)")


def muac_tag(value):
    if value is None:
        return
    if value < 11.5:
        st.caption("🔴 Severe wasting range (< 11.5 cm)")
    elif value < 12.5:
        st.caption("🟠 Moderate wasting range (11.5–12.5 cm)")
    else:
        st.caption("🟢 Within normal range (≥ 12.5 cm)")


def haz_tag(value):
    if value is None:
        return
    if value < -3:
        st.caption("🔴 Severe stunting range (HAZ < -3)")
    elif value < -2:
        st.caption("🟠 Moderate stunting range (-3 to -2)")
    else:
        st.caption("🟢 Within normal range (≥ -2)")

st.set_page_config(page_title="Wasting Relapse Risk Calculator", layout="centered", page_icon="🩺")


def inject_global_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 15% 20%, rgba(59,130,246,0.18) 0%, rgba(59,130,246,0) 40%),
                radial-gradient(circle at 85% 15%, rgba(139,92,246,0.16) 0%, rgba(139,92,246,0) 40%),
                radial-gradient(circle at 25% 85%, rgba(6,182,212,0.14) 0%, rgba(6,182,212,0) 40%),
                radial-gradient(circle at 80% 80%, rgba(236,72,153,0.12) 0%, rgba(236,72,153,0) 40%),
                #0B1220;
            background-size: 180% 180%, 180% 180%, 180% 180%, 180% 180%, 100% 100%;
            animation: driftBg 22s ease-in-out infinite alternate;
        }
        @keyframes driftBg {
            0%   { background-position: 0% 0%, 100% 0%, 0% 100%, 100% 100%, 0 0; }
            50%  { background-position: 30% 20%, 70% 30%, 20% 70%, 80% 90%, 0 0; }
            100% { background-position: 10% 40%, 90% 10%, 40% 90%, 60% 70%, 0 0; }
        }
        .section-card:hover {
            background: rgba(255,255,255,0.055) !important;
            transform: translateX(2px);
            box-shadow: 0 6px 24px rgba(0,0,0,0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_style()


def probability_chart(probabilities):
    classes = list(probabilities.keys())
    values = [probabilities[c] * 100 for c in classes]
    colors = [CLASS_COLORS[c] for c in classes]
    fig = go.Figure(go.Bar(
        x=values, y=classes, orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.0f}%" for v in values], textposition="outside",
        textfont=dict(color="#E5E7EB", size=14),
    ))
    fig.update_layout(
        height=220, margin=dict(l=10, r=40, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(tickfont=dict(color="#E5E7EB", size=14)),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


@st.cache_resource
def load_bundle():
    return joblib.load(ARTIFACT_PATH)


bundle = load_bundle()
cat = bundle["category_levels"]
rng = bundle["raw_ranges"]

st.markdown(
    """
    <div style="padding: 18px 0 6px 0;">
        <h1 style="margin-bottom:4px; color:#F1F5F9;">🩺 Wasting Relapse Risk Calculator</h1>
        <p style="color:#94A3B8; font-size:15px; margin-top:0;">
            Estimates whether a child is likely to relapse, and if so whether to MAM or SAM,
            within 6 months of discharge.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.header("At admission")
section_card_open(SECTION_ACCENTS["admission"])

c1, c2 = st.columns(2)
with c1:
    facility = st.selectbox("Facility (Optional)", bundle["facility_list"], index=None, placeholder="Select...")
    sex = st.selectbox(
        "Child's sex", cat["sex"], index=None, placeholder="Select...",
        format_func=lambda v: v.capitalize())
    cage_adm = st.number_input(
        "Child's age at admission (months)",
        min_value=float(rng["cage_adm"][0]), max_value=float(rng["cage_adm"][1]),
        value=None, placeholder=f"e.g. {rng['cage_adm'][0]:.0f}-{rng['cage_adm'][1]:.0f} months")
    sammam = st.selectbox("Admission type (SAM / MAM)", cat["SAMMAM"], index=None, placeholder="Select...")
    region = st.selectbox(
        "Region", cat["RegioCEGn"], index=None, placeholder="Select...",
        format_func=lambda v: v.replace("_", " "))
    mat_edu = st.selectbox(
        "Did the mother/caregiver ever attend school?",
        cat["mat_edu_dichot"], index=None, placeholder="Select...",
        format_func=lambda v: "Yes" if v == "any" else "No")

with c2:
    whz_adm = st.number_input(
        "Weight-for-Height Z-score at admission",
        min_value=float(rng["whz_adm_clean"][0]), max_value=float(rng["whz_adm_clean"][1]),
        value=None, placeholder=f"e.g. {rng['whz_adm_clean'][0]:.1f} to {rng['whz_adm_clean'][1]:.1f}")
    whz_tag(whz_adm)
    muac_adm = st.number_input(
        "Mid-Upper Arm Circumference at admission (cm)",
        min_value=float(rng["MUACofchildatadmission"][0]), max_value=float(rng["MUACofchildatadmission"][1]),
        value=None, placeholder=f"e.g. {rng['MUACofchildatadmission'][0]:.1f}-{rng['MUACofchildatadmission'][1]:.1f} cm")
    muac_tag(muac_adm)
    haz_adm = st.number_input(
        "Height-for-Age Z-score at admission",
        min_value=float(rng["haz_adm_clean"][0]), max_value=float(rng["haz_adm_clean"][1]),
        value=None, placeholder=f"e.g. {rng['haz_adm_clean'][0]:.1f} to {rng['haz_adm_clean'][1]:.1f}")
    haz_tag(haz_adm)
    hndwash_label = st.selectbox("Household handwashing access", ["Yes", "No"], index=None, placeholder="Select...")
    prev_wast = st.selectbox(
        "Previous wasting treatment history", cat["prev_wast_treat"], index=None, placeholder="Select...",
        format_func=lambda v: v.capitalize())

admission_vals = [facility, sex, cage_adm, sammam, region, mat_edu, whz_adm, muac_adm, haz_adm, hndwash_label, prev_wast]
section_card_close(sum(v is not None for v in admission_vals), len(admission_vals), SECTION_ACCENTS["admission"])

st.header("At discharge")
st.caption("These values are only known once the child has completed treatment and is discharged.")
section_card_open(SECTION_ACCENTS["discharge"])
c5, c6 = st.columns(2)
with c5:
    dur_trt = st.number_input(
        "Duration of treatment (days)",
        min_value=float(rng["dur_trt"][0]), max_value=float(rng["dur_trt"][1]),
        value=None, placeholder=f"e.g. {rng['dur_trt'][0]:.0f}-{rng['dur_trt'][1]:.0f} days")
    diar_t1 = st.selectbox(
        "Diarrhea at discharge", cat["diar_T1"], index=None, placeholder="Select...",
        format_func=lambda v: v.capitalize())
with c6:
    whz_t0 = st.number_input(
        "Weight-for-Height Z-score at discharge (measured by facility staff)",
        min_value=float(rng["whz_T0"][0]), max_value=float(rng["whz_T0"][1]),
        value=None, placeholder=f"e.g. {rng['whz_T0'][0]:.1f} to {rng['whz_T0'][1]:.1f}")
    whz_tag(whz_t0)

discharge_vals = [dur_trt, diar_t1, whz_t0]
section_card_close(sum(v is not None for v in discharge_vals), len(discharge_vals), SECTION_ACCENTS["discharge"])

st.divider()

if st.button("Predict relapse risk", type="primary"):
    raw_input = {
        "facility": facility,
        "sex": sex,
        "cage_adm": cage_adm,
        "SAMMAM": sammam,
        "RegioCEGn": region,
        "mat_edu_dichot": mat_edu,
        "whz_adm_clean": whz_adm,
        "MUACofchildatadmission": muac_adm,
        "haz_adm_clean": haz_adm,
        "hndwash": None if hndwash_label is None else (1 if hndwash_label == "Yes" else 0),
        "prev_wast_treat": prev_wast,
        "dur_trt": dur_trt,
        "diar_T1": diar_t1,
        "whz_T0": whz_t0,
    }

    field_labels = {
        "facility": "Facility", "sex": "Child's sex", "cage_adm": "Child's age at admission",
        "SAMMAM": "Admission type", "RegioCEGn": "Region",
        "mat_edu_dichot": "Maternal education (ever attended school?)",
        "whz_adm_clean": "WHZ at admission", "MUACofchildatadmission": "MUAC at admission",
        "haz_adm_clean": "HAZ at admission", "hndwash": "Household handwashing access",
        "prev_wast_treat": "Previous wasting treatment history",
        "dur_trt": "Duration of treatment", "diar_T1": "Diarrhea at discharge",
        "whz_T0": "WHZ at discharge",
    }
    # Facility is optional, left blank, the model falls back to the population-average
    # facility risk instead of blocking the prediction.
    missing = [field_labels[k] for k, v in raw_input.items() if v is None and k != "facility"]

    range_check_fields = {
        "cage_adm": "cage_adm",
        "whz_adm_clean": "whz_adm_clean",
        "MUACofchildatadmission": "MUACofchildatadmission",
        "haz_adm_clean": "haz_adm_clean",
        "dur_trt": "dur_trt",
        "whz_T0": "whz_T0",
    }
    out_of_range = []
    if not missing:
        for field_key, range_key in range_check_fields.items():
            value = raw_input[field_key]
            lo, hi = rng[range_key]
            if value < lo or value > hi:
                out_of_range.append(
                    f"{field_labels[field_key]} = {value} (valid range: {lo:.1f} to {hi:.1f})"
                )

    if missing:
        st.error(
            "Please fill in every field before predicting, the model isn't allowed to guess. "
            "Missing: " + ", ".join(missing)
        )
    elif out_of_range:
        st.error(
            "This model was never trained on values outside the ranges below, so it can't "
            "give a reliable estimate here. Out of range: " + "; ".join(out_of_range)
        )
    else:
        result = predict_relapse_risk(raw_input, bundle)
        predicted = result["predicted_class"]
        color = CLASS_COLORS[predicted]
        glow_dim = hex_to_rgba(color, 0.22)
        glow_bright = hex_to_rgba(color, 0.5)
        icon = {"No relapse": "✅", "Relapse to MAM": "⚠️", "Relapse to SAM": "🔴"}[predicted]

        st.markdown(
            f"""
            <style>
            @keyframes pulseGlowRelapse {{
                0%   {{ box-shadow: 0 0 25px {glow_dim}; }}
                50%  {{ box-shadow: 0 0 60px {glow_bright}; }}
                100% {{ box-shadow: 0 0 25px {glow_dim}; }}
            }}
            </style>
            <div style="background:{CARD_BG}; border:1px solid rgba(255,255,255,0.08);
                border-radius:18px; padding:28px 10px 10px 10px; margin-top:8px; text-align:center;
                animation: pulseGlowRelapse 2.4s ease-in-out infinite;">
                <div style="font-size:40px;">{icon}</div>
                <div style="margin-top:10px;">
                    <span style="background-color:{color}; color:#0B1220;
                        padding:6px 22px; border-radius:999px; font-weight:700; font-size:18px;">
                        {predicted}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(probability_chart(result["probabilities"]), width="stretch")

        if not result["used_facility"] and facility is not None:
            st.caption(
                "This facility wasn't in the model's training data, so the estimate uses "
                "region and child details only."
            )

        st.caption(
            "Estimated chance of no relapse, relapse to MAM, or relapse to SAM within 6 months "
            "of discharge, based on patterns learned from past cases in this program."
        )
