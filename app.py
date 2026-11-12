import os

import joblib
import streamlit as st

from inference import predict_relapse_risk

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model_bundle.joblib")

CLASS_COLORS = {"No relapse": "#22C55E", "Relapse to MAM": "#FB923C", "Relapse to SAM": "#F87171"}
CARD_BG = "#111827"
SECTION_ACCENTS = {"admission": "#3B82F6", "discharge": "#8B5CF6", "food": "#22C55E"}

IPC_MAP = {"Phase 1/2 - minimal/stressed": 1, "Phase 3 - crisis": 2, "Phase 4/5 - emergency/famine": 3}
HUNGER_MAP = {"Little to no hunger": 1, "Moderate household hunger": 2, "Severe household hunger": 3}


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

st.subheader("Household food security")
section_card_open(SECTION_ACCENTS["food"])
ipc_phase_label = st.selectbox("IPC food security phase", list(IPC_MAP.keys()), index=None, placeholder="Select...")
hunger_scale_label = st.selectbox("Household hunger scale", list(HUNGER_MAP.keys()), index=None, placeholder="Select...")

food_vals = [ipc_phase_label, hunger_scale_label]
section_card_close(sum(v is not None for v in food_vals), len(food_vals), SECTION_ACCENTS["food"])

st.header("At discharge")
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
    oral_rehydration_label = st.selectbox(
        "Oral rehydration given during treatment", ["Yes", "No"], index=None, placeholder="Select...")

discharge_vals = [dur_trt, diar_t1, whz_t0, oral_rehydration_label]
section_card_close(sum(v is not None for v in discharge_vals), len(discharge_vals), SECTION_ACCENTS["discharge"])

st.divider()

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
    "ipc_phase_num": IPC_MAP.get(ipc_phase_label),
    "hunger_scale": HUNGER_MAP.get(hunger_scale_label),
    "dur_trt": dur_trt,
    "diar_T1": diar_t1,
    "whz_T0": whz_t0,
    "medication_oral_rehydration": None if oral_rehydration_label is None else (1 if oral_rehydration_label == "Yes" else 0),
}

field_labels = {
    "facility": "Facility", "sex": "Child's sex", "cage_adm": "Child's age at admission",
    "SAMMAM": "Admission type", "RegioCEGn": "Region",
    "mat_edu_dichot": "Maternal education (ever attended school?)",
    "whz_adm_clean": "WHZ at admission", "MUACofchildatadmission": "MUAC at admission",
    "haz_adm_clean": "HAZ at admission", "hndwash": "Household handwashing access",
    "prev_wast_treat": "Previous wasting treatment history",
    "ipc_phase_num": "IPC food security phase", "hunger_scale": "Household hunger scale",
    "dur_trt": "Duration of treatment", "diar_T1": "Diarrhea at discharge",
    "whz_T0": "WHZ at discharge",
    "medication_oral_rehydration": "Oral rehydration given during treatment",
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

col_relapse, col_whz = st.columns(2)
with col_relapse:
    predict_relapse_clicked = st.button("Predict Relapse", type="primary", width="stretch")
with col_whz:
    predict_whz_clicked = st.button("Predict WHZ", type="primary", width="stretch")

if predict_relapse_clicked or predict_whz_clicked:
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

        if predict_relapse_clicked:
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

            if not result["used_facility"] and facility is not None:
                st.caption(
                    "This facility wasn't in the model's training data, so the estimate uses "
                    "region and child details only."
                )

        if predict_whz_clicked:
            if result["predicted_whz"] is not None:
                st.markdown(
                    f"""
                    <div style="background:{CARD_BG}; border:1px solid rgba(255,255,255,0.08);
                        border-radius:18px; padding:28px 10px; margin-top:8px; text-align:center;">
                        <div style="font-size:14px; color:#9AA5B8;">
                            Estimated WHZ: <strong style="font-size:20px; color:{color};">{result["predicted_whz"]:.2f}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    f"This child is predicted as “{predicted}”, so no WHZ estimate applies, "
                    "that number is only shown for children predicted to relapse."
                )
