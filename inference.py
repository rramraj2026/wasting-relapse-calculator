import numpy as np
import pandas as pd


def recovery_pct_single(admission, exit_, threshold, floor):
    if pd.isna(admission) or pd.isna(exit_):
        return np.nan
    if admission < threshold:
        denom = max(threshold - admission, floor)
        return (exit_ - admission) / denom
    if exit_ >= threshold:
        return 1.0
    return (exit_ - threshold) / floor


def percentile_rank(reference_array, value):
    return float((reference_array <= value).mean())


def build_feature_row(raw_input, bundle, use_facility):
    """raw_input: dict of raw field values collected from the Streamlit form.
    Returns a one-row DataFrame with the exact design-matrix columns the chosen
    (facility-given or facility-blank) classifier expects, in the right order.
    """
    facility = raw_input['facility']
    cage_adm = raw_input['cage_adm']
    whz_adm = raw_input['whz_adm_clean']
    muac_adm = raw_input['MUACofchildatadmission']
    haz_adm = raw_input['haz_adm_clean']
    dur_trt = raw_input['dur_trt']
    whz_t0 = raw_input['whz_T0']

    whz_rank = percentile_rank(bundle['whz_reference'], whz_adm)
    muac_rank = percentile_rank(bundle['muac_reference'], muac_adm)
    whz_muac_discordance = abs(whz_rank - muac_rank)
    wasting_severity_rank = 1 - (whz_rank + muac_rank) / 2

    whz_recovery_pct = recovery_pct_single(whz_adm, whz_t0, bundle['WHZ_THRESH'], bundle['FLOOR_WHZ'])
    whz_gain_per_day = (whz_t0 - whz_adm) / dur_trt if dur_trt else np.nan

    clus_scaled = bundle['cluster_scaler'].transform([[whz_adm, muac_adm, dur_trt, cage_adm]])
    cluster_label = int(bundle['cluster_model'].predict(clus_scaled)[0])
    cluster0_flag = 1 if cluster_label == 0 else 0

    row = {
        'wasting_severity_rank': wasting_severity_rank,
        'whz_adm_clean': whz_adm,
        'double_burden': 1.0 if (whz_adm < -2 and haz_adm < -2) else 0.0,
        'whz_adm_below_neg3': 1.0 if whz_adm < -3 else 0.0,
        'whz_followup_below_neg2': 1.0 if whz_t0 < -2 else 0.0,
        'MUACofchildatadmission': muac_adm,
        'cluster0_flag': cluster0_flag,
        'dur_trt': dur_trt,
        'whz_gain_per_day': whz_gain_per_day,
        'whz_recovery_pct': whz_recovery_pct,
        'muac_adm_below_125': 1.0 if muac_adm < bundle['MUAC_THRESH'] else 0.0,
        'haz_adm_clean': haz_adm,
        'young_child_lt12m': 1.0 if cage_adm < 12 else 0.0,
        'hndwash': int(raw_input['hndwash']),
        'age_x_whz_adm': cage_adm * whz_adm,
        'whz_followup': whz_t0,
        'whz_muac_discordance': whz_muac_discordance,
        'SAMMAM_SAM': 1 if raw_input['SAMMAM'] == 'SAM' else 0,
        'diar_T1_yes': 1 if raw_input['diar_T1'] == 'yes' else 0,
        'sex_male': 1 if raw_input['sex'] == 'male' else 0,
        'prev_wast_treat_yes': 1 if raw_input['prev_wast_treat'] == 'yes' else 0,
        'mat_edu_dichot_none': 1 if raw_input['mat_edu_dichot'] == 'none' else 0,
        'ipc_phase_num': int(raw_input['ipc_phase_num']),
        'hunger_scale': int(raw_input['hunger_scale']),
        'medication_oral_rehydration': int(raw_input['medication_oral_rehydration']),
    }
    for region in ['Bari', 'Bay', 'Galgadud', 'Hiran', 'Lower_Juba', 'Mudug']:
        row[f'RegioCEGn_{region}'] = 1 if raw_input['RegioCEGn'] == region else 0

    if use_facility:
        for name in ('fac_any_loo', 'fac_sam_loo', 'facility_recovery_speed_loo'):
            lut = bundle['facility_lookups'][name]
            row[name] = lut['by_facility'].get(facility, lut['population_rate'])

    knn_cols = bundle['knn_base_cols']
    values_by_name = {'whz_adm_clean': whz_adm, 'MUACofchildatadmission': muac_adm,
                       'dur_trt': dur_trt, 'cage_adm': cage_adm}
    knn_raw = np.array([[values_by_name[c] for c in knn_cols]])
    knn_scaled = bundle['knn_scaler'].transform(knn_raw)
    dist = np.linalg.norm(bundle['knn_reference_X'] - knn_scaled, axis=1)
    nearest_idx = np.argsort(dist)[:bundle['knn_k']]
    neighbor_labels = bundle['knn_reference_y'][nearest_idx]
    row['knn_any_loo'] = float((neighbor_labels > 0).mean())
    row['knn_sam_loo'] = float((neighbor_labels == 2).mean())

    columns = bundle['columns_facility_given'] if use_facility else bundle['columns_facility_blank']
    X = pd.DataFrame([row])
    X = X.reindex(columns=columns, fill_value=0)
    return X


def predict_relapse_risk(raw_input, bundle):
    facility = raw_input.get('facility')
    use_facility = facility is not None and facility in bundle['facility_lookups']['fac_any_loo']['by_facility']

    model = bundle['model_facility_given'] if use_facility else bundle['model_facility_blank']
    X = build_feature_row(raw_input, bundle, use_facility)

    proba = model.predict_proba(X)[0]
    class_names = bundle['class_names']
    probabilities = {name: float(p) for name, p in zip(class_names, proba)}
    predicted_class = class_names[int(np.argmax(proba))]

    predicted_whz = None
    if predicted_class != 'No relapse' and 'model_whz_facility_given' in bundle:
        whz_model = bundle['model_whz_facility_given'] if use_facility else bundle['model_whz_facility_blank']
        predicted_whz = float(whz_model.predict(X)[0])

    return {
        'probabilities': probabilities,
        'predicted_class': predicted_class,
        'used_facility': use_facility,
        'predicted_whz': predicted_whz,
    }
