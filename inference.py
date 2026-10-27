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


def build_feature_row(raw_input, bundle):
    """raw_input: dict of raw field values collected from the Streamlit form.
    Returns a one-row DataFrame with the exact design-matrix columns the RSF model expects.
    """
    facility = raw_input['facility']

    facility_relapse_rate_loo = bundle['facility_relapse_rate_lookup'].get(
        facility, bundle['facility_pop_rate'])
    facility_recovery_speed_loo = bundle['facility_recovery_speed_lookup'].get(
        facility, bundle['facility_pop_gain_mean'])

    knn_cols = bundle['knn_base_cols']
    knn_raw = pd.DataFrame([{c: raw_input[c] for c in knn_cols}])
    knn_scaled = bundle['knn_scaler'].transform(knn_raw)
    dist = np.linalg.norm(bundle['knn_reference_X'] - knn_scaled, axis=1)
    nearest_idx = np.argsort(dist)[:15]
    knn_local_relapse_risk = float(bundle['knn_reference_y'][nearest_idx].mean())

    clus_scaled = bundle['cluster_scaler'].transform(knn_raw)
    cluster_label = int(bundle['cluster_model'].predict(clus_scaled)[0])
    cluster0_flag = 1 if cluster_label == 0 else 0

    whz_rank = percentile_rank(bundle['whz_reference'], raw_input['whz_adm_clean'])
    muac_rank = percentile_rank(bundle['muac_reference'], raw_input['MUACofchildatadmission'])
    whz_muac_discordance = abs(whz_rank - muac_rank)

    whz_recovery_pct = recovery_pct_single(
        raw_input['whz_adm_clean'], raw_input['whz_T0'],
        bundle['WHZ_THRESH'], bundle['FLOOR_WHZ'])

    # Household Vulnerability Index: IPC food security phase plus handwashing access.
    # Hunger scale was dropped from this formula on 2026-10-27 after a direct test
    # showed it changed model fit by 0.0022 at most, confirmed in Analysis 6.
    ipc_num = bundle['ipc_map'][raw_input['ipc_phase_fcs']]
    vulnerability_index_no_hunger = ipc_num + (1 - raw_input['hndwash'])

    row = {
        'facility_relapse_rate_loo': facility_relapse_rate_loo,
        'knn_local_relapse_risk': knn_local_relapse_risk,
        'facility_recovery_speed_loo': facility_recovery_speed_loo,
        'whz_adm_clean': raw_input['whz_adm_clean'],
        'whz_recovery_pct': whz_recovery_pct,
        'MUACofchildatadmission': raw_input['MUACofchildatadmission'],
        'dur_trt': raw_input['dur_trt'],
        'haz_adm_clean': raw_input['haz_adm_clean'],
        'whz_muac_discordance': whz_muac_discordance,
        'vulnerability_index_no_hunger': vulnerability_index_no_hunger,
        'SAMMAM_SAM': 1 if raw_input['SAMMAM'] == 'SAM' else 0,
        'cluster0_flag_1': cluster0_flag,
        'sex_male': 1 if raw_input['sex'] == 'male' else 0,
        'diar_T1_yes': 1 if raw_input['diar_T1'] == 'yes' else 0,
        'prev_wast_treat_yes': 1 if raw_input['prev_wast_treat'] == 'yes' else 0,
        'mat_edu_dichot_none': 1 if raw_input['mat_edu_dichot'] == 'none' else 0,
        'hndwash_1.0': int(raw_input['hndwash']),
        'medication_oral_rehydration_1.0': int(raw_input['medication_oral_rehydration']),
    }
    for region in ['Bari', 'Bay', 'Galgadud', 'Hiran', 'Lower_Juba', 'Mudug']:
        row[f'RegioCEGn_{region}'] = 1 if raw_input['RegioCEGn'] == region else 0

    X = pd.DataFrame([row])
    X = X.reindex(columns=bundle['design_columns'], fill_value=0)
    return X


def predict_relapse_risk(raw_input, bundle, horizon_days=182):
    X = build_feature_row(raw_input, bundle)
    rsf = bundle['rsf_model']

    surv_funcs = rsf.predict_survival_function(X)
    sf = surv_funcs[0]
    survival_at_horizon = float(sf(horizon_days))
    risk_pct = (1 - survival_at_horizon) * 100

    band = 'Relapse' if risk_pct >= 50 else 'No Relapse'

    severity = None
    if band == 'Relapse':
        if risk_pct < 65:
            severity = 'Low'
        elif risk_pct < 80:
            severity = 'Medium'
        else:
            severity = 'High'

    return {
        'risk_pct': risk_pct,
        'band': band,
        'severity': severity,
        'horizon_days': horizon_days,
    }
