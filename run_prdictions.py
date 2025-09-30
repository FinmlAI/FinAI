# run_predictions.py
# -------------------------------------------------
# Required installs (if not already done):
# pip install pandas numpy scikit-learn lifelines lightgbm joblib
# -------------------------------------------------

import json
import math
from pprint import pprint
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import lightgbm as lgb


# -------------------------
# Helper utilities
# -------------------------
def safe_get(d: Dict, path: List[str], default: Any = None):
    """Traverse nested dict keys in path; return default if any key missing."""
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p, default)
        if cur is default:
            return default
    return cur


def coalesce(*args):
    """Return the first non-None value from args, else None."""
    for a in args:
        if a is not None:
            return a
    return None


def aggregate_individual_loans(ind_loans: List[Dict]) -> Dict:
    """Return aggregated statistics from individual_loans list."""
    total_emi = 0.0
    max_tenure_left = 0
    count = 0
    for l in (ind_loans or []):
        emi = l.get("emi_amount", 0) or 0
        tenure = l.get("tenure_left_months", 0) or 0
        total_emi += float(emi)
        max_tenure_left = max(max_tenure_left, int(tenure))
        count += 1
    return {"total_individual_emi": total_emi, "max_tenure_left": max_tenure_left, "loan_count": count}


# -------------------------------------------------
# 1) Feature Engineering + predict_single Function
# -------------------------------------------------
def feature_engineering(borrower: Dict) -> np.ndarray:
    """
    Create a flat numerical feature vector from borrower JSON.
    This can be used for models that expect numeric vectors.
    """
    # aliases / tolerant fetching
    credit_health = coalesce(
        safe_get(borrower, ["cibil", "credit_health"]),
        safe_get(borrower, ["cibil", "credit_health_score"]),
        safe_get(borrower, ["cibil", "credit_score"]),
        0
    )
    monthly_salary = coalesce(
        safe_get(borrower, ["aa", "bank", "monthly_salary"]),
        safe_get(borrower, ["aa", "bank", "monthly_inflow_pattern", list(safe_get(borrower, ["aa", "bank", "monthly_inflow_pattern"], {}).keys())[:1] if isinstance(safe_get(borrower, ["aa", "bank", "monthly_inflow_pattern"], {}), dict) else None]),
        0
    ) or 0
    consolidated_monthly_emi = coalesce(
        safe_get(borrower, ["cibil", "consolidated_monthly_emi"]),
        safe_get(borrower, ["aa", "bank", "aggregated_emi_amount"]),
        safe_get(borrower, ["aa", "bank", "consolidated_monthly_emi"]),
        0
    ) or 0

    credit_age_months = coalesce(safe_get(borrower, ["cibil", "credit_age_months"]), 0) or 0
    payment_history_12m = coalesce(safe_get(borrower, ["cibil", "payment_history_12m"]), 0) or 0
    filing_consistency_3y = coalesce(safe_get(borrower, ["aa", "itr", "filing_consistency_3y"]), 0) or 0
    gst_returns_last_3y = coalesce(safe_get(borrower, ["kyc", "gst_returns_last_3y"]), safe_get(borrower, ["kyc", "gst_returns_last_3y"]), 0) or 0
    mca_returns_count = coalesce(safe_get(borrower, ["kyc", "mca_returns_count"]), 0) or 0
    default_count = coalesce(safe_get(borrower, ["cibil", "default_count"]), 0) or 0
    bounce_count = coalesce(safe_get(borrower, ["cibil", "bounce_count"]), 0) or 0

    ind_loans = safe_get(borrower, ["cibil", "individual_loans"], []) or []
    ind_stats = aggregate_individual_loans(ind_loans)

    # create vector (order is important — save order to "features" list if using models)
    feature_list = [
        float(credit_health),
        float(monthly_salary),
        float(consolidated_monthly_emi),
        float(credit_age_months),
        float(payment_history_12m),
        float(filing_consistency_3y),
        float(gst_returns_last_3y),
        float(mca_returns_count),
        float(default_count),
        float(bounce_count),
        float(ind_stats["total_individual_emi"]),
        float(ind_stats["max_tenure_left"]),
        float(ind_stats["loan_count"]),
    ]
    return np.array(feature_list).reshape(1, -1)


def predict_single(x: Dict,
                   surv_model: Optional[Any] = None,
                   surv_kind: str = "cox",
                   features: Optional[List[str]] = None):
    """
    Build a consistent row from nested JSON, compute risk_score using the passed survival model,
    and return a structured dict similar to your earlier format. This version:
      - Accepts alternate JSON keys (credit_health, consolidated_monthly_emi, etc.)
      - Produces a heuristic fallback risk_score when no model or model fails
      - Computes behavior scores from payment history / filing consistency / repayment ratio
    """
    # --- build canonical features dict (keys used downstream) ---
    credit_health = coalesce(
        safe_get(x, ["cibil", "credit_health"]),
        safe_get(x, ["cibil", "credit_health_score"]),
        safe_get(x, ["cibil", "credit_score"]),
        0
    ) or 0.0

    monthly_salary = coalesce(safe_get(x, ["aa", "bank", "monthly_salary"]),
                              safe_get(x, ["aa", "bank", "monthly_inflow_pattern", max(safe_get(x, ["aa", "bank", "monthly_inflow_pattern"], {}).keys())] if isinstance(safe_get(x, ["aa", "bank", "monthly_inflow_pattern"], {}), dict) and safe_get(x, ["aa", "bank", "monthly_inflow_pattern"], {}) else None),
                              0) or 0.0

    consolidated_monthly_emi = coalesce(
        safe_get(x, ["cibil", "consolidated_monthly_emi"]),
        safe_get(x, ["aa", "bank", "aggregated_emi_amount"]),
        safe_get(x, ["aa", "bank", "consolidated_monthly_emi"]),
        0
    ) or 0.0

    income = coalesce(safe_get(x, ["aa", "itr", "annual_income"]), monthly_salary * 12, 0) or 0.0
    employment_length = coalesce(safe_get(x, ["aa", "service_record", "employment_years"]), 0) or 0.0
    existing_loans = coalesce(safe_get(x, ["cibil", "loans_closed"]), 0) or 0
    outstanding_balance = consolidated_monthly_emi or coalesce(safe_get(x, ["aa", "bank", "aggregated_emi_amount"]), 0) or 0.0

    gst_returns_last_3y = coalesce(safe_get(x, ["kyc", "gst_returns_last_3y"]), 0) or 0
    mca_returns_count = coalesce(safe_get(x, ["kyc", "mca_returns_count"]), 0) or 0
    default_count = coalesce(safe_get(x, ["cibil", "default_count"]), 0) or 0
    payment_history_12m = coalesce(safe_get(x, ["cibil", "payment_history_12m"]), 0) or 0
    filing_consistency_3y = coalesce(safe_get(x, ["aa", "itr", "filing_consistency_3y"]), 0) or 0
    bounce_count = coalesce(safe_get(x, ["cibil", "bounce_count"]), 0) or 0
    credit_age_months = coalesce(safe_get(x, ["cibil", "credit_age_months"]), 0) or 0

    ind_stats = aggregate_individual_loans(safe_get(x, ["cibil", "individual_loans"], []))

    row = {
        "income": float(income),
        "credit_score": float(credit_health),
        "employment_length": float(employment_length),
        "existing_loans": int(existing_loans),
        "outstanding_balance": float(outstanding_balance),
        "gst_returns_last_3y": int(gst_returns_last_3y),
        "mca_returns_count": int(mca_returns_count),
        "default_count": int(default_count),
        "payment_history_12m": float(payment_history_12m),
        "filing_consistency_3y": float(filing_consistency_3y),
        "bounce_count": int(bounce_count),
        "credit_age_months": int(credit_age_months),
        "individual_total_emi": float(ind_stats["total_individual_emi"]),
        "individual_max_tenure_left": int(ind_stats["max_tenure_left"]),
        "individual_loan_count": int(ind_stats["loan_count"]),
    }

    # --- produce DataFrame in requested feature order if provided ---
    X_row = pd.DataFrame([row])
    if features:
        # ensure same feature order + fill missing columns with 0
        for f in features:
            if f not in X_row.columns:
                X_row[f] = 0
        X_row = X_row[features]

    # --- model-based risk score (CoxPH / LightGBM fallback) ---
    risk_score = None
    try:
        if surv_model is not None:
            if surv_kind == "cox":
                ph = surv_model.predict_partial_hazard(X_row)
                if hasattr(ph, "iloc"):
                    risk_score = float(ph.iloc[0])
                else:
                    risk_score = float(ph.values.ravel()[0])
            else:
                score = surv_model.predict(X_row)
                risk_score = float(score[0])
    except Exception as e:
        # swallow model failure; will use heuristic fallback
        risk_score = None

    # --- heuristic fallback risk score (lower is better) ---
    if risk_score is None:
        # combine credit score (higher better), payment history, filing consistency, default_count, bounce_count,
        # outstanding_balance relative to income, repayment burden
        # We'll compute a positive risk number where larger means higher risk.
        # careful arithmetic: scale factors chosen to normalize ranges
        try:
            credit_factor = max(0.0, (800.0 - row["credit_score"]) / 800.0)  # 0..1 where higher = worse
            payment_factor = max(0.0, (1.0 - row["payment_history_12m"]))  # 0..1 where higher = worse
            filing_factor = max(0.0, (1.0 - row["filing_consistency_3y"]))  # 0..1
            default_factor = min(1.0, row["default_count"] / 5.0)  # saturate
            bounce_factor = min(1.0, row["bounce_count"] / 5.0)
            # debt to income (monthly): outstanding_balance is monthly consolidated EMI
            dti = (row["outstanding_balance"] * 12.0) / (row["income"] + 1e-9)
            dti_factor = min(2.0, dti) / 2.0  # 0..1 (cap at 2)
            individual_emi_factor = min(1.0, row["individual_total_emi"] / (row["income"] / 12.0 + 1e-9))
            # weighted sum -> risk_score in arbitrary positive range
            risk_score = float(
                3.0 * credit_factor +
                2.5 * payment_factor +
                1.5 * filing_factor +
                2.0 * default_factor +
                1.0 * bounce_factor +
                2.5 * dti_factor +
                1.0 * individual_emi_factor
            )
        except Exception:
            risk_score = 1.0  # safe fallback

    # --- convert risk_score to probabilities (simple monotonic transforms) ---
    # p_default: map risk_score -> [0,1] via logistic-ish transform
    try:
        # keep arithmetic explicit and stable
        p_default = 1.0 / (1.0 + math.exp(-0.8 * (risk_score - 1.5)))
        # clip
        p_default = max(0.0, min(1.0, p_default))
    except Exception:
        p_default = 0.1

    p_secure = max(0.0, 1.0 - p_default)

    # --- behavior scores derived heuristically ---
    timeliness = float(payment_history_12m) if 0 <= payment_history_12m <= 1 else (0.5 if payment_history_12m is None else max(0.0, min(1.0, payment_history_12m)))
    repayment = 1.0 - min(1.0, (row["outstanding_balance"] / (row["income"] / 12.0 + 1e-9))) if row["income"] > 0 else 0.5
    spending = max(0.0, min(1.0, 1.0 - (row["individual_total_emi"] / (row["outstanding_balance"] + 1e-9)))) if (row["outstanding_balance"] + 1e-9) > 0 else 0.5

    behavior_scores = {
        "timeliness": round(float(timeliness), 3),
        "spending": round(float(spending), 3),
        "repayment": round(float(repayment), 3),
    }

    # --- ten-year income & loan projections (same as before but safer numeric ops) ---
    income_now = float(row["income"] or 0.0)
    balance_now = float(row["outstanding_balance"] or 0.0)

    income_projection = {
        "conservative": [round(income_now * (1.02 ** year), 2) for year in range(1, 11)],
        "base": [round(income_now * (1.05 ** year), 2) for year in range(1, 11)],
        "optimistic": [round(income_now * (1.08 ** year), 2) for year in range(1, 11)],
    }

    loan_projection = [round(max(balance_now - (year * (balance_now / 10.0)), 0.0), 2) for year in range(1, 11)]

    # build result
    return {
        "risk_score": float(risk_score),
        "features_used": row,
        "ten_year_plan": {
            "income_projection": income_projection,
            "loan_projection": loan_projection,
            "employment_notes": f"{row['employment_length']} years of work history"
        },
        "probabilities": {
            "p_default": round(float(p_default), 6),
            "p_secure": round(float(p_secure), 6),
            "behavior_scores": behavior_scores
        }
    }


# -------------------------------------------------
# 2) Prepare training data
# -------------------------------------------------
def prepare_training_data(borrowers: List[Dict]):
    """
    Convert nested borrower JSON list into a flat DataFrame for training.
    Returns X (features), y (duration,event), and the list of feature column names used.
    Currently generates dummy survival labels (replace with real labels when available).
    """
    rows = []
    for b in borrowers:
        # use the same canonical keys as predict_single to keep training consistent
        res = predict_single(b, surv_model=None, surv_kind="cox", features=None)
        feat = res.get("features_used", {})
        rows.append(feat)

    df = pd.DataFrame(rows)

    # survival labels (dummy for now) — replace with real labels when available
    np.random.seed(42)
    df["event"] = np.random.randint(0, 2, size=len(df))   # 0=censored, 1=default
    df["duration"] = np.random.randint(24, 120, size=len(df))  # months till event/observation

    features = [c for c in rows[0].keys()]

    X = df[features]
    y = df[["duration", "event"]]
    return X, y, features


# -------------------------------------------------
# 3) Train CoxPH model
# -------------------------------------------------
def train_coxph(X: pd.DataFrame, y: pd.DataFrame):
    df = pd.concat([X, y], axis=1)
    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")
    return cph


# -------------------------------------------------
# 4) Train LightGBM fallback model (regression on duration)
# -------------------------------------------------
def train_lightgbm(X: pd.DataFrame, y: pd.DataFrame):
    y_duration = y["duration"].values
    train_data = lgb.Dataset(X, label=y_duration, free_raw_data=False)

    params = {
        "objective": "regression",
        "metric": "l2",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1
    }
    lgbm_model = lgb.train(params, train_data, num_boost_round=100)
    return lgbm_model


# -------------------------------------------------
# 5) Expand nested fields for CSV and derive user-friendly insights
# -------------------------------------------------
def expand_prediction(res: Dict):
    """
    Take the raw result dict from predict_single and expand nested structures.
    Returns a flat dict ready for CSV output.
    """
    flat = dict(res)  # shallow copy
    ten = res.get("ten_year_plan", {})
    income_proj = ten.get("income_projection", {})
    loan_proj = ten.get("loan_projection", [])
    employment_notes = ten.get("employment_notes", "")

    # expand income projection into year1..year10 for each scenario
    for scenario in ["conservative", "base", "optimistic"]:
        lst = income_proj.get(scenario, [])
        for i in range(1, 11):
            val = lst[i - 1] if i - 1 < len(lst) else None
            flat[f"year{i}_income_{scenario}"] = val

    # expand loan
    for i in range(1, 11):
        val = loan_proj[i - 1] if i - 1 < len(loan_proj) else None
        flat[f"year{i}_loan_outstanding"] = val

    flat["employment_notes"] = employment_notes

    # probabilities & behavior scores
    prob = res.get("probabilities", {}) if isinstance(res.get("probabilities", {}), dict) else {}
    flat["p_default"] = prob.get("p_default", None)
    flat["p_secure"] = prob.get("p_secure", None)

    b = prob.get("behavior_scores", {}) if isinstance(prob.get("behavior_scores", {}), dict) else {}
    for k in ["timeliness", "spending", "repayment"]:
        flat[f"behavior_score_{k}"] = b.get(k, None)

    # derived insights similar to previous logic but produce boolean financially_secure
    behavior_values = [v for v in [flat.get("behavior_score_timeliness"),
                                   flat.get("behavior_score_spending"),
                                   flat.get("behavior_score_repayment")] if v is not None]

    if behavior_values:
        avg = sum(behavior_values) / len(behavior_values)
        if avg >= 0.75:
            flat["user_behavior_shift"] = "Stable"
        elif avg >= 0.6:
            flat["user_behavior_shift"] = "Promising"
        elif avg >= 0.4:
            flat["user_behavior_shift"] = "Reactive"
        else:
            flat["user_behavior_shift"] = "Unstable"
    else:
        flat["user_behavior_shift"] = None

    p_def = flat.get("p_default")
    if p_def is None:
        flat["ability_to_default"] = None
        flat["default_timeframe_months"] = None
    else:
        if p_def < 0.10:
            flat["ability_to_default"] = "Low"
        elif p_def < 0.25:
            flat["ability_to_default"] = "Medium"
        else:
            flat["ability_to_default"] = "High"

        if p_def > 0.25:
            flat["default_timeframe_months"] = 6
        elif p_def > 0.10:
            flat["default_timeframe_months"] = 18
        else:
            flat["default_timeframe_months"] = 36

    p_sec = flat.get("p_secure")
    base_end = flat.get("year10_income_base")
    # financial security: boolean (True/False) to match your sample JSON
    flat["financially_secure"] = bool(p_sec is not None and p_sec > 0.7 and (base_end is None or base_end > 0))

    flat["income_projection_summary"] = {
        "conservative_end": flat.get("year10_income_conservative"),
        "base_end": flat.get("year10_income_base"),
        "optimistic_end": flat.get("year10_income_optimistic"),
    }

    outstanding = res.get("features_used", {}).get("outstanding_balance", 0) or 0
    repayment_strength = flat.get("behavior_score_repayment", 0) or 0
    if repayment_strength >= 0.7:
        flat["loan_projection"] = f"Outstanding likely to reduce from {outstanding:.2f} over the next 10 years"
    else:
        flat["loan_projection"] = f"Outstanding may persist around {outstanding:.2f} without stronger repayment discipline"

    # remove heavy nested fields
    flat.pop("ten_year_plan", None)
    flat.pop("probabilities", None)

    return flat


# -------------------------------------------------
# 6) Human-readable statement generator
# -------------------------------------------------
def generate_statement(row: Dict) -> str:
    """
    Accepts a flattened result row (dict) and returns a textual lender-friendly summary.
    """
    features = row.get("features_used", {}) or {}
    income_now = features.get("income", 0)
    credit_score = features.get("credit_score", 0)
    employment = row.get("employment_notes", "")
    loans = features.get("existing_loans", 0)
    defaults = features.get("default_count", 0)

    conservative = row.get("year10_income_conservative") or 0
    base = row.get("year10_income_base") or 0
    optimistic = row.get("year10_income_optimistic") or 0

    p_default = round((row.get("p_default") or 0) * 100, 1)
    p_secure = round((row.get("p_secure") or 0) * 100, 1)

    behaviour = row.get("user_behavior_shift") or "Unknown"
    ability = row.get("ability_to_default") or "Unknown"
    timeframe = row.get("default_timeframe_months") or "Unknown"
    secure_bool = row.get("financially_secure")
    secure_txt = "Yes" if secure_bool else "No"

    statement = f"""Borrower summary:
- Employment: {employment}
- Current income: ₹{income_now:,.2f}
- Credit score: {credit_score}
- Behavior: {behaviour}
- Ability to default: {ability} (estimated timeframe: {timeframe} months)
- Default probability: {p_default}% | Financially secure: {secure_txt} ({p_secure}%)

10-year income projection (end-year estimates):
- Conservative: ₹{conservative:,.2f}
- Base:        ₹{base:,.2f}
- Optimistic:  ₹{optimistic:,.2f}

Loan projection: {row.get('loan_projection')}

Other notes:
- Existing loans: {loans}
- Historical defaults: {defaults}
"""
    return statement


# -------------------------------------------------
# 7) Main Script
# -------------------------------------------------
if __name__ == "__main__":
    import joblib
    import sys

    # 1) Load training data (input.json should be a list of borrower objects)
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = "input.json"

    with open(input_path, "r", encoding="utf-8") as f:
        borrowers = json.load(f)

    # 2) Prepare training data (dummy labels if you don't have ground truth)
    X, y, features = prepare_training_data(borrowers)

    # 3) Train Cox and LightGBM (wrap in try/except if you already have models)
    print("Training CoxPH...")
    try:
        cox_model = train_coxph(X, y)
        print("CoxPH trained.")
    except Exception as e:
        print("CoxPH training failed:", str(e))
        cox_model = None

    print("Training LightGBM (regression fallback)...")
    try:
        lgbm_model = train_lightgbm(X, y)
        print("LightGBM trained.")
    except Exception as e:
        print("LightGBM training failed:", str(e))
        lgbm_model = None

    # 4) Save trained models + features
    try:
        joblib.dump(cox_model, "cox_model.pkl")
        joblib.dump(lgbm_model, "lgbm_model.pkl")
        joblib.dump(features, "features.pkl")
        print("Models saved: cox_model.pkl, lgbm_model.pkl, features.pkl")
    except Exception as e:
        print("Failed to save models:", str(e))

    # 5) Predictions and expand
    results = []
    for b in borrowers:
        res = predict_single(
            b,
            surv_model=cox_model if cox_model is not None else lgbm_model,
            surv_kind="cox" if cox_model is not None else "lgbm",
            features=features
        )
        flat_res = expand_prediction(res)
        results.append(flat_res)

    # 6) Output print
    print("\nPredictions (flattened):")
    pprint(results)

    # 7) Save predictions to CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv("predictions.csv", index=False)
    print("\nPredictions saved to predictions.csv")

    # 8) Save human-readable statements
    statements = [generate_statement(r) for r in results]
    with open("statements.txt", "w", encoding="utf-8") as f:
        for s in statements:
            f.write(s)
            f.write("\n" + ("-" * 80) + "\n")
    print("Human-readable statements saved to statements.txt")
