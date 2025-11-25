import json
import math
from pprint import pprint
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Helper function
def coalesce(*args):
    """Return the first non-None value from args, else None."""
    for a in args:
        if a is not None:
            return a
    return None

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



# Main functions
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
