import pandas as pd
import json

# Load Excel file
# df = pd.read_excel("Synthetic Data.xlsx")
df = pd.read_excel("FinAI\synthetic_credit_data (2).csv")

# Function to transform one row into nested JSON
def row_to_json(row):
    return {
        "kyc": {
            "pan": row["kyc_pan"],
            "aadhar_verified": bool(row["kyc_aadhar_verified"]),
            "gst_returns_last_3y": row["kyc_gst_returns_last_3y"],
            "gst_turnover": row["kyc_gst_turnover"],
            "mcc_code": row["kyc_mcc_code"],
            "company_name": row["kyc_company_name"],
            "date_of_incorp": str(row["kyc_date_of_incorp"].date()) if pd.notnull(row["kyc_date_of_incorp"]) else None,
            "mca_returns_count": row["kyc_mca_returns_count"],
            "mca_returns_amount": row["kyc_mca_returns_amount"],
        },
        "cibil": {
            "credit_health": row["cibil_credit_health_score"],   # renamed
            "credit_mix_ratio": row["cibil_credit_mix_ratio"],
            "payment_history_12m": row["cibil_payment_history_12m"],
            "default_count": row["cibil_default_count"],
            "bounce_count": row["cibil_bounce_count"],
            "loans_closed": row["cibil_loans_closed"],
            "consolidated_monthly_emi": row["cibil_consolidated_monthly_emi"],
            "individual_loans": [
                {
                    "emi_amount": row["cibil_individual_loan1_emi"],
                    "tenure_left_months": row["cibil_individual_loan1_tenure"]
                },
                {
                    "emi_amount": row["cibil_individual_loan2_emi"],
                    "tenure_left_months": row["cibil_individual_loan2_tenure"]
                }
            ] if "cibil_individual_loan1_emi" in df.columns else [],
            "credit_age_months": row["cibil_credit_age_months"] if "cibil_credit_age_months" in df.columns else None
        },
        "aa": {
            "bank": {
                "total_outflow": row["aa_bank_total_outflow"],
                "total_inflow": row["aa_bank_total_inflow"],
                "emi_count": row["aa_bank_emi_count"],
                "aggregated_emi_amount": row["aa_bank_aggregated_emi_amount"],
                "monthly_salary": row["aa_bank_monthly_salary"],
                "income_to_emi_ratio": row["aa_bank_income_to_emi_ratio"],
                "monthly_inflow_pattern": {
                    col.replace("aa_bank_monthly_inflow_pattern_", ""): row[col]
                    for col in df.columns if col.startswith("aa_bank_monthly_inflow_pattern_")
                },
                "annual_inflow_outflow": {
                    col.replace("aa_bank_annual_inflow_outflow_", ""): row[col]
                    for col in df.columns if col.startswith("aa_bank_annual_inflow_outflow_")
                }
            },
            "investments": {
                "mutual_fund_value": row["aa_investments_mutual_fund_value"],
                "nps_value": row["aa_investments_nps_value"],
                "pf_value": row["aa_investments_pf_value"] if "aa_investments_pf_value" in df.columns else None,
                "equity_value": row["aa_investments_equity_value"]
            },
            "itr": {
                "annual_income": row["aa_itr_annual_income"],
                "itr_filed": bool(row["aa_itr_itr_filed"]),
                "filing_consistency_3y": row["aa_itr_filing_consistency_3y"]
            },
            "service_record": {
                "employment_years": row["aa_service_record_employment_years"],
                "job_changes": row["aa_service_record_job_changes"]
            },
            "insurance": {
                "insured": bool(row["aa_insurance_insured"]),
                "insured_amount": row["aa_insurance_insured_amount"]
            }
        },
        "labels": {
            "user_behavior_shift": row["labels_user_behavior_shift"],
            "ability_to_default": row["labels_ability_to_default"],
            "default_timeframe_months": row["labels_default_timeframe_months"],
            "financially_secure": bool(row["labels_financially_secure"]) if not pd.isnull(row["labels_financially_secure"]) else None
        }
    }

# Convert all rows
json_data = [row_to_json(row) for _, row in df.iterrows()]

# Save to JSON file
with open("input1.json", "w") as f:
    json.dump(json_data, f, indent=4)
