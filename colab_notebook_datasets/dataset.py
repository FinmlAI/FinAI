"""
dataset_generator.py
----------------------------------------------------
Creates a synthetic credit-risk dataset that matches
the models.InputPayload schema AND adds the four
ground-truth targets expected from the LLM:

1. user_behavior_shift      -> Stable | Promising | Reactive | Unstable
2. ability_to_default       -> Low | Medium | High
3. default_timeframe_months -> int
4. financially_secure       -> bool

Run:  python dataset_generator.py
----------------------------------------------------
"""

import json
import random
from datetime import date
from pathlib import Path
from faker import Faker

fake = Faker("en_IN")        # realistic Indian data

# ---------------------------------------------------------------------
# 1.  HIGH-LEVEL PERSONAS  (drives correlation across fields)
# ---------------------------------------------------------------------
PERSONAS = {
    "stable_salaried": {
        "credit_score": (750, 860),
        "annual_income": (6e5, 25e5),
        "emi_ratio": (0.15, 0.40),
        "job_changes": (0, 2),
        "default_prob": 0.03
    },
    "young_professional": {
        "credit_score": (650, 760),
        "annual_income": (3e5, 9e5),
        "emi_ratio": (0.25, 0.60),
        "job_changes": (1, 5),
        "default_prob": 0.12
    },
    "business_owner": {
        "credit_score": (600, 820),
        "annual_income": (4e5, 60e5),
        "emi_ratio": (0.20, 0.70),
        "job_changes": (0, 2),
        "default_prob": 0.22
    },
    "risky_borrower": {
        "credit_score": (400, 650),
        "annual_income": (2e5, 6e5),
        "emi_ratio": (0.45, 1.30),
        "job_changes": (3, 8),
        "default_prob": 0.45
    },
}

MCC_CODES = ["5411","5912","5999","7011","5812","5045","5734","8011","8021"]
YEAR_LIST = [date.today().year - i for i in (0, 1, 2)]   # last 3 years

# ---------------------------------------------------------------------
# 2.  GENERATOR HELPERS
# ---------------------------------------------------------------------
def gen_pan() -> str:
    return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5)) + \
           "".join(random.choices("0123456789", k=4)) + \
           random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def gen_kyc(persona_key: str) -> dict:
    gst_turnover = random.uniform(10e5, 10e7) if persona_key == "business_owner" else 0
    gst_returns = random.randint(24, 36) if gst_turnover else random.randint(0, 36)
    company_name = fake.company() if persona_key == "business_owner" else ""
    incorp = str(fake.date_between(start_date="-10y", end_date="-1y"))

    return {
        "pan": gen_pan(),
        "aadhar_verified": random.random() > 0.05,  # via Digilocker
        "gst_returns_last_3y": gst_returns,
        "gst_turnover": round(gst_turnover, 2),
        "mcc_code": random.choice(MCC_CODES),
        "company_name": company_name,
        "date_of_incorp": incorp,
        "mca_returns_count": random.randint(0, 12) if company_name else 0,
        "mca_returns_amount": round(random.uniform(1e4, 1e6), 2) if company_name else 0.0,
    }

def gen_cibil(persona_key: str) -> dict:
    persona = PERSONAS[persona_key]
    score = random.uniform(*persona["credit_score"])
    payment_hist = random.uniform(0.85, 1.0) if score > 750 else random.uniform(0.4, 0.9)
    defaults = 0 if random.random() > persona["default_prob"] else random.randint(1, 6)

    annual_income = random.uniform(*persona["annual_income"])
    emi_ratio = random.uniform(*persona["emi_ratio"])
    consolidated_emi = (annual_income / 12) * emi_ratio

    return {
        "credit_health": round(score, 1),
        "credit_mix_ratio": round(random.uniform(0.3, 0.8), 2),
        "payment_history_12m": round(payment_hist, 2),
        "default_count": defaults,
        "bounce_count": random.randint(0, 10) if defaults else random.randint(0, 3),
        "loans_closed": random.randint(0, 6),
        "consolidated_monthly_emi": round(consolidated_emi, 2),
        "individual_loans": [
            {
                "emi_amount": round(random.uniform(3e3, 5e4), 2),
                "tenure_left_months": random.randint(6, 60)
            } for _ in range(random.randint(1, 4))
        ],
        "credit_age_months": random.randint(6, 180)
    }

def gen_bank(persona_key: str) -> dict:
    persona = PERSONAS[persona_key]
    annual_income = random.uniform(*persona["annual_income"])
    monthly_salary = annual_income / 12

    monthly_inflow_pattern, yearly_inflow_outflow = {}, {}
    total_inflow = total_outflow = 0.0

    for year in YEAR_LIST:
        inflow_year, outflow_year = 0, 0
        for month in range(1, 13):
            k = f"{year}-{month:02d}"
            inflow = monthly_salary * random.uniform(0.8, 1.2)
            outflow = inflow * random.uniform(0.7, 0.95)
            monthly_inflow_pattern[k] = round(inflow, 2)
            inflow_year += inflow; outflow_year += outflow
        yearly_inflow_outflow[year] = round(inflow_year / outflow_year, 3)
        total_inflow += inflow_year; total_outflow += outflow_year

    emi_ratio = random.uniform(*persona["emi_ratio"])
    aggregated_emi = monthly_salary * emi_ratio

    return {
        "total_outflow": round(total_outflow, 2),
        "total_inflow": round(total_inflow, 2),
        "emi_count": random.randint(1, 6),
        "aggregated_emi_amount": round(aggregated_emi, 2),
        "monthly_salary": round(monthly_salary, 2),
        "income_to_emi_ratio": round(emi_ratio, 2),
        "monthly_inflow_pattern": monthly_inflow_pattern,
        "annual_inflow_outflow": yearly_inflow_outflow,
    }

def gen_investments(persona_key: str) -> dict:
    income_base = random.uniform(*PERSONAS[persona_key]["annual_income"])
    mult = income_base / 5e5
    return {
        "mutual_fund_value": round(random.uniform(0, 4e5) * mult, 2),
        "nps_value": round(random.uniform(0, 2e5) * mult, 2),
        "pf_value": round(random.uniform(5e4, 10e5) * mult, 2),
        "equity_value": round(random.uniform(0, 20e5) * mult, 2),
    }

def gen_itr(persona_key: str) -> dict:
    income = random.uniform(*PERSONAS[persona_key]["annual_income"])
    itr_consistency = random.uniform(0.3, 0.7) if persona_key == "risky_borrower" else random.uniform(0.7, 1.0)
    return {
        "annual_income": round(income, 2),
        "itr_filed": itr_consistency > 0.6,
        "filing_consistency_3y": round(itr_consistency, 2),
    }

def gen_service_record(persona_key: str) -> dict:
    years = random.uniform(1, 25)
    return {
        "employment_years": round(years, 1),
        "job_changes": random.randint(*PERSONAS[persona_key]["job_changes"]),
    }

def gen_insurance() -> dict:
    insured = random.random() > 0.3
    return {"insured": insured, "insured_amount": round(random.uniform(1e5, 1e7), 2) if insured else 0}

# ---------------------------------------------------------------------
# 3.  RULE-BASED LABEL GENERATION
# ---------------------------------------------------------------------
def derive_labels(cibil: dict, bank: dict) -> dict:
    score = cibil["credit_health"]
    dti = bank["income_to_emi_ratio"]
    defaults, bounces = cibil["default_count"], cibil["bounce_count"]

    if score > 750 and dti < 0.4 and defaults == 0:
        ability = "Low"
    elif score < 600 or dti > 1.0 or defaults > 2:
        ability = "High"
    else:
        ability = "Medium"

    fin_secure = (ability == "Low") and (dti < 0.5)

    if bounces == 0 and defaults == 0:
        shift = "Stable"
    elif score > 700 and defaults <= 1:
        shift = "Promising"
    elif defaults > 0 or bounces > 3:
        shift = "Reactive"
    else:
        shift = "Unstable"

    timeframe = 36 if ability == "Low" else 12 if ability == "Medium" else random.randint(1, 6)

    return {
        "user_behavior_shift": shift,
        "ability_to_default": ability,
        "default_timeframe_months": timeframe,
        "financially_secure": fin_secure,
    }

# ---------------------------------------------------------------------
# 4.  MAIN LOOP
# ---------------------------------------------------------------------
def generate_sample() -> dict:
    persona_key = random.choice(list(PERSONAS.keys()))

    kyc = gen_kyc(persona_key)
    cibil = gen_cibil(persona_key)
    bank = gen_bank(persona_key)
    investments = gen_investments(persona_key)
    itr = gen_itr(persona_key)
    service = gen_service_record(persona_key)
    insurance = gen_insurance()

    aa = {
        "bank": bank,
        "investments": investments,
        "itr": itr,
        "service_record": service,
        "insurance": insurance,
    }

    labels = derive_labels(cibil, bank)

    return {
        "kyc": kyc,
        "cibil": cibil,
        "aa": aa,
        "labels": labels,
    }

def build_dataset(n: int) -> list:
    return [generate_sample() for _ in range(n)]

# ---------------------------------------------------------------------
# 5.  WRITE DATA
# ---------------------------------------------------------------------
n_samples = 1000
outfile_name = "synthetic_credit_data.jsonl"
data = build_dataset(n_samples)

path = Path(outfile_name)
with path.open("w", encoding="utf-8") as f:
    for row in data:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")

print(f"Wrote {n_samples} records to →  {path.resolve()}")
