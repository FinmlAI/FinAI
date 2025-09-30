import json
import pandas as pd
import joblib
from run_predictions import predict_single, expand_prediction, generate_statement

# Load trained models
cox_model = joblib.load("cox_model.pkl")
lgbm_model = joblib.load("lgbm_model.pkl")
features = joblib.load("features.pkl")

# Load new borrowers data
with open("input1.json", "r") as f:
    new_borrowers = json.load(f)

# Run predictions
results = []
for b in new_borrowers:
    res = predict_single(
        b,
        surv_model=cox_model,
        surv_kind="cox",   # or "lgbm" if you want fallback
        features=features
    )
    flat_res = expand_prediction(res)
    results.append(flat_res)

# Save results
df_out = pd.DataFrame(results)
df_out.to_csv("new_predictions.csv", index=False)

# Save statements
statements = [generate_statement(r) for r in results]
with open("new_statements.txt", "w", encoding="utf-8") as f:
    for s in statements:
        f.write(s)
        f.write("\n" + ("-" * 80) + "\n")

print("Predictions saved to new_predictions.csv and new_statements.txt")
