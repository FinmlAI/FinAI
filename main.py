# import json
# import pandas as pd
# import joblib
# from flask import Flask, request, jsonify
# # Importing your custom functions from test.py
# from utils import predict_single, expand_prediction, generate_statement

# app = Flask(__name__)

# # --- Load Models & Features Globaly ---
# # We load these once at startup so we don't reload them for every request
# print("Loading models...")
# try:
#     cox_model = joblib.load("cox_model.pkl")
#     lgbm_model = joblib.load("lgbm_model.pkl")
#     features = joblib.load("features.pkl")
#     print("Models loaded successfully.")
# except FileNotFoundError as e:
#     print(f"Error loading model files: {e}")
#     print("Ensure .pkl files are in the same directory.")

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Check if 'file' is present in the request files (form-data)
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part in the request. Use key 'file'."}), 400
            
#         file = request.files['file']
        
#         # If user does not select file, browser also submit an empty part without filename
#         if file.filename == '':
#             return jsonify({"error": "No selected file"}), 400

#         # Load the JSON data from the uploaded file
#         try:
#             new_borrowers = json.load(file)
#         except json.JSONDecodeError:
#             return jsonify({"error": "Invalid JSON format in file"}), 400

#         # Validation: Ensure input is a list
#         if not new_borrowers:
#             return jsonify({"error": "No data provided"}), 400
        
#         if not isinstance(new_borrowers, list):
#             # If user sends a single object, wrap it in a list
#             new_borrowers = [new_borrowers]

#         results = []

#         # Loop through borrowers just like the original script
#         for b in new_borrowers:
#             # 1. Run Prediction
#             res = predict_single(
#                 b,
#                 surv_model=cox_model,
#                 surv_kind="cox",   # using cox as primary per your script
#                 features=features
#             )
            
#             # 2. Expand Prediction
#             flat_res = expand_prediction(res)
            
#             # 3. Generate Statement
#             # We add the statement directly to the result object for the API response
#             statement_text = generate_statement(flat_res)
#             flat_res['generated_statement'] = statement_text
            
#             results.append(flat_res)

#         # Return the results as JSON
#         return jsonify({
#             "status": "success",
#             "count": len(results),
#             "predictions": results
#         })

#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500

# if __name__ == '__main__':
#     # Run the Flask app
#     app.run(debug=True, port=5000)





import json
import pandas as pd
import joblib
import secrets  # <--- ADDED: For generating API keys
from flask import Flask, request, jsonify
from pymongo import MongoClient  # <--- ADDED: For MongoDB connection
from utils import predict_single, expand_prediction, generate_statement

app = Flask(__name__)

# --- MongoDB Setup (ADDED) ---
# Connect to your local MongoDB (or replace string with cloud URI)
client = MongoClient("mongodb+srv://arbazkhanpathan0348:t7dKlC3AIGcypjD9@linkdin.r1tjdyf.mongodb.net/?retryWrites=true&w=majority") 
db = client["lending"]  # Database name
users_collection = db["users"]    # Collection name

# --- Load Models & Features Globaly ---
print("Loading models...")
try:
    cox_model = joblib.load("cox_model.pkl")
    lgbm_model = joblib.load("lgbm_model.pkl")
    features = joblib.load("features.pkl")
    print("Models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Ensure .pkl files are in the same directory.")

# --- NEW ROUTE: Generate API Key (ADDED) ---
@app.route('/generate_key', methods=['POST'])
def generate_key():
    try:
        data = request.get_json()
        email = data.get('email')

        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Check if user already exists
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            return jsonify({
                "message": "User already exists",
                "api_key": existing_user['api_key']
            }), 200

        # Generate new 32-character hex key
        api_key = secrets.token_hex(16)

        # Save to MongoDB with initial count of 0
        user_doc = {
            "email": email,
            "api_key": api_key,
            "usage_count": 0
        }
        users_collection.insert_one(user_doc)

        return jsonify({
            "message": "API Key created successfully",
            "api_key": api_key
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- AUTHENTICATION & TRACKING (ADDED) ---
        # 1. Get API Key from Headers (e.g., 'x-api-key: 12345...')
        api_key = request.headers.get('x-api-key')

        if not api_key:
            return jsonify({"error": "API Key is missing. Provide 'x-api-key' in headers."}), 401

        # 2. Check if key exists in MongoDB
        user = users_collection.find_one({"api_key": api_key})

        if not user:
            return jsonify({"error": "Invalid API Key"}), 401

        # 3. Increment usage count in MongoDB (Atomic operation)
        users_collection.update_one(
            {"api_key": api_key},
            {"$inc": {"usage_count": 1}}
        )
        # -----------------------------------------

        # Check if 'file' is present in the request files (form-data)
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request. Use key 'file'."}), 400
            
        file = request.files['file']
        
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Load the JSON data from the uploaded file
        try:
            new_borrowers = json.load(file)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format in file"}), 400

        # Validation: Ensure input is a list
        if not new_borrowers:
            return jsonify({"error": "No data provided"}), 400
        
        if not isinstance(new_borrowers, list):
            # If user sends a single object, wrap it in a list
            new_borrowers = [new_borrowers]

        results = []

        # Loop through borrowers just like the original script
        for b in new_borrowers:
            # 1. Run Prediction
            res = predict_single(
                b,
                surv_model=cox_model,
                surv_kind="cox",   # using cox as primary per your script
                features=features
            )
            
            # 2. Expand Prediction
            flat_res = expand_prediction(res)
            
            # 3. Generate Statement
            # We add the statement directly to the result object for the API response
            statement_text = generate_statement(flat_res)
            flat_res['generated_statement'] = statement_text
            
            results.append(flat_res)

        # Return the results as JSON
        return jsonify({
            "status": "success",
            "count": len(results),
            "predictions": results
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)