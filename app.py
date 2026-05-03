from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load trained model
model = joblib.load("model/xgb_model.pkl")

# Load dataset for dashboard statistics
df = pd.read_csv("dataset/paysim.csv", nrows=50000)

# Transaction type encoding
type_map = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username == "geetha" and password == "2001":
            session["user"] = username
            return redirect(url_for("dashboard"))

        else:
            return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect(url_for("login"))

    total_transactions = len(df)
    total_fraud = len(df[df["isFraud"] == 1])
    total_safe = len(df[df["isFraud"] == 0])
    total_amount = df["amount"].sum()

    recent_transactions = df.tail(5).to_dict(orient="records")

    return render_template(
        "dashboard.html",
        total_transactions=total_transactions,
        total_fraud=total_fraud,
        total_safe=total_safe,
        total_amount=total_amount,
        recent_transactions=recent_transactions
    )


# ---------------- FRAUD DETECTION ----------------
@app.route("/", methods=["GET", "POST"])
def detection():

    if "user" not in session:
        return redirect(url_for("login"))

    result = None
    form_data = {}

    if request.method == "POST":

        form_data = request.form.to_dict()

        try:

            step = float(form_data["step"])
            txn_type = type_map[form_data["type"]]
            amount = float(form_data["amount"])
            oldbalanceOrg = float(form_data["oldbalanceOrg"])
            newbalanceOrig = float(form_data["newbalanceOrig"])
            oldbalanceDest = float(form_data["oldbalanceDest"])
            newbalanceDest = float(form_data["newbalanceDest"])

            features = np.array([[step, txn_type, amount,
                                  oldbalanceOrg, newbalanceOrig,
                                  oldbalanceDest, newbalanceDest]])

            prediction = model.predict(features)

            if prediction[0] == 1:
                result = "🚨 Fraud Transaction Detected"
            else:
                result = "✅ Legitimate Transaction"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template(
        "detection.html",
        result=result,
        form_data=form_data
    )


# ---------------- SETTINGS ----------------
@app.route("/settings")
def settings():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("settings.html")


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():

    session.clear()
    return redirect(url_for("login"))


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)