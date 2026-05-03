# 💳 UPI Fraud Detection System using Machine Learning

## 📌 Project Overview
This project is a web-based application that detects fraudulent UPI transactions using Machine Learning. It uses the XGBoost algorithm trained on transaction data to classify whether a transaction is fraudulent or genuine.

With the rapid increase in digital payments, fraud detection systems are essential to prevent financial loss and improve security. This system helps identify suspicious transaction patterns in real-time.

---

## 🎯 Objective
- Detect fraudulent UPI transactions  
- Improve digital payment security  
- Build a real-time ML-based web application  

---

## 🧠 Machine Learning Model
- **Algorithm:** XGBoost Classifier  
- **Dataset:** PaySim dataset (simulated financial transactions)  
- **Output Classes:**  
  - Fraud  
  - Not Fraud  

---

## 🛠️ Tech Stack
- Python  
- Flask (Web Framework)  
- XGBoost  
- Pandas, NumPy  
- HTML, CSS  

---

## 📂 Project Structure
upi-fraud-detection/  
│  
├── app.py                # Main Flask application  
├── model.pkl             # Trained ML model  
├── templates/            # HTML files  
│   └── index.html  
├── static/               # CSS/JS files  
├── dataset/              # Dataset used  
├── requirements.txt      # Required libraries  
└── README.md  

---

## ⚙️ How to Run the Project

1. Clone the repository:
   git clone https://github.com/geethasree9697-stack/upi-fraud-detection.git

2. Navigate to project folder:
   cd upi-fraud-detection

3. Install dependencies:
   pip install -r requirements.txt

4. Run the application:
   python app.py

5. Open browser:
   http://127.0.0.1:5000/

---

## 📊 Input Features
- Transaction Type  
- Amount  
- Old Balance  
- New Balance  
- Other transaction-related details  

---

## 📷 Application Screenshots

### 🖥️ Input Interface
This interface allows users to enter transaction details such as amount, balance, and transaction type.

![Input Interface](https://github.com/user-attachments/assets/3df9a8eb-809e-45e7-8125-7ebb8e0d773c)

---

### 📊 Prediction Output
The system predicts whether the transaction is **Fraud** or **Not Fraud** based on input values.

![Prediction Output](https://github.com/user-attachments/assets/bcd5c9d5-dfb6-4a16-b8d4-0ec0eaf091e0)

---

## 🚀 Future Improvements
- Deploy on cloud (AWS / Render / Railway)  
- Improve model accuracy  
- Add user authentication system  
- Real-time API integration  

---

## 👩‍💻 Author
N. Geetha Sree  

---

## 🔗 GitHub Repository
https://github.com/geethasree9697-stack/upi-fraud-detection
