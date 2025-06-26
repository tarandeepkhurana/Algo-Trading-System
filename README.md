# Algo-Trading System

This repository contains a modular, Python-based Algo-Trading prototype for two major Indian stocks: **RELIANCE.NS** and **HDFCBANK.NS**. It fetches historical and live data, computes technical indicators, applies an ML model for buy/sell prediction, and logs all trades and summaries to Google Sheets for visualization and analysis.

---

## ✨ Features

- 📉 Fetches live & historical data via `yfinance`
- 📊 Computes indicators: RSI, MACD, MAs, Volume, Bollinger Width, etc.
- 🤖 Predicts buy/sell using a pre-trained ML model (`.pkl`)
- 📢 Sends a Telegram Alert to buy/sell.
- 🧾 Logs trades to **Google Sheets**
- 🧮 Auto-generates summary (PnL) and win-ratio metrics

---

## ⚙️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/tarandeepkhurana/Algo-Trading-System.git
cd Algo-Trading-System
```
### 2. Create & Activate Virtual Enviornment
```bash
python -m venv trading_env
trading_env\Scripts\activate   # On Windows
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Google Sheets Setup
- Go to [Google Developers Console](https://console.developers.google.com/)
- Create a project, enable Google Sheets API and Google Drive API
- Generate credentials (Service Account) and download `credentials.json`
- Share the target Google Sheet with the service account email `(e.g. your-bot@project.iam.gserviceaccount.com)`
- Place `credentials.json` in the project root
### 5. Telegram Bot Setup
- Create a Telegram Bot using `@BotFather`
- Get your `Bot Token` and `Chat ID`
- Place them in `.env` file for security reasons.
### 6. How to Run
```bash
python main.py
```
---
## ⚠️ Important Note

The trained ML model used for generating predictions is **not included** in this repository due to file size and privacy constraints. 

To use the prediction and trading functionalities, you must:
- Train your own model using the data preprocessing and training scripts (already present in the repo), **OR**
- Obtain the original `.pkl` model file and place it in the appropriate `/models/` directory.

Without the model file, the core prediction flow will not function.

---
## 📬 Contact
Reach out if you face any issues or want to enhance it with Flask dashboard or more stocks.  
Let's build smarter trading tools together.  

📧 Email: tarandeepkhurana2005@gmail.com

Thanks for stopping by! 🚀
