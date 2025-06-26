import gspread
from oauth2client.service_account import ServiceAccountCredentials

def connect_all_sheets(sheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("google-sheets-key.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open(sheet_name)

    return {
        "trade_log_reliance": sheet.worksheet("Trade Log RELIANCE"),
        "trade_log_hdfc": sheet.worksheet("Trade Log HDFC"),
        "summary_pnl": sheet.worksheet("Summary PnL"),
        "win_ratio": sheet.worksheet("Win Ratio")
    }

def clear_and_initialize_sheet(sheet):
    # Clear existing content
    sheet.clear()

    # Set headers
    headers = ["Date", "Symbol", "Action", "Price", "Quantity", "PnL", "Signal_Type"]
    sheet.append_row(headers)

