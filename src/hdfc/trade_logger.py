import pandas as pd

class TradeLogger:
    def __init__(self):
        self.holding = {}

    def log_trade_hdfc_batch(self, row, action):
        """
        It identifies the trades and return it to process_trades_reliance() fn
        """
        symbol = "HDFC.NS"
        date = row.iloc[0]['Date']
        price = row.iloc[0]['Close']
        quantity = 10
        pnl = ""
        signal_type = "ML-Predicted"

        if action == "Buy":
            self.holding[symbol] = (price, quantity)
            return [date, symbol, "Buy", price, quantity, "", signal_type]

        elif action == "Sell":
            if symbol in self.holding:
                buy_price, buy_qty = self.holding.pop(symbol)
                pnl = round((price - buy_price) * buy_qty, 2)
                return [date, symbol, "Sell", price, buy_qty, pnl, signal_type]
            return None

def process_trades_hdfc(model, df, sheet):
    """
    Makes the prediction for each row and passes it 
    to the log_trade_reliance_batch() fn
    to identify the trades and update the trade log sheet.
    """
    logger = TradeLogger()
    all_log_rows = []

    for i in range(len(df)):
        row = df.iloc[[i]]

        features = row[['RSI', 'MACD', 'MACD_Signal', 'MA10', 'MA20', 'MA50',
                        'Volume', 'Volume_Change_Pct', 'Daily_Return_Pct',
                        'BB_Width', 'Price_Momentum', 'MA_Diff']]

        pred_class = model.predict(features)[0]
        action = "Buy" if pred_class == 1 else "Sell"

        log_row = logger.log_trade_hdfc_batch(row, action)
        if log_row:
            all_log_rows.append(log_row)

    if all_log_rows:
        # Write all rows at once
        sheet.update(f"A2:G{1 + len(all_log_rows)}", all_log_rows)

def log_summary_pnl_hdfc(trade_log_sheet, summary_sheet):
    df = pd.DataFrame(trade_log_sheet.get_all_records())
    sell_df = df[df["Action"] == "Sell"]

    summary_data = [
        ["HDFC", ""],
        ["Metric", "Value"],
        ["Total Trades", len(df)],
        ["Total Buys", len(df[df["Action"] == "Buy"])],
        ["Total Sells", len(sell_df)],
        ["Net PnL", sell_df["PnL"].sum()],
        ["Avg PnL per Sell", round(sell_df["PnL"].mean(), 2) if not sell_df.empty else 0],
        ["Max Profit", sell_df["PnL"].max() if not sell_df.empty else 0],
        ["Max Loss", sell_df["PnL"].min() if not sell_df.empty else 0]
    ]
    summary_sheet.update("D1:E9", summary_data)

def log_win_ratio_hdfc(trade_log_sheet, win_ratio_sheet):
    df = pd.DataFrame(trade_log_sheet.get_all_records())
    sell_df = df[df["Action"] == "Sell"]
    win_count = len(sell_df[sell_df["PnL"] > 0])
    lose_count = len(sell_df[sell_df["PnL"] <= 0])
    total_sells = len(sell_df)
    win_ratio = round((win_count / total_sells) * 100, 2) if total_sells else 0

    win_data = [
        ["HDFC", ""],
        ["Metric", "Value"],
        ["Total Sell Trades", total_sells],
        ["Profitable Trades", win_count],
        ["Losing Trades", lose_count],
        ["Win Ratio (%)", win_ratio]
    ]
    win_ratio_sheet.update("D1:E6", win_data)
