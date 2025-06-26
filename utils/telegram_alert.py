import requests

def send_telegram_alert(message, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print("Telegram alert failed:", response.text)
    except Exception as e:
        print("Error sending Telegram alert:", str(e))

