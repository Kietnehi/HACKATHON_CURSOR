import requests


TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"


def send_sos(alert_message: str, danger_score: int, frame_path: str) -> bool:
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print(f"[SOS ALERT] {alert_message} | Score: {danger_score} | Frame: {frame_path}")
        return False
    
    try:
        message = f"🚨 SOS ALERT 🚨\n\n{alert_message}\n\nDanger Score: {danger_score}/100"
        
        send_message_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(send_message_url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        })
        
        send_photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open(frame_path, "rb") as photo:
            requests.post(send_photo_url, data={
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": f"Danger Score: {danger_score}"
            }, files={"photo": photo})
        
        return True
    
    except Exception as e:
        print(f"Failed to send SOS: {e}")
        return False

