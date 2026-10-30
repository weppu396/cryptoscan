"""alerts.py - Telegram, email, webhook. Deduped, reasons attached."""
import json
import pathlib
import smtplib
import ssl
from email.mime.text import MIMEText

import pandas as pd
import requests

CONF = pathlib.Path("alert_config.json")
SENT = pathlib.Path("alerts_sent.json")


def config():
    if CONF.exists():
        try:
            return json.loads(CONF.read_text())
        except Exception:
            pass
    return {}


def save_config(c):
    CONF.write_text(json.dumps(c, indent=2))


def _already(key):
    seen = json.loads(SENT.read_text()) if SENT.exists() else {}
    today = pd.Timestamp.now(tz="UTC").date().isoformat()
    if seen.get(key) == today:
        return True
    seen[key] = today
    SENT.write_text(json.dumps(seen))
    return False


def _telegram(c, text):
    tok, chat = c.get("telegram_bot_token"), c.get("telegram_chat_id")
    if not tok or not chat:
        return False
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={"chat_id": chat, "text": text,
                            "parse_mode": "Markdown"}, timeout=20)
        return True
    except Exception:
        return False


def _email(c, subject, text):
    host, user = c.get("smtp_host"), c.get("smtp_user")
    pwd, to = c.get("smtp_pass"), c.get("email_to")
    if not (host and user and pwd and to):
        return False
    try:
        msg = MIMEText(text)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to
        with smtplib.SMTP(host, int(c.get("smtp_port") or 587), timeout=25) as s:
            s.starttls(context=ssl.create_default_context())
            s.login(user, pwd)
            s.send_message(msg)
        return True
    except Exception:
        return False


def _webhook(c, text):
    url = c.get("webhook")
    if not url:
        return False
    try:
        requests.post(url, json={"content": text}, timeout=20)
        return True
    except Exception:
        return False


def send(kind, key, title, body, c=None):
    if key and _already(f"{kind}|{key}"):
        return []
    c = c or config()
    text = f"*{title}*\n{body}\n\n_{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC_"
    used = []
    if _telegram(c, text):
        used.append("telegram")
    if _email(c, title, body):
        used.append("email")
    if _webhook(c, text):
        used.append("webhook")
    return used
