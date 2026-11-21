"""app_alerts.py - Telegram poster. Credentials from secrets/env only."""
import os

import requests


def _cred(key: str) -> str:
    try:
        import streamlit as st
        if key in st.secrets and st.secrets[key]:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, "")


def post(text: str) -> tuple:
    """(ok, retry_after). Never raises."""
    token, chat = _cred("TELEGRAM_TOKEN"), _cred("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False, None
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat, "text": text,
                                "disable_web_page_preview": True}, timeout=15)
        if r.status_code == 429:
            try:
                return False, float(r.json()["parameters"]["retry_after"])
            except Exception:
                return False, 30.0
        if r.ok and r.json().get("ok"):
            return True, None
        return False, None
    except Exception:
        return False, None
