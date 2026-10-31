"""
auth.py - session login for CryptoScan.

Users live in state/users.json with per-user salt and a PBKDF2-SHA256 hash.
Seed the first admin from Streamlit secrets or env vars so a redeploy never
locks you out.
"""
import hashlib
import json
import os
import pathlib
import secrets as pysecrets
import time
from datetime import datetime, timezone

import streamlit as st

STATE = pathlib.Path("./state")
USERS = STATE / "users.json"
ITERATIONS = 200_000
MAX_FAILS = 5
LOCK_SECONDS = 60


# ---------------------------------------------------------------- storage
def _load() -> dict:
    STATE.mkdir(exist_ok=True)
    if not USERS.exists():
        return {}
    try:
        return json.loads(USERS.read_text())
    except Exception:
        return {}


def _save(db: dict) -> None:
    STATE.mkdir(exist_ok=True)
    USERS.write_text(json.dumps(db, indent=2))


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), ITERATIONS
    ).hex()


def _secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


def _seed() -> None:
    """Create the admin account once, from secrets or env."""
    db = _load()
    if db:
        return
    user = _secret("ADMIN_USER") or "admin"
    pw = _secret("ADMIN_PASS") or "changeme123"
    salt = pysecrets.token_hex(16)
    db[user] = {
        "salt": salt,
        "hash": _hash(pw, salt),
        "role": "admin",
        "created": datetime.now(timezone.utc).isoformat(),
    }
    _save(db)


# ---------------------------------------------------------------- accounts
def create_user(username: str, password: str, role: str = "user") -> tuple[bool, str]:
    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username needs at least 3 characters."
    if len(password) < 8:
        return False, "Password needs at least 8 characters."
    db = _load()
    if username in db:
        return False, "That username is taken."
    salt = pysecrets.token_hex(16)
    db[username] = {
        "salt": salt,
        "hash": _hash(password, salt),
        "role": role,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    _save(db)
    return True, "Account created."


def verify(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    db = _load()
    rec = db.get(username)
    if not rec:
        return False, "No such user."
    if pysecrets.compare_digest(rec["hash"], _hash(password, rec["salt"])):
        return True, username
    return False, "Wrong password."


def current_user() -> str | None:
    return st.session_state.get("user")


def is_admin() -> bool:
    u = current_user()
    if not u:
        return False
    return _load().get(u, {}).get("role") == "admin"


def logout() -> None:
    st.session_state.pop("user", None)
    st.session_state.pop("fails", None)
    st.rerun()


# ---------------------------------------------------------------- gate
def _locked() -> tuple[bool, int]:
    until = st.session_state.get("locked_until", 0)
    remaining = int(until - time.time())
    return (remaining > 0, max(remaining, 0))


def _try_login(username: str, password: str) -> None:
    ok, msg = verify(username, password)
    if ok:
        st.session_state["user"] = msg
        st.session_state["fails"] = 0
        st.session_state.pop("locked_until", None)
        st.rerun()
    fails = st.session_state.get("fails", 0) + 1
    st.session_state["fails"] = fails
    if fails >= MAX_FAILS:
        st.session_state["locked_until"] = time.time() + LOCK_SECONDS
        st.session_state["fails"] = 0
    st.session_state["auth_msg"] = msg


def gate() -> str:
    """Render the login screen and stop the script until authed."""
    if current_user():
        return current_user()

    _seed()
    locked, remaining = _locked()

    left, mid, right = st.columns([1, 1.15, 1])
    with mid:
        st.markdown(
            """
            <div class="auth-wrap">
              <div class="auth-logo">◈</div>
              <h1 class="auth-title">CryptoScan</h1>
              <p class="auth-sub">Break-of-structure scanner, backtester and risk desk.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if locked:
            st.warning(f"Too many attempts. Try again in {remaining}s.")

        tab_in, tab_up = st.tabs(["Sign in", "Create account"])

        with tab_in:
            with st.form("login"):
                u = st.text_input("Username", key="li_u")
                p = st.text_input("Password", type="password", key="li_p")
                if st.form_submit_button("Sign in", use_container_width=True, disabled=locked):
                    if u and p:
                        _try_login(u, p)
                    else:
                        st.session_state["auth_msg"] = "Fill both fields."
            msg = st.session_state.get("auth_msg")
            if msg:
                st.caption(msg)

        with tab_up:
            with st.form("signup"):
                nu = st.text_input("Username", key="su_u")
                npw = st.text_input("Password", type="password", key="su_p",
                                    help="At least 8 characters.")
                npw2 = st.text_input("Confirm", type="password", key="su_p2")
                if st.form_submit_button("Create account", use_container_width=True):
                    if npw != npw2:
                        st.caption("Passwords do not match.")
                    else:
                        ok, m = create_user(nu, npw)
                        st.caption(m)
                        if ok:
                            st.session_state["user"] = nu.strip().lower()
                            st.rerun()

        st.caption("First boot creates `admin` with password `changeme123` "
                   "unless you set ADMIN_USER / ADMIN_PASS in secrets.")

    st.stop()
