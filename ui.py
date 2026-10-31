"""ui.py - theme injection and reusable layout pieces."""
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root{
  --bg:#08090b; --panel:#0f1116; --panel2:#141821; --line:#1f2430;
  --txt:#e7eaf0; --mut:#7f8798; --acc:#14b8a6; --up:#16c784;
  --dn:#ea3943; --warn:#f0b90b;
}
html, body, [data-testid="stAppViewContainer"]{
  background:var(--bg); color:var(--txt);
  font-family:Inter,system-ui,-apple-system,sans-serif;
}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0b0d11,#080a0d);
  border-right:1px solid var(--line);
}
[data-testid="stSidebar"] *{color:var(--txt);}
h1,h2,h3,h4{letter-spacing:-.02em;font-weight:650;}
p, li, label, span{color:var(--txt);}
small, .stCaption, [data-testid="stCaptionContainer"]{color:var(--mut)!important;}

.hero{
  position:relative; overflow:hidden; border:1px solid var(--line);
  border-radius:18px; padding:22px 26px; margin-bottom:14px;
  background:
    radial-gradient(700px 220px at 12% -30%, rgba(20,184,166,.20), transparent 65%),
    radial-gradient(600px 200px at 88% -20%, rgba(240,185,11,.13), transparent 60%),
    linear-gradient(180deg,#10131a,#0b0d12);
}
.hero h1{margin:0;font-size:26px;}
.hero p{margin:6px 0 0;color:var(--mut);font-size:13.5px;}

.card{
  background:linear-gradient(180deg,var(--panel2),var(--panel));
  border:1px solid var(--line); border-radius:14px; padding:14px 16px;
}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--mut);}
.card .v{font-family:'JetBrains Mono',monospace;font-size:21px;font-weight:600;margin-top:6px;}
.up{color:var(--up);} .dn{color:var(--dn);} .warn{color:var(--warn);} .mut{color:var(--mut);}

.pill{
  display:inline-block;font-size:11px;font-weight:600;letter-spacing:.05em;
  padding:3px 9px;border-radius:999px;border:1px solid var(--line);
  background:#12151c;color:var(--mut);
}
.pill.up{border-color:rgba(22,199,132,.35);color:var(--up);background:rgba(22,199,132,.09);}
.pill.dn{border-color:rgba(234,57,67,.35);color:var(--dn);background:rgba(234,57,67,.09);}
.pill.warn{border-color:rgba(240,185,11,.35);color:var(--warn);background:rgba(240,185,11,.09);}

/* live strip */
.live-bar{display:flex;align-items:center;gap:9px;margin:0 0 8px 2px;}
.dot{width:8px;height:8px;border-radius:50%;background:var(--up);
  box-shadow:0 0 0 0 rgba(22,199,132,.65);animation:pulse 1.9s infinite;}
@keyframes pulse{
  0%{box-shadow:0 0 0 0 rgba(22,199,132,.6);}
  70%{box-shadow:0 0 0 9px rgba(22,199,132,0);}
  100%{box-shadow:0 0 0 0 rgba(22,199,132,0);}
}
.live-lbl{font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--mut);}

.mq-wrap{
  overflow:hidden;border:1px solid var(--line);border-radius:12px;
  background:#0b0e13;padding:10px 0;margin-bottom:16px;
}
.mq-track{display:flex;gap:30px;white-space:nowrap;width:max-content;
  animation:scroll 52s linear infinite;}
.mq-wrap:hover .mq-track{animation-play-state:paused;}
@keyframes scroll{from{transform:translateX(0);}to{transform:translateX(-50%);}}
.mq{font-family:'JetBrains Mono',monospace;font-size:12.5px;color:var(--mut);}
.mq b{color:var(--txt);font-weight:600;}

.stButton>button{
  border-radius:10px;border:1px solid var(--line);background:#141821;
  color:var(--txt);font-weight:600;transition:.15s;
}
.stButton>button:hover{border-color:var(--acc);color:var(--acc);}
.stButton>button[kind="primary"]{
  background:linear-gradient(180deg,#17c2ae,#0f9c8b);border-color:#0f9c8b;color:#04120f;
}
.stButton>button[kind="primary"]:hover{filter:brightness(1.08);color:#04120f;}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input{
  background:#0d1015;border:1px solid var(--line);border-radius:9px;color:var(--txt);
}
[data-baseweb="select"]>div{background:#0d1015;border-color:var(--line);border-radius:9px;}
[data-testid="stTabs"] [data-baseweb="tab-list"]{
  gap:4px;border-bottom:1px solid var(--line);background:transparent;
}
[data-testid="stTabs"] [data-baseweb="tab"]{
  border-radius:9px 9px 0 0;color:var(--mut);font-weight:600;font-size:13px;
}
[data-testid="stTabs"] [aria-selected="true"]{color:var(--acc);background:#10141b;}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;}
hr{border-color:var(--line);}
.auth-wrap{text-align:center;margin:40px 0 22px;}
.auth-logo{
  font-size:34px;color:var(--acc);line-height:1;
  text-shadow:0 0 26px rgba(20,184,166,.55);
}
.auth-title{font-size:27px;margin:10px 0 4px;}
.auth-sub{color:var(--mut);font-size:13px;margin:0;}
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace;}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, sub: str, chips: list[tuple[str, str]] | None = None) -> None:
    tags = "".join(f'<span class="pill {t}">{x}</span> ' for x, t in (chips or []))
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{sub}</p>'
        f'<div style="margin-top:12px">{tags}</div></div>',
        unsafe_allow_html=True,
    )


def stat(col, key: str, value: str, tone: str = "") -> None:
    col.markdown(
        f'<div class="card"><div class="k">{key}</div>'
        f'<div class="v {tone}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, sub: str = "") -> None:
    st.markdown(f"#### {title}")
    if sub:
        st.caption(sub)


def live_label(text: str = "live · 30s refresh") -> None:
    st.markdown(
        f'<div class="live-bar"><span class="dot"></span>'
        f'<span class="live-lbl">{text}</span></div>',
        unsafe_allow_html=True,
    )


def marquee(rows: list[dict]) -> None:
    if not rows:
        st.caption("Live prices unavailable.")
        return
    items = "".join(
        f'<span class="mq"><b>{r["symbol"]}</b> ${r["price"]:,.6g} '
        f'<span class="{"up" if r["chg"] >= 0 else "dn"}">'
        f'{r["chg"]:+.2f}%</span></span>'
        for r in rows
    )
    st.markdown(
        f'<div class="mq-wrap"><div class="mq-track">{items}{items}</div></div>',
        unsafe_allow_html=True,
    )
