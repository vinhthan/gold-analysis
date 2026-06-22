#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOLD PRICE TREND ANALYSIS v3 — Streamlit Web App
Phân tích đa yếu tố vĩ mô + kỹ thuật + mùa vụ
Dự báo xu hướng giá vàng thế giới (XAU/USD) 1–2 tháng tới
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Gold Trend Analysis",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    .gold-header { text-align:center; padding:16px 0 4px 0; }
    .gold-header h1 { color:#FFD700; font-size:1.75rem; font-weight:700;
                      margin:0; letter-spacing:1px; }
    .gold-header p  { color:#8b949e; font-size:0.86rem; margin:3px 0 0 0; }
    [data-testid="metric-container"] {
        background:#161b22; border:1px solid #30363d;
        border-radius:10px; padding:10px 14px; }
    [data-testid="stMetricLabel"] { color:#8b949e !important; font-size:0.78rem !important; }
    [data-testid="stMetricValue"] { color:#e6edf3 !important; font-size:1.35rem !important;
                                    font-weight:700 !important; }
    [data-testid="stMetricDelta"] { font-size:0.88rem !important; }
    .stButton > button { font-weight:700; font-size:1rem; border-radius:8px;
                         border:none; padding:9px 0; width:100%; }
    .signal-box { border-radius:10px; padding:12px 18px; margin:8px 0;
                  font-size:0.90rem; line-height:1.65; }
    .macro-card { background:#161b22; border:1px solid #30363d; border-radius:8px;
                  padding:10px 14px; margin:4px 0; font-size:0.85rem; }
    .stRadio label { color:#e6edf3 !important; }
    hr { border-color:#30363d; }
    h3, h4 { color:#e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

def hex_rgba(hex_c: str, alpha: float) -> str:
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# Trung bình lợi nhuận tháng lịch sử của vàng (dữ liệu 30+ năm)
SEASONAL_BIAS = {
    1: 0.013,   # Jan: mạnh (Chinese New Year, FOMO đầu năm)
    2: 0.004,   # Feb: nhẹ tích cực
    3: -0.002,  # Mar: trung tính
    4: 0.006,   # Apr: tích cực (nhu cầu trang sức Ấn Độ)
    5: -0.006,  # May: yếu
    6: -0.003,  # Jun: hơi yếu
    7: 0.002,   # Jul: trung tính
    8: 0.007,   # Aug: khá mạnh (mùa cưới Ấn Độ)
    9: 0.010,   # Sep: mạnh nhất (safe-haven + mùa cưới)
    10: 0.002,  # Oct: trung tính
    11: -0.002, # Nov: hơi yếu
    12: 0.005,  # Dec: tích cực (cuối năm)
}

MACRO_TICKERS = {
    "DX-Y.NYB": "dxy",       # USD Index — tương quan nghịch mạnh với vàng
    "^TNX":     "yield10y",  # 10Y Treasury Yield — cơ hội giữ vàng vs trái phiếu
    "^VIX":     "vix",       # Fear Index — vàng là safe-haven khi VIX cao
    "^GSPC":    "sp500",     # S&P 500 — risk-on/off indicator
}

# ══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gold() -> tuple[pd.Series, str]:
    """Lấy giá vàng Spot (ưu tiên) hoặc Futures."""
    for ticker in ("XAUUSD=X", "GC=F", "GLD", "IAU"):
        try:
            raw = yf.download(ticker, period="2y", interval="1d",
                              progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close = raw["Close"].dropna()
            if len(close) >= 100:
                return close, ticker
        except Exception:
            continue
    raise RuntimeError("Không tải được dữ liệu giá vàng. Kiểm tra Internet.")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro() -> dict[str, pd.Series]:
    """Lấy tất cả chỉ số vĩ mô một lần."""
    result = {}
    for ticker, key in MACRO_TICKERS.items():
        try:
            raw = yf.download(ticker, period="1y", interval="1d",
                              progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            s = raw["Close"].dropna()
            if len(s) >= 20:
                result[key] = s
        except Exception:
            pass
    return result


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_price():
    """
    Lấy giá vàng LIVE (real-time) từ Yahoo Finance API.
    Khác với yf.download() chỉ trả về giá đóng cửa phiên trước.
    Cùng nguồn XAU/USD như giavang.org → khớp hơn.
    Cache 60 giây để không gọi quá nhiều.
    """
    import urllib.request as _ur, json as _json, ssl as _ssl

    urls = [
        # Yahoo Finance real-time chart API
        "https://query1.finance.yahoo.com/v8/finance/chart/XAUUSD=X?interval=1m&range=1d",
        # Mirror 2
        "https://query2.finance.yahoo.com/v8/finance/chart/GC=F?interval=1m&range=1d",
    ]
    ctx = _ssl.create_default_context()

    for url in urls:
        try:
            req = _ur.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with _ur.urlopen(req, timeout=8, context=ctx) as r:
                data   = _json.load(r)
                meta   = data["chart"]["result"][0]["meta"]
                price  = meta.get("regularMarketPrice") or meta.get("previousClose")
                source = "Yahoo Finance (Live)"
                if price and 500 < float(price) < 20000:
                    return round(float(price), 2), source
        except Exception:
            pass

    return None, ""

# ══════════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def calc_rsi(s: pd.Series, period: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

# ══════════════════════════════════════════════════════════════════════════════
#  MACRO REGIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def macro_regime(macro: dict) -> tuple[int, str, list, dict]:
    """
    Phân tích 4 yếu tố vĩ mô, trả về:
    - score: -8 đến +8
    - label: nhãn môi trường
    - signals: danh sách tín hiệu chi tiết
    - metrics: dict giá trị hiện tại để hiển thị
    """
    score = 0
    signals = []
    metrics = {}

    # ── DXY (USD Index) ────────────────────────────────────────────────────
    if "dxy" in macro:
        d = macro["dxy"]
        cur   = float(d.iloc[-1])
        ma20  = float(d.rolling(20).mean().iloc[-1])
        ma50  = float(d.rolling(50).mean().iloc[-1]) if len(d) >= 50 else ma20
        chg1m = (cur / float(d.iloc[max(-22, -len(d))]) - 1) * 100
        metrics["dxy"] = {"val": cur, "chg": chg1m, "unit": ""}

        if cur < ma20 - 0.5 and cur < ma50:
            score += 2
            signals.append(("✅", f"USD rất yếu — DXY {cur:.1f} dưới cả MA20 & MA50 → hỗ trợ mạnh cho vàng", "green"))
        elif cur < ma20:
            score += 1
            signals.append(("✅", f"USD hơi yếu — DXY {cur:.1f} dưới MA20 → tích cực cho vàng", "green"))
        elif cur > ma20 + 0.5 and cur > ma50:
            score -= 2
            signals.append(("🔴", f"USD rất mạnh — DXY {cur:.1f} trên cả MA20 & MA50 → áp lực giảm vàng", "red"))
        elif cur > ma20:
            score -= 1
            signals.append(("⚠️", f"USD hơi mạnh — DXY {cur:.1f} trên MA20 → áp lực nhẹ", "orange"))
        else:
            signals.append(("➡️", f"USD trung tính — DXY {cur:.1f}", "gray"))
        signals.append(("📊", f"DXY thay đổi 1 tháng: {chg1m:+.1f}%", "gray"))

    # ── 10Y Treasury Yield ────────────────────────────────────────────────
    if "yield10y" in macro:
        y   = macro["yield10y"]
        cur = float(y.iloc[-1])
        chg = cur - float(y.iloc[max(-22, -len(y))])
        metrics["yield10y"] = {"val": cur, "chg": chg, "unit": "%"}

        if cur < 3.5:
            score += 2
            signals.append(("✅", f"Yield 10Y: {cur:.2f}% — thấp → chi phí cơ hội giữ vàng thấp, hỗ trợ tăng", "green"))
        elif cur < 4.0:
            score += 1
            signals.append(("✅", f"Yield 10Y: {cur:.2f}% — trung bình, tương đối tích cực cho vàng", "green"))
        elif cur < 4.5:
            score -= 1
            signals.append(("⚠️", f"Yield 10Y: {cur:.2f}% — hơi cao, áp lực nhẹ lên vàng", "orange"))
        else:
            score -= 2
            signals.append(("🔴", f"Yield 10Y: {cur:.2f}% — cao → trái phiếu hút tiền khỏi vàng", "red"))

        t = "⬆ tăng" if chg > 0.1 else ("⬇ giảm" if chg < -0.1 else "→ ổn định")
        signals.append(("📊", f"Yield 10Y đang {t} ({chg:+.2f}% so tháng trước)", "gray"))

    # ── VIX (Fear Index) ──────────────────────────────────────────────────
    if "vix" in macro:
        v   = macro["vix"]
        cur = float(v.iloc[-1])
        ma20 = float(v.rolling(20).mean().iloc[-1])
        metrics["vix"] = {"val": cur, "chg": None, "unit": ""}

        if cur > 35:
            score += 3
            signals.append(("✅", f"VIX {cur:.0f} — hoảng loạn thị trường → dòng tiền mạnh vào vàng (safe haven)", "green"))
        elif cur > 25:
            score += 2
            signals.append(("✅", f"VIX {cur:.0f} — sợ hãi cao → vàng được hỗ trợ tốt", "green"))
        elif cur > 18:
            score += 1
            signals.append(("✅", f"VIX {cur:.0f} — thị trường lo ngại, hỗ trợ vàng", "green"))
        elif cur < 12:
            score -= 2
            signals.append(("🔴", f"VIX {cur:.0f} — tham lam cực độ → ít dòng tiền vào vàng", "red"))
        elif cur < 16:
            score -= 1
            signals.append(("⚠️", f"VIX {cur:.0f} — thị trường tự tin, ít hỗ trợ vàng", "orange"))
        else:
            signals.append(("➡️", f"VIX {cur:.0f} — trung tính", "gray"))

    # ── S&P 500 (Risk Sentiment) ──────────────────────────────────────────
    if "sp500" in macro:
        s    = macro["sp500"]
        cur  = float(s.iloc[-1])
        ma50 = float(s.rolling(50).mean().iloc[-1]) if len(s) >= 50 else cur
        chg1m = (cur / float(s.iloc[max(-22, -len(s))]) - 1) * 100
        metrics["sp500"] = {"val": cur, "chg": chg1m, "unit": ""}

        if chg1m < -8:
            score += 3
            signals.append(("✅", f"S&P 500 sụt mạnh {chg1m:.1f}% → risk-off, tiền đổ vào vàng mạnh", "green"))
        elif chg1m < -4:
            score += 2
            signals.append(("✅", f"S&P 500 giảm {chg1m:.1f}% → risk-off, hỗ trợ vàng", "green"))
        elif chg1m < -1:
            score += 1
            signals.append(("✅", f"S&P 500 giảm nhẹ {chg1m:.1f}% → tích cực nhẹ cho vàng", "green"))
        elif chg1m > 6:
            score -= 2
            signals.append(("🔴", f"S&P 500 tăng mạnh {chg1m:.1f}% → risk-on, vốn rút khỏi vàng", "red"))
        elif chg1m > 2:
            score -= 1
            signals.append(("⚠️", f"S&P 500 tăng {chg1m:.1f}% → risk-on, vàng kém hấp dẫn hơn", "orange"))
        else:
            signals.append(("➡️", f"S&P 500 {chg1m:+.1f}% (1 tháng) — trung tính", "gray"))

    # ── Nhãn tổng hợp ─────────────────────────────────────────────────────
    if score >= 6:
        label = "RẤT TÍCH CỰC CHO VÀNG"
    elif score >= 3:
        label = "TÍCH CỰC CHO VÀNG"
    elif score <= -6:
        label = "RẤT TIÊU CỰC CHO VÀNG"
    elif score <= -3:
        label = "TIÊU CỰC CHO VÀNG"
    else:
        label = "TRUNG TÍNH"

    return score, label, signals, metrics

# ══════════════════════════════════════════════════════════════════════════════
#  SEASONAL FACTOR
# ══════════════════════════════════════════════════════════════════════════════

def seasonal_factor(last_date_str: str, days: int) -> float:
    """Tính lệch mùa vụ trung bình cho kỳ dự báo."""
    start = pd.Timestamp(last_date_str) + timedelta(days=1)
    month_days: dict[int, int] = {}
    for i in range(days):
        m = (start + timedelta(days=i)).month
        month_days[m] = month_days.get(m, 0) + 1
    return sum(SEASONAL_BIAS.get(m, 0) * cnt / days
               for m, cnt in month_days.items())

# ══════════════════════════════════════════════════════════════════════════════
#  FORECASTING  (3-model ensemble + macro + seasonal)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def forecast(_price_values: np.ndarray, last_date_str: str,
             days: int, macro_score: int = 0) -> tuple:
    price = pd.Series(_price_values)
    train = price.tail(min(365, len(price)))

    # ── Model A: Holt-Winters (trend dài hạn + damping) ──────────────────
    hw_vals = None
    try:
        hw = ExponentialSmoothing(
            train, trend="add", damped_trend=True, seasonal=None
        ).fit(optimized=True, use_brute=False)
        hw_vals = hw.forecast(days).values
    except Exception:
        pass

    # ── Model B: ARIMA(1,1,1) (cấu trúc chuỗi thời gian) ────────────────
    arima_vals = None
    try:
        fit = ARIMA(train, order=(1, 1, 1)).fit()
        arima_vals = fit.forecast(steps=days).values
    except Exception:
        pass

    # ── Model C: Log-return momentum (60 ngày gần, damped) ───────────────
    log_p = np.log(train.values)
    w     = min(60, len(log_p))
    slope = LinearRegression().fit(
        np.arange(w).reshape(-1, 1), log_p[-w:]
    ).coef_[0] * 0.45  # damped để tránh extrapolate quá mức
    lr_vals = np.exp(log_p[-1] + slope * np.arange(1, days + 1))

    # ── Ensemble ──────────────────────────────────────────────────────────
    if hw_vals is not None and arima_vals is not None:
        base = 0.45 * hw_vals + 0.35 * arima_vals + 0.20 * lr_vals
    elif hw_vals is not None:
        base = 0.60 * hw_vals + 0.40 * lr_vals
    elif arima_vals is not None:
        base = 0.60 * arima_vals + 0.40 * lr_vals
    else:
        base = lr_vals

    # ── Macro adjustment: ±0.35% / month / score point ───────────────────
    macro_adj = macro_score * 0.0035 * (days / 30)

    # ── Seasonal adjustment ───────────────────────────────────────────────
    seas_adj = seasonal_factor(last_date_str, days)

    # ── Apply adjustments ─────────────────────────────────────────────────
    combined = base * (1 + macro_adj + seas_adj)

    # ── Confidence interval (~90%) ────────────────────────────────────────
    vol    = train.pct_change().std()
    spread = combined * vol * np.sqrt(np.arange(1, days + 1)) * 1.65

    fut = pd.bdate_range(
        start=pd.Timestamp(last_date_str) + timedelta(days=1), periods=days
    )
    n   = min(len(fut), len(combined))
    idx = fut[:n]

    return (
        pd.Series(combined[:n], index=idx),
        pd.Series((combined - spread)[:n], index=idx),
        pd.Series((combined + spread)[:n], index=idx),
    )

# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_signal(price, ma20, ma50, ma200, rsi,
                   fc_mean, macro_score: int = 0) -> tuple:
    cur  = float(price.iloc[-1])
    r    = float(rsi.iloc[-1])
    fc_e = float(fc_mean.iloc[-1])
    chg  = (fc_e - cur) / cur * 100

    score = 0
    notes = []

    # ── Moving Averages ───────────────────────────────────────────────────
    if cur > float(ma20.iloc[-1]):  score += 1
    if cur > float(ma50.iloc[-1]):  score += 1
    if cur > float(ma200.iloc[-1]):
        score += 1
        notes.append("✅ Trên MA200 — xu hướng dài hạn **tăng**")
    else:
        notes.append("⚠️ Dưới MA200 — xu hướng dài hạn chưa rõ")

    if float(ma20.iloc[-1]) > float(ma50.iloc[-1]):
        score += 1
        notes.append("✅ MA20 > MA50 — tín hiệu tăng ngắn hạn")
    else:
        notes.append("⚠️ MA20 < MA50 — áp lực giảm ngắn hạn")

    # ── RSI ───────────────────────────────────────────────────────────────
    if 45 < r < 70:
        score += 1; notes.append(f"✅ RSI {r:.0f} — động lực tăng ổn định")
    elif r >= 70:
        score -= 1; notes.append(f"⚠️ RSI {r:.0f} — vùng quá mua, cẩn thận điều chỉnh")
    elif r <= 30:
        score += 1; notes.append(f"✅ RSI {r:.0f} — vùng quá bán, khả năng hồi phục")
    else:
        notes.append(f"➡️ RSI {r:.0f} — trung tính")

    # ── Macro score (clamped -2 to +2 để không át hết kỹ thuật) ──────────
    macro_contrib = max(-2, min(2, round(macro_score / 3)))
    score += macro_contrib

    # ── Forecast direction ────────────────────────────────────────────────
    if chg >= 4:
        score += 2; notes.append(f"✅ Dự báo tăng **{chg:.1f}%** (bao gồm điều chỉnh macro + mùa vụ)")
    elif chg >= 1:
        score += 1; notes.append(f"✅ Dự báo tăng nhẹ **{chg:.1f}%**")
    elif chg <= -4:
        score -= 2; notes.append(f"🔴 Dự báo giảm **{abs(chg):.1f}%** — tiêu cực")
    elif chg < -1:
        score -= 1; notes.append(f"⚠️ Dự báo giảm nhẹ **{abs(chg):.1f}%**")
    else:
        notes.append(f"➡️ Dự báo biến động nhẹ **{chg:+.1f}%**")

    if score >= 6:
        return "TĂNG MẠNH", "#3fb950", "🟢", notes
    elif score >= 3:
        return "CÓ XU HƯỚNG TĂNG", "#76ff03", "🟡", notes
    elif score <= -5:
        return "GIẢM MẠNH", "#f85149", "🔴", notes
    elif score <= -2:
        return "CÓ XU HƯỚNG GIẢM", "#ff7b54", "🟠", notes
    else:
        return "ĐI NGANG / TRUNG TÍNH", "#FFD700", "⚪", notes

# ══════════════════════════════════════════════════════════════════════════════
#  CHART
# ══════════════════════════════════════════════════════════════════════════════

def build_chart(p, m20, m50, m200, bbu, bbl, hi52, lo52,
                fc_mean, fc_lo, fc_hi, rsi_s,
                sig_color: str, forecast_days: int) -> go.Figure:

    H   = 150
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.68, 0.32],
        shared_xaxes=True, vertical_spacing=0.04,
    )

    def sl(s): return s.tail(H)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=sl(bbu).index, y=sl(bbu), name="BB Upper",
        line=dict(color="#79c0ff", width=0.5), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sl(bbl).index, y=sl(bbl), name="Bollinger Bands",
        line=dict(color="#79c0ff", width=0.5), fill="tonexty",
        fillcolor="rgba(121,192,255,0.06)"), row=1, col=1)

    # 52W levels
    fig.add_hline(y=hi52, line_dash="dot", line_color="#3fb950", line_width=0.9, opacity=0.5,
                  annotation_text=f"52W High ${hi52:,.0f}",
                  annotation_font_color="#3fb950", annotation_font_size=9, row=1, col=1)
    fig.add_hline(y=lo52, line_dash="dot", line_color="#f85149", line_width=0.9, opacity=0.5,
                  annotation_text=f"52W Low ${lo52:,.0f}",
                  annotation_font_color="#f85149", annotation_font_size=9,
                  annotation_position="bottom right", row=1, col=1)

    # MAs
    fig.add_trace(go.Scatter(x=sl(m200).index, y=sl(m200), name="MA200",
        line=dict(color="#ff7b54", width=1.0, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sl(m50).index, y=sl(m50), name="MA50",
        line=dict(color="#bc8cff", width=1.1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sl(m20).index, y=sl(m20), name="MA20",
        line=dict(color="#f9a825", width=1.1, dash="dash")), row=1, col=1)

    # Forecast CI band
    fig.add_trace(go.Scatter(x=fc_hi.index, y=fc_hi, line=dict(width=0),
        showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_lo.index, y=fc_lo,
        name="Vùng dự báo (90% CI)", line=dict(width=0),
        fill="tonexty", fillcolor=hex_rgba(sig_color, 0.16)), row=1, col=1)

    # Price history
    fig.add_trace(go.Scatter(x=sl(p).index, y=sl(p),
        name="Giá Vàng (XAU/USD)", line=dict(color="#c9d1d9", width=2.2),
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Giá: $%{y:,.0f}<extra></extra>"), row=1, col=1)

    # Forecast line
    ml = f"{forecast_days // 30} tháng"
    fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean, name=f"Dự báo {ml}",
        line=dict(color=sig_color, width=2.6),
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Dự báo: $%{y:,.0f}<extra></extra>"), row=1, col=1)

    # TODAY divider
    fig.add_vline(x=p.index[-1].timestamp() * 1000,
                  line_dash="dash", line_color="rgba(255,255,255,0.18)", line_width=1)

    # Current price marker
    cur_p = float(p.iloc[-1])
    fig.add_trace(go.Scatter(x=[p.index[-1]], y=[cur_p], mode="markers+text",
        marker=dict(color="#FFD700", size=10),
        text=[f"  Hôm nay ${cur_p:,.0f}"],
        textposition="middle right", textfont=dict(color="#FFD700", size=10),
        showlegend=False, hoverinfo="skip"), row=1, col=1)

    # Forecast end marker
    fc_e = float(fc_mean.iloc[-1])
    chg  = (fc_e - cur_p) / cur_p * 100
    sign = "+" if chg >= 0 else ""
    fig.add_trace(go.Scatter(x=[fc_mean.index[-1]], y=[fc_e], mode="markers+text",
        marker=dict(color=sig_color, size=9, symbol="diamond"),
        text=[f"  ${fc_e:,.0f} ({sign}{chg:.1f}%)"],
        textposition="middle right", textfont=dict(color=sig_color, size=10, family="Arial Black"),
        showlegend=False, hoverinfo="skip"), row=1, col=1)

    # RSI
    r = rsi_s.tail(H)
    fig.add_trace(go.Scatter(x=r.index, y=r, name="RSI (14)",
        line=dict(color="#ce93d8", width=1.5),
        hovertemplate="RSI: %{y:.1f}<extra></extra>"), row=2, col=1)
    fig.add_hrect(y0=70, y1=90, fillcolor="rgba(248,81,73,0.07)",  line_width=0, row=2, col=1)
    fig.add_hrect(y0=10, y1=30, fillcolor="rgba(63,185,80,0.07)",  line_width=0, row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#f85149", line_width=0.9, opacity=0.6,
                  annotation_text="Quá mua", annotation_font_color="#f85149", annotation_font_size=9,
                  row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#3fb950", line_width=0.9, opacity=0.6,
                  annotation_text="Quá bán", annotation_font_color="#3fb950", annotation_font_size=9,
                  annotation_position="bottom right", row=2, col=1)
    fig.add_hline(y=50, line_color="rgba(255,255,255,0.08)", line_width=0.5, row=2, col=1)
    fig.add_trace(go.Scatter(x=[r.index[-1]], y=[float(r.iloc[-1])], mode="markers",
        marker=dict(color="#ce93d8", size=7),
        showlegend=False, hoverinfo="skip"), row=2, col=1)

    # Layout
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", size=11, family="Segoe UI, Arial"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d",
                        font_color="#e6edf3", font_size=11),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                    font=dict(color="#c9d1d9", size=9),
                    orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=10, r=90, t=10, b=10),
        height=560, xaxis_rangeslider_visible=False, dragmode="pan",
    )
    gd = dict(gridcolor="rgba(255,255,255,0.05)", showline=False,
              zerolinecolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(**gd)
    fig.update_yaxes(**gd)
    fig.update_yaxes(title_text="USD / troy oz", row=1, col=1, title_font_size=9)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[10, 90], title_font_size=9)
    fig.update_xaxes(tickformat="%d/%m/%y", tickangle=-30, row=2, col=1)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="gold-header">
        <h1>🥇 GOLD PRICE TREND ANALYSIS</h1>
        <p>Phân tích đa yếu tố vĩ mô + kỹ thuật + mùa vụ · Dự báo xu hướng XAU/USD</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 4])
    with c1:
        forecast_days = st.radio("Kỳ dự báo:", [30, 60],
            format_func=lambda x: "1 tháng tới" if x == 30 else "2 tháng tới",
            horizontal=True, label_visibility="collapsed")
    with c2:
        if st.button("🔄 Làm mới", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Fetch ─────────────────────────────────────────────────────────────────
    with st.spinner("📡 Đang tải dữ liệu giá vàng & chỉ số vĩ mô..."):
        try:
            price, ticker = fetch_gold()
        except RuntimeError as e:
            st.error(str(e)); return
        macro = fetch_macro()
        live_price, live_src = fetch_live_price()

    # ── Technical indicators ──────────────────────────────────────────────────
    ma20   = price.rolling(20).mean()
    ma50   = price.rolling(50).mean()
    ma200  = price.rolling(200).mean()
    rsi    = calc_rsi(price)
    bb_mid = price.rolling(20).mean()
    bb_std = price.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    hi52   = float(price.tail(252).max())
    lo52   = float(price.tail(252).min())

    # ── Macro regime ──────────────────────────────────────────────────────────
    macro_score, macro_label, macro_signals, macro_metrics = macro_regime(macro)

    # ── Forecast (với macro + seasonal) ──────────────────────────────────────
    with st.spinner("🔮 Chạy mô hình dự báo (HW + ARIMA + Momentum + Macro + Mùa vụ)..."):
        fc_mean, fc_lo_s, fc_hi_s = forecast(
            price.values, str(price.index[-1]), forecast_days, macro_score
        )

    # ── Signal ────────────────────────────────────────────────────────────────
    signal, sig_color, sig_icon, tech_notes = compute_signal(
        price, ma20, ma50, ma200, rsi, fc_mean, macro_score
    )

    cur  = float(price.iloc[-1])
    fc_e = float(fc_mean.iloc[-1])
    chg  = (fc_e - cur) / cur * 100
    sign = "+" if chg >= 0 else ""
    r_cur = float(rsi.iloc[-1])

    # ── Seasonal context ──────────────────────────────────────────────────────
    seas = seasonal_factor(str(price.index[-1]), forecast_days)
    seas_note = f"{'📈' if seas > 0 else '📉'} Yếu tố mùa vụ: {seas*100:+.1f}% " \
                f"({'thuận lợi' if seas > 0 else 'bất lợi'} cho kỳ này)"

    # ═════════════════════════ DISPLAY ════════════════════════════════════════

    # ── Gold metrics ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    # Live price (real-time) ưu tiên hơn giá đóng cửa
    if live_price:
        live_diff = live_price - cur
        m1.metric(
            "⚡ Giá Live (Real-time)",
            f"${live_price:,.2f}",
            f"{live_diff:+.2f} so với phiên trước",
            delta_color="normal",
        )
    else:
        m1.metric("💰 Giá Spot (Close)", f"${cur:,.0f}", f"Nguồn: {ticker}")

    m2.metric(f"📅 Dự báo ({forecast_days//30} tháng)", f"${fc_e:,.0f}",
              f"{sign}{chg:.1f}%", delta_color="normal")
    m3.metric("📊 52W High / Low", f"${hi52:,.0f}", f"Low: ${lo52:,.0f}")
    m4.metric("📈 RSI (14)", f"{r_cur:.0f}",
              "Quá mua ⚠️" if r_cur > 70 else ("Quá bán ✅" if r_cur < 30 else "Bình thường"),
              delta_color="off")

    # Ghi chú về nguồn giá
    if live_price:
        diff_pct = abs(live_price - cur) / cur * 100
        note_color = "#f9a825" if diff_pct > 0.5 else "#8b949e"
        st.markdown(
            f"<p style='font-size:0.78rem;color:{note_color};margin:-8px 0 4px 0;'>"
            f"⚡ Giá live: <b>${live_price:,.2f}</b> ({live_src}) · "
            f"Giá đóng cửa phiên trước: <b>${cur:,.0f}</b> ({ticker}) · "
            f"Lệch: <b>{live_price - cur:+.2f} USD</b>"
            f"</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Macro dashboard ───────────────────────────────────────────────────────
    macro_score_color = "#3fb950" if macro_score >= 3 else \
                        "#f85149" if macro_score <= -3 else "#FFD700"

    st.markdown(
        f"#### 🌐 Môi trường vĩ mô &nbsp; "
        f"<span style='color:{macro_score_color};font-size:0.95rem;'>"
        f"[ {macro_label} · Điểm: {macro_score:+d}/10 ]</span>",
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    def macro_metric(col, label, key, fmt, low_good=False):
        if key not in macro_metrics:
            col.metric(label, "N/A"); return
        m = macro_metrics[key]
        val_str = f"{m['val']:{fmt}}"
        delta   = f"{m['chg']:+.2f}" if m["chg"] is not None else None
        col.metric(label, val_str + m["unit"], delta,
                   delta_color="inverse" if low_good else "normal")

    macro_metric(mc1, "💵 USD Index (DXY)",  "dxy",      ".1f", low_good=True)
    macro_metric(mc2, "📉 Yield 10Y",        "yield10y", ".2f", low_good=True)
    macro_metric(mc3, "😨 VIX (Fear Index)", "vix",      ".0f", low_good=False)
    macro_metric(mc4, "📈 S&P 500",          "sp500",    ",.0f", low_good=False)

    # Macro signals
    with st.expander("📋 Chi tiết phân tích vĩ mô", expanded=False):
        for icon, text, _ in macro_signals:
            st.markdown(f"{icon} {text}")
        st.markdown(f"---\n{seas_note}")

    st.markdown("---")

    # ── Chart ─────────────────────────────────────────────────────────────────
    ml = f"{forecast_days//30} tháng"
    st.markdown(
        f"#### Biểu đồ Giá Vàng · Dự báo {ml} tới · "
        f"Cập nhật {datetime.now():%H:%M %d/%m/%Y}"
    )
    fig = build_chart(price, ma20, ma50, ma200, bb_up, bb_lo, hi52, lo52,
                      fc_mean, fc_lo_s, fc_hi_s, rsi, sig_color, forecast_days)
    st.plotly_chart(fig, use_container_width=True,
                    config={"scrollZoom": True, "responsive": True,
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                            "toImageButtonOptions": {"filename": "gold_chart"}})

    st.markdown("---")

    # ── Signal & Analysis ─────────────────────────────────────────────────────
    sig_bg  = hex_rgba(sig_color, 0.12)
    sig_bdr = hex_rgba(sig_color, 0.45)
    st.markdown(
        f"""<div class="signal-box" style="background:{sig_bg};border:1px solid {sig_bdr};">
        <b style="color:{sig_color};font-size:1.05rem;">{sig_icon} NHẬN ĐỊNH TỔNG HỢP: {signal}</b><br>
        <span style="color:#8b949e;font-size:0.85rem;">
        Giá Spot: <b style="color:#e6edf3;">${cur:,.0f}</b> &nbsp;→&nbsp;
        Dự báo cuối kỳ: <b style="color:{sig_color};">${fc_e:,.0f}</b> ({sign}{chg:.1f}%)
        &nbsp;·&nbsp; Macro: <b style="color:{macro_score_color};">{macro_label}</b>
        </span></div>""",
        unsafe_allow_html=True,
    )

    col_t, col_m = st.columns(2)
    with col_t:
        st.markdown("**📐 Phân tích kỹ thuật:**")
        for note in tech_notes:
            st.markdown(f"- {note}")
    with col_m:
        st.markdown("**🌐 Tín hiệu vĩ mô chính:**")
        shown = 0
        for icon, text, _ in macro_signals:
            if icon not in ("📊",) and shown < 5:
                st.markdown(f"- {icon} {text}")
                shown += 1
        st.markdown(f"- {seas_note}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<p style='color:#484f58;font-size:0.76rem;text-align:center;'>"
        f"Nguồn: Yahoo Finance ({ticker}) · "
        f"Mô hình: Holt-Winters + ARIMA(1,1,1) + Momentum + Macro (DXY/VIX/Yield/SP500) + Mùa vụ · "
        f"⚠️ Chỉ mang tính tham khảo, không phải khuyến nghị đầu tư.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
