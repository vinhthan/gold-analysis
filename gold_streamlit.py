#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MULTI-ASSET PRICE TREND ANALYSIS v4 — Streamlit Web App
Phân tích đa yếu tố vĩ mô + kỹ thuật + mùa vụ
6 tài sản: 🥇 Vàng · 🥈 Bạc · 🟤 Đồng · 🛢️ Dầu WTI · 💵 USD/VND · ₿ Bitcoin
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
    page_title="Market Trend Analysis",
    page_icon="📊",
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

PERIOD_LABELS = {
    30:  "1 tháng tới",
    60:  "2 tháng tới",
    90:  "3 tháng tới",
    180: "6 tháng tới",
    365: "1 năm tới",
}

# ── Asset metadata ─────────────────────────────────────────────────────────
ASSETS = {
    "XAU": {
        "tab":    "🥇 Vàng",
        "name":   "Vàng (XAU/USD)",
        "short":  "Vàng",
        "unit":   "USD/troy oz",
        "color":  "#FFD700",
        "lo": 1000,   "hi": 20000,
        "prefix": "$", "suffix": "",
        "decimals": 2,
    },
    "XAG": {
        "tab":    "🥈 Bạc",
        "name":   "Bạc (XAG/USD)",
        "short":  "Bạc",
        "unit":   "USD/troy oz",
        "color":  "#A8B8C8",
        "lo": 10,     "hi": 500,
        "prefix": "$", "suffix": "",
        "decimals": 2,
    },
    "HG": {
        "tab":    "🟤 Đồng",
        "name":   "Đồng (HG/USD)",
        "short":  "Đồng",
        "unit":   "USD/lb",
        "color":  "#B87333",
        "lo": 1.5,    "hi": 20,
        "prefix": "$", "suffix": "/lb",
        "decimals": 4,
    },
    "CL": {
        "tab":    "🛢️ Dầu WTI",
        "name":   "Dầu WTI (USD/bbl)",
        "short":  "Dầu WTI",
        "unit":   "USD/bbl",
        "color":  "#4a90d9",
        "lo": 10,     "hi": 300,
        "prefix": "$", "suffix": "",
        "decimals": 2,
    },
    "USDVND": {
        "tab":    "💵 USD/VND",
        "name":   "USD/VND",
        "short":  "USD/VND",
        "unit":   "VND",
        "color":  "#2ecc71",
        "lo": 20000,  "hi": 35000,
        "prefix": "", "suffix": " ₫",
        "decimals": 0,
    },
    "BTC": {
        "tab":    "₿ Bitcoin",
        "name":   "Bitcoin (BTC/USD)",
        "short":  "Bitcoin",
        "unit":   "USD",
        "color":  "#F7931A",
        "lo": 5000,   "hi": 500000,
        "prefix": "$", "suffix": "",
        "decimals": 0,
    },
}

# ── Macro sign per asset (+1 = same direction as DXY-up/yield-up/vix-up/sp500-up)
# Positive sign means the raw macro state benefits this asset
MACRO_SIGNS = {
    "XAU":    {"dxy":-1, "yield10y":-1, "vix":+1, "sp500":-1, "oil":+1, "tips":+1},
    "XAG":    {"dxy":-1, "yield10y":-1, "vix":+1, "sp500":+1, "oil":+1, "tips":+1},
    "HG":     {"dxy":-1, "yield10y":-1, "vix":-1, "sp500":+1, "oil":+1, "tips":-1},
    "CL":     {"dxy":-1, "yield10y":-1, "vix":-1, "sp500":+1, "oil": 0, "tips":-1},
    "USDVND": {"dxy":+1, "yield10y":+1, "vix":-1, "sp500":+1, "oil":-1, "tips":+1},
    "BTC":    {"dxy":-1, "yield10y":-1, "vix":+1, "sp500":+1, "oil": 0, "tips":+1},
}

# ── Seasonal monthly bias per asset (historical average monthly return) ─────
SEASONAL_BIAS = {
    "XAU": {
        1: 0.013, 2: 0.004, 3:-0.002, 4: 0.006, 5:-0.006, 6:-0.003,
        7: 0.002, 8: 0.007, 9: 0.010, 10: 0.002, 11:-0.002, 12: 0.005,
    },
    "XAG": {
        1: 0.010, 2: 0.006, 3:-0.003, 4: 0.008, 5: 0.005, 6:-0.003,
        7:-0.004, 8:-0.002, 9: 0.005, 10: 0.004, 11:-0.002, 12: 0.003,
    },
    "HG": {
        1: 0.018, 2: 0.012, 3: 0.006, 4: 0.002, 5:-0.002, 6:-0.006,
        7:-0.007, 8:-0.003, 9: 0.004, 10: 0.007, 11: 0.003, 12:-0.002,
    },
    "CL": {
        1:-0.003, 2: 0.004, 3: 0.010, 4: 0.014, 5: 0.010, 6: 0.006,
        7: 0.001, 8:-0.005, 9:-0.009, 10:-0.005, 11:-0.003, 12: 0.000,
    },
    "USDVND": {
        1: 0.003, 2: 0.002, 3: 0.000, 4:-0.001, 5: 0.001, 6: 0.002,
        7: 0.001, 8: 0.000, 9:-0.001, 10: 0.000, 11: 0.002, 12:-0.001,
    },
    "BTC": {
        1: 0.030, 2:-0.010, 3: 0.020, 4: 0.015, 5:-0.020, 6:-0.010,
        7: 0.010, 8: 0.005, 9:-0.030, 10: 0.030, 11: 0.020, 12: 0.025,
    },
}

MACRO_TICKERS = {
    "DX-Y.NYB": "dxy",
    "^TNX":     "yield10y",
    "^VIX":     "vix",
    "^GSPC":    "sp500",
    "CL=F":     "oil",
    "TIP":      "tips",
}

# ── Personality Profiles — encoded từ hành vi lịch sử quan sát được ────────
# Scale: -3 (rất tiêu cực) đến +3 (rất tích cực) cho từng đặc điểm
TRUMP_PROFILE = {
    "rate_preference":     -3,  # Luôn muốn lãi suất thấp nhất có thể
    "growth_priority":      3,  # GDP & thị trường > mọi mục tiêu khác
    "inflation_tolerance":  2,  # Chấp nhận lạm phát cao hơn mức thông thường
    "market_as_scorecard":  3,  # S&P 500 = điểm số nhiệm kỳ
    "tariff_weapon":        3,  # Thuế quan = vũ khí đàm phán chính
    "fed_pressure":         3,  # Sẵn sàng tấn công Fed công khai
    "unpredictability":     3,  # Bất ngờ có chủ đích là chiến thuật
    "dollar_preference":   -2,  # Muốn USD yếu để xuất khẩu tốt
    "deal_orientation":     3,  # Mọi thứ đều là deal — sẽ xuống thang nếu có lợi
}

WARSH_PROFILE = {
    "hawkish_bias":         2,  # Tự nhiên là diều hâu (hawk)
    "rules_based":          2,  # Thích Taylor Rule hơn discretion
    "qe_skepticism":        3,  # Phê phán QE mạnh nhất lịch sử Fed
    "credibility_priority": 3,  # Fed credibility > áp lực chính trị
    "wall_street_dna":      2,  # Morgan Stanley background → market-savvy
    "political_savvy":      2,  # Từng ở Nhà Trắng → biết survive chính trị
    "druckenmiller_view":   2,  # Ảnh hưởng từ Druckenmiller: macro dài hạn
    "communication":        2,  # Trực tiếp và rõ ràng hơn Powell
    "trump_resistance":     1,  # Sẽ kháng cự Trump nhưng không đối đầu công khai
}

# Tóm tắt xung đột cấu trúc Trump-Warsh cho UI
TRUMP_WARSH_DYNAMIC = {
    "conflict_level":   3,   # Mâu thuẫn cao (Trump muốn cắt, Warsh muốn giữ)
    "trump_wins_prob":  35,  # Xác suất Trump thuyết phục được Warsh (%)
    "warsh_wins_prob":  50,  # Xác suất Warsh giữ vững lập trường (%)
    "compromise_prob":  15,  # Xác suất thỏa hiệp giữa chừng (%)
}

def period_label(days: int) -> str:
    return PERIOD_LABELS.get(days, f"{days} ngày tới")

def hex_rgba(hex_c: str, alpha: float) -> str:
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def fmt_price(p: float, asset_key: str) -> str:
    """Format a price value for display."""
    a = ASSETS[asset_key]
    d = a["decimals"]
    if d == 0:
        return f"{a['prefix']}{p:,.0f}{a['suffix']}"
    elif d == 2:
        return f"{a['prefix']}{p:,.2f}{a['suffix']}"
    else:
        return f"{a['prefix']}{p:.4f}{a['suffix']}"

# ══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gold() -> tuple[pd.Series, str]:
    """
    Lấy giá vàng Spot lịch sử 2 năm.
    Thứ tự:
      1) Yahoo v8 Chart API trực tiếp (XAUUSD=X)
      2) yfinance XAUUSD=X
      3) GLD ETF → quy đổi sang spot (chính xác, luôn tải được)
      4) Stooq qua pandas_datareader
      5) GC=F với điều chỉnh basis lãi suất
    """
    import urllib.request as _ur, json as _json, ssl as _ssl
    from datetime import datetime as _dt, timedelta as _td

    ctx = _ssl.create_default_context()
    ua  = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    def _raw_to_close(raw):
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.copy(); raw.columns = raw.columns.get_level_values(0)
        s = raw["Close"].dropna()
        return s if len(s) >= 100 else None

    # ── 1) Yahoo Finance v8 Chart API (urllib trực tiếp) ─────────────────────
    def _series_from_v8(data):
        res    = data["chart"]["result"][0]
        dates  = pd.to_datetime(res["timestamp"], unit="s").normalize()
        closes = res["indicators"]["quote"][0]["close"]
        s = pd.Series(closes, index=dates, dtype=float).dropna()
        return s[s > 100]

    for host in ("query1", "query2"):
        try:
            url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/"
                   f"XAUUSD%3DX?interval=1d&range=2y")
            req = _ur.Request(url, headers={"User-Agent": ua, "Accept": "application/json"})
            with _ur.urlopen(req, timeout=20, context=ctx) as r:
                s = _series_from_v8(_json.load(r))
            if len(s) >= 100:
                return s, "XAUUSD=X (spot)"
        except Exception:
            pass

    # ── 2) yfinance XAUUSD=X ─────────────────────────────────────────────────
    for _m in ("download", "ticker"):
        try:
            raw = (yf.download("XAUUSD=X", period="2y", interval="1d",
                               progress=False, auto_adjust=True) if _m == "download"
                   else yf.Ticker("XAUUSD=X").history(period="2y", auto_adjust=True))
            s = _raw_to_close(raw)
            if s is not None:
                return s, "XAUUSD=X (spot)"
        except Exception:
            continue

    # ── 3) GLD ETF → quy đổi spot  (ETF luôn tải được, không bị block) ───────
    #    GLD inception: 18/11/2004, initial ratio 0.10 oz/share, ER=0.40%/năm
    #    Spot ≈ GLD_price / (0.10 × 0.996^years_since_inception)
    try:
        raw_gld = yf.download("GLD", period="2y", interval="1d",
                               progress=False, auto_adjust=True)
        s_gld = _raw_to_close(raw_gld)
        if s_gld is not None:
            inception  = _dt(2004, 11, 18)
            yrs        = (_dt.today() - inception).days / 365.25
            gld_ratio  = 0.10 * (0.996 ** yrs)   # oz vàng per GLD share
            spot       = (s_gld / gld_ratio).dropna()
            spot       = spot[spot > 100]
            if len(spot) >= 100:
                return spot, "GLD→XAU/USD (spot)"
    except Exception:
        pass

    # ── 4) pandas_datareader + Stooq ─────────────────────────────────────────
    try:
        from pandas_datareader import data as _pdr
        start = _dt.today() - _td(days=730)
        df    = _pdr.DataReader("XAUUSD", "stooq", start=start, end=_dt.today())
        s     = df.sort_index()["Close"].dropna()
        s     = s[s > 100]
        if len(s) >= 100:
            return s, "Stooq (XAU/USD spot)"
    except Exception:
        pass

    # ── 5) GC=F với điều chỉnh basis (last resort) ───────────────────────────
    #    Spot ≈ GC=F / (1 + r × 30/365)  — khử premium futures ~$15-25
    try:
        raw_gcf = yf.download("GC=F", period="2y", interval="1d",
                               progress=False, auto_adjust=True)
        s_gcf = _raw_to_close(raw_gcf)
        if s_gcf is not None:
            # Lấy lãi suất ngắn hạn từ ^IRX (13-week T-bill)
            r = 0.045  # fallback 4.5%
            try:
                irx = yf.download("^IRX", period="5d", interval="1d",
                                   progress=False, auto_adjust=True)
                if not irx.empty:
                    r = float(irx["Close"].dropna().iloc[-1]) / 100
            except Exception:
                pass
            basis  = 1 + r * 30 / 365     # ~30 ngày bình quân đến hạn hợp đồng
            spot   = (s_gcf / basis).dropna()
            if len(spot) >= 100:
                return spot, "XAU/USD spot (adj.)"
    except Exception:
        pass

    # ── 6) GC=F thô (absolute last resort) ───────────────────────────────────
    for ticker in ("GC=F", "IAU"):
        try:
            raw = yf.download(ticker, period="2y", interval="1d",
                               progress=False, auto_adjust=True)
            s = _raw_to_close(raw)
            if s is not None:
                return s, ticker
        except Exception:
            continue

    raise RuntimeError("Không tải được dữ liệu giá vàng. Kiểm tra Internet.")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price(asset_key: str) -> tuple[pd.Series, str]:
    """
    Generic price fetcher cho tất cả asset.
    Tự động chọn nguồn phù hợp theo asset_key.
    """
    if asset_key == "XAU":
        return fetch_gold()

    import urllib.request as _ur, json as _json, ssl as _ssl
    from datetime import datetime as _dt, timedelta as _td

    info = ASSETS[asset_key]
    lo, hi = info["lo"], info["hi"]
    ctx = _ssl.create_default_context()
    ua  = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    def _raw_to_close(raw):
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.copy(); raw.columns = raw.columns.get_level_values(0)
        s = raw["Close"].dropna()
        return s if len(s) >= 60 else None

    def _valid(s):
        if s is None or len(s) < 60:
            return False
        v = float(s.iloc[-1])
        return lo * 0.3 < v < hi * 3

    def _v8_series(host, ticker):
        enc = ticker.replace("=", "%3D").replace("-", "%2D")
        url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{enc}?interval=1d&range=2y"
        req = _ur.Request(url, headers={"User-Agent": ua, "Accept": "application/json"})
        with _ur.urlopen(req, timeout=20, context=ctx) as r:
            data = _json.load(r)
        res   = data["chart"]["result"][0]
        dates = pd.to_datetime(res["timestamp"], unit="s").normalize()
        closes = res["indicators"]["quote"][0]["close"]
        s = pd.Series(closes, index=dates, dtype=float).dropna()
        return s[s > 0]

    # ── Bitcoin — BTC-USD luôn tải được ──────────────────────────────────────
    if asset_key == "BTC":
        try:
            raw = yf.download("BTC-USD", period="2y", interval="1d",
                              progress=False, auto_adjust=True)
            s = _raw_to_close(raw)
            if _valid(s):
                return s, "BTC-USD (Yahoo Finance)"
        except Exception:
            pass
        # IBIT ETF fallback (≈0.001 BTC/share, ER 0.25%)
        try:
            raw = yf.download("IBIT", period="1y", interval="1d",
                              progress=False, auto_adjust=True)
            s_etf = _raw_to_close(raw)
            if s_etf is not None:
                inception = _dt(2024, 1, 11)
                yrs = (_dt.today() - inception).days / 365.25
                ratio = 0.001 * (0.9975 ** yrs)
                spot = (s_etf / ratio).dropna()
                if _valid(spot):
                    return spot, "IBIT→BTC (Yahoo Finance)"
        except Exception:
            pass
        raise RuntimeError("Không tải được dữ liệu Bitcoin.")

    # ── USD/VND — Yahoo forex ─────────────────────────────────────────────────
    if asset_key == "USDVND":
        for ticker in ("USDVND=X",):
            for host in ("query1", "query2"):
                try:
                    s = _v8_series(host, ticker)
                    s = s[(s > 20000) & (s < 40000)]
                    if len(s) >= 60:
                        return s, f"{ticker} (Yahoo Finance)"
                except Exception:
                    pass
            try:
                raw = yf.download(ticker, period="2y", interval="1d",
                                  progress=False, auto_adjust=True)
                s = _raw_to_close(raw)
                if s is not None:
                    s = s[(s > 20000) & (s < 40000)]
                    if len(s) >= 60:
                        return s, f"{ticker} (Yahoo Finance)"
            except Exception:
                pass
        raise RuntimeError("Không tải được dữ liệu USD/VND.")

    # ── Bạc (XAG) — XAGUSD=X → SLV ETF → SI=F ───────────────────────────────
    if asset_key == "XAG":
        # 1) XAGUSD=X v8 API
        for host in ("query1", "query2"):
            try:
                s = _v8_series(host, "XAGUSD=X")
                s = s[(s > 10) & (s < 500)]
                if len(s) >= 100:
                    return s, "XAGUSD=X (spot)"
            except Exception:
                pass
        # 2) yfinance XAGUSD=X
        for _m in ("download", "ticker"):
            try:
                raw = (yf.download("XAGUSD=X", period="2y", interval="1d",
                                   progress=False, auto_adjust=True) if _m == "download"
                       else yf.Ticker("XAGUSD=X").history(period="2y", auto_adjust=True))
                s = _raw_to_close(raw)
                if s is not None:
                    s = s[(s > 10) & (s < 500)]
                    if len(s) >= 100:
                        return s, "XAGUSD=X (spot)"
            except Exception:
                continue
        # 3) SLV ETF → spot (inception 21/04/2006, 0.9434 oz/share, ER≈0.5%/năm)
        try:
            raw = yf.download("SLV", period="2y", interval="1d",
                              progress=False, auto_adjust=True)
            s_etf = _raw_to_close(raw)
            if s_etf is not None:
                inception = _dt(2006, 4, 21)
                yrs = (_dt.today() - inception).days / 365.25
                ratio = 0.9434 * (0.995 ** yrs)
                spot = (s_etf / ratio).dropna()
                spot = spot[(spot > 10) & (spot < 500)]
                if len(spot) >= 100:
                    return spot, "SLV→XAG (spot)"
        except Exception:
            pass
        # 4) SI=F futures với basis adjustment
        try:
            raw = yf.download("SI=F", period="2y", interval="1d",
                              progress=False, auto_adjust=True)
            s = _raw_to_close(raw)
            if s is not None:
                r = 0.045
                try:
                    irx = yf.download("^IRX", period="5d", interval="1d",
                                      progress=False, auto_adjust=True)
                    if not irx.empty:
                        r = float(irx["Close"].dropna().iloc[-1]) / 100
                except Exception:
                    pass
                basis = 1 + r * 30 / 365
                spot = (s / basis).dropna()
                if _valid(spot):
                    return spot, "SI=F (adj.)"
        except Exception:
            pass
        raise RuntimeError("Không tải được dữ liệu giá Bạc.")

    # ── Đồng (HG) — HG=F futures ─────────────────────────────────────────────
    if asset_key == "HG":
        # HG=F là hợp đồng tương lai đồng COMEX, đơn vị USD/lb
        for period in ("2y", "1y"):
            try:
                raw = yf.download("HG=F", period=period, interval="1d",
                                  progress=False, auto_adjust=True)
                s = _raw_to_close(raw)
                if s is not None:
                    s = s[(s > 1.0) & (s < 25)]
                    if len(s) >= 60:
                        return s, "HG=F (COMEX)"
            except Exception:
                pass
        # COPX ETF (copper miners) — không phải giá đồng trực tiếp, bỏ qua
        raise RuntimeError("Không tải được dữ liệu giá Đồng.")

    # ── Dầu WTI (CL) — CL=F futures ──────────────────────────────────────────
    if asset_key == "CL":
        try:
            raw = yf.download("CL=F", period="2y", interval="1d",
                              progress=False, auto_adjust=True)
            s = _raw_to_close(raw)
            if s is not None:
                s = s[(s > 5) & (s < 400)]
                if len(s) >= 100:
                    return s, "CL=F (WTI Futures)"
        except Exception:
            pass
        raise RuntimeError("Không tải được dữ liệu giá Dầu WTI.")

    raise RuntimeError(f"Asset không hỗ trợ: {asset_key}")


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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_rates() -> dict:
    """
    Lấy dữ liệu lãi suất từ FRED (St. Louis Fed) — miễn phí, không cần API key.
    Series: FEDFUNDS, DGS2, T10Y2Y, T10Y3M
    """
    import requests as _req
    from io import StringIO as _SI

    series = {
        "fedfunds":  "FEDFUNDS",   # Lãi suất Fed Funds hiện tại
        "yield2y":   "DGS2",       # 2-Year Treasury (predictor tốt nhất của Fed)
        "curve":     "T10Y2Y",     # 10Y - 2Y spread (yield curve)
        "curve_3m":  "T10Y3M",     # 10Y - 3M spread (recession indicator)
    }
    result = {}
    for key, sid in series.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            r = _req.get(url, timeout=12,
                        headers={"User-Agent": "Mozilla/5.0"})
            df = pd.read_csv(_SI(r.text), parse_dates=["DATE"], index_col="DATE")
            s = df.iloc[:, 0]
            s = pd.to_numeric(s, errors="coerce").dropna()
            if len(s) >= 10:
                result[key] = s
        except Exception:
            pass
    return result


def fed_policy_analysis(fred_data: dict, macro: dict) -> dict:
    """
    Phân tích chính sách Fed kết hợp ngoại cảm:
    1. Yield curve thực tế (FRED data)
    2. Momentum 2Y Treasury (thị trường đang bet gì?)
    3. Khoảng cách Fed rate vs Neutral rate (2.5%)
    4. Warsh Hawkish Factor (tính cách diều hâu)
    5. Trump Pressure Factor (áp lực từ Nhà Trắng)

    Returns dict với score, direction, signals, metrics.
    Score: -5 (cắt lãi mạnh) → +5 (tăng lãi mạnh)
    """
    score = 0
    signals = []
    metrics = {}

    # ── 1. Yield Curve (T10Y2Y) — tín hiệu thị trường số 1 ──────────────
    curve_val = None
    for key in ("curve", "curve_3m"):
        if key in fred_data and len(fred_data[key]) > 0:
            curve_val = float(fred_data[key].iloc[-1])
            break

    if curve_val is not None:
        metrics["curve"] = curve_val
        if curve_val < -0.75:
            score += 3
            signals.append(("✅", f"Yield Curve đảo ngược sâu {curve_val:.2f}% → thị trường kỳ vọng cắt lãi mạnh trong 6-12T", "green"))
        elif curve_val < -0.25:
            score += 2
            signals.append(("✅", f"Yield Curve âm {curve_val:.2f}% → áp lực cắt lãi đang tích tụ", "green"))
        elif curve_val < 0:
            score += 1
            signals.append(("✅", f"Yield Curve hơi âm {curve_val:.2f}% → nhẹ nghiêng về cắt lãi", "green"))
        elif curve_val > 1.5:
            score -= 2
            signals.append(("🔴", f"Yield Curve dốc mạnh +{curve_val:.2f}% → chưa cần cắt lãi, tăng trưởng ổn", "red"))
        else:
            signals.append(("➡️", f"Yield Curve +{curve_val:.2f}% — flat/trung tính", "gray"))

    # ── 2. 2Y Treasury Momentum — best forward predictor ─────────────────
    if "yield2y" in fred_data and len(fred_data["yield2y"]) >= 20:
        y2 = fred_data["yield2y"]
        cur_2y  = float(y2.iloc[-1])
        prev_2y = float(y2.iloc[-20])
        chg_2y  = cur_2y - prev_2y
        metrics["yield2y"]     = cur_2y
        metrics["yield2y_chg"] = chg_2y

        if chg_2y < -0.4:
            score += 3
            signals.append(("✅", f"2Y Yield giảm mạnh {chg_2y:.2f}% — thị trường đang bet mạnh cắt lãi", "green"))
        elif chg_2y < -0.15:
            score += 2
            signals.append(("✅", f"2Y Yield giảm {chg_2y:.2f}% — kỳ vọng cắt lãi tăng lên", "green"))
        elif chg_2y < -0.05:
            score += 1
            signals.append(("✅", f"2Y Yield giảm nhẹ {chg_2y:.2f}%", "green"))
        elif chg_2y > 0.4:
            score -= 3
            signals.append(("🔴", f"2Y Yield tăng mạnh {chg_2y:.2f}% — thị trường bet tăng/giữ lãi", "red"))
        elif chg_2y > 0.15:
            score -= 2
            signals.append(("🔴", f"2Y Yield tăng {chg_2y:.2f}% — kỳ vọng tăng lãi", "red"))
        elif chg_2y > 0.05:
            score -= 1
            signals.append(("⚠️", f"2Y Yield tăng nhẹ {chg_2y:.2f}%", "orange"))

    # ── 3. Fed Rate vs Neutral (2.5%) ─────────────────────────────────────
    current_rate = None
    if "fedfunds" in fred_data and len(fred_data["fedfunds"]) > 0:
        current_rate = float(fred_data["fedfunds"].iloc[-1])
        neutral = 2.5
        gap = current_rate - neutral
        metrics["fedfunds"]    = current_rate
        metrics["neutral_gap"] = gap

        if gap > 2.5:
            score += 3
            signals.append(("✅", f"Fed rate {current_rate:.2f}% — cách neutral {gap:.1f}% → áp lực cắt lãi rất lớn", "green"))
        elif gap > 1.5:
            score += 2
            signals.append(("✅", f"Fed rate {current_rate:.2f}% — trên neutral {gap:.1f}%", "green"))
        elif gap > 0.5:
            score += 1
            signals.append(("✅", f"Fed rate {current_rate:.2f}% — hơi trên neutral", "green"))
        elif gap < -0.5:
            score -= 1
            signals.append(("🔴", f"Fed rate {current_rate:.2f}% — dưới neutral → còn dư địa tăng lãi", "red"))

    # ── 4. WARSH HAWKISH FACTOR (ngoại cảm tính cách) ────────────────────
    # Warsh diều hâu → luôn giữ lãi cao hơn kỳ vọng thị trường 1 bậc
    w_hawkish = WARSH_PROFILE["hawkish_bias"]     # +2
    w_cred    = WARSH_PROFILE["credibility_priority"]  # +3 (coi trọng uy tín > áp lực)
    w_resist  = WARSH_PROFILE["trump_resistance"]  # +1
    warsh_net = -(w_hawkish * 0.5 + w_cred * 0.2 + w_resist * 0.1)  # → âm = hawkish
    score += round(warsh_net)
    signals.append(("🦅", f"Warsh Factor: hawkish DNA (bias {warsh_net:.1f}) → lãi cao hơn thị trường kỳ vọng", "orange"))

    # ── 5. TRUMP PRESSURE FACTOR (ngoại cảm áp lực chính trị) ────────────
    trump_pressure = 0
    trump_signal   = ""

    # Trump áp lực mạnh hơn khi thị trường giảm
    if "sp500" in macro and len(macro["sp500"]) >= 22:
        sp_chg = (float(macro["sp500"].iloc[-1]) / float(macro["sp500"].iloc[-22]) - 1) * 100
        metrics["sp500_chg"] = sp_chg
        if sp_chg < -12:
            trump_pressure = 3
            trump_signal   = f"S&P {sp_chg:.1f}% → Trump áp lực tối đa, gần như chắc chắn gây áp lực công khai Warsh"
        elif sp_chg < -6:
            trump_pressure = 2
            trump_signal   = f"S&P {sp_chg:.1f}% → Trump sẽ công khai chỉ trích Fed"
        elif sp_chg < -2:
            trump_pressure = 1
            trump_signal   = f"S&P {sp_chg:.1f}% → Trump bắt đầu gây áp lực ngầm"
        elif sp_chg > 8:
            trump_pressure = -1
            trump_signal   = f"S&P +{sp_chg:.1f}% → Trump hài lòng, ít áp lực Fed"

    # Trump pressure offset bởi Warsh resistance:
    # Warsh sẽ kháng cự được khoảng 50-60% áp lực
    warsh_resistance = WARSH_PROFILE["trump_resistance"]  # 1
    effective_trump  = trump_pressure * (1 - warsh_resistance * 0.3)
    score += round(effective_trump)
    if trump_signal:
        signals.append(("⚡", f"Trump Pressure: {trump_signal}", "orange"))

    metrics["trump_pressure"]   = trump_pressure
    metrics["warsh_net"]        = warsh_net
    metrics["effective_trump"]  = effective_trump

    # ── Clamp & label ─────────────────────────────────────────────────────
    score = max(-5, min(5, score))

    if score >= 3:
        direction = "CẮT LÃI SỚM (6-9T)"
        d_color   = "#3fb950"
        prob_cut  = min(88, 55 + score * 6)
    elif score >= 1:
        direction = "NGHIÊNG VỀ CẮT LÃI"
        d_color   = "#76c3a0"
        prob_cut  = min(70, 52 + score * 6)
    elif score <= -3:
        direction = "GIỮ HOẶC TĂNG LÃI"
        d_color   = "#f85149"
        prob_cut  = max(12, 48 + score * 6)
    elif score <= -1:
        direction = "NGHIÊNG VỀ GIỮ LÃI"
        d_color   = "#ff7b54"
        prob_cut  = max(30, 48 + score * 6)
    else:
        direction = "KHÔNG XÁC ĐỊNH / PHỤ THUỘC DATA MỚI"
        d_color   = "#FFD700"
        prob_cut  = 50

    return {
        "score":        score,
        "direction":    direction,
        "color":        d_color,
        "prob_cut":     round(prob_cut),
        "prob_hold":    round(100 - prob_cut),
        "current_rate": current_rate,
        "curve_val":    curve_val,
        "signals":      signals,
        "metrics":      metrics,
    }


def _comex_days_to_expiry() -> int:
    """Tính số ngày đến khi hết hạn hợp đồng COMEX vàng front-month."""
    import calendar as _cal
    from datetime import date as _date, timedelta as _td
    today = _date.today()
    # Các tháng giao dịch chính của vàng COMEX: 2,4,6,8,10,12
    for year in [today.year, today.year + 1]:
        for month in [2, 4, 6, 8, 10, 12]:
            if year == today.year and month < today.month:
                continue
            # Ngày hết hạn = ngày làm việc thứ 3 từ cuối tháng
            last_day = _cal.monthrange(year, month)[1]
            d, biz = _date(year, month, last_day), 0
            while biz < 3:
                if d.weekday() < 5:
                    biz += 1
                    if biz == 3:
                        break
                d -= _td(days=1)
            days_left = (d - today).days
            if days_left >= 5:
                return days_left
    return 45  # fallback


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_price():
    """
    Lấy giá vàng LIVE spot (XAU/USD).
    Dùng requests (luôn có trên Streamlit Cloud) + nhiều nguồn fallback.
    Cache 60 giây.
    """
    import requests as _req
    from datetime import datetime as _dt2

    def _valid(p):
        try:
            return p is not None and 1000 < float(p) < 20000
        except Exception:
            return False

    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

    session = _req.Session()
    session.headers.update({"User-Agent": ua, "Accept": "application/json",
                             "Accept-Language": "en-US,en;q=0.9"})

    # ── 1) TradingView Scanner — FX_IDC:XAUUSD (cùng nguồn giavang.org) ──────
    try:
        r = session.post(
            "https://scanner.tradingview.com/forex/scan",
            json={"symbols": {"tickers": ["FX_IDC:XAUUSD"], "query": {"types": []}},
                  "columns": ["close"]},
            headers={"Origin": "https://www.tradingview.com",
                     "Referer": "https://www.tradingview.com/"},
            timeout=8
        )
        price = r.json()["data"][0]["d"][0]
        if _valid(price):
            return round(float(price), 2), "TradingView (ICE spot)"
    except Exception:
        pass

    # ── 2) Yahoo Finance XAUUSD=X fast_info (endpoint v7/quote) ─────────────
    try:
        price = yf.Ticker("XAUUSD=X").fast_info.last_price
        if _valid(price):
            return round(float(price), 2), "Yahoo Finance (XAU/USD spot)"
    except Exception:
        pass

    # ── 3) Yahoo Finance v8 chart — XAUUSD=X ─────────────────────────────────
    for host in ("query1", "query2"):
        try:
            r = session.get(
                f"https://{host}.finance.yahoo.com/v8/finance/chart/XAUUSD%3DX"
                f"?interval=1m&range=1d", timeout=8)
            meta  = r.json()["chart"]["result"][0]["meta"]
            price = meta.get("regularMarketPrice") or meta.get("previousClose")
            if _valid(price):
                return round(float(price), 2), "Yahoo Finance (XAU/USD spot)"
        except Exception:
            pass

    # ── 4) Swissquote forex feed (institutional broker, real-time 24/5) ──────
    try:
        r = session.get(
            "https://forex-data-feed.swissquote.com/public-quotes/bboquotes"
            "/instrument/XAU/USD", timeout=8)
        d = r.json()[0]
        price = (float(d["bid"]) + float(d["ask"])) / 2
        if _valid(price):
            return round(price, 2), "Swissquote (XAU/USD spot)"
    except Exception:
        pass

    # ── 5) goldprice.org API ──────────────────────────────────────────────────
    try:
        r = session.get("https://data-asg.goldprice.org/dbXRates/USD",
                        headers={"Origin": "https://goldprice.org",
                                 "Referer": "https://goldprice.org/"},
                        timeout=8)
        for item in r.json().get("items", []):
            if item.get("curr") == "USD":
                price = item.get("xauPrice")
                if _valid(price):
                    return round(float(price), 2), "goldprice.org"
    except Exception:
        pass

    # ── 6) GC=F v8 chart 1m (COMEX giao dịch 24/5) + basis adjustment ─────────
    #    Dùng OHLCV thực từ chart (không phải fast_info/close ngày trước)
    #    Spot ≈ GC=F_live / (1 + (r_rf + r_storage) × t/365)
    for host in ("query1", "query2"):
        try:
            r = session.get(
                f"https://{host}.finance.yahoo.com/v8/finance/chart/GC%3DF"
                f"?interval=1m&range=1d", timeout=10)
            result = r.json()["chart"]["result"][0]
            # Lấy giá 1-phút cuối cùng có dữ liệu (không null)
            closes = result["indicators"]["quote"][0]["close"]
            valid_c = [c for c in closes if c is not None]
            if valid_c:
                gcf_price = valid_c[-1]
                t     = _comex_days_to_expiry()
                basis = 1 + (0.045 + 0.0015) * t / 365
                spot  = gcf_price / basis
                if _valid(spot):
                    return round(spot, 2), f"XAU/USD spot (COMEX-{t}d)"
        except Exception:
            pass

    # ── 7) GC=F fast_info (fallback cuối, có thể stale sau 5 PM ET) ──────────
    try:
        gcf_price = yf.Ticker("GC=F").fast_info.last_price
        if gcf_price and gcf_price > 0:
            t     = _comex_days_to_expiry()
            basis = 1 + (0.045 + 0.0015) * t / 365
            spot  = gcf_price / basis
            if _valid(spot):
                return round(spot, 2), f"XAU/USD spot (COMEX adj.)"
    except Exception:
        pass

    return None, ""


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live(asset_key: str) -> tuple:
    """
    Generic live price fetcher cho tất cả asset.
    """
    if asset_key == "XAU":
        return fetch_live_price()

    import requests as _req
    info = ASSETS[asset_key]
    lo, hi = info["lo"], info["hi"]

    def _valid(p):
        try:
            return p is not None and lo * 0.3 < float(p) < hi * 3
        except Exception:
            return False

    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    session = _req.Session()
    session.headers.update({"User-Agent": ua, "Accept": "application/json"})

    # TradingView scanner symbol per asset
    tv_cfg = {
        "XAG":    ("https://scanner.tradingview.com/forex/scan",   "FX_IDC:XAGUSD"),
        "HG":     ("https://scanner.tradingview.com/futures/scan",  "COMEX:HG1!"),
        "CL":     ("https://scanner.tradingview.com/futures/scan",  "NYMEX:CL1!"),
        "USDVND": ("https://scanner.tradingview.com/forex/scan",   "FX_IDC:USDVND"),
        "BTC":    ("https://scanner.tradingview.com/crypto/scan",   "COINBASE:BTCUSD"),
    }
    if asset_key in tv_cfg:
        tv_url, tv_sym = tv_cfg[asset_key]
        try:
            r = session.post(
                tv_url,
                json={"symbols": {"tickers": [tv_sym], "query": {"types": []}},
                      "columns": ["close"]},
                headers={"Origin": "https://www.tradingview.com",
                         "Referer": "https://www.tradingview.com/"},
                timeout=8
            )
            price = r.json()["data"][0]["d"][0]
            if _valid(price):
                return round(float(price), info["decimals"]), "TradingView"
        except Exception:
            pass

    # Yahoo Finance fast_info
    yf_map = {
        "XAG":    "XAGUSD=X",
        "HG":     "HG=F",
        "CL":     "CL=F",
        "USDVND": "USDVND=X",
        "BTC":    "BTC-USD",
    }
    yf_tk = yf_map.get(asset_key)
    if yf_tk:
        try:
            price = yf.Ticker(yf_tk).fast_info.last_price
            if _valid(price):
                return round(float(price), info["decimals"]), f"Yahoo ({yf_tk})"
        except Exception:
            pass

    # Yahoo v8 chart 1m
    if yf_tk:
        for host in ("query1", "query2"):
            try:
                enc = yf_tk.replace("=", "%3D")
                r = session.get(
                    f"https://{host}.finance.yahoo.com/v8/finance/chart/{enc}"
                    f"?interval=1m&range=1d", timeout=10)
                meta = r.json()["chart"]["result"][0]["meta"]
                price = meta.get("regularMarketPrice") or meta.get("previousClose")
                if _valid(price):
                    return round(float(price), info["decimals"]), "Yahoo Finance v8"
            except Exception:
                pass

    # Bitcoin — CoinGecko (public, luôn accessible)
    if asset_key == "BTC":
        try:
            r = session.get(
                "https://api.coingecko.com/api/v3/simple/price"
                "?ids=bitcoin&vs_currencies=usd", timeout=8)
            price = r.json()["bitcoin"]["usd"]
            if _valid(price):
                return round(float(price), 0), "CoinGecko"
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

def macro_regime(macro: dict, price: pd.Series = None,
                 asset_key: str = "XAU") -> tuple[int, str, list, dict]:
    """
    Phân tích 6 yếu tố vĩ mô + momentum + mean reversion cho bất kỳ asset.
    Dùng MACRO_SIGNS[asset_key] để flip dấu đóng góp score.
    """
    signs  = MACRO_SIGNS.get(asset_key, MACRO_SIGNS["XAU"])
    a_name = ASSETS[asset_key]["short"]
    score  = 0
    signals = []
    metrics = {}

    def _si(contrib):
        """Icon/color theo đóng góp thực (sau khi nhân sign)."""
        return ("✅", "green")   if contrib > 0 else \
               ("🔴", "red")    if contrib < 0 else \
               ("➡️", "gray")

    def _apply(raw_score, key):
        """Nhân raw_score với MACRO_SIGNS và cộng vào score."""
        nonlocal score
        c = raw_score * signs.get(key, 0)
        score += c
        return c

    # ── DXY (USD Index) ────────────────────────────────────────────────────
    if "dxy" in macro:
        d = macro["dxy"]
        cur   = float(d.iloc[-1])
        ma20  = float(d.rolling(20).mean().iloc[-1])
        ma50  = float(d.rolling(50).mean().iloc[-1]) if len(d) >= 50 else ma20
        chg1m = (cur / float(d.iloc[max(-22, -len(d))]) - 1) * 100
        metrics["dxy"] = {"val": cur, "chg": chg1m, "unit": ""}

        if cur < ma20 - 0.5 and cur < ma50:
            c = _apply(2, "dxy"); ico, col = _si(c)
            signals.append((ico, f"USD rất yếu — DXY {cur:.1f} dưới MA20 & MA50"
                            + (f" → hỗ trợ {a_name}" if c > 0 else f" → bất lợi {a_name}"), col))
        elif cur < ma20:
            c = _apply(1, "dxy"); ico, col = _si(c)
            signals.append((ico, f"USD hơi yếu — DXY {cur:.1f} dưới MA20"
                            + (f" → tích cực cho {a_name}" if c > 0 else f" → bất lợi nhẹ {a_name}"), col))
        elif cur > ma20 + 0.5 and cur > ma50:
            c = _apply(-2, "dxy"); ico, col = _si(c)
            signals.append((ico, f"USD rất mạnh — DXY {cur:.1f} trên MA20 & MA50"
                            + (f" → hỗ trợ {a_name}" if c > 0 else f" → áp lực lớn lên {a_name}"), col))
        elif cur > ma20:
            c = _apply(-1, "dxy"); ico, col = _si(c)
            signals.append((ico, f"USD hơi mạnh — DXY {cur:.1f} trên MA20"
                            + (f" → tích cực {a_name}" if c > 0 else f" → áp lực nhẹ {a_name}"), col))
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
            c = _apply(2, "yield10y"); ico, col = _si(c)
            signals.append((ico, f"Yield 10Y: {cur:.2f}% — thấp → chi phí cơ hội thấp, "
                            + (f"hỗ trợ {a_name}" if c > 0 else f"bất lợi {a_name}"), col))
        elif cur < 4.0:
            c = _apply(1, "yield10y"); ico, col = _si(c)
            signals.append((ico, f"Yield 10Y: {cur:.2f}% — trung bình thấp", col))
        elif cur < 4.5:
            c = _apply(-1, "yield10y"); ico, col = _si(c)
            signals.append((ico, f"Yield 10Y: {cur:.2f}% — hơi cao"
                            + (f" → tích cực {a_name}" if c > 0 else f" → áp lực nhẹ {a_name}"), col))
        else:
            c = _apply(-2, "yield10y"); ico, col = _si(c)
            signals.append((ico, f"Yield 10Y: {cur:.2f}% — cao"
                            + (f" → hỗ trợ {a_name}" if c > 0 else f" → áp lực lớn {a_name}"), col))

        t = "⬆ tăng" if chg > 0.1 else ("⬇ giảm" if chg < -0.1 else "→ ổn định")
        signals.append(("📊", f"Yield 10Y đang {t} ({chg:+.2f}% so tháng trước)", "gray"))

    # ── VIX (Fear Index) ──────────────────────────────────────────────────
    if "vix" in macro:
        v   = macro["vix"]
        cur = float(v.iloc[-1])
        metrics["vix"] = {"val": cur, "chg": None, "unit": ""}
        # VIX cao = panic → hỗ trợ safe-haven (vàng, bạc, BTC); bất lợi risk assets (đồng, dầu)
        vix_sign = signs.get("vix", 1)

        if cur > 35:
            c = _apply(3, "vix"); ico, col = _si(c)
            lbl = (f"dòng tiền mạnh vào {a_name} (safe haven)" if c > 0
                   else f"nhu cầu công nghiệp/{a_name} giảm mạnh khi hoảng loạn")
            signals.append((ico, f"VIX {cur:.0f} — hoảng loạn thị trường → {lbl}", col))
        elif cur > 25:
            c = _apply(2, "vix"); ico, col = _si(c)
            signals.append((ico, f"VIX {cur:.0f} — sợ hãi cao → "
                            + (f"{a_name} được hỗ trợ" if c > 0 else f"bất lợi {a_name}"), col))
        elif cur > 18:
            c = _apply(1, "vix"); ico, col = _si(c)
            signals.append((ico, f"VIX {cur:.0f} — thị trường lo ngại → "
                            + (f"hỗ trợ {a_name}" if c > 0 else f"nhẹ bất lợi {a_name}"), col))
        elif cur < 12:
            c = _apply(-2, "vix"); ico, col = _si(c)
            signals.append((ico, f"VIX {cur:.0f} — tham lam cực độ → "
                            + (f"ít dòng tiền vào {a_name}" if c < 0 else f"tích cực {a_name}"), col))
        elif cur < 16:
            c = _apply(-1, "vix"); ico, col = _si(c)
            signals.append((ico, f"VIX {cur:.0f} — thị trường tự tin, "
                            + (f"ít hỗ trợ {a_name}" if c < 0 else f"tốt cho {a_name}"), col))
        else:
            signals.append(("➡️", f"VIX {cur:.0f} — trung tính", "gray"))

    # ── S&P 500 (Risk Sentiment) ──────────────────────────────────────────
    if "sp500" in macro:
        s     = macro["sp500"]
        cur   = float(s.iloc[-1])
        chg1m = (cur / float(s.iloc[max(-22, -len(s))]) - 1) * 100
        metrics["sp500"] = {"val": cur, "chg": chg1m, "unit": ""}
        # S&P tăng = risk-on: xấu cho vàng/XAU, tốt cho đồng/dầu/BTC

        if chg1m < -8:
            c = _apply(3, "sp500"); ico, col = _si(c)
            signals.append((ico, f"S&P 500 sụt mạnh {chg1m:.1f}% → risk-off, "
                            + (f"tiền đổ vào {a_name}" if c > 0 else f"nhu cầu {a_name} giảm theo"), col))
        elif chg1m < -4:
            c = _apply(2, "sp500"); ico, col = _si(c)
            signals.append((ico, f"S&P 500 giảm {chg1m:.1f}% → "
                            + (f"hỗ trợ {a_name}" if c > 0 else f"bất lợi {a_name}"), col))
        elif chg1m < -1:
            c = _apply(1, "sp500"); ico, col = _si(c)
            signals.append((ico, f"S&P 500 giảm nhẹ {chg1m:.1f}%", col))
        elif chg1m > 6:
            c = _apply(-2, "sp500"); ico, col = _si(c)
            signals.append((ico, f"S&P 500 tăng mạnh {chg1m:.1f}% → risk-on, "
                            + (f"tốt cho {a_name}" if c > 0 else f"vốn rút khỏi {a_name}"), col))
        elif chg1m > 2:
            c = _apply(-1, "sp500"); ico, col = _si(c)
            signals.append((ico, f"S&P 500 tăng {chg1m:.1f}% → "
                            + (f"tích cực {a_name}" if c > 0 else f"{a_name} kém hấp dẫn hơn"), col))
        else:
            signals.append(("➡️", f"S&P 500 {chg1m:+.1f}% (1 tháng) — trung tính", "gray"))

    # ── Dầu thô WTI (proxy lạm phát) ─────────────────────────────────────
    if "oil" in macro:
        o     = macro["oil"]
        cur   = float(o.iloc[-1])
        n3m   = max(-66, -len(o))
        ret3m = (cur / float(o.iloc[n3m]) - 1) * 100
        metrics["oil"] = {"val": cur, "chg": ret3m, "unit": "$/bbl"}
        # Với CL chính nó (oil sign = 0) → không tự tính vào score
        if signs.get("oil", 0) != 0:
            if ret3m > 15:
                c = _apply(2, "oil"); ico, col = _si(c)
                signals.append((ico, f"Dầu WTI +{ret3m:.1f}% (3T) → lạm phát tăng → "
                                + (f"hỗ trợ {a_name}" if c > 0 else f"áp lực {a_name}"), col))
            elif ret3m > 5:
                c = _apply(1, "oil"); ico, col = _si(c)
                signals.append((ico, f"Dầu WTI +{ret3m:.1f}% (3T) → áp lực lạm phát", col))
            elif ret3m < -15:
                c = _apply(-2, "oil"); ico, col = _si(c)
                signals.append((ico, f"Dầu WTI {ret3m:.1f}% (3T) → lạm phát hạ nhiệt", col))
            elif ret3m < -5:
                c = _apply(-1, "oil"); ico, col = _si(c)
                signals.append((ico, f"Dầu WTI {ret3m:.1f}% (3T) → giảm áp lực lạm phát", col))
            else:
                signals.append(("➡️", f"Dầu WTI {ret3m:+.1f}% (3T) — ổn định, trung tính", "gray"))
        else:
            signals.append(("➡️", f"Dầu WTI {cur:.1f} USD/bbl (3T: {ret3m:+.1f}%)", "gray"))

    # ── TIPS ETF (lãi suất thực — TIP tăng = lãi suất thực giảm) ────────
    if "tips" in macro:
        t     = macro["tips"]
        cur   = float(t.iloc[-1])
        n3m   = max(-66, -len(t))
        ret3m = (cur / float(t.iloc[n3m]) - 1) * 100
        metrics["tips"] = {"val": cur, "chg": ret3m, "unit": ""}

        if ret3m > 3:
            c = _apply(3, "tips"); ico, col = _si(c)
            signals.append((ico, f"TIPS ETF +{ret3m:.1f}% (3T) → lãi suất thực giảm mạnh → "
                            + (f"hỗ trợ rất lớn {a_name}" if c > 0 else f"áp lực lớn {a_name}"), col))
        elif ret3m > 1:
            c = _apply(2, "tips"); ico, col = _si(c)
            signals.append((ico, f"TIPS ETF +{ret3m:.1f}% (3T) → lãi suất thực hạ", col))
        elif ret3m > 0:
            c = _apply(1, "tips"); ico, col = _si(c)
            signals.append((ico, f"TIPS ETF +{ret3m:.1f}% (3T) → lãi suất thực ổn định/giảm nhẹ", col))
        elif ret3m < -3:
            c = _apply(-3, "tips"); ico, col = _si(c)
            signals.append((ico, f"TIPS ETF {ret3m:.1f}% (3T) → lãi suất thực tăng mạnh → "
                            + (f"hỗ trợ {a_name}" if c > 0 else f"áp lực lớn {a_name}"), col))
        elif ret3m < -1:
            c = _apply(-2, "tips"); ico, col = _si(c)
            signals.append((ico, f"TIPS ETF {ret3m:.1f}% (3T) → lãi suất thực tăng", col))
        else:
            c = _apply(-1, "tips"); ico, col = _si(c)
            signals.append((ico, f"TIPS ETF {ret3m:.1f}% (3T) → lãi suất thực hơi tăng", col))

    # ── Momentum giá tài sản (3T & 6T) ────────────────────────────────────
    if price is not None and len(price) >= 20:
        cur_p  = float(price.iloc[-1])
        n3m    = max(-66,  -len(price))
        n6m    = max(-130, -len(price))
        ret3m  = (cur_p / float(price.iloc[n3m])  - 1) * 100
        ret6m  = (cur_p / float(price.iloc[n6m])  - 1) * 100
        metrics["momentum"] = {"ret3m": ret3m, "ret6m": ret6m}

        if ret3m > 12:
            score += 2
            signals.append(("✅", f"Momentum {a_name} 3 tháng: +{ret3m:.1f}% — đà tăng mạnh", "green"))
        elif ret3m > 4:
            score += 1
            signals.append(("✅", f"Momentum {a_name} 3 tháng: +{ret3m:.1f}% — tích cực", "green"))
        elif ret3m < -8:
            score -= 2
            signals.append(("🔴", f"Momentum {a_name} 3 tháng: {ret3m:.1f}% — đà giảm rõ", "red"))
        elif ret3m < -3:
            score -= 1
            signals.append(("⚠️", f"Momentum {a_name} 3 tháng: {ret3m:.1f}% — áp lực giảm", "orange"))

        if ret6m > 15:
            score += 1
            signals.append(("✅", f"Momentum 6 tháng: +{ret6m:.1f}% — xu hướng tăng trung hạn vững", "green"))
        elif ret6m < -10:
            score -= 1
            signals.append(("🔴", f"Momentum 6 tháng: {ret6m:.1f}% — xu hướng trung hạn tiêu cực", "red"))

    # ── Độ lệch MA200 (mean reversion) ────────────────────────────────────
    if price is not None and len(price) >= 200:
        cur_p  = float(price.iloc[-1])
        ma200v = float(price.rolling(200).mean().iloc[-1])
        dev    = (cur_p / ma200v - 1) * 100
        metrics["ma200_dev"] = dev

        if dev > 25:
            score -= 2
            signals.append(("⚠️", f"Giá cao hơn MA200 {dev:.1f}% — quá mua dài hạn, rủi ro điều chỉnh", "orange"))
        elif dev > 15:
            score -= 1
            signals.append(("⚠️", f"Giá cao hơn MA200 {dev:.1f}% — hơi overbought so lịch sử", "orange"))
        elif dev < -15:
            score += 1
            signals.append(("✅", f"Giá thấp hơn MA200 {abs(dev):.1f}% — vùng giá trị hấp dẫn dài hạn", "green"))

    # ── Nhãn tổng hợp ─────────────────────────────────────────────────────
    AU = a_name.upper()
    if score >= 8:
        label = f"RẤT TÍCH CỰC CHO {AU}"
    elif score >= 4:
        label = f"TÍCH CỰC CHO {AU}"
    elif score <= -8:
        label = f"RẤT TIÊU CỰC CHO {AU}"
    elif score <= -4:
        label = f"TIÊU CỰC CHO {AU}"
    else:
        label = "TRUNG TÍNH"

    return score, label, signals, metrics

# ══════════════════════════════════════════════════════════════════════════════
#  SEASONAL FACTOR
# ══════════════════════════════════════════════════════════════════════════════

def seasonal_factor(last_date_str: str, days: int, asset_key: str = "XAU") -> float:
    """Tính lệch mùa vụ trung bình cho kỳ dự báo."""
    bias  = SEASONAL_BIAS.get(asset_key, SEASONAL_BIAS["XAU"])
    start = pd.Timestamp(last_date_str) + timedelta(days=1)
    month_days: dict[int, int] = {}
    for i in range(days):
        m = (start + timedelta(days=i)).month
        month_days[m] = month_days.get(m, 0) + 1
    return sum(bias.get(m, 0) * cnt / days for m, cnt in month_days.items())

# ══════════════════════════════════════════════════════════════════════════════
#  FORECASTING  (3-model ensemble + macro + seasonal)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def forecast(_price_values: np.ndarray, last_date_str: str,
             days: int, macro_score: int = 0, asset_key: str = "XAU") -> tuple:
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

    # ── Macro adjustment: sqrt-damped để tránh cộng tuyến tính quá mức ──
    # Cũ: score * 0.0035 * (days/30) → 1 năm score=8 ra +34% (sai)
    # Mới: sqrt-damped → 1 năm score=8 ra ~11%, hợp lý hơn
    macro_adj = np.clip(
        macro_score * 0.004 * np.sqrt(days / 30),
        -0.18, 0.18
    )

    # ── Seasonal adjustment ───────────────────────────────────────────────
    seas_adj = seasonal_factor(last_date_str, days, asset_key)

    # ── Momentum adjustment (3 tháng gần nhất của giá vàng) ──────────────
    n3m      = min(66, len(train))
    ret_3m   = float(train.iloc[-1]) / float(train.iloc[-n3m]) - 1
    # Momentum đóng góp tối đa 12%, giảm dần cho kỳ dài (mean reversion)
    mom_weight = 0.12 * (30 / max(days, 30)) ** 0.5
    mom_adj  = np.clip(ret_3m * mom_weight, -0.08, 0.08)

    # ── Mean reversion từ MA200 ───────────────────────────────────────────
    if len(price) >= 200:
        ma200v   = float(price.rolling(200).mean().iloc[-1])
        cur_p    = float(price.iloc[-1])
        dev      = cur_p / ma200v - 1          # +0.20 = đang cao hơn MA200 20%
        # Áp lực quay về trung bình: tỷ lệ với độ lệch và thời gian
        rev_adj  = np.clip(-dev * 0.10 * (days / 365), -0.06, 0.06)
    else:
        rev_adj  = 0.0

    # ── Apply adjustments ─────────────────────────────────────────────────
    combined = base * (1 + macro_adj + seas_adj + mom_adj + rev_adj)

    # ── Confidence interval (~90%) — mở rộng theo căn bậc 2 thời gian ───
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
#  NARRATIVE EXPLANATION  (giải thích tại sao tăng/giảm — tiếng Việt)
# ══════════════════════════════════════════════════════════════════════════════

_MONTH_VN = {
    1:"Tháng 1", 2:"Tháng 2", 3:"Tháng 3", 4:"Tháng 4",
    5:"Tháng 5", 6:"Tháng 6", 7:"Tháng 7", 8:"Tháng 8",
    9:"Tháng 9", 10:"Tháng 10", 11:"Tháng 11", 12:"Tháng 12",
}
_SEASON_REASON = {
    1:  "đầu năm nhu cầu vàng thường tăng do Tết Nguyên Đán và phong tục tặng vàng",
    2:  "sau Tết nhu cầu trang sức còn dư, tâm lý mua vàng đầu năm vẫn cao",
    3:  "giai đoạn chuyển tiếp, nhu cầu theo mùa chưa có động lực rõ",
    4:  "mùa cưới Ấn Độ bắt đầu, nhu cầu vàng trang sức tăng đáng kể",
    5:  "thường là tháng yếu nhất trong năm về nhu cầu vàng toàn cầu",
    6:  "nhu cầu thấp, thanh khoản thị trường giảm trong mùa hè Bắc bán cầu",
    7:  "thị trường hè, giao dịch tương đối trầm lắng",
    8:  "nhu cầu bắt đầu tăng chuẩn bị mùa cưới lớn tháng 9–10",
    9:  "mùa cưới Ấn Độ đỉnh điểm và lễ Diwali — nhu cầu vàng cao nhất năm",
    10: "lễ Diwali và mùa tặng quà, nhu cầu vàng trang sức tiếp tục mạnh",
    11: "chuẩn bị Giáng Sinh, nhu cầu trang sức và quà tặng tăng",
    12: "mua vàng cuối năm để bảo toàn tài sản, tặng quà dịp lễ",
}


def generate_narrative(price, ma20, ma50, ma200, rsi,
                       fc_mean, fc_lo, fc_hi,
                       macro_score, macro_signals, macro_metrics,
                       forecast_days, seas) -> str:
    """Tạo đoạn giải thích tại sao giá vàng dự báo tăng/giảm."""

    cur   = float(price.iloc[-1])
    fc_e  = float(fc_mean.iloc[-1])
    fc_l  = float(fc_lo.iloc[-1])
    fc_h  = float(fc_hi.iloc[-1])
    chg   = (fc_e - cur) / cur * 100
    r     = float(rsi.iloc[-1])
    lbl   = period_label(forecast_days)
    m_now = pd.Timestamp(str(price.index[-1])).month

    parts = []

    # ── 1. Kỹ thuật ─────────────────────────────────────────────────────────
    tech_lines = []
    above = [m for m, s in [("MA20", ma20), ("MA50", ma50), ("MA200", ma200)]
             if cur > float(s.iloc[-1])]
    below = [m for m, s in [("MA20", ma20), ("MA50", ma50), ("MA200", ma200)]
             if cur <= float(s.iloc[-1])]

    if len(above) == 3:
        tech_lines.append(
            f"Về kỹ thuật, giá vàng hiện tại (${cur:,.0f}) đang giao dịch "
            f"**trên cả MA20, MA50 và MA200** — đây là cấu hình rất tích cực, "
            f"cho thấy xu hướng tăng được xác nhận trên mọi khung thời gian."
        )
    elif len(above) >= 2:
        tech_lines.append(
            f"Về kỹ thuật, giá (${cur:,.0f}) đang nằm **trên {' và '.join(above)}** "
            f"nhưng dưới {' và '.join(below)}, phản ánh xu hướng tăng "
            f"{'ngắn-trung hạn' if 'MA200' in below else 'chưa đủ mạnh trên dài hạn'}."
        )
    elif len(below) >= 2:
        tech_lines.append(
            f"Về kỹ thuật, giá (${cur:,.0f}) đang **dưới {' và '.join(below)}** "
            f"— áp lực giảm {'rất lớn' if len(below) == 3 else 'đáng kể'} từ góc độ kỹ thuật."
        )
    else:
        tech_lines.append(
            f"Về kỹ thuật, giá (${cur:,.0f}) nằm gần các đường MA, "
            f"chưa có tín hiệu xu hướng rõ ràng."
        )

    if r > 70:
        tech_lines.append(
            f"RSI đang ở **{r:.0f} (vùng quá mua)**: thị trường tăng nhanh, "
            f"nguy cơ điều chỉnh ngắn hạn tăng lên, dù xu hướng chính vẫn là tăng."
        )
    elif r < 30:
        tech_lines.append(
            f"RSI ở **{r:.0f} (vùng quá bán)**: lực bán đã cạn kiệt, "
            f"tạo điều kiện thuận lợi cho đợt hồi phục kỹ thuật."
        )
    elif r > 55:
        tech_lines.append(
            f"RSI **{r:.0f}** phản ánh động lực tăng ổn định, "
            f"chưa vào vùng quá mua — còn nhiều dư địa để tiếp tục tăng."
        )
    elif r < 45:
        tech_lines.append(
            f"RSI **{r:.0f}** cho thấy momentum đang nghiêng về phía giảm, "
            f"người mua chưa chiếm ưu thế rõ ràng."
        )
    else:
        tech_lines.append(f"RSI **{r:.0f}** đang ở vùng trung tính, chưa xác nhận hướng rõ ràng.")

    parts.append("**📐 Phân tích kỹ thuật**\n\n" + " ".join(tech_lines))

    # ── 2. Vĩ mô ─────────────────────────────────────────────────────────────
    macro_lines = []

    if "dxy" in macro_metrics:
        dxy_v = macro_metrics["dxy"]["val"]
        dxy_c = macro_metrics["dxy"]["chg"] or 0
        if dxy_c < -1:
            macro_lines.append(
                f"**USD suy yếu rõ rệt** (DXY {dxy_v:.1f}, giảm {abs(dxy_c):.1f}% trong tháng): "
                f"vàng được định giá bằng USD — khi USD yếu, giá vàng tăng tự nhiên "
                f"vì cùng một lượng vàng cần nhiều USD hơn để mua."
            )
        elif dxy_c > 1:
            macro_lines.append(
                f"**USD mạnh lên** (DXY {dxy_v:.1f}, tăng {dxy_c:.1f}% trong tháng): "
                f"đây là lực cản chính với vàng — USD mạnh làm vàng đắt hơn với người mua "
                f"nước ngoài, giảm nhu cầu toàn cầu và tạo áp lực giảm giá."
            )
        else:
            macro_lines.append(
                f"**USD tương đối ổn định** (DXY {dxy_v:.1f}): "
                f"yếu tố tỷ giá chưa tạo áp lực lớn theo hướng nào cho vàng."
            )

    if "yield10y" in macro_metrics:
        y_v = macro_metrics["yield10y"]["val"]
        y_c = macro_metrics["yield10y"]["chg"] or 0
        if y_v < 3.5:
            macro_lines.append(
                f"**Lợi suất trái phiếu 10Y Mỹ thấp ({y_v:.2f}%)**: "
                f"chi phí cơ hội giữ vàng (vốn không sinh lãi) rất thấp, "
                f"giúp vàng hấp dẫn hơn so với trái phiếu."
            )
        elif y_v > 4.5:
            macro_lines.append(
                f"**Lợi suất trái phiếu 10Y Mỹ cao ({y_v:.2f}%)**: "
                f"nhà đầu tư có thể nhận {y_v:.2f}% mỗi năm từ trái phiếu an toàn, "
                f"trong khi vàng không trả lãi — điều này cạnh tranh trực tiếp và kéo "
                f"dòng tiền rút khỏi vàng."
            )
        else:
            macro_lines.append(
                f"**Lợi suất 10Y ở mức {y_v:.2f}%** "
                f"({'đang tăng' if y_c > 0.1 else 'đang giảm' if y_c < -0.1 else 'ổn định'}): "
                f"áp lực lên vàng ở mức trung bình."
            )

    if "vix" in macro_metrics:
        vix_v = macro_metrics["vix"]["val"]
        if vix_v > 30:
            macro_lines.append(
                f"**Chỉ số sợ hãi VIX ở {vix_v:.0f} — rất cao**: "
                f"thị trường đang hoảng loạn, nhà đầu tư toàn cầu đổ tiền vào vàng "
                f"như một \"hầm trú ẩn\" an toàn — đây là yếu tố đẩy giá vàng tăng mạnh nhất."
            )
        elif vix_v > 20:
            macro_lines.append(
                f"**VIX {vix_v:.0f} — thị trường lo lắng**: "
                f"tâm lý phòng thủ đang chiếm ưu thế, hỗ trợ dòng tiền vào vàng."
            )
        elif vix_v < 14:
            macro_lines.append(
                f"**VIX {vix_v:.0f} — thị trường quá tự tin**: "
                f"khi tất cả đều lạc quan và mua cổ phiếu, vàng mất đi sức hút "
                f"\"tài sản trú ẩn\" — dòng tiền ưu tiên chảy vào rủi ro."
            )
        else:
            macro_lines.append(
                f"**VIX {vix_v:.0f}** — tâm lý thị trường trung tính, "
                f"chưa có cú sốc đủ mạnh để đẩy dòng tiền mạnh vào hoặc ra khỏi vàng."
            )

    if "sp500" in macro_metrics:
        sp_c = macro_metrics["sp500"]["chg"] or 0
        sp_v = macro_metrics["sp500"]["val"]
        if sp_c < -5:
            macro_lines.append(
                f"**Chứng khoán Mỹ (S&P 500) sụt giảm mạnh {sp_c:.1f}%**: "
                f"khi cổ phiếu rớt giá mạnh, nhà đầu tư bán tháo và chuyển sang vàng "
                f"để bảo vệ tài sản — tạo làn sóng mua vàng đáng kể."
            )
        elif sp_c > 5:
            macro_lines.append(
                f"**Chứng khoán Mỹ (S&P 500) tăng mạnh {sp_c:.1f}%**: "
                f"tâm lý risk-on mạnh — nhà đầu tư tự tin mua cổ phiếu thay vì vàng, "
                f"làm giảm nhu cầu với vàng."
            )
        else:
            macro_lines.append(
                f"**S&P 500 biến động {sp_c:+.1f}%** — chưa có xu hướng rõ ràng "
                f"tác động lớn đến dòng tiền vào/ra vàng."
            )

    parts.append("**🌐 Yếu tố vĩ mô**\n\n" + "\n\n".join(macro_lines))

    # ── 3. Mùa vụ ─────────────────────────────────────────────────────────────
    seas_pct = seas * 100
    seas_reason = _SEASON_REASON.get(m_now, "")
    if abs(seas_pct) >= 0.2:
        if seas_pct > 0:
            seas_text = (
                f"Kỳ dự báo trùng với giai đoạn **thuận lợi theo mùa vụ** "
                f"(+{seas_pct:.1f}% bias lịch sử): {seas_reason}. "
                f"Đây là lực đẩy bổ sung, đã được tính vào dự báo."
            )
        else:
            seas_text = (
                f"Kỳ dự báo rơi vào giai đoạn **kém thuận lợi theo mùa** "
                f"({seas_pct:.1f}% bias lịch sử): {seas_reason}. "
                f"Yếu tố này tạo lực cản nhẹ, đã được trừ vào dự báo."
            )
        parts.append(f"**📅 Yếu tố mùa vụ ({_MONTH_VN.get(m_now, '')})**\n\n{seas_text}")

    # ── 4. Kết luận ───────────────────────────────────────────────────────────
    if chg >= 4:
        verdict = (
            f"Tổng hợp lại, **tất cả các yếu tố đều nghiêng về phía tăng**: "
            f"kỹ thuật tích cực, môi trường vĩ mô hỗ trợ"
            f"{' và mùa vụ thuận lợi' if seas_pct > 0 else ''}. "
            f"Mô hình dự báo giá vàng đạt **${fc_e:,.0f}** sau {lbl} "
            f"(tăng {chg:.1f}%), với vùng dao động kỳ vọng **${fc_l:,.0f} – ${fc_h:,.0f}**."
        )
    elif chg >= 1:
        verdict = (
            f"Nhìn chung, **nhiều yếu tố hỗ trợ xu hướng tăng nhẹ**, "
            f"dù chưa có catalyst đặc biệt mạnh. "
            f"Dự báo giá vàng đạt khoảng **${fc_e:,.0f}** sau {lbl} "
            f"(+{chg:.1f}%), vùng dao động **${fc_l:,.0f} – ${fc_h:,.0f}**."
        )
    elif chg <= -4:
        verdict = (
            f"Tổng hợp lại, **áp lực giảm đang chiếm ưu thế**: "
            f"môi trường vĩ mô bất lợi"
            f"{' kết hợp kỹ thuật yếu' if len(below) >= 2 else ''}. "
            f"Dự báo giá vàng giảm về **${fc_e:,.0f}** sau {lbl} "
            f"({chg:.1f}%), vùng dao động **${fc_l:,.0f} – ${fc_h:,.0f}**."
        )
    elif chg <= -1:
        verdict = (
            f"Nhìn chung, **áp lực nhẹ về phía giảm** đang hiện diện, "
            f"chưa đủ mạnh để tạo xu hướng giảm rõ ràng. "
            f"Dự báo giá khoảng **${fc_e:,.0f}** sau {lbl} "
            f"({chg:.1f}%), vùng dao động **${fc_l:,.0f} – ${fc_h:,.0f}**."
        )
    else:
        verdict = (
            f"Các yếu tố **tương đối cân bằng**, chưa có lực đẩy rõ ràng theo hướng nào. "
            f"Dự báo giá vàng dao động quanh **${fc_e:,.0f}** sau {lbl} "
            f"({chg:+.1f}%), trong vùng **${fc_l:,.0f} – ${fc_h:,.0f}**."
        )

    if forecast_days >= 180:
        verdict += (
            f"\n\n> ⚠️ **Lưu ý quan trọng với dự báo {lbl}:** Độ bất định tăng rất cao "
            f"theo thời gian. Vùng giá rộng (${fc_l:,.0f} – ${fc_h:,.0f}) là hoàn toàn bình thường. "
            f"Các sự kiện bất ngờ như thay đổi lãi suất Fed, xung đột địa chính trị, "
            f"hay khủng hoảng kinh tế đều có thể đảo chiều hoàn toàn dự báo này."
        )

    parts.append(f"**🎯 Kết luận ({lbl})**\n\n{verdict}")

    return "\n\n---\n\n".join(parts)

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
                sig_color: str, forecast_days: int,
                asset_key: str = "XAU") -> go.Figure:

    # Lịch sử hiển thị = 2× kỳ dự báo, tối thiểu 150, tối đa 500 ngày
    H   = min(max(150, forecast_days * 2), 500)
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
                  annotation_text=f"52W High {fmt_price(hi52, asset_key)}",
                  annotation_font_color="#3fb950", annotation_font_size=9, row=1, col=1)
    fig.add_hline(y=lo52, line_dash="dot", line_color="#f85149", line_width=0.9, opacity=0.5,
                  annotation_text=f"52W Low {fmt_price(lo52, asset_key)}",
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

    a_info = ASSETS[asset_key]
    a_name = a_info["name"]

    # Price history
    fig.add_trace(go.Scatter(x=sl(p).index, y=sl(p),
        name=a_name, line=dict(color="#c9d1d9", width=2.2),
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Giá: %{y:,.4g}<extra></extra>"), row=1, col=1)

    # Forecast line
    ml = period_label(forecast_days)
    fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean, name=f"Dự báo {ml}",
        line=dict(color=sig_color, width=2.6),
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Dự báo: %{y:,.4g}<extra></extra>"), row=1, col=1)

    # TODAY divider
    fig.add_vline(x=p.index[-1].timestamp() * 1000,
                  line_dash="dash", line_color="rgba(255,255,255,0.18)", line_width=1)

    # Current price marker
    cur_p = float(p.iloc[-1])
    cur_label = fmt_price(cur_p, asset_key)
    fig.add_trace(go.Scatter(x=[p.index[-1]], y=[cur_p], mode="markers+text",
        marker=dict(color=a_info["color"], size=10),
        text=[f"  Hôm nay {cur_label}"],
        textposition="middle right", textfont=dict(color=a_info["color"], size=10),
        showlegend=False, hoverinfo="skip"), row=1, col=1)

    # Forecast end marker
    fc_e = float(fc_mean.iloc[-1])
    chg  = (fc_e - cur_p) / cur_p * 100
    sign = "+" if chg >= 0 else ""
    fc_label = fmt_price(fc_e, asset_key)
    fig.add_trace(go.Scatter(x=[fc_mean.index[-1]], y=[fc_e], mode="markers+text",
        marker=dict(color=sig_color, size=9, symbol="diamond"),
        text=[f"  {fc_label} ({sign}{chg:.1f}%)"],
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
    fig.update_yaxes(title_text=a_info["unit"], row=1, col=1, title_font_size=9)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[10, 90], title_font_size=9)
    fig.update_xaxes(tickformat="%d/%m/%y", tickangle=-30, row=2, col=1)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  WHALE (INSTITUTIONAL POSITIONING) DATA
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_whale_data(asset_key: str) -> dict:
    """
    Dữ liệu định vị cá mập (institutional positioning):
    - ETF AUM flow proxy (price × volume trend của GLD/SLV/IBIT/USO/COPX)
    - Volume momentum (smart money footprint)
    - Futures Open Interest (yfinance)
    Returns dict với etf_flow, vol, oi, score, signals
    """
    etf_map = {
        "XAU":    ("GLD",  "vàng"),
        "XAG":    ("SLV",  "bạc"),
        "HG":     ("COPX", "đồng"),
        "CL":     ("USO",  "dầu WTI"),
        "USDVND": (None,   "USD/VND"),
        "BTC":    ("IBIT", "Bitcoin"),
    }
    fut_map = {
        "XAU":    "GC=F",
        "XAG":    "SI=F",
        "HG":     "HG=F",
        "CL":     "CL=F",
        "USDVND": None,
        "BTC":    None,
    }

    etf_ticker, asset_vn = etf_map.get(asset_key, (None, asset_key))
    fut_ticker = fut_map.get(asset_key)
    score  = 0
    signals = []
    result  = {"etf_flow": None, "vol": None, "oi": None, "score": 0, "signals": []}

    # ── 1. ETF AUM Flow Proxy ────────────────────────────────────────────────
    if etf_ticker:
        try:
            raw = yf.download(etf_ticker, period="3mo", interval="1d",
                              progress=False, auto_adjust=True)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                price_s = raw["Close"].dropna()
                vol_s   = raw["Volume"].dropna()

                if len(price_s) >= 20:
                    # AUM proxy: 20-day MA now vs 20-day MA from 60 days ago
                    p20  = float(price_s.rolling(20).mean().iloc[-1])
                    tail60 = price_s.tail(60)
                    p60  = float(tail60.rolling(20).mean().dropna().iloc[0]) if len(tail60) >= 20 else float(tail60.iloc[0])
                    aum_chg = (p20 / p60 - 1) * 100 if p60 > 0 else 0
                    result["etf_flow"] = {
                        "ticker": etf_ticker, "aum_chg": aum_chg,
                        "price": float(price_s.iloc[-1]),
                    }

                    if aum_chg > 5:
                        score += 2
                        signals.append(("🐋", f"ETF {etf_ticker} AUM +{aum_chg:.1f}% (20d MA now vs 60d ago) → tổ chức đang tích lũy {asset_vn} mạnh", "green"))
                    elif aum_chg > 2:
                        score += 1
                        signals.append(("📈", f"ETF {etf_ticker} AUM +{aum_chg:.1f}% → tổ chức mua nhẹ {asset_vn}", "green"))
                    elif aum_chg < -5:
                        score -= 2
                        signals.append(("📉", f"ETF {etf_ticker} AUM {aum_chg:.1f}% → tổ chức phân phối {asset_vn}", "red"))
                    elif aum_chg < -2:
                        score -= 1
                        signals.append(("⚠️", f"ETF {etf_ticker} AUM {aum_chg:.1f}% → xả hàng nhẹ {asset_vn}", "orange"))
                    else:
                        signals.append(("➡️", f"ETF {etf_ticker}: AUM ổn định ({aum_chg:+.1f}%) — cá mập chưa hành động rõ ràng", "gray"))

                    # Volume surge (5d vs 90d)
                    if len(vol_s) >= 10:
                        avg_vol  = float(vol_s.tail(60).mean()) if len(vol_s) >= 60 else float(vol_s.mean())
                        rec_vol  = float(vol_s.tail(5).mean())
                        vol_ratio = rec_vol / avg_vol if avg_vol > 0 else 1.0
                        result["vol"] = {"ratio": vol_ratio, "ticker": etf_ticker}

                        if vol_ratio > 2.0:
                            score += 1
                            signals.append(("⚡", f"Khối lượng {etf_ticker} tăng vọt {vol_ratio:.1f}× avg → có thể block trading (cá mập vào lệnh lớn)", "green"))
                        elif vol_ratio > 1.5:
                            signals.append(("📊", f"Khối lượng {etf_ticker}: {vol_ratio:.1f}× avg — hoạt động tổ chức gia tăng", "green"))
                        elif vol_ratio < 0.5:
                            signals.append(("🔇", f"Khối lượng {etf_ticker} rất thấp ({vol_ratio:.1f}× avg) — cá mập vắng mặt", "gray"))
                        else:
                            signals.append(("➡️", f"Khối lượng {etf_ticker}: {vol_ratio:.1f}× avg — bình thường", "gray"))
        except Exception:
            pass

    # ── 2. Futures Open Interest ─────────────────────────────────────────────
    if fut_ticker:
        try:
            tk  = yf.Ticker(fut_ticker)
            inf = tk.info
            oi  = inf.get("openInterest")
            if oi and int(oi) > 0:
                result["oi"] = {"value": int(oi), "ticker": fut_ticker}
                oi_label = "tập trung lớn → sắp có biến động mạnh" if oi > 400000 else "mức trung bình"
                signals.append(("📌", f"Open Interest {fut_ticker}: {oi:,} hợp đồng — {oi_label}", "gray"))
        except Exception:
            pass

    result["score"]   = max(-5, min(5, score))
    result["signals"] = signals
    return result


def whale_regime(whale_data: dict, asset_key: str) -> dict:
    """
    Phân tích định vị cá mập và đưa ra nhận định tổng hợp.
    Trả về score, label, color, signals, cot_note.
    """
    score    = whale_data.get("score", 0)
    signals  = whale_data.get("signals", [])
    a_name   = ASSETS[asset_key]["short"]

    etf  = whale_data.get("etf_flow")
    vol  = whale_data.get("vol")
    cot_signal = ""

    # ── Double confirmation (ETF + Volume) ───────────────────────────────────
    if etf and vol:
        if etf["aum_chg"] > 3 and vol["ratio"] > 1.3:
            score += 1
            cot_signal = (f"📣 XÁC NHẬN KÉP: ETF tăng +{etf['aum_chg']:.1f}% "
                          f"+ volume {vol['ratio']:.1f}× → cá mập đang tích lũy {a_name} mạnh")
        elif etf["aum_chg"] < -3 and vol["ratio"] > 1.3:
            score -= 1
            cot_signal = (f"📣 XÁC NHẬN KÉP: ETF giảm {etf['aum_chg']:.1f}% "
                          f"+ volume {vol['ratio']:.1f}× → cá mập đang phân phối {a_name}")

    score = max(-5, min(5, score))

    if score >= 3:
        label = f"CÁ MẬP TÍCH LŨY {a_name.upper()}"
        color = "#3fb950"
    elif score >= 1:
        label = "TÍCH CỰC NHẸ (CÁ MẬP MUA)"
        color = "#76c3a0"
    elif score <= -3:
        label = f"CÁ MẬP PHÂN PHỐI {a_name.upper()}"
        color = "#f85149"
    elif score <= -1:
        label = "TIÊU CỰC NHẸ (CÁ MẬP BÁN)"
        color = "#ff7b54"
    else:
        label = "TRUNG TÍNH (CÁ MẬP CHƯA HÀNH ĐỘNG)"
        color = "#FFD700"

    return {
        "score":    score,
        "label":    label,
        "color":    color,
        "signals":  signals,
        "cot_note": cot_signal,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ASSET TAB RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_asset_tab(asset_key: str, macro: dict, forecast_days: int):
    """Render toàn bộ nội dung phân tích cho một asset tab."""
    info = ASSETS[asset_key]
    a_color  = info["color"]
    a_name   = info["name"]
    a_short  = info["short"]
    a_unit   = info["unit"]

    # ── Fetch price + live ────────────────────────────────────────────────
    with st.spinner(f"📡 Đang tải dữ liệu {a_short}..."):
        try:
            price, ticker = fetch_price(asset_key)
        except RuntimeError as e:
            st.error(str(e)); return
        live_price, live_src = fetch_live(asset_key)

    # ── Technical indicators ──────────────────────────────────────────────
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

    # ── Macro regime ──────────────────────────────────────────────────────
    macro_score, macro_label, macro_signals, macro_metrics = macro_regime(
        macro, price, asset_key
    )

    # ── Fed Policy Analysis (FRED + personality) ──────────────────────────
    with st.spinner("🏦 Phân tích chính sách Fed..."):
        fred_data  = fetch_fred_rates()
        fed_result = fed_policy_analysis(fred_data, macro)

    # ── Whale Positioning (ETF flow + OI + volume) ────────────────────────
    with st.spinner("🐋 Tải dữ liệu cá mập..."):
        whale_data_raw = fetch_whale_data(asset_key)
        whale_result   = whale_regime(whale_data_raw, asset_key)

    # ── Combined macro score (Macro + Fed + Whale) ───────────────────────
    fed_contrib    = round(fed_result["score"] * 0.35)
    whale_contrib  = round(whale_result["score"] * 0.25)
    combined_macro = max(-12, min(12, macro_score + fed_contrib + whale_contrib))

    # ── Forecast ──────────────────────────────────────────────────────────
    with st.spinner(f"🔮 Chạy mô hình dự báo {a_short}..."):
        fc_mean, fc_lo_s, fc_hi_s = forecast(
            price.values, str(price.index[-1]), forecast_days,
            combined_macro, asset_key
        )

    # ── Signal ────────────────────────────────────────────────────────────
    signal, sig_color, sig_icon, tech_notes = compute_signal(
        price, ma20, ma50, ma200, rsi, fc_mean, combined_macro
    )

    cur   = float(price.iloc[-1])
    fc_e  = float(fc_mean.iloc[-1])
    chg   = (fc_e - cur) / cur * 100
    sign  = "+" if chg >= 0 else ""
    r_cur = float(rsi.iloc[-1])
    seas  = seasonal_factor(str(price.index[-1]), forecast_days, asset_key)
    seas_note = (f"{'📈' if seas > 0 else '📉'} Yếu tố mùa vụ: {seas*100:+.1f}% "
                 f"({'thuận lợi' if seas > 0 else 'bất lợi'} cho kỳ này)")

    # ── Metrics row ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    if live_price:
        live_diff = live_price - cur
        m1.metric(
            f"⚡ Giá Live ({a_short})",
            fmt_price(live_price, asset_key),
            f"{live_diff:+.{info['decimals']}f} so với phiên trước",
            delta_color="normal",
        )
    else:
        m1.metric(f"💰 Giá ({a_short})", fmt_price(cur, asset_key),
                  f"Nguồn: {ticker}")

    m2.metric(f"📅 Dự báo ({period_label(forecast_days)})",
              fmt_price(fc_e, asset_key),
              f"{sign}{chg:.1f}%", delta_color="normal")
    m3.metric("📊 52W High / Low",
              fmt_price(hi52, asset_key), f"Low: {fmt_price(lo52, asset_key)}")
    m4.metric("📈 RSI (14)", f"{r_cur:.0f}",
              "Quá mua ⚠️" if r_cur > 70 else ("Quá bán ✅" if r_cur < 30 else "Bình thường"),
              delta_color="off")

    # Ghi chú nguồn giá
    if live_price:
        diff_pct = abs(live_price - cur) / cur * 100
        note_color = "#f9a825" if diff_pct > 0.5 else "#8b949e"
        delay_note = ""
        if "COMEX" in live_src or "Yahoo" in live_src:
            delay_note = " · ⏱ <i>Delay ~15 phút so với giá real-time</i>"
        st.markdown(
            f"<p style='font-size:0.78rem;color:{note_color};margin:-8px 0 4px 0;'>"
            f"⚡ Live: <b>{fmt_price(live_price, asset_key)}</b> ({live_src}) · "
            f"Phiên trước: <b>{fmt_price(cur, asset_key)}</b> ({ticker}) · "
            f"Lệch: <b>{live_price - cur:+.{info['decimals']}f}</b>"
            f"{delay_note}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Macro dashboard ───────────────────────────────────────────────────
    macro_score_color = ("#3fb950" if macro_score >= 3 else
                         "#f85149" if macro_score <= -3 else "#FFD700")
    st.markdown(
        f"#### 🌐 Môi trường vĩ mô &nbsp; "
        f"<span style='color:{macro_score_color};font-size:0.95rem;'>"
        f"[ {macro_label} · Điểm: {macro_score:+d} ]</span>",
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3, mc4 = st.columns(4)

    def _mm(col, label, key, fmt, low_good=False, pct_delta=False):
        if key not in macro_metrics:
            col.metric(label, "N/A"); return
        m = macro_metrics[key]
        val_str = f"{m['val']:{fmt}}"
        if m["chg"] is not None:
            delta = f"{m['chg']:+.1f}%" if pct_delta else f"{m['chg']:+.2f}"
        else:
            delta = None
        col.metric(label, val_str + m["unit"], delta,
                   delta_color="inverse" if low_good else "normal")

    _mm(mc1, "💵 DXY (USD Index)",        "dxy",      ".1f",  low_good=True, pct_delta=True)
    _mm(mc2, "📉 Yield 10Y (%)",           "yield10y", ".2f",  low_good=True)
    _mm(mc3, "😨 VIX (Fear Index)",        "vix",      ".0f",  low_good=False)
    _mm(mc4, "📈 S&P 500",                 "sp500",    ",.0f", low_good=False, pct_delta=True)

    mc5, mc6, mc7, mc8 = st.columns(4)
    _mm(mc5, "🛢️ Dầu WTI ($/bbl)",        "oil",      ".1f",  low_good=False, pct_delta=True)
    _mm(mc6, "📊 TIPS ETF (lãi suất thực)", "tips",    ".2f",  low_good=False, pct_delta=True)

    if "momentum" in macro_metrics:
        mom = macro_metrics["momentum"]
        mc7.metric(f"⚡ Momentum {a_short} (3T)",
                   f"{mom['ret3m']:+.1f}%", f"6 tháng: {mom['ret6m']:+.1f}%",
                   delta_color="normal")
    else:
        mc7.metric(f"⚡ Momentum {a_short} (3T)", "N/A")

    if "ma200_dev" in macro_metrics:
        dev = macro_metrics["ma200_dev"]
        lbl = "⚠️ Overbought" if dev > 15 else ("✅ Giá trị tốt" if dev < -10 else "Bình thường")
        mc8.metric("📐 Lệch so MA200", f"{dev:+.1f}%", lbl,
                   delta_color="inverse" if dev > 15 else "normal")
    else:
        mc8.metric("📐 Lệch so MA200", "N/A")

    with st.expander("📋 Chi tiết phân tích vĩ mô", expanded=False):
        for icon, text, _ in macro_signals:
            st.markdown(f"{icon} {text}")
        st.markdown(f"---\n{seas_note}")

    st.markdown("---")

    # ── Fed Policy Radar ──────────────────────────────────────────────────
    fed_score     = fed_result["score"]
    fed_color     = fed_result["color"]
    fed_direction = fed_result["direction"]
    fed_prob_cut  = fed_result["prob_cut"]
    fed_prob_hold = fed_result["prob_hold"]
    fed_signals   = fed_result["signals"]
    cur_rate      = fed_result.get("current_rate")
    curve_val     = fed_result.get("curve_val")

    st.markdown(
        f"#### 🏦 Fed Radar — Trump · Warsh · Lãi suất &nbsp;"
        f"<span style='color:{fed_color};font-size:0.95rem;'>"
        f"[ {fed_direction} · Điểm: {fed_score:+d} ]</span>",
        unsafe_allow_html=True,
    )

    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("🎯 Fed Hướng đi", fed_direction,
               f"Điểm tổng hợp: {fed_score:+d}/5", delta_color="off")
    fc2.metric("✂️ Xác suất Cắt lãi", f"{fed_prob_cut}%",
               f"Giữ/Tăng lãi: {fed_prob_hold}%",
               delta_color="normal" if fed_prob_cut > 50 else "inverse")
    if cur_rate is not None:
        fc3.metric("🏛️ Fed Funds Rate", f"{cur_rate:.2f}%",
                   f"Neutral 2.50% · Gap {cur_rate - 2.5:+.2f}%",
                   delta_color="inverse")
    else:
        fc3.metric("🏛️ Fed Funds Rate", "N/A")
    if curve_val is not None:
        fc4.metric("📐 Yield Curve (10Y–2Y)", f"{curve_val:+.2f}%",
                   "Đảo ngược → kỳ vọng cắt lãi" if curve_val < 0 else "Bình thường/dốc",
                   delta_color="normal" if curve_val < 0 else "inverse")
    else:
        fc4.metric("📐 Yield Curve (10Y–2Y)", "N/A")

    with st.expander("🔍 Ngoại cảm: Trump · Warsh · Xung đột cấu trúc", expanded=False):
        conflict = TRUMP_WARSH_DYNAMIC["conflict_level"]
        t_win    = TRUMP_WARSH_DYNAMIC["trump_wins_prob"]
        w_win    = TRUMP_WARSH_DYNAMIC["warsh_wins_prob"]
        comp     = TRUMP_WARSH_DYNAMIC["compromise_prob"]
        st.markdown(
            f"**⚡ Xung đột Trump–Warsh** (cấp độ {conflict}/3 — MÂU THUẪN CAO)\n\n"
            f"| Kịch bản | Xác suất |\n"
            f"|---|---|\n"
            f"| Trump thuyết phục Warsh cắt lãi | **{t_win}%** |\n"
            f"| Warsh giữ vững lập trường hawkish | **{w_win}%** |\n"
            f"| Thỏa hiệp ở giữa | **{comp}%** |\n\n"
            f"**🦅 Kevin Warsh** *(Fed Chair từ 22/5/2026)*: Diều hâu (hawkish) · Rules-based "
            f"(Taylor Rule) · QE skeptic mạnh nhất lịch sử · Morgan Stanley + Nhà Trắng "
            f"+ Druckenmiller background · Ưu tiên uy tín Fed > áp lực chính trị\n\n"
            f"**🇺🇸 Donald Trump**: Luôn muốn lãi suất thấp nhất có thể · Coi S&P 500 là "
            f"điểm số nhiệm kỳ · Thuế quan = vũ khí đàm phán · Sẵn sàng tấn công Fed "
            f"công khai · Muốn USD yếu để hỗ trợ xuất khẩu"
        )
        st.markdown("---\n**📊 Tín hiệu lãi suất thực tế (FRED Data):**")
        for icon, text, _ in fed_signals:
            st.markdown(f"{icon} {text}")
        st.markdown(
            f"\n*Đóng góp vào dự báo: Fed score {fed_score:+d} × 35% = {fed_contrib:+d} điểm*"
        )

    st.markdown("---")

    # ── Whale Positioning ─────────────────────────────────────────────────
    w_score   = whale_result["score"]
    w_color   = whale_result["color"]
    w_label   = whale_result["label"]
    w_signals = whale_result["signals"]
    w_cot     = whale_result.get("cot_note", "")
    etf_flow  = whale_data_raw.get("etf_flow")
    vol_data  = whale_data_raw.get("vol")
    oi_data   = whale_data_raw.get("oi")

    st.markdown(
        f"#### 🐋 Cá Mập (Institutional Positioning) &nbsp;"
        f"<span style='color:{w_color};font-size:0.95rem;'>"
        f"[ {w_label} · Điểm: {w_score:+d} ]</span>",
        unsafe_allow_html=True,
    )

    wc1, wc2, wc3 = st.columns(3)
    if etf_flow:
        etf_chg = etf_flow["aum_chg"]
        wc1.metric(
            f"🏦 ETF {etf_flow['ticker']} (AUM Proxy)",
            f"{etf_chg:+.1f}%",
            "Tổ chức đang MUA" if etf_chg > 2 else ("Tổ chức đang BÁN" if etf_chg < -2 else "Trung tính"),
            delta_color="normal" if etf_chg > 0 else "inverse",
        )
    else:
        wc1.metric("🏦 ETF Flow", "Không có dữ liệu")

    if vol_data:
        vr = vol_data["ratio"]
        wc2.metric(
            f"📊 Volume Ratio ({vol_data.get('ticker','')} 5d/avg)",
            f"{vr:.1f}×",
            "Block trading!" if vr > 2 else ("Tổ chức hoạt động" if vr > 1.5 else ("Bình thường" if vr > 0.7 else "Thanh khoản kém")),
            delta_color="normal" if vr > 1.3 else "off",
        )
    else:
        wc2.metric("📊 Volume Ratio", "N/A")

    if oi_data:
        wc3.metric("📌 Open Interest", f"{oi_data['value']:,}",
                   oi_data["ticker"], delta_color="off")
    else:
        wc3.metric("📌 Open Interest", "N/A")

    if w_signals or w_cot:
        with st.expander("🔍 Chi tiết định vị cá mập (ETF flow · Volume · OI)", expanded=False):
            if w_cot:
                st.markdown(f"**{w_cot}**\n")
                st.markdown("---")
            for icon, text, _ in w_signals:
                st.markdown(f"{icon} {text}")
            st.markdown(
                f"\n*Đóng góp vào dự báo: Whale score {w_score:+d} × 25% = {whale_contrib:+d} điểm*\n\n"
                f"*Lưu ý: COT (Commitment of Traders) chính thức từ CFTC cập nhật mỗi thứ Sáu. "
                f"App dùng ETF AUM flow + volume momentum + Open Interest làm proxy thời gian thực.*"
            )

    st.markdown("---")

    # ── Chart ─────────────────────────────────────────────────────────────
    ml = period_label(forecast_days)
    st.markdown(
        f"#### Biểu đồ {a_name} · Dự báo {ml} · "
        f"Cập nhật {datetime.now():%H:%M %d/%m/%Y}"
    )
    fig = build_chart(price, ma20, ma50, ma200, bb_up, bb_lo, hi52, lo52,
                      fc_mean, fc_lo_s, fc_hi_s, rsi, sig_color, forecast_days,
                      asset_key)
    st.plotly_chart(fig, use_container_width=True,
                    config={"scrollZoom": True, "responsive": True,
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                            "toImageButtonOptions": {"filename": f"{asset_key}_chart"}})

    st.markdown("---")

    # ── Signal box ────────────────────────────────────────────────────────
    sig_bg  = hex_rgba(sig_color, 0.12)
    sig_bdr = hex_rgba(sig_color, 0.45)
    st.markdown(
        f"""<div class="signal-box" style="background:{sig_bg};border:1px solid {sig_bdr};">
        <b style="color:{sig_color};font-size:1.05rem;">{sig_icon} NHẬN ĐỊNH TỔNG HỢP: {signal}</b><br>
        <span style="color:#8b949e;font-size:0.85rem;">
        Giá hiện tại: <b style="color:#e6edf3;">{fmt_price(cur, asset_key)}</b> &nbsp;→&nbsp;
        Dự báo cuối kỳ: <b style="color:{sig_color};">{fmt_price(fc_e, asset_key)}</b> ({sign}{chg:.1f}%)
        &nbsp;·&nbsp; Macro: <b style="color:{macro_score_color};">{macro_label}</b>
        </span></div>""",
        unsafe_allow_html=True,
    )

    col_t, col_m = st.columns(2)
    with col_t:
        st.markdown("**📐 Tín hiệu kỹ thuật:**")
        for note in tech_notes:
            st.markdown(f"- {note}")
    with col_m:
        st.markdown("**🌐 Tín hiệu vĩ mô:**")
        shown = 0
        for icon, text, _ in macro_signals:
            if icon not in ("📊",) and shown < 5:
                st.markdown(f"- {icon} {text}")
                shown += 1
        st.markdown(f"- {seas_note}")

    st.markdown("---")

    # ── Narrative ─────────────────────────────────────────────────────────
    st.markdown(f"#### 📝 Giải thích chi tiết — Tại sao {a_short} dự báo như vậy trong {period_label(forecast_days)}?")
    narrative = generate_narrative(
        price, ma20, ma50, ma200, rsi,
        fc_mean, fc_lo_s, fc_hi_s,
        macro_score, macro_signals, macro_metrics,
        forecast_days, seas,
    )
    sections = narrative.split("\n\n---\n\n")
    sec_icons = ["📐", "🌐", "📅", "🎯"]
    for i, sec in enumerate(sections):
        title_end = sec.find("\n\n")
        if title_end == -1:
            st.markdown(sec); continue
        title = sec[:title_end].replace("**", "").strip()
        body  = sec[title_end + 2:]
        icon  = sec_icons[i] if i < len(sec_icons) else "📌"
        with st.expander(title, expanded=(i == len(sections) - 1)):
            st.markdown(body)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<p style='color:#484f58;font-size:0.76rem;text-align:center;'>"
        f"Nguồn: {ticker} · Mô hình: HW + ARIMA + Momentum + Macro + Fed (Trump/Warsh) + Cá Mập + Mùa vụ · "
        f"Điểm tổng hợp: Macro {macro_score:+d} + Fed {fed_contrib:+d} + Whale {whale_contrib:+d} = {combined_macro:+d} · "
        f"⚠️ Chỉ mang tính tham khảo, không phải khuyến nghị đầu tư.</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="gold-header">
        <h1>📊 MARKET TREND ANALYSIS</h1>
        <p>Phân tích đa yếu tố vĩ mô + kỹ thuật + mùa vụ · Vàng · Bạc · Đồng · Dầu · USD/VND · Bitcoin</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([3, 1, 3])
    with c1:
        forecast_days = st.radio(
            "Kỳ dự báo:", list(PERIOD_LABELS.keys()),
            format_func=lambda x: PERIOD_LABELS[x],
            horizontal=True, label_visibility="collapsed"
        )
    with c2:
        if st.button("🔄 Làm mới", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Fetch macro một lần, chia sẻ cho tất cả tabs ─────────────────────────
    with st.spinner("📡 Đang tải chỉ số vĩ mô..."):
        macro = fetch_macro()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_keys   = list(ASSETS.keys())
    tab_labels = [ASSETS[k]["tab"] for k in tab_keys]
    tabs = st.tabs(tab_labels)

    for tab, asset_key in zip(tabs, tab_keys):
        with tab:
            render_asset_tab(asset_key, macro, forecast_days)


if __name__ == "__main__":
    main()
