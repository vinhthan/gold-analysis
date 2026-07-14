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
    1:    "Ngày mai",
    3:    "3 ngày tới",
    7:    "1 tuần tới",
    14:   "2 tuần tới",
    21:   "3 tuần tới",
    30:   "1 tháng tới",
    60:   "2 tháng tới",
    90:   "3 tháng tới",
    180:  "6 tháng tới",
    270:  "9 tháng tới",
    365:  "1 năm tới",
    730:  "2 năm tới",
    1095: "3 năm tới",
    1825: "5 năm tới",
}

SHORT_TERM_DAYS = {1, 3, 7, 14, 21}   # Kỳ giao dịch ngắn hạn

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

# ── Leader Profiles Database — Tự động tra cứu theo tên phát hiện ───────────
# Mỗi profile chứa: ngoại cảm tính cách + ảnh hưởng lên chính sách tiền tệ/vàng
# cut_pressure (0-3): áp lực cắt lãi từ TT;  rate_direction (+/-): hawk/dove của Fed Chair
LEADER_PROFILES: dict = {
    # ════════ US PRESIDENTS ══════════════════════════════════════════════════
    "donald trump": {
        "role": "president", "name_vn": "Donald Trump",
        "term_info": "Tổng thống thứ 47 (2025–hiện tại)",
        "emoji": "🇺🇸",
        "personality_vn": [
            "Luôn muốn lãi suất thấp nhất có thể — coi S&P 500 là điểm số nhiệm kỳ",
            "Thuế quan = vũ khí đàm phán chính → rủi ro lạm phát cao",
            "Sẵn sàng tấn công Fed công khai qua mạng xã hội",
            "Muốn USD yếu để hỗ trợ xuất khẩu và sản xuất nội địa",
            "Bất ngờ có chủ đích — deal-maker: sẽ xuống thang nếu có lợi",
        ],
        "cut_pressure":     3,   # áp lực cắt lãi (0=không, 3=tối đa)
        "fed_pressure":     3,   # sẵn sàng tấn công Fed công khai
        "inflation_risk":   2,   # rủi ro lạm phát từ chính sách thuế quan
        "gold_bias":       +2,   # thuế quan + USD yếu + lạm phát → vàng tăng
        "usd_bias":        -2,   # muốn USD yếu
        # Dùng để tính Warsh/Fed chair factor
        "sp500_sensitivity": 3,  # áp lực Fed tỷ lệ với S&P drops
        "resistance_kw":    0.3, # Fed chair kháng cự được ~30% áp lực Trump
    },
    "joe biden": {
        "role": "president", "name_vn": "Joe Biden",
        "term_info": "Tổng thống thứ 46 (2021–2025)",
        "emoji": "🇺🇸",
        "personality_vn": [
            "Tôn trọng tính độc lập của Fed — không gây áp lực công khai",
            "Chi tiêu xã hội lớn (IRA, CHIPS Act) → rủi ro lạm phát vừa phải",
            "Chính sách đối ngoại truyền thống, đa phương",
            "Ủng hộ kinh tế xanh và chuyển dịch năng lượng",
        ],
        "cut_pressure":     0, "fed_pressure": 0,
        "inflation_risk":   1, "gold_bias": 0, "usd_bias": 0,
        "sp500_sensitivity": 0, "resistance_kw": 0.0,
    },
    "kamala harris": {
        "role": "president", "name_vn": "Kamala Harris",
        "term_info": "Phó Tổng thống (2021–2025)",
        "emoji": "🇺🇸",
        "personality_vn": [
            "Chính sách tiến bộ, chi tiêu xã hội cao",
            "Tôn trọng độc lập Fed, không can thiệp công khai",
            "Ủng hộ năng lượng sạch, kiểm soát giá",
        ],
        "cut_pressure":     1, "fed_pressure": 0,
        "inflation_risk":   1, "gold_bias": 0, "usd_bias": 0,
        "sp500_sensitivity": 0, "resistance_kw": 0.0,
    },
    "barack obama": {
        "role": "president", "name_vn": "Barack Obama",
        "term_info": "Tổng thống thứ 44 (2009–2017)",
        "emoji": "🇺🇸",
        "personality_vn": [
            "Chính sách ổn định, truyền thống — tôn trọng Fed",
            "Phục hồi hậu khủng hoảng 2008 — ủng hộ QE",
            "Tăng thuế người giàu, chi tiêu hạ tầng vừa phải",
        ],
        "cut_pressure":     0, "fed_pressure": 0,
        "inflation_risk":   0, "gold_bias": 0, "usd_bias": 0,
        "sp500_sensitivity": 0, "resistance_kw": 0.0,
    },
    # ════════ FED CHAIRS ═════════════════════════════════════════════════════
    "kevin warsh": {
        "role": "fed_chair", "name_vn": "Kevin Warsh",
        "term_info": "Fed Chair từ 22/5/2026",
        "emoji": "🦅",
        "personality_vn": [
            "Diều hâu (hawkish) bẩm sinh — phê phán QE mạnh nhất lịch sử",
            "Rules-based: ưu tiên Taylor Rule hơn discretion",
            "Uy tín Fed > áp lực chính trị — sẽ kháng cự nhưng không đối đầu",
            "Morgan Stanley + Nhà Trắng + Druckenmiller → market-savvy",
            "Trực tiếp, rõ ràng hơn Powell; không thích forward guidance mơ hồ",
        ],
        "hawkish_bias":         2,
        "credibility_priority": 3,
        "qe_skepticism":        3,
        "political_resistance": 1,   # kháng cự áp lực TT (0=dễ bị ảnh hưởng, 3=độc lập hoàn toàn)
        "rate_direction":      +2,   # +số = hawkish, -số = dovish
        "gold_bias":           -1,   # hawkish → lãi thực cao → vàng bị áp lực
        "score_adj":           -2,   # điều chỉnh fed_score (âm = khó cắt lãi)
    },
    "jerome powell": {
        "role": "fed_chair", "name_vn": "Jerome Powell",
        "term_info": "Fed Chair 2018–2026",
        "emoji": "⚖️",
        "personality_vn": [
            "Data-dependent — quyết định theo dữ liệu kinh tế, không định trước",
            "Thận trọng, không vội vã, tránh gây surprise cho thị trường",
            "Dovish khi kinh tế cần, hawkish khi lạm phát cao — linh hoạt",
            "Giao tiếp rõ ràng qua forward guidance; ưu tiên ổn định thị trường",
        ],
        "hawkish_bias":         0,
        "credibility_priority": 2,
        "qe_skepticism":        1,
        "political_resistance": 2,
        "rate_direction":       0,
        "gold_bias":            0,
        "score_adj":            0,
    },
    "janet yellen": {
        "role": "fed_chair", "name_vn": "Janet Yellen",
        "term_info": "Fed Chair 2014–2018",
        "emoji": "🕊️",
        "personality_vn": [
            "Bồ câu (dovish) — ưu tiên việc làm hơn lạm phát",
            "Ủng hộ chính sách nới lỏng khi kinh tế yếu",
            "Kinh tế học lao động là chuyên môn — quan tâm thất nghiệp",
        ],
        "hawkish_bias":        -2,
        "credibility_priority": 2,
        "qe_skepticism":       -1,
        "political_resistance": 1,
        "rate_direction":      -2,
        "gold_bias":           +1,
        "score_adj":           +2,
    },
    "michelle bowman": {
        "role": "fed_chair", "name_vn": "Michelle Bowman",
        "term_info": "Thành viên Fed Board",
        "emoji": "🦅",
        "personality_vn": [
            "Diều hâu — phản đối cắt lãi sớm khi lạm phát chưa về target",
            "Ưu tiên kiểm soát lạm phát trước khi kích thích",
            "Bỏ phiếu chống cắt lãi 2024 — người duy nhất phản đối",
        ],
        "hawkish_bias":         2, "credibility_priority": 2,
        "qe_skepticism":        2, "political_resistance": 1,
        "rate_direction":      +1, "gold_bias": -1, "score_adj": -1,
    },
    "christopher waller": {
        "role": "fed_chair", "name_vn": "Christopher Waller",
        "term_info": "Thành viên Fed Board",
        "emoji": "📊",
        "personality_vn": [
            "Diều hâu vừa phải — ủng hộ cắt lãi khi dữ liệu cho phép",
            "Kinh tế học tiền tệ thực chứng, hướng data",
        ],
        "hawkish_bias":         1, "credibility_priority": 2,
        "qe_skepticism":        1, "political_resistance": 1,
        "rate_direction":       0, "gold_bias": 0, "score_adj": 0,
    },
    "philip jefferson": {
        "role": "fed_chair", "name_vn": "Philip Jefferson",
        "term_info": "Phó Chủ tịch Fed",
        "emoji": "🕊️",
        "personality_vn": [
            "Trung dung — linh hoạt giữa hawkish và dovish",
            "Ưu tiên ổn định kép: lạm phát + việc làm",
        ],
        "hawkish_bias":         0, "credibility_priority": 2,
        "qe_skepticism":        0, "political_resistance": 1,
        "rate_direction":       0, "gold_bias": 0, "score_adj": 0,
    },
    # ─── Default fallback cho lãnh đạo chưa được lập trình ───────────────────
    "_unknown_president": {
        "role": "president", "name_vn": "Tổng thống Mỹ", "term_info": "Đang xác định",
        "emoji": "🇺🇸", "personality_vn": ["Đang phân tích ngoại cảm..."],
        "cut_pressure": 0, "fed_pressure": 0, "inflation_risk": 0,
        "gold_bias": 0, "usd_bias": 0, "sp500_sensitivity": 1, "resistance_kw": 0.3,
    },
    "_unknown_fedchair": {
        "role": "fed_chair", "name_vn": "Chủ tịch Fed", "term_info": "Đang xác định",
        "emoji": "🏦", "personality_vn": ["Đang phân tích ngoại cảm..."],
        "hawkish_bias": 0, "credibility_priority": 2, "qe_skepticism": 0,
        "political_resistance": 1, "rate_direction": 0, "gold_bias": 0, "score_adj": 0,
    },
}


def get_leader_profile(name: str, role: str) -> dict:
    """Tra cứu profile lãnh đạo theo tên (fuzzy match). Fallback về unknown nếu không có."""
    if not name:
        return LEADER_PROFILES[f"_unknown_{role}"]
    nl = name.lower().strip()
    if nl in LEADER_PROFILES and LEADER_PROFILES[nl]["role"] == role:
        return LEADER_PROFILES[nl]
    # Partial match — so khớp từng từ trong tên
    name_parts = set(nl.split())
    for key, prof in LEADER_PROFILES.items():
        if key.startswith("_") or prof.get("role") != role:
            continue
        if name_parts & set(key.split()):   # có từ trùng nhau
            return prof
    # Không tìm thấy — tạo unknown với tên thực
    fallback = dict(LEADER_PROFILES[f"_unknown_{role}"])
    fallback["name_vn"] = name
    return fallback


def analyze_leader_dynamic(pres: dict, chair: dict) -> dict:
    """Phân tích động lực xung đột / hợp tác giữa Tổng thống và Chủ tịch Fed."""
    pres_cut  = pres.get("cut_pressure", 0)
    chair_hk  = chair.get("rate_direction", 0)   # >0 = hawkish
    chair_res = chair.get("political_resistance", 1)

    if pres_cut > 0 and chair_hk > 0:
        # TT muốn cắt lãi ↔ Chair hawkish = MÂU THUẪN
        conflict = min(3, (pres_cut + chair_hk + 1) // 2)
        resist_factor = chair_res / 3          # 0→1
        pres_wins  = max(10, round(50 - resist_factor * 30 - chair_hk * 5))
        chair_wins = min(80, round(25 + resist_factor * 30 + chair_hk * 5))
        narrative  = "MÂU THUẪN CAO" if conflict >= 2 else "MÂU THUẪN NHẸ"
    elif pres_cut <= 0 and chair_hk <= 0:
        # Cả hai dovish = ĐỒNG THUẬN
        conflict   = 0
        pres_wins  = 75; chair_wins = 75
        narrative  = "ĐỒNG THUẬN — Cả hai ủng hộ nới lỏng"
    elif pres_cut <= 0 and chair_hk > 0:
        # TT trung tính, Chair hawkish
        conflict   = 0
        pres_wins  = 50; chair_wins = 60
        narrative  = "ĐỒNG THUẬN — Ưu tiên kiểm soát lạm phát"
    else:
        conflict   = 1
        pres_wins  = 50; chair_wins = 45
        narrative  = "TRUNG TÍNH"

    comp = max(5, 100 - pres_wins - chair_wins)
    return {
        "conflict_level": conflict,
        "narrative":      narrative,
        "pres_wins_prob":  pres_wins,
        "chair_wins_prob": chair_wins,
        "compromise_prob": comp,
    }

# ── Bảng giải thích thuật ngữ — hiển thị trong tab app ─────────────────────
GLOSSARY = {
    # ── Kỹ thuật ──────────────────────────────────────────────────────────
    "MA20 / MA50 / MA200":
        "Moving Average (Đường trung bình động) 20/50/200 ngày. Tính bằng trung bình cộng giá "
        "đóng cửa trong N ngày gần nhất. MA20 = ngắn hạn, MA50 = trung hạn, MA200 = dài hạn. "
        "Giá nằm TRÊN MA200 = xu hướng tăng dài hạn được xác nhận.",
    "RSI (14)":
        "Relative Strength Index — Chỉ báo động lượng. Đo tốc độ và độ lớn thay đổi giá trong 14 "
        "ngày. Thang 0–100. >70 = Quá mua (overbought) → nguy cơ điều chỉnh. <30 = Quá bán "
        "(oversold) → khả năng hồi phục. 40–60 = Trung tính.",
    "Bollinger Bands (BB)":
        "Dải Bollinger = MA20 ± 2 độ lệch chuẩn. Khi giá chạm dải trên = overbought, "
        "chạm dải dưới = oversold. Dải co lại = biến động thấp, sắp có đột phá lớn.",
    "52W High / Low":
        "52-Week High/Low — Giá cao nhất/thấp nhất trong 52 tuần (1 năm) qua. "
        "Giá phá vỡ 52W High = tín hiệu tăng mạnh. Giá gần 52W Low = vùng hỗ trợ quan trọng.",

    # ── Vĩ mô ─────────────────────────────────────────────────────────────
    "DXY (Dollar Index)":
        "US Dollar Index — Chỉ số sức mạnh USD so với rổ 6 đồng tiền chính "
        "(EUR 57.6%, JPY 13.6%, GBP 11.9%, CAD 9.1%, SEK 4.2%, CHF 3.6%). "
        "DXY tăng = USD mạnh → áp lực giảm vàng/bạc. DXY giảm = USD yếu → hỗ trợ hàng hóa.",
    "Yield 10Y (TNX)":
        "10-Year US Treasury Yield — Lợi suất trái phiếu kho bạc Mỹ kỳ hạn 10 năm. "
        "Là lãi suất tham chiếu quan trọng nhất thế giới. Yield tăng = chi phí vay vốn tăng, "
        "USD hấp dẫn hơn → áp lực giảm vàng. Yield giảm → hỗ trợ vàng.",
    "VIX (Fear Index)":
        "CBOE Volatility Index — Chỉ số sợ hãi thị trường. Đo biến động kỳ vọng 30 ngày "
        "của S&P 500 từ giá options. <15 = Tự tin/tham lam. 15–25 = Trung tính. "
        "25–35 = Lo lắng. >35 = Hoảng loạn → tiền chạy vào vàng (safe haven).",
    "S&P 500 (SPX)":
        "Standard & Poor's 500 — Chỉ số cổ phiếu 500 công ty lớn nhất Mỹ. "
        "Đại diện cho tâm lý risk-on/risk-off. S&P tăng = risk-on → tiền rời khỏi vàng. "
        "S&P giảm mạnh = risk-off → tiền đổ vào vàng và tài sản an toàn.",
    "TIPS ETF (TIP)":
        "iShares TIPS Bond ETF — Quỹ trái phiếu bảo vệ lạm phát (Treasury Inflation-Protected). "
        "TIP tăng giá = lãi suất thực đang giảm → hỗ trợ vàng. Proxy gián tiếp cho lãi suất thực.",

    # ── Lãi suất & Fed ────────────────────────────────────────────────────
    "FEDFUNDS":
        "Federal Funds Effective Rate — Lãi suất liên ngân hàng qua đêm (overnight) tại Mỹ. "
        "Đây là lãi suất chính sách của Fed, quyết định chi phí vốn toàn cầu. "
        "Hiện tại Fed target 2.5% là lãi suất trung tính (neutral rate).",
    "DFII10 (Lãi suất thực)":
        "10-Year Treasury Inflation-Indexed Security Yield — Lãi suất thực (real interest rate) "
        "kỳ hạn 10 năm. = Yield danh nghĩa − Lạm phát kỳ vọng. "
        "ĐÂY LÀ INDICATOR QUAN TRỌNG NHẤT CỦA VÀNG: Tương quan âm -0.90. "
        "DFII10 < 0% = vàng không có chi phí cơ hội → BUY mạnh. "
        "DFII10 > 2% = chi phí cơ hội cao → áp lực bán vàng.",
    "T5YIFR (Kỳ vọng lạm phát)":
        "5-Year, 5-Year Forward Inflation Expectation Rate — Kỳ vọng lạm phát trung bình "
        "5 năm, bắt đầu từ 5 năm nữa. Đây là thước đo mà Fed theo dõi sát nhất. "
        "Target của Fed là 2%. >2.75% = lạm phát mất kiểm soát → Fed không thể cắt lãi. "
        "<2.0% = áp lực deflation → Fed có lý do cắt lãi.",
    "T10Y2Y (Yield Curve)":
        "10Y Treasury Yield trừ 2Y Treasury Yield — Đường cong lợi suất. "
        "BÌNH THƯỜNG: 10Y > 2Y (dương) = kinh tế khỏe mạnh. "
        "ĐẢO NGƯỢC (âm): 2Y > 10Y = thị trường kỳ vọng kinh tế suy thoái và Fed sẽ cắt lãi. "
        "Lịch sử: đảo ngược → 6-18 tháng sau thường xảy ra suy thoái.",
    "WALCL (Fed Balance Sheet)":
        "Weekly Assets of Large Commercial Banks — Tổng tài sản trên bảng cân đối của Fed (nghìn tỷ USD). "
        "QE (Quantitative Easing = nới lỏng định lượng): Fed mua trái phiếu → bảng tăng → thanh khoản nhiều → hỗ trợ vàng/BTC. "
        "QT (Quantitative Tightening = thắt chặt): Fed bán trái phiếu → bảng giảm → rút thanh khoản → áp lực giảm tài sản.",
    "Neutral Rate (Lãi suất trung tính)":
        "Mức lãi suất mà ở đó nền kinh tế không tăng trưởng quá nóng cũng không suy thoái. "
        "Fed ước tính neutral rate ≈ 2.5%. Nếu FEDFUNDS > 2.5% = chính sách thắt chặt (restrictive). "
        "Nếu FEDFUNDS < 2.5% = chính sách nới lỏng (accommodative).",

    # ── Cá mập & ETF ──────────────────────────────────────────────────────
    "ETF (Exchange-Traded Fund)":
        "Quỹ giao dịch trên sàn. Cho phép nhà đầu tư tổ chức mua/bán hàng hóa gián tiếp. "
        "GLD (SPDR Gold) = 1 share ≈ 0.094 oz vàng. SLV (iShares Silver) ≈ 0.94 oz bạc. "
        "IBIT (BlackRock Bitcoin) ≈ 0.001 BTC. COPX = cổ phiếu công ty đồng. USO = dầu WTI.",
    "AUM (Assets Under Management)":
        "Tổng tài sản đang được quản lý. Dùng như proxy cho dòng tiền tổ chức. "
        "AUM của GLD tăng = tổ chức đang mua vàng. AUM giảm = tổ chức đang bán vàng.",
    "Open Interest (OI)":
        "Số lượng hợp đồng futures đang mở chưa được tất toán. "
        "OI tăng = tổ chức đang vào vị thế mới (bullish hoặc bearish). "
        "OI giảm + giá giảm = tổ chức đang thoát lệnh mua (bearish). "
        "OI rất cao = sắp có biến động mạnh khi hợp đồng đáo hạn.",
    "COT (Commitment of Traders)":
        "Báo cáo định vị nhà giao dịch từ CFTC (Commodity Futures Trading Commission — Mỹ). "
        "Phát hành mỗi thứ Sáu. Phân loại: Commercial (hedgers), Non-Commercial (hedge funds/speculators). "
        "Large Speculators Net Long tăng = cá mập đang mua → bullish signal.",
    "Fear & Greed Index":
        "Chỉ số sợ hãi và tham lam của thị trường crypto (Alternative.me). "
        "Tổng hợp từ: volatility, market momentum, social media, surveys, dominance, trends. "
        "0–25 = Cực kỳ sợ hãi (mua khi người khác sợ). 75–100 = Cực kỳ tham lam (cẩn thận đỉnh).",

    # ── Tỷ lệ & tín hiệu ──────────────────────────────────────────────────
    "Gold/Silver Ratio (G/S Ratio)":
        "Số ounce bạc cần để mua 1 ounce vàng. Trung bình lịch sử: 65–70. "
        "Tỷ lệ > 80: Bạc đang rẻ tương đối so với vàng → nhiều khả năng bạc sẽ tăng mạnh hơn. "
        "Tỷ lệ < 50: Vàng đang rẻ hơn tương đối. Hiện tại khi tỷ lệ cao → cơ hội mua bạc tốt.",
    "Momentum":
        "Tốc độ và sức mạnh của xu hướng giá. Momentum dương = giá đang tăng nhanh. "
        "Momentum 3T (3 tháng) và 6T (6 tháng) đo đà tăng/giảm trung hạn. "
        "Momentum mạnh thường tiếp tục — nhưng khi quá mạnh → rủi ro đảo chiều đột ngột.",
    "Lệch MA200 (Deviation from MA200)":
        "Phần trăm giá hiện tại cao hơn hoặc thấp hơn đường MA200. "
        "> +25%: Overbought dài hạn → lịch sử thường xảy ra điều chỉnh. "
        "< -15%: Giá trị hấp dẫn dài hạn → cơ hội mua tốt. "
        "MA200 là 'chân trời' của xu hướng dài hạn.",

    # ── Mô hình dự báo ────────────────────────────────────────────────────
    "Holt-Winters (HW)":
        "Mô hình dự báo chuỗi thời gian — bộ ba số mũ mũ hóa (triple exponential smoothing). "
        "Bắt được trend dài hạn có damping (xu hướng yếu dần theo thời gian). "
        "Trọng số trong ensemble: 45%.",
    "ARIMA (1,1,1)":
        "AutoRegressive Integrated Moving Average — Mô hình tự hồi quy tích hợp trung bình động. "
        "Bắt được cấu trúc tự tương quan của chuỗi giá. Trọng số: 35%.",
    "Momentum Model":
        "Mô hình hồi quy log-return trên 60 ngày gần nhất, có damping 45%. "
        "Đại diện cho xu hướng kỹ thuật ngắn-trung hạn. Trọng số: 20%.",
    "CI 90% (Confidence Interval)":
        "Khoảng tin cậy 90% — Vùng giá mà xác suất 90% giá thực tế sẽ nằm trong. "
        "Vùng rộng = bất định cao (dự báo dài). Vùng hẹp = dự báo ngắn hạn tự tin hơn. "
        "KHÔNG phải đảm bảo — sự kiện bất ngờ (geopolitical, Fed shock) có thể phá vỡ mọi CI.",

    # ── Futures tickers ───────────────────────────────────────────────────
    "GC=F":
        "COMEX Gold Futures — Hợp đồng tương lai vàng tại sở COMEX New York. "
        "Đơn vị: troy ounce. Cao hơn giá spot ~$10-30 do chi phí lưu trữ và lãi suất.",
    "SI=F":
        "COMEX Silver Futures — Hợp đồng tương lai bạc tại COMEX.",
    "HG=F":
        "COMEX Copper Futures — Hợp đồng tương lai đồng. Đơn vị USD/pound (lb). "
        "'Dr. Copper': Đồng được coi là chỉ báo sức khỏe kinh tế toàn cầu vì ứng dụng công nghiệp rộng.",
    "CL=F":
        "NYMEX WTI Crude Oil Futures — Hợp đồng tương lai dầu thô WTI (West Texas Intermediate) "
        "tại sở NYMEX. Benchmark dầu Mỹ. Đơn vị: USD/barrel (thùng, 159 lít).",
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
            with _ur.urlopen(req, timeout=12, context=ctx) as r:
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
        with _ur.urlopen(req, timeout=12, context=ctx) as r:
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
        "fedfunds":    "FEDFUNDS",  # Lãi suất Fed Funds hiện tại
        "yield2y":     "DGS2",      # 2-Year Treasury (predictor tốt nhất của Fed)
        "curve":       "T10Y2Y",    # 10Y - 2Y spread (yield curve)
        "curve_3m":    "T10Y3M",    # 10Y - 3M spread (recession indicator)
        "real_rate":   "DFII10",    # 10Y Real Interest Rate — lãi suất thực, driver số 1 của vàng
        "inflation5y": "T5YIFR",    # 5Y5Y Forward Inflation Expectation — kỳ vọng lạm phát
        "fed_balance": "WALCL",     # Fed Balance Sheet (nghìn tỷ USD) — QT vs QE cycle
        "breakeven5y": "T5YIE",     # 5-Year Breakeven Inflation — kỳ vọng lạm phát thị trường (daily)
        "m2":          "M2SL",      # M2 Money Supply — cung tiền (monthly), bullish cho vàng khi tăng
        "credit_oas":  "BAMLH0A0HYM2",  # ICE BofA US High Yield OAS — rủi ro tín dụng (credit stress)
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


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_current_leaders() -> dict:
    """
    Tự động xác định Tổng thống Mỹ và Chủ tịch Fed hiện tại qua Wikidata SPARQL.
    Fallback: Wikipedia REST API.  Cache 24h — thông tin thay đổi rất hiếm.
    """
    import urllib.request, urllib.parse, json as _json, re as _re

    result = {"president": None, "fed_chair": None, "ok": False, "source": None}
    hdrs   = {"User-Agent": "GoldAnalysisStreamlit/2.0 (educational; contact=github.com/vinhthan)"}

    def _sparql(query: str) -> list[str]:
        url = ("https://query.wikidata.org/sparql?query="
               + urllib.parse.quote(query.strip()) + "&format=json")
        req = urllib.request.Request(url, headers={**hdrs, "Accept": "application/sparql-results+json"})
        with urllib.request.urlopen(req, timeout=14) as r:
            data = _json.loads(r.read())
        return [b.get("personLabel", {}).get("value", "")
                for b in data["results"]["bindings"] if b.get("personLabel")]

    def _wiki_extract(page: str) -> str:
        url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
               + urllib.parse.quote(page.replace(" ", "_")))
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=10) as r:
            return _json.loads(r.read()).get("extract", "")

    # ── 1. Wikidata SPARQL (ưu tiên — dữ liệu chuẩn nhất) ──────────────
    PRES_QUERY = """
    SELECT ?personLabel WHERE {
      wd:Q30 p:P35 ?stmt .
      ?stmt ps:P35 ?person .
      FILTER NOT EXISTS { ?stmt pq:P582 ?end . }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    } LIMIT 3"""

    FED_QUERY = """
    SELECT ?personLabel WHERE {
      wd:Q146190 p:P488 ?stmt .
      ?stmt ps:P488 ?person .
      FILTER NOT EXISTS { ?stmt pq:P582 ?end . }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    } LIMIT 3"""

    try:
        names = _sparql(PRES_QUERY)
        if names and not names[0].startswith("Q"):
            result["president"] = names[0]
            result["source"]    = "Wikidata"
    except Exception:
        pass

    try:
        names = _sparql(FED_QUERY)
        if names and not names[0].startswith("Q"):
            result["fed_chair"] = names[0]
            result["source"]    = result["source"] or "Wikidata"
    except Exception:
        pass

    # ── 2. Wikipedia REST fallback (nếu Wikidata chậm / lỗi) ───────────
    if not result["president"]:
        try:
            ext = _wiki_extract("President of the United States")
            for pat in [
                r"(\w[\w\-']+(?:\s+\w[\w\-']+){1,3})\s+is the\s+(?:\d+\w+\s+and\s+)?current president",
                r"(\w[\w\-']+(?:\s+\w[\w\-']+){1,3})\s+is the\s+\d+\w+\s+president",
            ]:
                m = _re.search(pat, ext, _re.IGNORECASE)
                if m:
                    result["president"] = m.group(1).strip()
                    result["source"]    = result["source"] or "Wikipedia"
                    break
        except Exception:
            pass

    if not result["fed_chair"]:
        try:
            ext = _wiki_extract("Chair of the Federal Reserve")
            for pat in [
                r"(?:current|incumbent)\s+chair(?:man|person)?\s+is\s+(\w[\w\-']+(?:\s+\w[\w\-']+){1,2})",
                r"(\w[\w\-']+(?:\s+\w[\w\-']+){1,2})\s+(?:is|serves as)\s+(?:the\s+)?(?:current\s+)?chair",
            ]:
                m = _re.search(pat, ext, _re.IGNORECASE)
                if m:
                    result["fed_chair"] = m.group(1).strip()
                    result["source"]    = result["source"] or "Wikipedia"
                    break
        except Exception:
            pass

    result["ok"] = bool(result["president"] or result["fed_chair"])
    return result


def fed_policy_analysis(fred_data: dict, macro: dict, leaders: dict = None) -> dict:
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

    # ── 4. FED CHAIR NGOẠI CẢM FACTOR (tự động từ profile) ───────────────
    # Lấy profile động: ưu tiên leaders truyền vào, fallback về Warsh (mặc định hiện tại)
    if leaders and leaders.get("fed_chair"):
        chair_prof = get_leader_profile(leaders["fed_chair"], "fed_chair")
    else:
        chair_prof = get_leader_profile("kevin warsh", "fed_chair")  # default hiện tại

    w_hawkish = chair_prof.get("hawkish_bias", 0)
    w_cred    = chair_prof.get("credibility_priority", 2)
    w_resist  = chair_prof.get("political_resistance", 1)
    chair_net = -(w_hawkish * 0.5 + w_cred * 0.2 + w_resist * 0.1)   # âm = hawkish
    score += round(chair_net)
    chair_emoji  = chair_prof.get("emoji", "🏦")
    chair_namevn = chair_prof.get("name_vn", "Fed Chair")
    if chair_net < -0.5:
        signals.append((chair_emoji, f"{chair_namevn} Factor: hawkish DNA (bias {chair_net:.1f}) → lãi cao hơn thị trường kỳ vọng", "orange"))
    elif chair_net > 0.5:
        signals.append((chair_emoji, f"{chair_namevn} Factor: dovish DNA (bias {chair_net:.1f}) → nghiêng về cắt lãi sớm hơn", "green"))
    else:
        signals.append((chair_emoji, f"{chair_namevn} Factor: trung dung (bias {chair_net:.1f}) — theo dữ liệu thực tế", "gray"))

    # ── 5. PRESIDENT PRESSURE FACTOR (ngoại cảm áp lực chính trị) ────────
    if leaders and leaders.get("president"):
        pres_prof = get_leader_profile(leaders["president"], "president")
    else:
        pres_prof = get_leader_profile("donald trump", "president")  # default hiện tại

    pres_cut_pref = pres_prof.get("cut_pressure", 0)   # TT muốn cắt lãi mạnh đến đâu
    pres_sp_sens  = pres_prof.get("sp500_sensitivity", 1)
    pres_namevn   = pres_prof.get("name_vn", "Tổng thống")
    resist_factor = chair_prof.get("political_resistance", 1)  # chair kháng cự TT được k

    pres_pressure = 0
    pres_signal   = ""
    if "sp500" in macro and len(macro["sp500"]) >= 22:
        sp_chg = (float(macro["sp500"].iloc[-1]) / float(macro["sp500"].iloc[-22]) - 1) * 100
        metrics["sp500_chg"] = sp_chg
        if pres_cut_pref >= 2:   # TT có xu hướng áp lực Fed mạnh
            if sp_chg < -12:
                pres_pressure = 3
                pres_signal   = f"S&P {sp_chg:.1f}% → {pres_namevn} áp lực tối đa lên Fed"
            elif sp_chg < -6:
                pres_pressure = 2
                pres_signal   = f"S&P {sp_chg:.1f}% → {pres_namevn} sẽ công khai chỉ trích Fed"
            elif sp_chg < -2:
                pres_pressure = 1
                pres_signal   = f"S&P {sp_chg:.1f}% → {pres_namevn} bắt đầu gây áp lực"
            elif sp_chg > 8:
                pres_pressure = -1
                pres_signal   = f"S&P +{sp_chg:.1f}% → {pres_namevn} hài lòng, ít áp lực Fed"
        elif pres_cut_pref == 1:
            if sp_chg < -8:
                pres_pressure = 1
                pres_signal   = f"S&P {sp_chg:.1f}% → {pres_namevn} nhẹ nhàng gợi ý cắt lãi"
        # else pres_cut_pref == 0: TT tôn trọng Fed → không tính áp lực

    effective_pres = pres_pressure * (1 - resist_factor * 0.3)
    score += round(effective_pres)
    if pres_signal:
        signals.append(("⚡", f"{pres_namevn} Pressure: {pres_signal}", "orange"))

    metrics["pres_pressure"]   = pres_pressure
    metrics["chair_net"]       = chair_net
    metrics["effective_pres"]  = effective_pres
    metrics["chair_name"]      = chair_namevn
    metrics["pres_name"]       = pres_namevn

    # ── 6. REAL INTEREST RATE — DFII10 (driver số 1 của vàng) ────────────
    # Lãi suất thực = Yield danh nghĩa − Lạm phát kỳ vọng
    # Tương quan âm -0.90 với giá vàng — đây là indicator quan trọng nhất
    real_rate = None
    if "real_rate" in fred_data and len(fred_data["real_rate"]) > 0:
        real_rate = float(fred_data["real_rate"].iloc[-1])
        metrics["real_rate"] = real_rate
        if real_rate < 0:
            score += 2
            signals.append(("✅", f"Lãi suất thực (DFII10): {real_rate:.2f}% — ÂM → vàng/bạc không có chi phí cơ hội, tương quan -0.90 với giá vàng → BUY signal mạnh", "green"))
        elif real_rate < 0.5:
            score += 1
            signals.append(("✅", f"Lãi suất thực (DFII10): {real_rate:.2f}% — thấp → hỗ trợ tài sản không sinh lãi (vàng, bạc, BTC)", "green"))
        elif real_rate > 2.5:
            score -= 2
            signals.append(("🔴", f"Lãi suất thực (DFII10): {real_rate:.2f}% — rất cao → chi phí cơ hội giữ vàng lớn → áp lực giảm mạnh", "red"))
        elif real_rate > 1.5:
            score -= 1
            signals.append(("⚠️", f"Lãi suất thực (DFII10): {real_rate:.2f}% — cao → áp lực nhẹ lên vàng/bạc", "orange"))
        else:
            signals.append(("➡️", f"Lãi suất thực (DFII10): {real_rate:.2f}% — trung tính (0.5–1.5%)", "gray"))

    # ── 7. INFLATION EXPECTATIONS — T5YIFR (kỳ vọng lạm phát 5 năm) ─────
    # 5Y5Y = thị trường kỳ vọng lạm phát trung bình 5 năm, bắt đầu từ 5 năm nữa
    # Fed target 2% — nếu > 2.75% thi Fed không thể cắt lãi dù áp lực từ Trump
    inflation_exp = None
    if "inflation5y" in fred_data and len(fred_data["inflation5y"]) > 0:
        inflation_exp = float(fred_data["inflation5y"].iloc[-1])
        metrics["inflation_exp"] = inflation_exp
        if inflation_exp > 2.75:
            score -= 1  # Lạm phát cao → Fed không thể cắt lãi
            signals.append(("⚠️", f"Kỳ vọng lạm phát 5Y5Y (T5YIFR): {inflation_exp:.2f}% — trên target, Fed khó cắt lãi. Nhưng vàng ĐƯỢC lợi kép: hedge lạm phát", "orange"))
        elif inflation_exp > 2.25:
            signals.append(("➡️", f"Kỳ vọng lạm phát 5Y5Y: {inflation_exp:.2f}% — được neo ổn định gần target 2%", "gray"))
        elif inflation_exp < 2.0:
            score += 1  # Lạm phát thấp → Fed có thể cắt lãi
            signals.append(("✅", f"Kỳ vọng lạm phát 5Y5Y: {inflation_exp:.2f}% — thấp → áp lực deflation → Fed có lý do cắt lãi sớm hơn", "green"))

    # ── 8. FED BALANCE SHEET — WALCL (QT vs QE) ──────────────────────────
    # WALCL = tổng tài sản Fed (nghìn tỷ USD). Tăng = QE (bơm tiền) = bullish vàng.
    # Giảm = QT (rút tiền) = bearish vàng. So sánh YoY (52 tuần = ~1 năm dữ liệu weekly).
    if "fed_balance" in fred_data and len(fred_data["fed_balance"]) >= 10:
        wb      = fred_data["fed_balance"]
        wb_now  = float(wb.iloc[-1])
        wb_ref  = float(wb.iloc[-52]) if len(wb) >= 52 else float(wb.iloc[0])
        wb_chg  = (wb_now / wb_ref - 1) * 100 if wb_ref > 0 else 0
        metrics["walcl"]     = round(wb_now, 2)
        metrics["walcl_chg"] = round(wb_chg, 1)
        if wb_chg < -10:
            score -= 2
            signals.append(("🔴", f"Fed Balance Sheet: {wb_now:.1f}T USD, giảm {abs(wb_chg):.1f}% YoY — QT mạnh → rút thanh khoản khỏi hệ thống, bearish vàng", "red"))
        elif wb_chg < -5:
            score -= 1
            signals.append(("⚠️", f"Fed Balance Sheet: {wb_now:.1f}T USD, QT -{abs(wb_chg):.1f}% YoY — siết thanh khoản, áp lực nhẹ", "orange"))
        elif wb_chg > 10:
            score += 2
            signals.append(("✅", f"Fed Balance Sheet: {wb_now:.1f}T USD, +{wb_chg:.1f}% YoY — QE / bơm thanh khoản → bullish vàng mạnh", "green"))
        elif wb_chg > 5:
            score += 1
            signals.append(("✅", f"Fed Balance Sheet: {wb_now:.1f}T USD, +{wb_chg:.1f}% YoY — nới lỏng định lượng nhẹ", "green"))
        else:
            signals.append(("➡️", f"Fed Balance Sheet: {wb_now:.1f}T USD, ổn định ({wb_chg:+.1f}% YoY) — không QE cũng không QT tích cực", "gray"))

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
        "score":         score,
        "direction":     direction,
        "color":         d_color,
        "prob_cut":      round(prob_cut),
        "prob_hold":     round(100 - prob_cut),
        "current_rate":  current_rate,
        "curve_val":     curve_val,
        "real_rate":     real_rate,
        "inflation_exp": inflation_exp,
        "signals":       signals,
        "metrics":       metrics,
        # Thông tin lãnh đạo động (để UI dùng)
        "pres_name":     metrics.get("pres_name", "Tổng thống"),
        "chair_name":    metrics.get("chair_name", "Fed Chair"),
        "pres_profile":  pres_prof,
        "chair_profile": chair_prof,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fear_greed() -> dict:
    """
    Fear & Greed Index từ alternative.me — miễn phí, không cần API key.
    Dùng cho BTC tab. Cũng hữu ích cho gold (nghịch chiều với sợ hãi).
    Score: 0–25 Extreme Fear · 26–45 Fear · 46–55 Neutral · 56–75 Greed · 76–100 Extreme Greed
    """
    import requests as _req
    try:
        r = _req.get("https://api.alternative.me/fng/?limit=1",
                     timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        data = r.json()["data"][0]
        val   = int(data["value"])
        label = data["value_classification"]
        if val <= 25:
            color = "#f85149"; vn = "Cực kỳ sợ hãi"
        elif val <= 45:
            color = "#ff7b54"; vn = "Sợ hãi"
        elif val <= 55:
            color = "#FFD700"; vn = "Trung tính"
        elif val <= 75:
            color = "#76c3a0"; vn = "Tham lam"
        else:
            color = "#3fb950"; vn = "Cực kỳ tham lam"
        return {"value": val, "label": vn, "label_en": label, "color": color, "ok": True}
    except Exception:
        return {"value": None, "label": "N/A", "label_en": "N/A", "color": "#8b949e", "ok": False}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ohlcv(asset_key: str) -> pd.DataFrame:
    """
    Lấy dữ liệu OHLCV (Open/High/Low/Close/Volume) từ yfinance.
    Dùng cho ADX, Stochastic, ATR thực (cần High/Low).
    Trả về DataFrame rỗng nếu không lấy được.
    """
    ticker_map = {
        "XAU":    "GC=F",
        "XAG":    "SI=F",
        "HG":     "HG=F",
        "CL":     "CL=F",
        "USDVND": "USDVND=X",
        "BTC":    "BTC-USD",
    }
    ticker = ticker_map.get(asset_key, "GC=F")
    try:
        raw = yf.download(ticker, period="2y", interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
        df = raw[cols].dropna(subset=["Close"])
        return df if len(df) >= 50 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gvz() -> dict:
    """
    Gold Volatility Index (^GVZ) — chỉ số sợ hãi / biến động của thị trường vàng.
    Tương tự VIX nhưng dành riêng cho vàng. Cao → bất ổn, cần mở rộng stop loss.
    GVZ < 12: bình tĩnh · 12–18: bình thường · 18–25: lo ngại · >25: biến động mạnh.
    """
    try:
        raw = yf.download("^GVZ", period="60d", interval="1d",
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        s = raw["Close"].dropna()
        if len(s) < 3:
            return {"value": None, "ok": False}
        val  = float(s.iloc[-1])
        prev = float(s.iloc[-6]) if len(s) >= 6 else val
        chg  = val - prev

        if   val < 12: level, color = "Thấp — Thị trường bình tĩnh", "#3fb950"
        elif val < 18: level, color = "Bình thường", "#8b949e"
        elif val < 25: level, color = "Cao — Lo ngại tăng", "#f9a825"
        else:          level, color = "Rất cao ⚠️ — Biến động mạnh", "#f85149"

        return {"value": val, "chg": chg, "level": level, "color": color, "ok": True}
    except Exception:
        return {"value": None, "level": "N/A", "color": "#8b949e", "ok": False}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_cot_data(asset_key: str) -> dict:
    """
    COT Disaggregated — dữ liệu vị thế Managed Money (quỹ đầu cơ) từ CFTC.
    Nguồn: CFTC Public Reporting API (miễn phí, cập nhật thứ Sáu hàng tuần).
    Tín hiệu: MM tăng mua ròng → Bullish · MM giảm vị thế → Bearish.
    Vị thế cực đoan (>80th / <20th percentile) → tín hiệu đảo chiều (contrarian).
    """
    import requests as _req

    # Tên thị trường trong dữ liệu CFTC
    market_kw = {
        "XAU":    "GOLD",
        "XAG":    "SILVER",
        "HG":     "COPPER",
        "CL":     "CRUDE OIL, LIGHT",
    }.get(asset_key)

    if not market_kw:
        return {"ok": False, "reason": "Không có COT cho asset này"}

    try:
        # CFTC Socrata API — disaggregated futures (Managed Money positions)
        url = (
            "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
            f"?$where=market_and_exchange_names+like+%27{market_kw.replace(' ','+')}%25%27"
            "&$limit=52&$order=report_date_as_mm_dd_yyyy+DESC"
        )
        r    = _req.get(url, timeout=12,
                        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        data = r.json()

        if not isinstance(data, list) or len(data) < 2:
            return {"ok": False, "reason": "API trả về dữ liệu rỗng"}

        records = []
        for row in data:
            try:
                date_str = str(row.get("report_date_as_mm_dd_yyyy", ""))[:10]
                mm_long  = float(row.get("m_money_positions_long_all",  0) or 0)
                mm_short = float(row.get("m_money_positions_short_all", 0) or 0)
                if mm_long == 0 and mm_short == 0:
                    continue
                records.append({"date": date_str, "long": mm_long, "short": mm_short,
                                 "net": mm_long - mm_short})
            except Exception:
                continue

        if len(records) < 2:
            return {"ok": False, "reason": "Không đủ dữ liệu COT"}

        records.sort(key=lambda x: x["date"], reverse=True)
        latest  = records[0]
        prev1   = records[1]
        prev4   = records[4]  if len(records) > 4  else records[-1]
        prev12  = records[12] if len(records) > 12 else records[-1]
        prev52  = records[51] if len(records) > 51 else records[-1]

        net_now  = latest["net"]
        chg_1w   = net_now - prev1["net"]
        chg_4w   = net_now - prev4["net"]
        chg_12w  = net_now - prev12["net"]
        chg_52w  = net_now - prev52["net"]

        # Percentile so với toàn bộ 52 tuần (1 chu kỳ đầy đủ)
        all_nets = [r["net"] for r in records]
        pct_rank = sum(1 for n in all_nets if n <= net_now) / len(all_nets) * 100

        # Scoring
        score = 0
        signals = []

        # Trend momentum ngắn hạn (1W vs 4W)
        if chg_1w > 5000 and chg_4w > 10000:
            score += 2
            signals.append(f"✅ MM tăng mua mạnh 1W +{chg_1w:,.0f} / 4W +{chg_4w:,.0f} → Bullish")
        elif chg_1w > 0:
            score += 1
            signals.append(f"✅ MM tăng mua nhẹ 1W +{chg_1w:,.0f} → Nghiêng Bullish")
        elif chg_1w < -5000 and chg_4w < -10000:
            score -= 2
            signals.append(f"🔴 MM giảm bán mạnh 1W {chg_1w:,.0f} / 4W {chg_4w:,.0f} → Bearish")
        elif chg_1w < 0:
            score -= 1
            signals.append(f"🔴 MM giảm nhẹ 1W {chg_1w:,.0f} → Nghiêng Bearish")

        # Xu hướng dài hạn 52W — whale đang build hay liquidate?
        if chg_52w > 50000:
            score += 1
            signals.append(f"✅ Xu hướng 52W: MM tăng mua +{chg_52w:,.0f} hợp đồng — Bull cycle dài hạn đang hình thành")
        elif chg_52w < -50000:
            score -= 1
            signals.append(f"🔴 Xu hướng 52W: MM giảm {abs(chg_52w):,.0f} hợp đồng — Bear cycle dài hạn, cá mập đang rút lui")

        # Contrarian extreme signals (so với 52W percentile — chính xác hơn 20W)
        if pct_rank >= 85 and chg_1w < 0:
            score -= 1
            signals.append(f"⚠️ Vị thế cực đoan LONG ({pct_rank:.0f}/100 percentile-52W) và đang giảm → Nguy cơ điều chỉnh mạnh")
        elif pct_rank <= 15 and chg_1w > 0:
            score += 1
            signals.append(f"✅ Vị thế cực đoan SHORT ({pct_rank:.0f}/100 percentile-52W) và đang đảo chiều → Short squeeze tiềm năng")

        score = max(-3, min(3, score))

        return {
            "ok":       True,
            "date":     latest["date"],
            "net":      net_now,
            "long":     latest["long"],
            "short":    latest["short"],
            "chg_1w":   chg_1w,
            "chg_4w":   chg_4w,
            "chg_12w":  chg_12w,
            "chg_52w":  chg_52w,
            "pct_rank": round(pct_rank, 1),
            "score":    score,
            "signals":  signals,
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}


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


def calc_macd(s: pd.Series, fast: int = 12, slow: int = 26,
              sig: int = 9) -> tuple:
    """Trả về (macd_line, signal_line, histogram)."""
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    signal   = macd.ewm(span=sig, adjust=False).mean()
    return macd, signal, macd - signal


def kalman_smooth(prices: np.ndarray, Q: float = 1e-4, R: float = 5e-3) -> np.ndarray:
    """
    Bộ lọc Kalman 1 chiều — khử nhiễu ngắn hạn, giữ xu hướng thật.
    Q = process noise (nhỏ → smooth hơn, phản ứng chậm hơn với đảo chiều)
    R = measurement noise (lớn → tin model nhiều hơn, smooth hơn)
    Không cần thư viện ngoài, chỉ dùng numpy.
    """
    n = len(prices)
    x = float(prices[0])
    P = 1.0
    out = np.empty(n)
    for i, z in enumerate(prices):
        P   = P + Q
        K   = P / (P + R)
        x   = x + K * (float(z) - x)
        P   = (1.0 - K) * P
        out[i] = x
    return out


def calc_adx(df: pd.DataFrame, period: int = 14) -> dict:
    """
    Wilder's ADX (Average Directional Index) — đo độ mạnh xu hướng.
    Cần DataFrame với cột High, Low, Close.
    Trả về: adx (0–100), plus_di, minus_di, trend, strong (bool).
    ADX < 20 → đi ngang, tín hiệu kỹ thuật kém tin cậy.
    ADX 20–25 → xu hướng hình thành. ADX > 25 → xu hướng mạnh.
    """
    default = {"adx": 20.0, "plus_di": 20.0, "minus_di": 20.0,
               "trend": "Không rõ xu hướng", "direction": "neutral", "strong": False}
    if df.empty or len(df) < period + 5:
        return default
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return default

    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    close = df["Close"].astype(float)

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up   = high.diff()
    down = -low.diff()
    pdm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=close.index)
    mdm  = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=close.index)

    # Wilder smoothing (EMA với alpha = 1/period)
    alpha  = 1.0 / period
    atr14  = tr.ewm(alpha=alpha, adjust=False).mean()
    pdi14  = 100.0 * pdm.ewm(alpha=alpha, adjust=False).mean() / (atr14 + 1e-9)
    mdi14  = 100.0 * mdm.ewm(alpha=alpha, adjust=False).mean() / (atr14 + 1e-9)
    dx     = 100.0 * (pdi14 - mdi14).abs() / (pdi14 + mdi14 + 1e-9)
    adx    = dx.ewm(alpha=alpha, adjust=False).mean()

    adx_v = float(adx.iloc[-1])
    pdi_v = float(pdi14.iloc[-1])
    mdi_v = float(mdi14.iloc[-1])

    if adx_v >= 40:
        strength, strong = "Xu hướng RẤT MẠNH", True
    elif adx_v >= 25:
        strength, strong = "Xu hướng MẠNH", True
    elif adx_v >= 20:
        strength, strong = "Xu hướng vừa", False
    else:
        strength, strong = "Đi ngang / Tích lũy", False

    direction = "up" if pdi_v > mdi_v else "down"
    dir_vn    = "TĂNG" if direction == "up" else "GIẢM"

    return {
        "adx": adx_v, "plus_di": pdi_v, "minus_di": mdi_v,
        "trend": f"{strength} · {dir_vn}",
        "direction": direction, "strong": strong,
    }


def calc_stoch(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> dict:
    """
    Stochastic Oscillator %K và %D.
    Cần DataFrame với High, Low, Close (hoặc chỉ Close — dùng rolling min/max).
    Trả về: k, d, signal ('oversold'/'overbought'/'buy'/'sell'/'neutral').
    """
    default = {"k": 50.0, "d": 50.0, "signal": "neutral"}
    if df.empty or len(df) < k_period + d_period:
        return default

    close = df["Close"].astype(float)
    high  = df["High"].astype(float)  if "High" in df.columns else close
    low   = df["Low"].astype(float)   if "Low"  in df.columns else close

    lo_k  = low.rolling(k_period).min()
    hi_k  = high.rolling(k_period).max()
    k_pct = 100.0 * (close - lo_k) / (hi_k - lo_k + 1e-9)
    d_pct = k_pct.rolling(d_period).mean()

    k_v, d_v = float(k_pct.iloc[-1]), float(d_pct.iloc[-1])

    if   k_v < 20 and d_v < 20:              sig = "oversold"
    elif k_v > 80 and d_v > 80:              sig = "overbought"
    elif k_v > d_v and k_v < 50:             sig = "buy"
    elif k_v < d_v and k_v > 50:             sig = "sell"
    else:                                    sig = "neutral"

    return {"k": k_v, "d": d_v, "signal": sig}


def calc_cci(df: pd.DataFrame, period: int = 20) -> dict:
    """
    Commodity Channel Index — xác định điểm đảo chiều theo chu kỳ.
    CCI > +100 = quá mua / CCI < -100 = quá bán.
    Cần High, Low, Close.
    """
    if df.empty or len(df) < period + 1 or not {"High", "Low", "Close"}.issubset(df.columns):
        return {"cci": 0.0, "signal": "neutral", "ok": False}

    tp  = (df["High"].astype(float) + df["Low"].astype(float) + df["Close"].astype(float)) / 3
    ma  = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - ma) / (0.015 * mad + 1e-9)
    v   = float(cci.iloc[-1])

    if   v >  200: sig = "extreme_overbought"
    elif v >  100: sig = "overbought"
    elif v < -200: sig = "extreme_oversold"
    elif v < -100: sig = "oversold"
    else:          sig = "neutral"

    return {"cci": round(v, 1), "signal": sig, "ok": True}


def calc_obv(df: pd.DataFrame) -> dict:
    """
    On Balance Volume — theo dõi dòng tiền tổ chức qua volume.
    OBV tăng trong khi giá đi ngang / giảm = tích lũy ngầm → tín hiệu MUA.
    OBV giảm trong khi giá đi ngang / tăng = phân phối ngầm → tín hiệu BÁN.
    """
    if df.empty or "Volume" not in df.columns:
        return {"signal": "neutral", "divergence": None, "slope_pct": 0.0, "ok": False}

    close  = df["Close"].astype(float)
    volume = df["Volume"].astype(float).replace(0, np.nan).ffill()

    if volume.isna().all() or float(volume.sum()) < 1:
        return {"signal": "neutral", "divergence": None, "slope_pct": 0.0, "ok": False}

    direction = np.sign(close.diff().fillna(0))
    obv       = (volume * direction).cumsum()
    obv_ma20  = obv.rolling(20).mean()

    obv_v     = float(obv.iloc[-1])
    obv_ma_v  = float(obv_ma20.iloc[-1])

    # Slope % — OBV thay đổi bao nhiêu trong 10 ngày qua
    n = min(10, len(obv) - 1)
    denom     = abs(float(obv.iloc[-n - 1])) + 1e-9
    obv_slope = (float(obv.iloc[-1]) - float(obv.iloc[-n - 1])) / denom * 100

    # Price slope — giá thay đổi bao nhiêu trong 10 ngày qua
    p_slope   = (float(close.iloc[-1]) / float(close.iloc[-n - 1]) - 1) * 100

    # Divergence detection
    divergence = None
    if   p_slope >  2.0 and obv_slope < -1.0: divergence = "bearish"   # Giá tăng, OBV giảm → fake rally
    elif p_slope < -2.0 and obv_slope >  1.0: divergence = "bullish"   # Giá giảm, OBV tăng → tích lũy

    # Trend signal
    if   obv_v > obv_ma_v and obv_slope > 0: signal = "bullish"
    elif obv_v < obv_ma_v and obv_slope < 0: signal = "bearish"
    else:                                    signal = "neutral"

    # Score contribution
    score = 0
    if signal   == "bullish":   score += 1
    elif signal == "bearish":   score -= 1
    if divergence == "bullish": score += 1
    elif divergence == "bearish": score -= 1

    return {
        "signal": signal, "divergence": divergence,
        "slope_pct": round(obv_slope, 1),
        "p_slope": round(p_slope, 1),
        "score": score, "ok": True,
    }


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
             days: int, macro_score: int = 0, asset_key: str = "XAU",
             ml_prob: float = 0.5) -> tuple:
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

    # ── Model C: Log-return momentum (damped, ngắn/dài hạn khác nhau) ───
    log_p = np.log(train.values)
    if days <= 14:
        # Ngắn hạn: dùng 10 ngày gần, damping nhẹ hơn để bám sát trend
        w, damp = min(10, len(log_p)), 0.75
    else:
        # Dài hạn: 60 ngày, damping mạnh tránh extrapolate quá mức
        w, damp = min(60, len(log_p)), 0.45
    slope   = LinearRegression().fit(
        np.arange(w).reshape(-1, 1), log_p[-w:]
    ).coef_[0] * damp
    lr_vals = np.exp(log_p[-1] + slope * np.arange(1, days + 1))

    # ── Ensemble ──────────────────────────────────────────────────────────
    if days <= 14:
        # Ngắn hạn: nghiêng về momentum (kỹ thuật), ít HW hơn
        if hw_vals is not None and arima_vals is not None:
            base = 0.20 * hw_vals + 0.35 * arima_vals + 0.45 * lr_vals
        elif hw_vals is not None:
            base = 0.30 * hw_vals + 0.70 * lr_vals
        elif arima_vals is not None:
            base = 0.45 * arima_vals + 0.55 * lr_vals
        else:
            base = lr_vals
    else:
        # Dài hạn: giữ nguyên trọng số gốc
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

    # ── Kalman-smoothed price (giảm nhiễu trước khi tính momentum) ──────
    kal_prices = pd.Series(kalman_smooth(train.values), index=train.index)

    # ── Momentum adjustment ───────────────────────────────────────────────
    if days <= 14:
        n_mom      = min(5, len(train))    # Momentum 5 ngày cho giao dịch ngắn hạn
        mom_weight = 0.08                  # Tác động thấp — kỹ thuật chiếm chủ đạo
        mom_cap    = 0.03                  # Cap ±3% cho ngắn hạn
    else:
        n_mom      = min(66, len(train))
        mom_weight = 0.12 * (30 / max(days, 30)) ** 0.5
        mom_cap    = 0.08
    # Dùng Kalman-smoothed prices để momentum ít bị nhiễu hơn
    ret_mom = float(kal_prices.iloc[-1]) / float(kal_prices.iloc[-n_mom]) - 1
    mom_adj = np.clip(ret_mom * mom_weight, -mom_cap, mom_cap)

    # ── Mean reversion từ MA200 ───────────────────────────────────────────
    if len(price) >= 200:
        ma200v   = float(price.rolling(200).mean().iloc[-1])
        cur_p    = float(price.iloc[-1])
        dev      = cur_p / ma200v - 1          # +0.20 = đang cao hơn MA200 20%
        # Áp lực quay về trung bình: tỷ lệ với độ lệch và thời gian
        rev_adj  = np.clip(-dev * 0.10 * (days / 365), -0.06, 0.06)
    else:
        rev_adj  = 0.0

    # ── ML directional bias ───────────────────────────────────────────────
    # (ml_prob - 0.5) → [-0.5, +0.5] → scale → max ±5% adjustment
    # Giảm ảnh hưởng cho dài hạn: ML chủ yếu đáng tin ngắn–trung hạn
    ml_scale = 0.10 * min(1.0, 30 / max(days, 7))   # 10% at 7d, 3% at 90d, ...
    ml_adj   = np.clip((ml_prob - 0.5) * ml_scale, -0.05, 0.05)

    # ── Apply adjustments ─────────────────────────────────────────────────
    combined = base * (1 + macro_adj + seas_adj + mom_adj + rev_adj + ml_adj)

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
                   fc_mean, macro_score: int = 0,
                   ml_result: dict = None) -> tuple:
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

    # ── ML Directional Model ──────────────────────────────────────────────
    if ml_result and ml_result.get("ok"):
        ml_prob = ml_result["prob_up"]
        ml_conf = ml_result["confidence"]
        ml_sig  = ml_result["signal"]
        ml_sc   = 0
        if   ml_sig == "strong_buy":  ml_sc = +2
        elif ml_sig == "buy":         ml_sc = +1
        elif ml_sig == "strong_sell": ml_sc = -2
        elif ml_sig == "sell":        ml_sc = -1
        score += ml_sc
        prob_pct = ml_prob * 100
        conf_pct = ml_conf * 100
        if ml_sc > 0:
            notes.append(f"🤖 ML Model: {prob_pct:.0f}% xác suất TĂNG (tin cậy {conf_pct:.0f}%) — tín hiệu MUA")
        elif ml_sc < 0:
            notes.append(f"🤖 ML Model: {100-prob_pct:.0f}% xác suất GIẢM (tin cậy {conf_pct:.0f}%) — tín hiệu BÁN")
        else:
            notes.append(f"🤖 ML Model: {prob_pct:.0f}% xác suất tăng — Trung tính")

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

    # ── 1b. IAU cross-confirm (XAU only) ─────────────────────────────────────
    if asset_key == "XAU":
        try:
            raw_iau = yf.download("IAU", period="3mo", interval="1d",
                                  progress=False, auto_adjust=True)
            if not raw_iau.empty:
                if isinstance(raw_iau.columns, pd.MultiIndex):
                    raw_iau.columns = raw_iau.columns.get_level_values(0)
                p_iau = raw_iau["Close"].dropna()
                if len(p_iau) >= 20:
                    p20_iau = float(p_iau.rolling(20).mean().iloc[-1])
                    tail60_iau = p_iau.tail(60)
                    p60_iau = (float(tail60_iau.rolling(20).mean().dropna().iloc[0])
                               if len(tail60_iau) >= 20 else float(p_iau.iloc[0]))
                    iau_chg = (p20_iau / p60_iau - 1) * 100 if p60_iau > 0 else 0
                    result["etf_iau"] = {"aum_chg": iau_chg}
                    etf = result.get("etf_flow")
                    if etf and abs(etf["aum_chg"]) > 2 and abs(iau_chg) > 2:
                        avg_chg = (etf["aum_chg"] + iau_chg) / 2
                        if etf["aum_chg"] > 0 and iau_chg > 0:
                            score += 1
                            signals.append(("🏦", f"XÁC NHẬN ĐA ETF: GLD+IAU đều tăng (~{avg_chg:.1f}%) → dòng tiền tổ chức vào vàng đồng loạt", "green"))
                        elif etf["aum_chg"] < 0 and iau_chg < 0:
                            score -= 1
                            signals.append(("🏦", f"XÁC NHẬN ĐA ETF: GLD+IAU đều giảm (~{avg_chg:.1f}%) → outflow đồng loạt", "red"))
        except Exception:
            pass

    # ── 2. Futures Open Interest ─────────────────────────────────────────────
    OI_THRESHOLDS = {
        "XAU": (400000, 600000),
        "XAG": (100000, 150000),
        "HG":  (150000, 250000),
        "CL":  (1200000, 1800000),
    }
    if fut_ticker:
        try:
            tk  = yf.Ticker(fut_ticker)
            inf = tk.info
            oi  = inf.get("openInterest")
            if oi and int(oi) > 0:
                oi_val = int(oi)
                result["oi"] = {"value": oi_val, "ticker": fut_ticker}
                oi_mid, oi_high = OI_THRESHOLDS.get(asset_key, (200000, 350000))
                if oi_val > oi_high:
                    signals.append(("📌", f"OI {fut_ticker}: {oi_val:,} — CỰC CAO (>{oi_high:,}) → thị trường rất crowded, nguy cơ squeeze 2 chiều", "orange"))
                elif oi_val > oi_mid:
                    signals.append(("📌", f"OI {fut_ticker}: {oi_val:,} — Cao → vị thế lớn, theo dõi breakout", "gray"))
                else:
                    signals.append(("📌", f"OI {fut_ticker}: {oi_val:,} — Bình thường", "gray"))
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
#  SENTIMENT INTELLIGENCE LAYER  (News · Options · VIX Term · Credit Stress)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_sentiment(asset_key: str) -> dict:
    """
    Sentiment tin tức từ Google News RSS — miễn phí, không cần API key.
    Phân tích tích cực/tiêu cực trong 20 tiêu đề bài báo mới nhất.
    Score: -3 (rất tiêu cực) → +3 (rất tích cực).
    """
    import urllib.request, urllib.parse
    import xml.etree.ElementTree as _ET

    QUERIES = {
        "XAU":    "gold price market",
        "XAG":    "silver price market",
        "HG":     "copper price market",
        "CL":     "crude oil price market",
        "BTC":    "bitcoin price crypto",
        "USDVND": "USD VND dollar exchange rate",
    }
    # Từ tích cực và tiêu cực — được chọn lọc cho tài sản hàng hoá/vàng
    POS = {
        "surge","rally","rise","rises","rising","soar","soars","soaring","climb",
        "climbs","gains","gain","bullish","record","high","safe haven","demand",
        "strong","boost","upside","breakout","accumulate","inflow","inflows",
        "rally","rebound","optimism","buy","buying","inflation hedge",
        "central bank","rate cut","pivot","weaker dollar","geopolitical",
        "uncertainty","war","conflict","crisis",
    }
    NEG = {
        "fall","falls","falling","drop","drops","plunge","plunges","decline",
        "declines","bearish","sell","crash","pressure","weakness","outflow",
        "outflows","selloff","hawkish","tightening","rate hike","rate hikes",
        "dollar strength","strong dollar","risk on","equities rally",
        "crypto winter","regulation",
    }

    query = QUERIES.get(asset_key, "gold price")
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            xml_bytes = r.read()

        root = _ET.fromstring(xml_bytes)
        titles = [
            el.text.lower()
            for el in root.findall(".//item/title")
            if el.text
        ][:20]

        if not titles:
            return {"ok": False, "score": 0, "label": "N/A", "color": "#8b949e", "n": 0}

        pos_hits = neg_hits = 0
        pos_words_seen: set = set()
        neg_words_seen: set = set()

        for title in titles:
            for w in POS:
                if w in title:
                    pos_hits += 1
                    pos_words_seen.add(w)
                    break
            for w in NEG:
                if w in title:
                    neg_hits += 1
                    neg_words_seen.add(w)
                    break

        net   = pos_hits - neg_hits
        total = pos_hits + neg_hits
        ratio = net / total if total > 0 else 0.0
        score = max(-3, min(3, round(ratio * 3)))

        # Confidence weight: cần ít nhất 5 bài có tín hiệu để score đáng tin
        confidence = min(1.0, total / 8)

        if score >= 2:   label, color = "Rất tích cực 📈", "#3fb950"
        elif score == 1: label, color = "Hơi tích cực",    "#76c3a0"
        elif score == -1:label, color = "Hơi tiêu cực",    "#ff7b54"
        elif score <= -2:label, color = "Rất tiêu cực 📉", "#f85149"
        else:            label, color = "Trung tính",       "#FFD700"

        return {
            "ok": True, "score": score, "label": label, "color": color,
            "pos": pos_hits, "neg": neg_hits, "n": len(titles),
            "confidence": round(confidence, 2),
            "pos_words": sorted(pos_words_seen)[:6],
            "neg_words": sorted(neg_words_seen)[:6],
            "query": query,
        }
    except Exception:
        return {"ok": False, "score": 0, "label": "N/A", "color": "#8b949e", "n": 0}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_options_pcr(asset_key: str) -> dict:
    """
    Put/Call Ratio từ thị trường quyền chọn ETF proxy — chỉ báo sentiment tổ chức.
    PCR cao (nhiều put) = bi quan = tín hiệu MUA ngược chiều (contrarian).
    PCR thấp (nhiều call) = lạc quan quá mức = tín hiệu BÁN ngược chiều.
    """
    ETF_MAP = {
        "XAU": "GLD",   # SPDR Gold Shares — ETF vàng lớn nhất
        "XAG": "SLV",   # iShares Silver Trust
        "HG":  "CPER",  # United States Copper Index Fund
        "CL":  "USO",   # United States Oil Fund
        "BTC": "GBTC",  # Grayscale Bitcoin Trust
    }
    etf = ETF_MAP.get(asset_key)
    if not etf:
        return {"ok": False, "score": 0, "label": "N/A", "color": "#8b949e"}

    try:
        ticker = yf.Ticker(etf)
        exp_dates = ticker.options
        if not exp_dates:
            return {"ok": False, "score": 0, "label": "N/A", "color": "#8b949e"}

        # Lấy expiry gần nhất (trong 2-5 tuần)
        exp_date = exp_dates[min(1, len(exp_dates) - 1)]
        chain    = ticker.option_chain(exp_date)
        calls    = chain.calls
        puts     = chain.puts

        call_vol = float(calls["volume"].fillna(0).sum())
        put_vol  = float(puts["volume"].fillna(0).sum())
        call_oi  = float(calls["openInterest"].fillna(0).sum())
        put_oi   = float(puts["openInterest"].fillna(0).sum())

        if call_vol < 50 and call_oi < 100:
            return {"ok": False, "score": 0, "label": "Không đủ dữ liệu", "color": "#8b949e"}

        pcr_vol = put_vol / call_vol if call_vol > 0 else 1.0
        pcr_oi  = put_oi  / call_oi  if call_oi  > 0 else pcr_vol
        # Trung bình có trọng số: OI đáng tin hơn vol trong ngắn hạn
        avg_pcr = pcr_vol * 0.4 + pcr_oi * 0.6

        # Scoring contrarian:
        # PCR > 1.5 = thị trường cực kỳ bi quan → contrarian BUY mạnh
        # PCR > 1.2 = bi quan → nhẹ ủng hộ mua
        # PCR < 0.7 = lạc quan thái quá → contrarian SELL mạnh
        # PCR < 0.9 = hơi lạc quan → nhẹ ủng hộ bán
        if avg_pcr > 1.5:
            score, label, color = +2, f"PCR={avg_pcr:.2f} — Bi quan cực độ → MUA ngược chiều", "#3fb950"
        elif avg_pcr > 1.2:
            score, label, color = +1, f"PCR={avg_pcr:.2f} — Bi quan → Hỗ trợ tích lũy",        "#76c3a0"
        elif avg_pcr < 0.7:
            score, label, color = -2, f"PCR={avg_pcr:.2f} — Lạc quan cực độ → BÁN ngược chiều","#f85149"
        elif avg_pcr < 0.9:
            score, label, color = -1, f"PCR={avg_pcr:.2f} — Lạc quan → Cảnh báo đảo chiều",    "#ff7b54"
        else:
            score, label, color =  0, f"PCR={avg_pcr:.2f} — Trung tính",                        "#FFD700"

        return {
            "ok": True, "score": score, "label": label, "color": color,
            "pcr_vol": round(pcr_vol, 3), "pcr_oi": round(pcr_oi, 3),
            "avg_pcr": round(avg_pcr, 3), "etf": etf, "expiry": exp_date,
            "call_vol": int(call_vol), "put_vol": int(put_vol),
        }
    except Exception:
        return {"ok": False, "score": 0, "label": "N/A", "color": "#8b949e"}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_vix_term() -> dict:
    """
    VIX Term Structure: VIX spot (^VIX) vs VIX 3-tháng (^VIX3M).
    Backwardation (VIX > VIX3M): sợ hãi cấp tính → vàng là safe haven → Bullish.
    Contango (VIX < VIX3M): thị trường bình tĩnh → vàng ít được cầu → Bearish nhẹ.
    """
    try:
        raw = yf.download(["^VIX", "^VIX3M"], period="5d", interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return {"ok": False, "score": 0}

        # Handle multi-level columns
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["^VIX", "^VIX3M"]]

        vix_s   = close["^VIX"].dropna()
        vix3m_s = close["^VIX3M"].dropna()

        if vix_s.empty or vix3m_s.empty:
            return {"ok": False, "score": 0}

        vix_val  = float(vix_s.iloc[-1])
        vix3m_val = float(vix3m_s.iloc[-1])
        ratio = vix_val / vix3m_val

        if ratio > 1.15:
            score, label, color = +2, "Backwardation mạnh — Sợ hãi cực độ → Vàng safe haven BUY", "#3fb950"
        elif ratio > 1.03:
            score, label, color = +1, "Backwardation — Căng thẳng thị trường → Hỗ trợ vàng",      "#76c3a0"
        elif ratio < 0.85:
            score, label, color = -1, "Contango sâu — Lạc quan cao → Ít cầu vàng phòng thủ",      "#ff7b54"
        elif ratio < 0.95:
            score, label, color =  0, "Contango nhẹ — Thị trường bình tĩnh — Trung tính",          "#FFD700"
        else:
            score, label, color =  0, "Flat — Trung tính",                                          "#FFD700"

        # Thêm điểm nếu VIX tuyệt đối cao (>30 = khủng hoảng → vàng tăng mạnh)
        if vix_val > 35:
            score = min(3, score + 1)
            label += " · VIX >35 = KHỦNG HOẢNG"
        elif vix_val > 25:
            score = min(3, score + 1)
            label += f" · VIX={vix_val:.0f} cao"

        return {
            "ok": True, "score": score, "label": label, "color": color,
            "vix": round(vix_val, 2), "vix3m": round(vix3m_val, 2),
            "ratio": round(ratio, 3),
        }
    except Exception:
        return {"ok": False, "score": 0, "label": "N/A", "color": "#8b949e"}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_credit_stress(fred_data: dict) -> dict:
    """
    Rủi ro tín dụng từ ICE BofA US High Yield OAS (BAMLH0A0HYM2) — FRED.
    Spread cao = căng thẳng tín dụng = vàng safe haven tăng.
    Spread thấp = bình thường / lạc quan = neutral/nhẹ bearish vàng.
    """
    try:
        series = fred_data.get("credit_oas")
        if series is None or len(series) < 5:
            return {"ok": False, "score": 0}

        val  = float(series.iloc[-1])
        prev = float(series.iloc[-5])
        chg  = val - prev   # 1-tuần thay đổi

        # Scoring:
        # Spread > 6%: khủng hoảng tín dụng → vàng safe haven mạnh (+2)
        # Spread > 4.5%: căng thẳng cao (+1)
        # Spread < 3%: bình thường, thị trường lạc quan (-1)
        # Spread < 2.5%: cực kỳ bình tĩnh (-2, complacency)
        if val > 6.0:
            score, label, color = +2, f"OAS {val:.2f}% — Khủng hoảng tín dụng → Vàng safe haven", "#3fb950"
        elif val > 4.5:
            score, label, color = +1, f"OAS {val:.2f}% — Căng thẳng tín dụng → Hỗ trợ vàng",     "#76c3a0"
        elif val < 2.5:
            score, label, color = -2, f"OAS {val:.2f}% — Bình tĩnh tuyệt đối → Ít cầu safe haven","#f85149"
        elif val < 3.2:
            score, label, color = -1, f"OAS {val:.2f}% — Bình thường → Trung tính / nhẹ bearish", "#ff7b54"
        else:
            score, label, color =  0, f"OAS {val:.2f}% — Bình thường trung bình",                  "#FFD700"

        # Nếu spread đang nới rộng nhanh → tín hiệu mạnh hơn
        if chg > 0.5:
            score = min(3, score + 1)
            label += f" · Nới rộng mạnh +{chg:.2f}% (1W) ⚠️"

        return {
            "ok": True, "score": score, "label": label, "color": color,
            "oas": round(val, 2), "chg_1w": round(chg, 2),
        }
    except Exception:
        return {"ok": False, "score": 0, "label": "N/A", "color": "#8b949e"}


# ══════════════════════════════════════════════════════════════════════════════
#  HEDGE FUND FRAMEWORK — Macro Regime + Cross-Asset Alignment
# ══════════════════════════════════════════════════════════════════════════════

def detect_macro_regime(fred_data: dict, macro: dict) -> dict:
    """
    Phân loại môi trường kinh tế theo khung Bridgewater All Weather:
    2 chiều độc lập — GROWTH (tăng trưởng) × INFLATION (lạm phát).

    5 chế độ:
    ┌────────────────┬─────────────────┬────────────────────┐
    │                │  Lạm phát cao   │  Lạm phát thấp     │
    ├────────────────┼─────────────────┼────────────────────┤
    │ Tăng trưởng ↑  │ TĂNG TRƯỞNG NÓNG│ GOLDILOCKS         │
    │ Tăng trưởng ↓  │ ĐÌNH LẠM 🔥     │ RỦI RO GIẢM PHÁT  │
    │ Khủng hoảng    │      CRISIS 🚨  │                    │
    └────────────────┴─────────────────┴────────────────────┘
    Vàng mạnh nhất: ĐÌNH LẠM > CRISIS > TĂNG TRƯỞNG NÓNG
    """
    g_pts = 0   # growth points
    i_pts = 0   # inflation points
    details: list[str] = []
    vix_val = None

    # ── GROWTH INDICATORS ──────────────────────────────────────────────────
    # 1. S&P 500 momentum 3 tháng
    if "sp500" in macro and len(macro["sp500"]) >= 63:
        sp_chg = (float(macro["sp500"].iloc[-1]) / float(macro["sp500"].iloc[-63]) - 1) * 100
        if   sp_chg > 12: g_pts += 2; details.append(f"S&P 3M +{sp_chg:.1f}% → Tăng trưởng rất mạnh")
        elif sp_chg > 4:  g_pts += 1; details.append(f"S&P 3M +{sp_chg:.1f}% → Tăng trưởng ổn định")
        elif sp_chg < -12:g_pts -= 2; details.append(f"S&P 3M {sp_chg:.1f}% → Suy giảm nghiêm trọng ⚠️")
        elif sp_chg < -4: g_pts -= 1; details.append(f"S&P 3M {sp_chg:.1f}% → Tăng trưởng đang yếu")

    # 2. Yield Curve (10Y-2Y)
    for k in ("curve", "curve_3m"):
        if k in fred_data and len(fred_data[k]) > 0:
            cv = float(fred_data[k].iloc[-1])
            if   cv >  0.75: g_pts += 1; details.append(f"Yield Curve +{cv:.2f}% → Tăng trưởng dài hạn kỳ vọng")
            elif cv < -0.75: g_pts -= 2; details.append(f"Yield Curve {cv:.2f}% → Đảo ngược sâu = Cảnh báo suy thoái")
            elif cv < -0.25: g_pts -= 1; details.append(f"Yield Curve {cv:.2f}% → Đảo ngược nhẹ")
            break

    # 3. High Yield Credit Spread
    if "credit_oas" in fred_data and len(fred_data["credit_oas"]) > 0:
        oas = float(fred_data["credit_oas"].iloc[-1])
        if   oas < 3.2: g_pts += 1; details.append(f"HY OAS {oas:.2f}% → Tín dụng bình thường, doanh nghiệp khoẻ")
        elif oas > 6.5: g_pts -= 2; details.append(f"HY OAS {oas:.2f}% → KHỦNG HOẢNG tín dụng ⚠️")
        elif oas > 4.5: g_pts -= 1; details.append(f"HY OAS {oas:.2f}% → Căng thẳng tín dụng đáng kể")

    # 4. VIX (fear gauge)
    if "vix" in macro and len(macro["vix"]) > 0:
        vix_val = float(macro["vix"].iloc[-1])
        if   vix_val < 14: g_pts += 1; details.append(f"VIX {vix_val:.0f} → Thị trường cực kỳ bình tĩnh")
        elif vix_val > 40: g_pts -= 2; details.append(f"VIX {vix_val:.0f} → KHỦNG HOẢNG hệ thống ⚠️")
        elif vix_val > 28: g_pts -= 1; details.append(f"VIX {vix_val:.0f} → Sợ hãi cao")

    # ── INFLATION INDICATORS ───────────────────────────────────────────────
    # 5. 5Y Breakeven Inflation (thị trường kỳ vọng)
    if "breakeven5y" in fred_data and len(fred_data["breakeven5y"]) > 0:
        bei = float(fred_data["breakeven5y"].iloc[-1])
        if   bei > 3.0:  i_pts += 2; details.append(f"5Y Breakeven {bei:.2f}% → Kỳ vọng lạm phát rất cao")
        elif bei > 2.5:  i_pts += 1; details.append(f"5Y Breakeven {bei:.2f}% → Lạm phát trên target 2%")
        elif bei < 1.8:  i_pts -= 1; details.append(f"5Y Breakeven {bei:.2f}% → Nguy cơ giảm phát")

    # 6. 5Y5Y Forward Inflation Expectation
    if "inflation5y" in fred_data and len(fred_data["inflation5y"]) > 0:
        t5 = float(fred_data["inflation5y"].iloc[-1])
        if   t5 > 2.75:  i_pts += 1; details.append(f"T5YIFR {t5:.2f}% → Lạm phát dài hạn mất neo")
        elif t5 < 2.0:   i_pts -= 1; details.append(f"T5YIFR {t5:.2f}% → Kỳ vọng lạm phát được neo")

    # 7. Dầu WTI momentum (supply-side inflation proxy)
    if "oil" in macro and len(macro["oil"]) >= 63:
        oil_chg = (float(macro["oil"].iloc[-1]) / float(macro["oil"].iloc[-63]) - 1) * 100
        if   oil_chg > 30:  i_pts += 2; details.append(f"Dầu WTI 3M +{oil_chg:.1f}% → Sốc lạm phát chi phí đẩy")
        elif oil_chg > 12:  i_pts += 1; details.append(f"Dầu WTI 3M +{oil_chg:.1f}% → Áp lực lạm phát rõ ràng")
        elif oil_chg < -20: i_pts -= 1; details.append(f"Dầu WTI 3M {oil_chg:.1f}% → Giảm áp lực lạm phát")

    g = max(-4, min(4, g_pts))
    i = max(-4, min(4, i_pts))

    # ── Phân loại chế độ ─────────────────────────────────────────────────
    is_crisis = (vix_val is not None and vix_val > 42) or (g <= -3)

    if is_crisis:
        regime, name, bias  = "crisis",           "KHỦNG HOẢNG",         +3
        emoji, color        = "🚨", "#f85149"
        gold_outlook        = "Rất thuận lợi — Vàng là tài sản trú ẩn số 1 trong khủng hoảng"
        desc = "VIX cực cao + tín dụng bùng nổ + thị trường sụp. Lịch sử: vàng tăng mạnh nhất giai đoạn này."
    elif g <= 0 and i > 0:
        regime, name, bias  = "stagflation",      "ĐÌNH LẠM",            +3
        emoji, color        = "🔥", "#3fb950"
        gold_outlook        = "Rất thuận lợi — Môi trường TỐT NHẤT cho vàng trong lịch sử"
        desc = "Tăng trưởng yếu + Lạm phát cao (1970s style, 2022). Cổ phiếu + trái phiếu đều bị áp lực. Vàng là kênh trú ẩn lý tưởng nhất."
    elif g > 0 and i > 0:
        regime, name, bias  = "inflationary_boom","TĂNG TRƯỞNG NÓNG",    +1
        emoji, color        = "📈", "#76c3a0"
        gold_outlook        = "Thuận lợi — Vàng được hỗ trợ qua kênh lạm phát, cạnh tranh với cổ phiếu"
        desc = "Kinh tế mạnh + Lạm phát cao. Vàng hoạt động tốt như hedge lạm phát nhưng cổ phiếu cũng tăng mạnh. Phân bổ hợp lý giữa hai kênh."
    elif g > 0 and i <= 0:
        regime, name, bias  = "goldilocks",       "GOLDILOCKS",          -2
        emoji, color        = "🌟", "#ff7b54"
        gold_outlook        = "Bất lợi — Kinh tế tốt, lạm phát thấp = môi trường xấu nhất cho vàng"
        desc = "Kinh tế phát triển ổn định + Lạm phát thấp. USD mạnh, lãi suất thực cao. Nhà đầu tư ưu tiên cổ phiếu. Vàng thiếu catalyst."
    else:
        regime, name, bias  = "deflation_risk",   "RỦI RO GIẢM PHÁT",   +0
        emoji, color        = "❄️", "#8b949e"
        gold_outlook        = "Trung tính — Phức tạp: ngắn hạn bất lợi, dài hạn kích thích"
        desc = "Tăng trưởng yếu + Lạm phát thấp. Ngắn hạn: DXY mạnh, commodities yếu. Dài hạn: chính phủ/Fed sẽ tung gói kích thích → bullish vàng sau đó."

    return {
        "regime":         regime, "name_vn": name, "gold_bias": bias,
        "growth_score":   g,      "inflation_score": i,
        "gold_outlook":   gold_outlook, "desc_vn": desc,
        "emoji":          emoji,  "color": color, "details": details,
    }


def calc_cross_asset_alignment(macro: dict, fred_data: dict) -> dict:
    """
    Đo lường sự đồng thuận của 7 yếu tố cross-asset theo chiều tăng của vàng.
    Hedge fund dùng loại tín hiệu này để lọc false positives từ từng chỉ báo đơn lẻ.

    Nguyên tắc: Khi nhiều yếu tố độc lập cùng nói một điều → độ tin cậy tăng
    Khi phân kỳ → thận trọng, giảm size vị thế.
    """
    bulls: list[str] = []
    bears: list[str] = []
    neutrals: list[str] = []

    def _chg3m(series: pd.Series) -> float | None:
        if len(series) >= 63:
            return (float(series.iloc[-1]) / float(series.iloc[-63]) - 1) * 100
        return None

    def _chg1m(series: pd.Series) -> float | None:
        if len(series) >= 22:
            return (float(series.iloc[-1]) / float(series.iloc[-22]) - 1) * 100
        return None

    # 1. DXY — USD yếu → vàng tăng (tương quan âm ~-0.80)
    if "dxy" in macro and len(macro["dxy"]) > 0:
        chg = _chg3m(macro["dxy"])
        if chg is not None:
            if   chg < -3:  bulls.append(f"💵 DXY {chg:.1f}% (3M) — USD yếu → Vàng được định giá lại cao hơn")
            elif chg > +3:  bears.append(f"💵 DXY +{chg:.1f}% (3M) — USD mạnh → Áp lực giảm vàng")
            else:           neutrals.append(f"💵 DXY {chg:+.1f}% (3M) — USD trung tính")

    # 2. 10Y Treasury Yield — giảm → chi phí cơ hội vàng giảm
    if "yield10y" in macro and len(macro["yield10y"]) > 0:
        if len(macro["yield10y"]) >= 63:
            d = float(macro["yield10y"].iloc[-1]) - float(macro["yield10y"].iloc[-63])
            if   d < -0.4:  bulls.append(f"📉 10Y Yield {d:+.2f}% (3M) — Lãi suất giảm → Chi phí cơ hội vàng giảm")
            elif d > +0.4:  bears.append(f"📈 10Y Yield {d:+.2f}% (3M) — Lãi suất tăng → Áp lực vàng")
            else:           neutrals.append(f"📊 10Y Yield {d:+.2f}% (3M) — Lãi suất trung tính")

    # 3. S&P 500 — giảm → risk-off → cầu safe haven tăng
    if "sp500" in macro and len(macro["sp500"]) > 0:
        chg = _chg3m(macro["sp500"])
        if chg is not None:
            if   chg < -8:  bulls.append(f"📉 S&P {chg:.1f}% (3M) — Risk-off mạnh → Dòng tiền vào vàng")
            elif chg < -3:  bulls.append(f"📉 S&P {chg:.1f}% (3M) — Risk-off → Hỗ trợ safe haven")
            elif chg > +12: bears.append(f"📈 S&P +{chg:.1f}% (3M) — Risk-on mạnh → Ưu tiên cổ phiếu")
            else:           neutrals.append(f"📊 S&P {chg:+.1f}% (3M) — Cổ phiếu trung tính")

    # 4. VIX — cao → sợ hãi → cầu safe haven
    if "vix" in macro and len(macro["vix"]) > 0:
        vix = float(macro["vix"].iloc[-1])
        if   vix > 28:  bulls.append(f"😨 VIX {vix:.0f} — Sợ hãi cao → Cầu vàng safe haven")
        elif vix > 20:  bulls.append(f"😰 VIX {vix:.0f} — Căng thẳng — Hỗ trợ nhẹ")
        elif vix < 14:  bears.append(f"😊 VIX {vix:.0f} — Thị trường cực kỳ bình tĩnh → Ít cầu vàng")
        else:           neutrals.append(f"😐 VIX {vix:.0f} — Trung tính")

    # 5. Lãi suất thực DFII10 — âm là driver số 1 của vàng
    if "real_rate" in fred_data and len(fred_data["real_rate"]) > 0:
        rr = float(fred_data["real_rate"].iloc[-1])
        if   rr < 0:    bulls.append(f"🔑 Lãi suất thực {rr:.2f}% — ÂM = Không có chi phí cơ hội giữ vàng")
        elif rr < 0.5:  bulls.append(f"🔑 Lãi suất thực {rr:.2f}% — Thấp → Thuận lợi cho vàng")
        elif rr > 2.0:  bears.append(f"🔑 Lãi suất thực {rr:.2f}% — Cao → Chi phí cơ hội lớn")
        else:           neutrals.append(f"🔑 Lãi suất thực {rr:.2f}% — Trung tính")

    # 6. Dầu WTI — tăng → lạm phát → vàng tăng theo
    if "oil" in macro and len(macro["oil"]) > 0:
        chg = _chg1m(macro["oil"])
        if chg is not None:
            if   chg > 12:  bulls.append(f"🛢️ Dầu +{chg:.1f}% (1M) — Lạm phát chi phí đẩy → Hỗ trợ vàng")
            elif chg < -15: bears.append(f"🛢️ Dầu {chg:.1f}% (1M) — Giảm phát + risk-off → Áp lực vàng")
            else:           neutrals.append(f"🛢️ Dầu {chg:+.1f}% (1M) — Trung tính")

    # 7. HY Credit OAS — nới rộng → căng thẳng → safe haven
    if "credit_oas" in fred_data and len(fred_data["credit_oas"]) >= 5:
        oas = float(fred_data["credit_oas"].iloc[-1])
        oas_chg = oas - float(fred_data["credit_oas"].iloc[-5])
        if   oas > 4.5 or oas_chg > 0.5:
            bulls.append(f"🏦 HY OAS {oas:.2f}% (Δ{oas_chg:+.2f}W) — Credit stress → Safe haven demand")
        elif oas < 3.0 and oas_chg < 0:
            bears.append(f"🏦 HY OAS {oas:.2f}% — Thu hẹp → Risk-on, ít cầu safe haven")
        else:
            neutrals.append(f"🏦 HY OAS {oas:.2f}% — Trung tính")

    b = len(bulls); br = len(bears); n = len(neutrals)
    total_directional = b + br

    if total_directional == 0:
        score, alignment_pct = 0, 50
    else:
        net   = b - br
        score = max(-3, min(3, round(net / max(total_directional, 1) * 3)))
        alignment_pct = round(b / total_directional * 100)

    if b >= 6:   label, color = f"🟢 SIÊU HỘI TỤ — {b}/7 yếu tố đồng thuận BULLISH vàng", "#3fb950"
    elif b >= 5: label, color = f"🟢 Hội tụ mạnh — {b}/7 yếu tố ủng hộ vàng",              "#3fb950"
    elif b >= 4: label, color = f"🟡 Hội tụ vừa — {b}/7 ủng hộ · {br}/7 phản đối",          "#FFD700"
    elif br >= 5:label, color = f"🔴 Phân kỳ mạnh — {br}/7 yếu tố bất lợi vàng",            "#f85149"
    elif br >= 4:label, color = f"🟠 Phân kỳ vừa — {br}/7 bất lợi · {b}/7 ủng hộ",          "#ff7b54"
    else:        label, color = f"⚪ Phân kỳ hỗn hợp — {b} bull / {br} bear / {n} neutral",  "#8b949e"

    return {
        "score": score, "label": label, "color": color,
        "bullish": b, "bearish": br, "neutral": n,
        "alignment_pct": alignment_pct,
        "bull_signals": bulls, "bear_signals": bears, "neutral_signals": neutrals,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ML DIRECTIONAL MODEL (GradientBoosting — dự báo hướng giá)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def ml_directional_signal(_price_values: np.ndarray, days: int) -> dict:
    """
    GradientBoosting Classifier dự báo hướng giá sau `days` ngày.
    Features: RSI14, RSI5, MACD hist, BB%B, momentum 3/5/10/20d,
              MA20/MA50 deviation, volatility 10/20d, Kalman trend slope.
    Train trên lịch sử — không có lookahead bias.
    Kết quả: prob_up (0–1), signal, confidence.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    prices = pd.Series(_price_values)
    n      = len(prices)

    if n < max(180, days * 4):
        return {"prob_up": 0.5, "signal": "neutral", "confidence": 0.0, "ok": False}

    # ── Feature engineering ───────────────────────────────────────────────
    rsi14    = calc_rsi(prices, 14)
    rsi5     = calc_rsi(prices, 5)
    _, _, mh = calc_macd(prices)

    bb_mid   = prices.rolling(20).mean()
    bb_std   = prices.rolling(20).std()
    bb_pct   = (prices - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    ma20 = prices.rolling(20).mean()
    ma50 = prices.rolling(50).mean()

    # Kalman-smoothed slope (10-day)
    kal    = pd.Series(kalman_smooth(prices.values), index=prices.index)
    k_slope = kal.pct_change(5)

    feat = pd.DataFrame({
        "rsi14":     rsi14,
        "rsi5":      rsi5,
        "macd_hist": mh,
        "bb_pct":    bb_pct.clip(0, 1),
        "mom3":      prices.pct_change(3),
        "mom5":      prices.pct_change(5),
        "mom10":     prices.pct_change(10),
        "mom20":     prices.pct_change(20),
        "ma20_dev":  prices / (ma20 + 1e-9) - 1,
        "ma50_dev":  prices / (ma50 + 1e-9) - 1,
        "vol10":     prices.pct_change().rolling(10).std(),
        "vol20":     prices.pct_change().rolling(20).std(),
        "k_slope":   k_slope,
    }, index=prices.index)

    # Target: giá sau `days` ngày cao hơn hiện tại?
    target = (prices.shift(-days) > prices).astype(int)

    combined = feat.join(target.rename("target")).dropna()
    if len(combined) < 150 or combined["target"].nunique() < 2:
        return {"prob_up": 0.5, "signal": "neutral", "confidence": 0.0, "ok": False}

    # Tránh lookahead bias: chỉ train đến trước `days` ngày cuối
    X_train  = combined.iloc[:-days][feat.columns]
    y_train  = combined.iloc[:-days]["target"]
    X_pred   = combined.iloc[[-1]][feat.columns]

    if len(X_train) < 100 or y_train.nunique() < 2:
        return {"prob_up": 0.5, "signal": "neutral", "confidence": 0.0, "ok": False}

    try:
        from sklearn.ensemble import (GradientBoostingClassifier,
                                      RandomForestClassifier, VotingClassifier)
        from sklearn.neural_network import MLPClassifier

        scaler   = StandardScaler()
        Xtr      = scaler.fit_transform(X_train)
        Xpr      = scaler.transform(X_pred)

        # ── 3-Model Soft Voting Ensemble ──────────────────────────────────
        gb  = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.07,
            subsample=0.8, min_samples_leaf=10, random_state=42
        )
        rf  = RandomForestClassifier(
            n_estimators=50, max_depth=4, min_samples_leaf=10,
            random_state=42, n_jobs=1   # n_jobs=1 tránh fork trên Streamlit Cloud
        )
        mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu",
            max_iter=200, random_state=42, early_stopping=True,
            alpha=0.01    # L2 regularisation để tránh overfitting
        )
        ensemble = VotingClassifier(
            estimators=[("gb", gb), ("rf", rf), ("mlp", mlp)],
            voting="soft"
        )
        ensemble.fit(Xtr, y_train)

        prob_up    = float(ensemble.predict_proba(Xpr)[0, 1])
        confidence = abs(prob_up - 0.5) * 2   # 0 = không chắc, 1 = rất chắc

        if   prob_up >= 0.65: signal = "strong_buy"
        elif prob_up >= 0.55: signal = "buy"
        elif prob_up <= 0.35: signal = "strong_sell"
        elif prob_up <= 0.45: signal = "sell"
        else:                 signal = "neutral"

        return {
            "prob_up":    round(prob_up, 3),
            "prob_down":  round(1 - prob_up, 3),
            "signal":     signal,
            "confidence": round(confidence, 3),
            "ok":         True,
            "model":      "Ensemble (GBM + RF + MLP)",
        }
    except Exception as e:
        return {"prob_up": 0.5, "signal": "neutral", "confidence": 0.0,
                "ok": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  SHORT-TERM TECHNICAL SIGNALS (giao dịch 1–14 ngày)
# ══════════════════════════════════════════════════════════════════════════════

def short_term_signals(price: pd.Series, asset_key: str,
                       ohlcv: pd.DataFrame = None,
                       ml_result: dict = None) -> dict:
    """
    Phân tích kỹ thuật ngắn hạn cho giao dịch 1–14 ngày.
    Dùng RSI(5), MACD(12/26/9), Bollinger Band, Support/Resistance 20 ngày,
    ADX (trend filter), Stochastic, ML directional signal.
    Trả về dict: action, entry, target, stop, R:R, signals, adx, stoch, ...
    """
    if ohlcv is None: ohlcv = pd.DataFrame()
    cur = float(price.iloc[-1])

    # ── RSI(5) — nhạy hơn RSI(14) cho giao dịch ngắn ─────────────────────
    rsi5_val = float(calc_rsi(price, 5).iloc[-1])

    # ── MACD(12/26/9) ─────────────────────────────────────────────────────
    macd_line, macd_sig_line, macd_hist = calc_macd(price)
    macd_val   = float(macd_line.iloc[-1])
    mhist_now  = float(macd_hist.iloc[-1])
    mhist_prev = float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else 0.0

    if   mhist_now > 0 and mhist_prev <= 0: macd_cross = "golden"
    elif mhist_now < 0 and mhist_prev >= 0: macd_cross = "dead"
    elif mhist_now > 0:                      macd_cross = "up"
    else:                                    macd_cross = "down"

    # ── Bollinger Band position (20 ngày) ─────────────────────────────────
    bb_mid_v = float(price.rolling(20).mean().iloc[-1])
    bb_std_v = float(price.rolling(20).std().iloc[-1])
    bb_up_v  = bb_mid_v + 2 * bb_std_v
    bb_lo_v  = bb_mid_v - 2 * bb_std_v
    bb_width = bb_up_v - bb_lo_v
    bb_pos   = (cur - bb_lo_v) / bb_width if bb_width > 0 else 0.5

    # ── Support / Resistance (swing high/low 20 ngày) ─────────────────────
    recent20   = price.tail(20)
    support    = float(recent20.min())
    resistance = float(recent20.max())

    # ── ATR proxy (std dev 14 ngày) — dùng để đặt target/stop ────────────
    atr = max(float(price.tail(14).std()), cur * 0.003)

    # ── Momentum 3 ngày ───────────────────────────────────────────────────
    mom3 = (cur / float(price.iloc[-4]) - 1) * 100 if len(price) >= 4 else 0.0

    # ── ADX — đo độ mạnh xu hướng (cần OHLCV) ────────────────────────────
    adx_data  = calc_adx(ohlcv)     # returns default if ohlcv empty
    adx_val   = adx_data["adx"]
    adx_strong = adx_data["strong"]
    adx_dir   = adx_data["direction"]   # "up" / "down"

    # ── Stochastic Oscillator ─────────────────────────────────────────────
    stoch_data = calc_stoch(ohlcv)
    stoch_k    = stoch_data["k"]
    stoch_sig  = stoch_data["signal"]

    # ── Scoring ───────────────────────────────────────────────────────────
    score   = 0
    signals = []

    # RSI(5)
    if rsi5_val < 20:
        score += 3
        signals.append(("✅", f"RSI(5) = {rsi5_val:.0f} — Quá bán cực mạnh → Cơ hội MUA rất tốt", "green"))
    elif rsi5_val < 35:
        score += 2
        signals.append(("✅", f"RSI(5) = {rsi5_val:.0f} — Quá bán → Xem xét MUA", "green"))
    elif rsi5_val < 45:
        score += 1
        signals.append(("✅", f"RSI(5) = {rsi5_val:.0f} — Hơi quá bán → Nghiêng MUA", "green"))
    elif rsi5_val > 80:
        score -= 3
        signals.append(("🔴", f"RSI(5) = {rsi5_val:.0f} — Quá mua cực mạnh → Cơ hội BÁN / chốt lời", "red"))
    elif rsi5_val > 65:
        score -= 2
        signals.append(("🔴", f"RSI(5) = {rsi5_val:.0f} — Quá mua → Cẩn thận, xem xét BÁN", "red"))
    elif rsi5_val > 55:
        score -= 1
        signals.append(("🔴", f"RSI(5) = {rsi5_val:.0f} — Hơi quá mua → Nghiêng BÁN", "red"))
    else:
        signals.append(("➡️", f"RSI(5) = {rsi5_val:.0f} — Vùng trung tính (45–55)", "gray"))

    # MACD
    if macd_cross == "golden":
        score += 2
        signals.append(("✅", "MACD Golden Cross — vừa cắt lên → Tín hiệu MUA mạnh", "green"))
    elif macd_cross == "dead":
        score -= 2
        signals.append(("🔴", "MACD Dead Cross — vừa cắt xuống → Tín hiệu BÁN mạnh", "red"))
    elif macd_cross == "up":
        score += 1
        signals.append(("✅", f"MACD dương ({macd_val:+.4g}) — Đà tăng đang duy trì", "green"))
    else:
        score -= 1
        signals.append(("🔴", f"MACD âm ({macd_val:+.4g}) — Đà giảm đang duy trì", "red"))

    # Bollinger Band
    if bb_pos <= 0.10:
        score += 2
        signals.append(("✅", f"Giá sát BB dưới ({bb_pos*100:.0f}%) — Quá bán ngắn hạn, thường hồi phục", "green"))
    elif bb_pos <= 0.25:
        score += 1
        signals.append(("✅", f"Giá gần BB dưới ({bb_pos*100:.0f}%) — Vùng hỗ trợ Bollinger", "green"))
    elif bb_pos >= 0.90:
        score -= 2
        signals.append(("🔴", f"Giá sát BB trên ({bb_pos*100:.0f}%) — Quá mua, nguy cơ đảo chiều", "red"))
    elif bb_pos >= 0.75:
        score -= 1
        signals.append(("🔴", f"Giá gần BB trên ({bb_pos*100:.0f}%) — Cẩn thận áp lực bán", "red"))
    else:
        signals.append(("➡️", f"Giá giữa BB ({bb_pos*100:.0f}%) — Trung tính", "gray"))

    # Price vs S/R 20 ngày
    if cur <= support + atr * 0.2:
        score += 1
        signals.append(("✅", "Giá đang ở vùng hỗ trợ 20 ngày — Điểm MUA tiềm năng", "green"))
    elif cur >= resistance - atr * 0.2:
        score -= 1
        signals.append(("🔴", "Giá đang ở vùng kháng cự 20 ngày — Điểm BÁN tiềm năng", "red"))

    # Momentum 3 ngày
    if mom3 < -2.5:
        score -= 1
        signals.append(("🔴", f"Momentum 3 ngày: {mom3:+.1f}% — Đà giảm ngắn hạn mạnh", "red"))
    elif mom3 > 2.5:
        score += 1
        signals.append(("✅", f"Momentum 3 ngày: {mom3:+.1f}% — Đà tăng ngắn hạn mạnh", "green"))
    else:
        signals.append(("➡️", f"Momentum 3 ngày: {mom3:+.1f}%", "gray"))

    # ── ADX — Bộ lọc xu hướng (tín hiệu đáng tin hơn khi ADX > 25) ───────
    if adx_val < 20:
        signals.append(("⚠️", f"ADX = {adx_val:.0f} — Thị trường đang ĐI NGANG. Tín hiệu kỹ thuật kém tin cậy hơn. Nên chờ breakout.", "gray"))
        # Giảm score về 0 nếu thị trường sideways (không có xu hướng rõ)
        score = round(score * 0.5)
    elif adx_val >= 25:
        dir_match = (adx_dir == "up" and score > 0) or (adx_dir == "down" and score < 0)
        if dir_match:
            score += 1
            signals.append(("✅", f"ADX = {adx_val:.0f} — Xu hướng MẠNH, xác nhận tín hiệu {adx_data['direction'].upper()}", "green"))
        else:
            signals.append(("⚠️", f"ADX = {adx_val:.0f} — Xu hướng mạnh nhưng NGƯỢC chiều tín hiệu. Thận trọng!", "red"))
    else:
        signals.append(("➡️", f"ADX = {adx_val:.0f} — Xu hướng vừa phải", "gray"))

    # ── Stochastic Oscillator ─────────────────────────────────────────────
    stoch_label_map = {
        "oversold":   ("✅", "Stochastic %K={:.0f} — Quá bán (dưới 20) → Cơ hội MUA", "green"),
        "overbought": ("🔴", "Stochastic %K={:.0f} — Quá mua (trên 80) → Cơ hội BÁN", "red"),
        "buy":        ("✅", "Stochastic %K={:.0f} — %K cắt lên dưới 50 → Tín hiệu MUA", "green"),
        "sell":       ("🔴", "Stochastic %K={:.0f} — %K cắt xuống trên 50 → Tín hiệu BÁN", "red"),
        "neutral":    ("➡️", "Stochastic %K={:.0f} — Trung tính", "gray"),
    }
    s_ico, s_txt, s_col = stoch_label_map.get(stoch_sig, stoch_label_map["neutral"])
    signals.append((s_ico, s_txt.format(stoch_k), s_col))
    if stoch_sig == "oversold":   score += 1
    elif stoch_sig == "overbought": score -= 1
    elif stoch_sig == "buy":      score += 1
    elif stoch_sig == "sell":     score -= 1

    # ── OBV (On Balance Volume) ───────────────────────────────────────────
    obv_data = calc_obv(ohlcv)
    if obv_data.get("ok"):
        obv_sig = obv_data["signal"]
        obv_div = obv_data["divergence"]
        if obv_div == "bullish":
            score += 2
            signals.append(("✅", f"OBV Divergence BULLISH — Giá giảm nhưng volume MUA tăng → Tích lũy ngầm", "green"))
        elif obv_div == "bearish":
            score -= 2
            signals.append(("🔴", f"OBV Divergence BEARISH — Giá tăng nhưng volume BÁN tăng → Phân phối ngầm", "red"))
        elif obv_sig == "bullish":
            score += 1
            signals.append(("✅", f"OBV Trend BULLISH — Dòng tiền tổ chức đang chảy vào", "green"))
        elif obv_sig == "bearish":
            score -= 1
            signals.append(("🔴", f"OBV Trend BEARISH — Dòng tiền tổ chức đang rút ra", "red"))
        else:
            signals.append(("➡️", f"OBV trung tính — Dòng tiền chưa rõ hướng", "gray"))

    # ── CCI (Commodity Channel Index) ─────────────────────────────────────
    cci_data = calc_cci(ohlcv)
    if cci_data.get("ok"):
        cv = cci_data["cci"]
        if   cv > 200:  score -= 1; signals.append(("🔴", f"CCI = {cv:.0f} — Cực kỳ quá mua → Nguy cơ đảo chiều giảm", "red"))
        elif cv > 100:  score -= 0; signals.append(("⚠️", f"CCI = {cv:.0f} — Quá mua → Thận trọng", "orange"))
        elif cv < -200: score += 1; signals.append(("✅", f"CCI = {cv:.0f} — Cực kỳ quá bán → Cơ hội tăng", "green"))
        elif cv < -100: signals.append(("✅", f"CCI = {cv:.0f} — Quá bán → Xem xét mua", "green"))
        else:           signals.append(("➡️", f"CCI = {cv:.0f} — Trung tính", "gray"))

    # ── ML Directional Signal (3-model Ensemble) ──────────────────────────
    if ml_result and ml_result.get("ok"):
        ml_prob  = ml_result["prob_up"]
        ml_conf  = ml_result["confidence"]
        ml_sig   = ml_result["signal"]
        if ml_sig in ("strong_buy", "buy"):
            ml_sc = 2 if ml_sig == "strong_buy" else 1
            score += ml_sc
            signals.append(("🤖", f"ML Ensemble: {ml_prob*100:.0f}% xác suất TĂNG (tin cậy {ml_conf*100:.0f}%) → MUA", "green"))
        elif ml_sig in ("strong_sell", "sell"):
            ml_sc = -2 if ml_sig == "strong_sell" else -1
            score += ml_sc
            signals.append(("🤖", f"ML Ensemble: {(1-ml_prob)*100:.0f}% xác suất GIẢM (tin cậy {ml_conf*100:.0f}%) → BÁN", "red"))
        else:
            signals.append(("🤖", f"ML Ensemble: {ml_prob*100:.0f}% xác suất tăng — Trung tính", "gray"))

    # ── Action ────────────────────────────────────────────────────────────
    if score >= 5:
        action = "MUA MẠNH"; a_color = "#3fb950"; a_icon = "🚀"
    elif score >= 2:
        action = "MUA"; a_color = "#76c3a0"; a_icon = "📈"
    elif score <= -5:
        action = "BÁN MẠNH"; a_color = "#f85149"; a_icon = "🔻"
    elif score <= -2:
        action = "BÁN / CHỐT LỜI"; a_color = "#ff7b54"; a_icon = "📉"
    else:
        action = "QUAN SÁT / TRUNG TÍNH"; a_color = "#FFD700"; a_icon = "⏸️"

    # ── Entry / Target / Stop ─────────────────────────────────────────────
    if score >= 2:          # Buy bias
        entry  = cur
        target = min(resistance, cur + atr * 2.0)
        stop   = max(support * 0.998, cur - atr * 1.2)
    elif score <= -2:       # Sell bias
        entry  = cur
        target = max(support, cur - atr * 2.0)
        stop   = min(resistance * 1.002, cur + atr * 1.2)
    else:                   # Neutral
        entry  = cur
        target = cur + atr * 0.8
        stop   = cur - atr * 0.6

    gain = abs(target - entry)
    loss = abs(entry  - stop)
    rr   = round(gain / loss, 2) if loss > 1e-9 else 0.0

    return {
        "score":        score,
        "action":       action,
        "action_color": a_color,
        "action_icon":  a_icon,
        "rsi5":         rsi5_val,
        "macd_val":     macd_val,
        "macd_cross":   macd_cross,
        "macd_hist":    mhist_now,
        "bb_pos":       bb_pos,
        "support":      support,
        "resistance":   resistance,
        "entry":        entry,
        "target":       target,
        "stop":         stop,
        "rr":           rr,
        "atr":          atr,
        "mom3":         mom3,
        "adx":          adx_data,
        "stoch":        stoch_data,
        "signals":      signals,
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

    # ── Auto-detect current leaders (Wikidata) ────────────────────────────
    with st.spinner("🔍 Xác định lãnh đạo hiện tại (Wikidata)..."):
        leaders_data = fetch_current_leaders()

    # ── Fed Policy Analysis (FRED + personality) ──────────────────────────
    with st.spinner("🏦 Phân tích chính sách Fed..."):
        fred_data  = fetch_fred_rates()
        fed_result = fed_policy_analysis(fred_data, macro, leaders=leaders_data)

    # ── Whale Positioning (ETF flow + OI + volume) ────────────────────────
    with st.spinner("🐋 Tải dữ liệu cá mập..."):
        whale_data_raw = fetch_whale_data(asset_key)
        whale_result   = whale_regime(whale_data_raw, asset_key)

    # ── OHLCV + ADX + Stochastic + OBV + CCI ─────────────────────────────
    with st.spinner("📐 Tải OHLCV + tính ADX / OBV / CCI..."):
        ohlcv    = fetch_ohlcv(asset_key)
        adx_data = calc_adx(ohlcv)
        stoch_d  = calc_stoch(ohlcv)
        obv_d    = calc_obv(ohlcv)
        cci_d    = calc_cci(ohlcv)

    # ── COT (Commitment of Traders — CFTC Official) ───────────────────────
    with st.spinner("📋 Tải dữ liệu COT từ CFTC..."):
        cot_result = fetch_cot_data(asset_key)

    # ── Sentiment Intelligence Layer ──────────────────────────────────────
    with st.spinner("📰 Phân tích sentiment tin tức (Google News)..."):
        news_result = fetch_news_sentiment(asset_key)

    with st.spinner("📊 Tải Put/Call Ratio (Options Market)..."):
        pcr_result  = fetch_options_pcr(asset_key)

    with st.spinner("😨 Phân tích VIX Term Structure..."):
        vix_term    = fetch_vix_term()

    # Credit stress + Regime + Cross-Asset (pure compute, không cần network)
    credit_result  = fetch_credit_stress(fred_data)
    mkt_regime     = detect_macro_regime(fred_data, macro)
    cross_asset    = calc_cross_asset_alignment(macro, fred_data)

    # ── ML Directional Signal (3-model Ensemble) ──────────────────────────
    with st.spinner("🤖 Chạy ML Ensemble (GBM + RF + MLP)..."):
        ml_result = ml_directional_signal(price.values, forecast_days)

    # ── GVZ (Gold Volatility) — chỉ cho XAU ──────────────────────────────
    gvz_data = fetch_gvz() if asset_key == "XAU" else {"ok": False}

    # ── Combined macro score (Macro + Fed + Whale + COT + Sentiment) ──────
    fed_contrib    = round(fed_result["score"]  * 0.35)
    whale_contrib  = round(whale_result["score"] * 0.25)
    cot_contrib    = round(cot_result.get("score", 0)    * 0.30) if cot_result.get("ok")    else 0
    obv_contrib    = round(obv_d.get("score", 0)          * 0.20) if obv_d.get("ok")         else 0
    news_contrib   = round(news_result.get("score", 0)    * 0.20) if news_result.get("ok")   else 0
    pcr_contrib    = round(pcr_result.get("score", 0)     * 0.15) if pcr_result.get("ok")    else 0
    vix_contrib    = round(vix_term.get("score", 0)       * 0.15) if vix_term.get("ok")      else 0
    credit_contrib = round(credit_result.get("score", 0)  * 0.15) if credit_result.get("ok") else 0
    regime_contrib = mkt_regime["gold_bias"]                        # trực tiếp: -2 → +3
    xasset_contrib = round(cross_asset["score"]            * 0.20)
    combined_macro = max(-12, min(12,
        macro_score + fed_contrib + whale_contrib + cot_contrib + obv_contrib
        + news_contrib + pcr_contrib + vix_contrib + credit_contrib
        + regime_contrib + xasset_contrib))

    # ── Forecast (tích hợp ML bias) ───────────────────────────────────────
    with st.spinner(f"🔮 Chạy mô hình dự báo {a_short}..."):
        ml_prob_val = ml_result.get("prob_up", 0.5) if ml_result.get("ok") else 0.5
        fc_mean, fc_lo_s, fc_hi_s = forecast(
            price.values, str(price.index[-1]), forecast_days,
            combined_macro, asset_key, ml_prob_val
        )

    # ── Signal (tích hợp ML) ──────────────────────────────────────────────
    signal, sig_color, sig_icon, tech_notes = compute_signal(
        price, ma20, ma50, ma200, rsi, fc_mean, combined_macro, ml_result
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
    m1, m2, m3, m4, m5 = st.columns(5)

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

    # ML probability metric
    if ml_result.get("ok"):
        ml_p  = ml_result["prob_up"]
        ml_lbl = "TĂNG" if ml_p >= 0.55 else ("GIẢM" if ml_p <= 0.45 else "TRUNG TÍNH")
        ml_clr = "normal" if ml_p >= 0.55 else ("inverse" if ml_p <= 0.45 else "off")
        m5.metric("🤖 ML Probability",
                  f"{ml_p*100:.0f}% {ml_lbl}",
                  f"Độ tin: {ml_result['confidence']*100:.0f}%",
                  delta_color=ml_clr,
                  help="GradientBoosting Classifier dự báo xác suất giá tăng. "
                       "Features: RSI, MACD, Bollinger, Momentum, MA deviation, Kalman slope. "
                       "Train trên lịch sử không có lookahead bias. "
                       "≥65% = MUA · ≤35% = BÁN · 45–55% = Trung tính.")
    else:
        m5.metric("🤖 ML Probability", "N/A", "Chưa đủ dữ liệu", delta_color="off")

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

    # ══ MARKET INTELLIGENCE DASHBOARD (Hedge Fund Framework) ══════════════
    r = mkt_regime
    xa = cross_asset

    # Tính Signal Consensus (tổng hợp toàn bộ signals)
    all_signals_scores = [
        macro_score, fed_contrib, whale_contrib, cot_contrib, obv_contrib,
        news_contrib, pcr_contrib, vix_contrib, credit_contrib,
        regime_contrib, xasset_contrib,
    ]
    bull_count  = sum(1 for s in all_signals_scores if s > 0)
    bear_count  = sum(1 for s in all_signals_scores if s < 0)
    neut_count  = sum(1 for s in all_signals_scores if s == 0)
    total_dir   = bull_count + bear_count
    consensus_pct = round(bull_count / total_dir * 100) if total_dir > 0 else 50

    if consensus_pct >= 80: cons_label, cons_color = "Đồng thuận mạnh MUA", "#3fb950"
    elif consensus_pct >= 65: cons_label, cons_color = "Nghiêng về MUA",   "#76c3a0"
    elif consensus_pct <= 20: cons_label, cons_color = "Đồng thuận mạnh BÁN","#f85149"
    elif consensus_pct <= 35: cons_label, cons_color = "Nghiêng về BÁN",   "#ff7b54"
    else:                     cons_label, cons_color = "Phân kỳ / Không rõ","#FFD700"

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        f"#### 🎯 Market Intelligence — Macro Regime & Signal Alignment &nbsp;"
        f"<span style='color:{r['color']};font-size:0.95rem;'>"
        f"[ {r['emoji']} {r['name_vn']} ]</span>",
        unsafe_allow_html=True,
    )

    ri1, ri2, ri3, ri4 = st.columns(4)

    # Col 1: Macro Regime
    ri1.metric(
        f"{r['emoji']} Macro Regime",
        r["name_vn"],
        r["gold_outlook"].split(" — ")[0],
        delta_color="normal" if r["gold_bias"] >= 0 else "inverse",
        help="Phân loại môi trường kinh tế theo mô hình Bridgewater All Weather: Growth × Inflation."
             " Xác định CHIỀU dài hạn của vàng trong môi trường hiện tại."
    )

    # Col 2: Growth vs Inflation
    g_arrow = "▲" * abs(r["growth_score"]) if r["growth_score"] > 0 else ("▼" * abs(r["growth_score"]) if r["growth_score"] < 0 else "→")
    i_arrow = "▲" * abs(r["inflation_score"]) if r["inflation_score"] > 0 else ("▼" * abs(r["inflation_score"]) if r["inflation_score"] < 0 else "→")
    ri2.metric(
        "📊 Growth × Inflation Quadrant",
        f"Growth {g_arrow} · Inflation {i_arrow}",
        f"Bias vàng từ regime: {r['gold_bias']:+d} điểm",
        delta_color="normal" if r["gold_bias"] >= 0 else "inverse",
        help="Trục tăng trưởng: S&P trend + Yield Curve + Credit OAS + VIX. "
             "Trục lạm phát: T5YIE breakeven + T5YIFR + Dầu WTI 3M momentum."
    )

    # Col 3: Cross-Asset Alignment
    ri3.metric(
        "🔗 Cross-Asset Alignment",
        xa["label"].split(" — ")[0] if " — " in xa["label"] else xa["label"],
        f"{xa['bullish']}/7 bull · {xa['bearish']}/7 bear",
        delta_color="normal" if xa["score"] >= 0 else "inverse",
        help="7 yếu tố cross-asset: DXY + 10Y Yield + S&P + VIX + Lãi suất thực + Dầu + HY OAS. "
             "Khi nhiều yếu tố đồng thuận → tín hiệu đáng tin hơn."
    )

    # Col 4: Signal Consensus
    ri4.metric(
        "🧭 Signal Consensus (11 nguồn)",
        cons_label,
        f"{bull_count} bull / {bear_count} bear / {neut_count} neutral",
        delta_color="normal" if consensus_pct >= 50 else "inverse",
        help="Đếm số nguồn tín hiệu đang bullish vs bearish trên toàn bộ 11 layers: "
             "Macro + Fed + Whale + COT + OBV + News + PCR + VIX + Credit + Regime + Cross-Asset."
    )

    # ── Expander chi tiết ─────────────────────────────────────────────────
    with st.expander("🔍 Chi tiết Market Intelligence — Regime phân tích & Cross-Asset", expanded=False):
        # Regime Description
        st.markdown(
            f"**{r['emoji']} Diễn giải Regime: {r['name_vn']}**\n\n"
            f"{r['desc_vn']}\n\n"
            f"**Triển vọng vàng:** {r['gold_outlook']}\n\n"
            f"**Điểm Growth:** {r['growth_score']:+d}/4 · **Điểm Inflation:** {r['inflation_score']:+d}/4"
        )
        if r["details"]:
            st.markdown("**Các yếu tố phát hiện:**")
            for d in r["details"]:
                st.markdown(f"  - {d}")

        st.markdown("---")

        # Cross-Asset Breakdown
        st.markdown(
            f"**🔗 Cross-Asset Alignment: {xa['alignment_pct']}% bullish "
            f"({xa['bullish']} / {xa['bullish'] + xa['bearish']} định hướng)**"
        )
        if xa["bull_signals"]:
            st.markdown("**✅ Bullish factors:**")
            for sig in xa["bull_signals"]:
                st.markdown(f"  - {sig}")
        if xa["bear_signals"]:
            st.markdown("**🔴 Bearish factors:**")
            for sig in xa["bear_signals"]:
                st.markdown(f"  - {sig}")
        if xa["neutral_signals"]:
            st.markdown("**➡️ Neutral:**")
            for sig in xa["neutral_signals"]:
                st.markdown(f"  - {sig}")

        st.markdown("---")

        # Full Signal Consensus breakdown
        st.markdown("**🧭 Signal Consensus — Tất cả 11 layers:**")
        layers = [
            ("Macro tổng hợp",   macro_score),
            ("Fed Policy",        fed_contrib),
            ("Whale (ETF/OI/Vol)",whale_contrib),
            ("COT (CFTC)",        cot_contrib),
            ("OBV Divergence",    obv_contrib),
            ("News Sentiment",    news_contrib),
            ("Options PCR",       pcr_contrib),
            ("VIX Term Structure",vix_contrib),
            ("Credit Stress",     credit_contrib),
            ("Macro Regime",      regime_contrib),
            ("Cross-Asset Align", xasset_contrib),
        ]
        for name_l, val_l in layers:
            bar_icon = "🟢" if val_l > 0 else ("🔴" if val_l < 0 else "⚪")
            st.markdown(f"  {bar_icon} **{name_l}**: {val_l:+d}")
        st.markdown(
            f"\n**Tổng hợp: {bull_count} bullish · {bear_count} bearish · {neut_count} neutral "
            f"→ Consensus: {consensus_pct}% ({cons_label})**\n\n"
            f"*Đóng góp Regime {regime_contrib:+d} + Cross-Asset {xasset_contrib:+d} = "
            f"{regime_contrib + xasset_contrib:+d} điểm vào combined_macro*"
        )

    st.markdown("---")

    # ── Short-term Trading Panel (chỉ hiển thị khi kỳ dự báo ≤ 14 ngày) ──
    if forecast_days in SHORT_TERM_DAYS:
        st_sig = short_term_signals(price, asset_key, ohlcv, ml_result)
        ac     = st_sig["action_color"]
        ac_bg  = hex_rgba(ac, 0.13)
        ac_bd  = hex_rgba(ac, 0.50)

        st.markdown(
            f"### ⚡ Bảng Giao Dịch Ngắn Hạn — {period_label(forecast_days)}",
        )

        # ── Action banner ─────────────────────────────────────────────────
        st.markdown(
            f"""<div style="background:{ac_bg};border:2px solid {ac_bd};border-radius:12px;
            padding:14px 20px;margin-bottom:12px;">
            <span style="color:{ac};font-size:1.4rem;font-weight:800;">
            {st_sig['action_icon']} {st_sig['action']}</span>
            &nbsp;&nbsp;
            <span style="color:#8b949e;font-size:0.9rem;">
            Điểm kỹ thuật: {st_sig['score']:+d} &nbsp;·&nbsp;
            {a_short} hiện tại: <b style="color:#e6edf3;">{fmt_price(cur, asset_key)}</b>
            </span></div>""",
            unsafe_allow_html=True,
        )

        # ── Key metrics: Entry / Target / Stop / R:R ──────────────────────
        t1, t2, t3, t4 = st.columns(4)
        t1.metric(
            "📍 Giá vào lệnh (Entry)",
            fmt_price(st_sig["entry"], asset_key),
            "Giá hiện tại",
            delta_color="off",
            help="Vùng giá đề xuất vào lệnh. Nên vào dần (DCA) thay vì all-in một lần.",
        )
        tgt_chg = (st_sig["target"] - cur) / cur * 100
        t2.metric(
            "🎯 Mục tiêu lợi nhuận (Target)",
            fmt_price(st_sig["target"], asset_key),
            f"{tgt_chg:+.1f}%",
            delta_color="normal",
            help="Kháng cự gần nhất hoặc mức giá mục tiêu dựa trên ATR. Chốt lời dần khi chạm.",
        )
        stp_chg = (st_sig["stop"] - cur) / cur * 100
        t3.metric(
            "🛑 Cắt lỗ (Stop Loss)",
            fmt_price(st_sig["stop"], asset_key),
            f"{stp_chg:+.1f}%",
            delta_color="inverse",
            help="Mức giá cắt lỗ. PHẢI đặt lệnh stop loss ngay khi vào lệnh để bảo vệ vốn.",
        )
        rr_color = "normal" if st_sig["rr"] >= 1.5 else "off"
        t4.metric(
            "⚖️ Tỷ lệ Lợi nhuận/Rủi ro (R:R)",
            f"1 : {st_sig['rr']:.1f}",
            "Tốt ✅" if st_sig["rr"] >= 2 else ("Chấp nhận ⚠️" if st_sig["rr"] >= 1.5 else "Thấp 🔴"),
            delta_color=rr_color,
            help="Tỷ lệ R:R ≥ 2 là lý tưởng. Tối thiểu ≥ 1.5. Dưới 1.0 nên bỏ qua giao dịch.",
        )

        # ── Support / Resistance + Technical indicators ───────────────────
        sr1, sr2 = st.columns(2)
        with sr1:
            st.markdown("**📊 Hỗ trợ & Kháng cự (20 ngày):**")
            s_color = "#3fb950"; r_color = "#f85149"
            st.markdown(
                f"- 🟢 **Hỗ trợ (Support):** {fmt_price(st_sig['support'], asset_key)}"
                f" &nbsp;({(st_sig['support']-cur)/cur*100:+.1f}%)\n"
                f"- 🔴 **Kháng cự (Resistance):** {fmt_price(st_sig['resistance'], asset_key)}"
                f" &nbsp;({(st_sig['resistance']-cur)/cur*100:+.1f}%)\n"
                f"- 📏 **ATR proxy (14 ngày):** {fmt_price(st_sig['atr'], asset_key)}"
            )
        with sr2:
            st.markdown("**📈 Chỉ báo kỹ thuật ngắn hạn:**")
            rsi5_emoji = "🔴 Quá mua" if st_sig["rsi5"] > 65 else ("🟢 Quá bán" if st_sig["rsi5"] < 35 else "⚪ Trung tính")
            macd_emoji = {"golden": "🟢 Golden Cross ↑", "dead": "🔴 Dead Cross ↓",
                          "up": "↗️ MACD dương", "down": "↘️ MACD âm"}.get(st_sig["macd_cross"], "—")
            bb_emoji   = "🟢 Gần BB dưới" if st_sig["bb_pos"] < 0.25 else \
                         ("🔴 Gần BB trên" if st_sig["bb_pos"] > 0.75 else "⚪ Giữa BB")
            # ADX + Stochastic info
            adx_r  = st_sig["adx"]
            stch_r = st_sig["stoch"]
            adx_emoji  = "🟢 Mạnh" if adx_r["strong"] else "⚪ Yếu/Sideways"
            stch_emoji = {"oversold":"🟢 Quá bán","overbought":"🔴 Quá mua",
                          "buy":"↗️ Mua","sell":"↘️ Bán","neutral":"⚪ Trung tính"}.get(stch_r["signal"],"⚪")
            # ML info
            ml_info = ""
            if ml_result and ml_result.get("ok"):
                ml_info = f"\n- **ML Model:** {ml_result['prob_up']*100:.0f}% tăng — {ml_result['signal'].replace('_',' ').upper()}"
            # GVZ info (XAU only)
            gvz_info = ""
            if gvz_data.get("ok"):
                gvz_info = f"\n- **GVZ (Vàng Volatility):** {gvz_data['value']:.1f} — {gvz_data['level']}"

            st.markdown(
                f"- **RSI(5):** {st_sig['rsi5']:.0f} — {rsi5_emoji}\n"
                f"- **MACD:** {macd_emoji}\n"
                f"- **Bollinger Band:** {bb_emoji} ({st_sig['bb_pos']*100:.0f}%)\n"
                f"- **ADX:** {adx_r['adx']:.0f} — {adx_emoji} ({adx_r['trend']})\n"
                f"- **Stochastic %K:** {stch_r['k']:.0f} — {stch_emoji}\n"
                f"- **Momentum 3 ngày:** {st_sig['mom3']:+.2f}%"
                + ml_info + gvz_info
            )

        # ── Signal detail expander ────────────────────────────────────────
        with st.expander("🔍 Chi tiết tín hiệu kỹ thuật ngắn hạn", expanded=False):
            for icon, text, _ in st_sig["signals"]:
                st.markdown(f"{icon} {text}")
            st.markdown(
                "\n---\n"
                f"*⚠️ Lưu ý quan trọng: Phân tích kỹ thuật ngắn hạn chỉ mang tính tham khảo. "
                "Vàng/bạc/hàng hoá có thể biến động mạnh bất ngờ do tin tức vĩ mô (Fed, xung đột, "
                "dữ liệu CPI...). Luôn dùng stop loss và không vào quá 10-15% danh mục cho một giao dịch. "
                "Kết hợp với phân tích vĩ mô bên dưới để có quyết định tốt hơn.*"
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

    _mm(mc1, "💵 DXY — Sức mạnh USD",      "dxy",      ".1f",  low_good=True, pct_delta=True)
    _mm(mc2, "📉 Yield 10Y — Lợi suất TP","yield10y", ".2f",  low_good=True)
    _mm(mc3, "😨 VIX — Chỉ số sợ hãi",    "vix",      ".0f",  low_good=False)
    _mm(mc4, "📈 S&P 500 — CK Mỹ",        "sp500",    ",.0f", low_good=False, pct_delta=True)

    mc5, mc6, mc7, mc8 = st.columns(4)
    _mm(mc5, "🛢️ Dầu WTI — Giá dầu",      "oil",      ".1f",  low_good=False, pct_delta=True)
    _mm(mc6, "📊 TIPS — Lãi suất thực proxy", "tips",  ".2f",  low_good=False, pct_delta=True)

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

    # Tên lãnh đạo từ kết quả phân tích (đã tra cứu dynamic)
    _pres_name_ui  = fed_result.get("pres_name",  leaders_data.get("president",  "Tổng thống"))
    _chair_name_ui = fed_result.get("chair_name", leaders_data.get("fed_chair", "Fed Chair"))
    _detect_src    = leaders_data.get("source", "")
    _detect_badge  = (f" <span style='color:#8b949e;font-size:0.78rem;'>"
                      f"[🔍 Tự động: {_detect_src}]</span>" if _detect_src else "")

    st.markdown(
        f"#### 🏦 Fed Radar — {_pres_name_ui} · {_chair_name_ui} · Lãi suất{_detect_badge} &nbsp;"
        f"<span style='color:{fed_color};font-size:0.95rem;'>"
        f"[ {fed_direction} · Điểm: {fed_score:+d} ]</span>",
        unsafe_allow_html=True,
    )

    real_rate_val   = fed_result.get("real_rate")
    inflation_val   = fed_result.get("inflation_exp")
    walcl_val       = None
    if "fed_balance" in fred_data and len(fred_data["fed_balance"]) >= 2:
        walcl_s   = fred_data["fed_balance"]
        walcl_val = float(walcl_s.iloc[-1]) / 1e6  # convert to tỷ USD (original in triệu)

    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("🎯 Fed Hướng đi", fed_direction,
               f"Điểm tổng hợp: {fed_score:+d}/5", delta_color="off",
               help="Hướng chính sách lãi suất dự kiến dựa trên yield curve, momentum 2Y, Warsh/Trump factor, lãi suất thực và kỳ vọng lạm phát.")
    fc2.metric("✂️ Xác suất Cắt lãi", f"{fed_prob_cut}%",
               f"Giữ/Tăng lãi: {fed_prob_hold}%",
               delta_color="normal" if fed_prob_cut > 50 else "inverse",
               help=f"Xác suất Fed CẮT lãi trong 6–9 tháng tới. Tính từ 7 yếu tố: dữ liệu FRED + ngoại cảm {_chair_name_ui} & {_pres_name_ui}.")
    if cur_rate is not None:
        fc3.metric("🏛️ FEDFUNDS — Lãi suất Fed", f"{cur_rate:.2f}%",
                   f"Neutral 2.50% · Gap {cur_rate - 2.5:+.2f}%",
                   delta_color="inverse",
                   help="Federal Funds Rate: Lãi suất liên ngân hàng qua đêm. Neutral rate = 2.5% (không kích thích cũng không thắt chặt kinh tế).")
    else:
        fc3.metric("🏛️ FEDFUNDS — Lãi suất Fed", "N/A")
    if curve_val is not None:
        fc4.metric("📐 T10Y2Y — Yield Curve", f"{curve_val:+.2f}%",
                   "⚠️ Đảo ngược → suy thoái sắp tới" if curve_val < -0.5 else ("Đảo ngược nhẹ" if curve_val < 0 else "Bình thường"),
                   delta_color="normal" if curve_val < 0 else "inverse",
                   help="10Y Treasury Yield trừ 2Y Treasury Yield. Âm = đảo ngược = thị trường kỳ vọng suy thoái và Fed cắt lãi. Lịch sử: đảo ngược → 6-18T sau thường có suy thoái.")
    else:
        fc4.metric("📐 T10Y2Y — Yield Curve", "N/A")

    fc5, fc6, fc7, fc8 = st.columns(4)
    if real_rate_val is not None:
        rr_delta = "⚡ BUY vàng/bạc mạnh!" if real_rate_val < 0 else ("Tốt" if real_rate_val < 0.5 else ("Áp lực cao" if real_rate_val > 2.0 else "Trung tính"))
        fc5.metric("🔑 DFII10 — Lãi suất thực", f"{real_rate_val:.2f}%",
                   rr_delta,
                   delta_color="normal" if real_rate_val < 0.5 else "inverse",
                   help="10Y Real Interest Rate = Yield danh nghĩa − Lạm phát kỳ vọng. DRIVER SỐ 1 CỦA VÀNG: tương quan âm -0.90. Âm = không có chi phí cơ hội giữ vàng → BUY. Cao = chi phí cơ hội lớn → SELL.")
    else:
        fc5.metric("🔑 DFII10 — Lãi suất thực", "N/A",
                   help="FRED DFII10: 10-Year Real Treasury Yield. Driver số 1 của vàng.")
    if inflation_val is not None:
        inf_delta = "Trên target 2% → Fed thắt" if inflation_val > 2.5 else ("Neo ổn định" if inflation_val > 2.0 else "Thấp → Fed có thể cắt")
        fc6.metric("📊 T5YIFR — Kỳ vọng lạm phát", f"{inflation_val:.2f}%",
                   inf_delta,
                   delta_color="inverse" if inflation_val > 2.5 else "normal",
                   help="5-Year, 5-Year Forward Inflation Expectation: Kỳ vọng lạm phát trung bình 5 năm (bắt đầu từ 5 năm nữa). Fed target 2%. >2.75% = lạm phát mất neo → Fed không thể cắt lãi.")
    else:
        fc6.metric("📊 T5YIFR — Kỳ vọng lạm phát", "N/A",
                   help="FRED T5YIFR: 5Y5Y Forward Inflation Expectation. Chỉ số lạm phát dài hạn mà Fed theo dõi sát.")
    if walcl_val is not None:
        walcl_chg = None
        if len(fred_data["fed_balance"]) >= 5:
            walcl_prev = float(fred_data["fed_balance"].iloc[-5]) / 1e6
            walcl_chg  = walcl_val - walcl_prev
        fc7.metric("🏦 WALCL — Bảng cân đối Fed", f"${walcl_val:.2f}T",
                   f"{'QT: rút {abs(walcl_chg):.2f}T' if walcl_chg and walcl_chg < 0 else ('QE: bơm +{walcl_chg:.2f}T' if walcl_chg and walcl_chg > 0 else 'Ổn định')}",
                   delta_color="normal" if walcl_chg and walcl_chg > 0 else "inverse",
                   help="Fed Balance Sheet (nghìn tỷ USD). QE = Fed mua TP, bơm tiền → tốt cho vàng/BTC. QT = Fed bán TP, hút tiền → áp lực giảm tài sản rủi ro.")
    else:
        fc7.metric("🏦 WALCL — Bảng cân đối Fed", "N/A",
                   help="FRED WALCL: Tổng tài sản trên bảng cân đối của Fed. QE vs QT.")

    # Gold/Silver ratio (hiển thị trong cả XAU và XAG tab)
    if asset_key in ("XAU", "XAG"):
        try:
            _gold_p   = float(fetch_price("XAU")[0].iloc[-1])
            _silver_p = float(fetch_price("XAG")[0].iloc[-1])
            gs_ratio  = _gold_p / _silver_p
            gs_delta  = "Bạc rẻ so với vàng → cơ hội mua bạc" if gs_ratio > 80 else ("Bình thường" if gs_ratio > 60 else "Vàng rẻ tương đối")
            fc8.metric("⚖️ Gold/Silver Ratio", f"{gs_ratio:.1f}",
                       gs_delta,
                       delta_color="inverse" if gs_ratio > 80 else "off",
                       help="Số ounce bạc cần để mua 1 ounce vàng. TB lịch sử 65–70. >80 = bạc đang rất rẻ so với vàng → lịch sử thường kéo về trung bình bằng cách bạc tăng mạnh hơn.")
        except Exception:
            fc8.metric("⚖️ Gold/Silver Ratio", "N/A")
    else:
        fc8.empty()

    with st.expander(f"🔍 Ngoại cảm: {_pres_name_ui} · {_chair_name_ui} · Xung đột cấu trúc", expanded=False):
        # ── Lấy profile động đã được tính trong fed_policy_analysis ─────────
        _pres_prof  = fed_result.get("pres_profile",  {})
        _chair_prof = fed_result.get("chair_profile", {})
        _dynamic    = analyze_leader_dynamic(_pres_prof, _chair_prof)
        conflict    = _dynamic["conflict_level"]
        pres_win    = _dynamic["pres_wins_prob"]
        chair_win   = _dynamic["chair_wins_prob"]
        comp        = _dynamic["compromise_prob"]
        narrative   = _dynamic["narrative"]
        conflict_color = "#f85149" if conflict >= 2 else ("#FFD700" if conflict == 1 else "#3fb950")

        # ── Bảng kịch bản xung đột ──────────────────────────────────────────
        st.markdown(
            f"**⚡ Động lực {_pres_name_ui}–{_chair_name_ui}** "
            f"<span style='color:{conflict_color};'>(Cấp độ {conflict}/3 — {narrative})</span>\n\n"
            f"| Kịch bản | Xác suất |\n"
            f"|---|---|\n"
            f"| {_pres_name_ui} thuyết phục {_chair_name_ui} thay đổi chính sách | **{pres_win}%** |\n"
            f"| {_chair_name_ui} giữ vững lập trường độc lập | **{chair_win}%** |\n"
            f"| Thỏa hiệp / trung gian | **{comp}%** |\n",
            unsafe_allow_html=True,
        )

        # ── Ngoại cảm Fed Chair ─────────────────────────────────────────────
        chair_emoji = _chair_prof.get("emoji", "🏦")
        chair_term  = _chair_prof.get("term_info", "")
        chair_traits = _chair_prof.get("personality_vn", [])
        if chair_traits:
            st.markdown(
                f"**{chair_emoji} {_chair_name_ui}** *({chair_term})*:\n"
                + "\n".join(f"- {t}" for t in chair_traits)
            )

        # ── Ngoại cảm Tổng thống ────────────────────────────────────────────
        pres_emoji  = _pres_prof.get("emoji", "🇺🇸")
        pres_term   = _pres_prof.get("term_info", "")
        pres_traits = _pres_prof.get("personality_vn", [])
        if pres_traits:
            st.markdown(
                f"\n**{pres_emoji} {_pres_name_ui}** *({pres_term})*:\n"
                + "\n".join(f"- {t}" for t in pres_traits)
            )

        # ── Nguồn dữ liệu ───────────────────────────────────────────────────
        if _detect_src:
            st.caption(f"🔍 Lãnh đạo được tự động xác định qua {_detect_src} — cập nhật mỗi 24h")
        else:
            st.caption("⚠️ Không kết nối được Wikidata/Wikipedia — dùng dữ liệu mặc định")

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

    # ── COT — Commitment of Traders (CFTC) ────────────────────────────────
    if cot_result.get("ok"):
        cot_score  = cot_result["score"]
        cot_color  = "#3fb950" if cot_score > 0 else ("#f85149" if cot_score < 0 else "#FFD700")
        cot_label  = (
            "Tích lũy mạnh" if cot_score >= 2 else
            "Tích lũy nhẹ"  if cot_score == 1 else
            "Trung tính"    if cot_score == 0 else
            "Phân phối nhẹ" if cot_score == -1 else
            "Phân phối mạnh"
        )
        st.markdown(
            f"#### 📋 COT — Quỹ đầu cơ (CFTC Chính thức) &nbsp;"
            f"<span style='color:{cot_color};font-size:0.95rem;'>"
            f"[ {cot_label} · Điểm: {cot_score:+d} ]</span>",
            unsafe_allow_html=True,
        )
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric(
            "📅 Báo cáo COT",
            cot_result["date"],
            "Cập nhật thứ Sáu hàng tuần",
            delta_color="off",
        )
        cc2.metric(
            "📊 Net Long (Hedge Fund)",
            f"{cot_result['net']:,.0f}",
            f"1W: {cot_result['chg_1w']:+,.0f}",
            delta_color="normal" if cot_result["chg_1w"] >= 0 else "inverse",
        )
        cc3.metric(
            "📈 Thay đổi 4 tuần",
            f"{cot_result['chg_4w']:+,.0f}",
            "Tích lũy" if cot_result["chg_4w"] > 0 else "Phân phối",
            delta_color="normal" if cot_result["chg_4w"] >= 0 else "inverse",
        )
        cc4.metric(
            "📐 Percentile (20 tuần)",
            f"{cot_result['pct_rank']:.0f}th",
            "Extreme Long" if cot_result["pct_rank"] >= 85 else (
                "Extreme Short" if cot_result["pct_rank"] <= 15 else "Trung bình"),
            delta_color="off",
        )
        with st.expander("🔍 Chi tiết COT (Commitment of Traders)", expanded=False):
            for sig in cot_result.get("signals", []):
                st.markdown(sig)
            st.markdown(
                f"\n*Đóng góp vào dự báo: COT score {cot_score:+d} × 30% = {cot_contrib:+d} điểm*\n\n"
                f"*Dữ liệu: CFTC (Commodity Futures Trading Commission) — Managed Money "
                f"Long/Short positions. Cập nhật mỗi thứ Sáu lúc 15:30 ET.*"
            )
    else:
        st.info("📋 COT: Không tải được dữ liệu CFTC (chỉ có cho XAU/XAG/HG/CL)", icon="ℹ️")

    st.markdown("---")

    # ── Sentiment Intelligence Layer ───────────────────────────────────────
    n_ok = sum([news_result.get("ok"), pcr_result.get("ok"),
                vix_term.get("ok"), credit_result.get("ok")])
    intel_score = news_contrib + pcr_contrib + vix_contrib + credit_contrib
    intel_color = "#3fb950" if intel_score > 0 else ("#f85149" if intel_score < 0 else "#FFD700")

    st.markdown(
        f"#### 🔮 Sentiment Intelligence — Tin tức · Options · VIX · Tín dụng &nbsp;"
        f"<span style='color:{intel_color};font-size:0.95rem;'>"
        f"[ Điểm tổng hợp: {intel_score:+d} ]</span>",
        unsafe_allow_html=True,
    )

    ic1, ic2, ic3, ic4 = st.columns(4)

    # News Sentiment
    if news_result.get("ok"):
        ic1.metric(
            "📰 Sentiment Tin tức",
            news_result["label"],
            f"{news_result['pos']} tích cực / {news_result['neg']} tiêu cực ({news_result['n']} bài)",
            delta_color="normal" if news_result["score"] >= 0 else "inverse",
            help="Phân tích sentiment tiêu đề bài báo từ Google News RSS. Đếm từ khóa tích cực/tiêu cực trong 20 bài mới nhất."
        )
    else:
        ic1.metric("📰 Sentiment Tin tức", "Không tải được", delta_color="off")

    # Options PCR
    if pcr_result.get("ok"):
        pcr_delta_color = "normal" if pcr_result["score"] >= 0 else "inverse"
        ic2.metric(
            f"📊 Put/Call Ratio ({pcr_result['etf']})",
            f"PCR = {pcr_result['avg_pcr']:.2f}",
            pcr_result["label"].split(" — ")[-1] if " — " in pcr_result["label"] else pcr_result["label"],
            delta_color=pcr_delta_color,
            help=f"Put/Call Ratio từ options {pcr_result['etf']} (expiry {pcr_result['expiry']}). PCR cao = bi quan = tín hiệu MUA ngược chiều (contrarian). PCR thấp = lạc quan thái quá = cảnh báo đảo chiều."
        )
    else:
        ic2.metric("📊 Put/Call Ratio", "N/A", delta_color="off",
                   help="Chỉ có cho XAU/XAG/HG/CL/BTC qua ETF proxy")

    # VIX Term Structure
    if vix_term.get("ok"):
        vix_ratio = vix_term["ratio"]
        vix_struct = "Backwardation" if vix_ratio > 1.0 else "Contango"
        ic3.metric(
            "😨 VIX Term Structure",
            f"VIX {vix_term['vix']:.0f} / VIX3M {vix_term['vix3m']:.0f}",
            f"{vix_struct} ({vix_ratio:.2f}x) — {vix_term['label'].split(' — ')[0]}",
            delta_color="normal" if vix_term["score"] >= 0 else "inverse",
            help="So sánh VIX spot và VIX 3 tháng. Backwardation (VIX>VIX3M) = sợ hãi cấp tính → vàng là safe haven. Contango = thị trường bình tĩnh."
        )
    else:
        ic3.metric("😨 VIX Term Structure", "N/A", delta_color="off")

    # Credit Stress
    if credit_result.get("ok"):
        ic4.metric(
            "🏦 Credit Stress (HY OAS)",
            f"{credit_result['oas']:.2f}%",
            credit_result["label"].split(" — ")[-1] if " — " in credit_result["label"] else credit_result["label"],
            delta_color="normal" if credit_result["score"] >= 0 else "inverse",
            help="ICE BofA US High Yield OAS (FRED: BAMLH0A0HYM2). Spread cao = căng thẳng tín dụng → vàng tăng. Bình thường <3.5%. Khủng hoảng >6%."
        )
    else:
        ic4.metric("🏦 Credit Stress (HY OAS)", "N/A", delta_color="off",
                   help="FRED BAMLH0A0HYM2: High Yield Option-Adjusted Spread")

    if n_ok > 0:
        with st.expander("🔍 Chi tiết Sentiment Intelligence", expanded=False):
            if news_result.get("ok"):
                conf_pct = int(news_result.get("confidence", 0) * 100)
                st.markdown(
                    f"**📰 Google News Sentiment** (query: '{news_result['query']}')\n\n"
                    f"- {news_result['n']} bài phân tích · Confidence: {conf_pct}%\n"
                    f"- Từ tích cực ({news_result['pos']}): `{', '.join(news_result.get('pos_words', []))}`\n"
                    f"- Từ tiêu cực ({news_result['neg']}): `{', '.join(news_result.get('neg_words', []))}`\n"
                    f"- *Đóng góp: News score {news_result['score']:+d} × 20% = {news_contrib:+d} điểm*"
                )
                st.markdown("---")
            if pcr_result.get("ok"):
                st.markdown(
                    f"**📊 Options Put/Call Ratio** ({pcr_result['etf']} · expiry {pcr_result['expiry']})\n\n"
                    f"- Volume PCR: {pcr_result['pcr_vol']:.3f} · Open Interest PCR: {pcr_result['pcr_oi']:.3f}\n"
                    f"- Weighted avg PCR: **{pcr_result['avg_pcr']:.3f}**\n"
                    f"- Put vol: {pcr_result['put_vol']:,} · Call vol: {pcr_result['call_vol']:,}\n"
                    f"- *Đóng góp: PCR score {pcr_result['score']:+d} × 15% = {pcr_contrib:+d} điểm*\n\n"
                    f"*Lý thuyết contrarian: PCR cao → thị trường đã mua bảo hiểm (put) nhiều "
                    f"→ downside được hấp thụ → giá có xu hướng phục hồi.*"
                )
                st.markdown("---")
            if vix_term.get("ok"):
                st.markdown(
                    f"**😨 VIX Term Structure** (^VIX / ^VIX3M)\n\n"
                    f"- VIX spot: **{vix_term['vix']:.2f}** · VIX 3M: **{vix_term['vix3m']:.2f}**\n"
                    f"- Ratio: {vix_term['ratio']:.3f} "
                    f"({'Backwardation = sợ hãi cấp tính' if vix_term['ratio'] > 1 else 'Contango = bình tĩnh'})\n"
                    f"- *Đóng góp: VIX score {vix_term['score']:+d} × 15% = {vix_contrib:+d} điểm*"
                )
                st.markdown("---")
            if credit_result.get("ok"):
                st.markdown(
                    f"**🏦 High Yield Credit Spread** (FRED: BAMLH0A0HYM2)\n\n"
                    f"- OAS hiện tại: **{credit_result['oas']:.2f}%** · Thay đổi 1 tuần: {credit_result['chg_1w']:+.2f}%\n"
                    f"- Ngưỡng: Bình thường <3.5% · Căng thẳng >4.5% · Khủng hoảng >6%\n"
                    f"- *Đóng góp: Credit score {credit_result['score']:+d} × 15% = {credit_contrib:+d} điểm*"
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
        # ADX
        adx_str = f"ADX = {adx_data['adx']:.0f} — {adx_data['trend']}"
        adx_ico = "✅" if adx_data["strong"] else "➡️"
        st.markdown(f"- {adx_ico} {adx_str}")
        # Stochastic
        stch_vn = {"oversold":"Quá bán","overbought":"Quá mua",
                   "buy":"Mua","sell":"Bán","neutral":"Trung tính"}.get(stoch_d["signal"],"")
        st.markdown(f"- 📊 Stochastic %K = {stoch_d['k']:.0f} · %D = {stoch_d['d']:.0f} — {stch_vn}")
        # GVZ (XAU only)
        if gvz_data.get("ok"):
            gvz_ico = "⚠️" if gvz_data["value"] >= 18 else "✅"
            st.markdown(f"- {gvz_ico} GVZ (Vàng Volatility): {gvz_data['value']:.1f} — {gvz_data['level']}")
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

    # ── BTC: Fear & Greed Index ───────────────────────────────────────────
    if asset_key == "BTC":
        fg = fetch_fear_greed()
        st.markdown("#### 😱 Crypto Fear & Greed Index")
        fg1, fg2, fg3 = st.columns(3)
        if fg["ok"]:
            fg_val = fg["value"]
            fg_lbl = fg["label"]
            fg_clr = fg["color"]
            fg1.metric(
                "😱 Fear & Greed Index",
                f"{fg_val} / 100",
                fg_lbl,
                delta_color="off",
                help="Chỉ số sợ hãi và tham lam crypto từ alternative.me. 0–25 = Cực sợ (mua khi người khác sợ). 75–100 = Cực tham lam (cẩn thận đỉnh). Tổng hợp từ: volatility, momentum, social media, surveys, dominance, Google Trends.",
            )
            fg2.markdown(
                f"<div style='padding:12px;border-radius:8px;background:{hex_rgba(fg_clr,0.15)};"
                f"border:1px solid {hex_rgba(fg_clr,0.4)};margin-top:8px;'>"
                f"<span style='color:{fg_clr};font-size:1.8rem;font-weight:700;'>{fg_val}</span>"
                f"<br><span style='color:{fg_clr};font-size:0.9rem;'>{fg_lbl}</span>"
                f"<br><span style='color:#8b949e;font-size:0.75rem;'>0 = Cực sợ · 100 = Cực tham lam</span></div>",
                unsafe_allow_html=True,
            )
            fg3.markdown(
                f"*Ý nghĩa với BTC:*\n"
                f"- Cực sợ hãi ({fg_val} ≤ 25): Thường là cơ hội mua tốt\n"
                f"- Cực tham lam ({fg_val} ≥ 75): Thị trường có thể đang đỉnh\n"
                f"- Kết hợp với macro + whale để quyết định"
            )
        else:
            fg1.metric("😱 Fear & Greed", "N/A", "Không tải được")
        st.markdown("---")

    # ── Glossary (Bảng giải thích thuật ngữ) ─────────────────────────────
    with st.expander("📖 Giải thích thuật ngữ — Các từ viết tắt trong app là gì?", expanded=False):
        st.markdown("*Nhấp vào mỗi mục để đọc giải thích chi tiết*\n")
        for term, explanation in GLOSSARY.items():
            st.markdown(f"**{term}**")
            st.markdown(f"&nbsp;&nbsp;&nbsp;{explanation}\n")
            st.markdown("---")

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<p style='color:#484f58;font-size:0.76rem;text-align:center;'>"
        f"Nguồn: {ticker} · Mô hình: HW + ARIMA + Momentum + Ensemble ML (GBM+RF+MLP) + Kalman + "
        f"Macro + Fed (DFII10/T5YIFR/T5YIE/M2SL) + Whale + COT (CFTC) + OBV + CCI + ADX + Stochastic + GVZ "
        f"+ News Sentiment + Options PCR + VIX Term + Credit Stress (HY OAS) + Auto Leader Detection (Wikidata) · "
        f"Điểm: Macro {macro_score:+d} + Fed {fed_contrib:+d} + Whale {whale_contrib:+d} "
        f"+ COT {cot_contrib:+d} + OBV {obv_contrib:+d} "
        f"+ News {news_contrib:+d} + PCR {pcr_contrib:+d} + VIX {vix_contrib:+d} + Credit {credit_contrib:+d} "
        f"+ Regime {regime_contrib:+d} + X-Asset {xasset_contrib:+d} "
        f"= {combined_macro:+d} · "
        f"⚠️ Chỉ mang tính tham khảo, không phải khuyến nghị đầu tư.</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERT ANALYSIS TAB — 17 MODULES · 4 BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fed_news() -> list:
    """
    Lấy tin tức mới nhất từ Federal Reserve RSS feed.
    Trả về list các dict {title, date, link}.
    """
    import urllib.request
    import xml.etree.ElementTree as _ET

    url = "https://www.federalreserve.gov/feeds/press_all.xml"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as r:
            xml_bytes = r.read()
        root = _ET.fromstring(xml_bytes)
        items = []
        for item in root.findall(".//item")[:15]:
            title_el = item.find("title")
            date_el  = item.find("pubDate")
            link_el  = item.find("link")
            if title_el is not None and title_el.text:
                items.append({
                    "title": title_el.text.strip(),
                    "date":  date_el.text.strip() if date_el is not None and date_el.text else "",
                    "link":  link_el.text.strip() if link_el is not None and link_el.text else "",
                })
        return items
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def translate_to_vi_batch(texts: tuple) -> list:
    """
    Dịch nhiều câu tiếng Anh → tiếng Việt bằng Google Translate free endpoint.
    Nhận tuple (để hashable với st.cache_data), trả về list cùng độ dài.
    Fallback về text gốc nếu lỗi.
    """
    import urllib.request, urllib.parse, json as _json

    if not texts:
        return []

    # Ghép tất cả câu bằng ký tự phân tách đặc biệt để dịch 1 lần
    SEP = " ||| "
    combined = SEP.join(texts)
    try:
        encoded = urllib.parse.quote(combined)
        url = (
            "https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl=en&tl=vi&dt=t&q={encoded}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = _json.loads(r.read())

        # Ghép các segment dịch lại
        translated_combined = "".join(
            seg[0] for seg in data[0] if seg[0]
        )
        # Tách lại theo SEP (Google có thể dịch SEP thành nhiều dạng khác nhau)
        # → tách theo ||| (dạng phổ biến nhất sau dịch)
        parts = translated_combined.split("|||")
        parts = [p.strip() for p in parts]

        # Nếu số lượng khớp thì trả về, không thì fallback
        if len(parts) == len(texts):
            return parts
        # Nếu lệch số lượng, thử tách lại
        parts2 = [p.strip() for p in translated_combined.split("|")]
        parts2 = [p for p in parts2 if p and p.strip() not in ("", "||")]
        if len(parts2) == len(texts):
            return parts2
        return list(texts)  # fallback: giữ nguyên tiếng Anh
    except Exception:
        return list(texts)  # fallback nếu không có internet hoặc bị block


def analyze_fedspeak(news_items: list) -> dict:
    """
    Phân tích ngôn ngữ của Fed (Fedspeak Decoder).
    Phân loại từng cụm từ thành Hawkish / Dovish / Hidden signal.
    Returns dict: score (-3..+3), label, signals, color.
    """
    HAWK_PHRASES = {
        # ── Cốt lõi (7 cũ) ──────────────────────────────────────────────────
        "higher for longer":        ("🦅 Hawkish", "Fed muốn duy trì lãi suất cao — thực ra đang cân nhắc cắt giảm"),
        "data dependent":           ("🦅 Hawkish", "Fed chuẩn bị thay đổi hướng — tín hiệu pivot sắp tới"),
        "remain vigilant":          ("🦅 Hawkish", "Fed lo ngại lạm phát — nhưng thường là peak hawkishness"),
        "inflation too high":       ("🦅 Hawkish", "Lạm phát vẫn là ưu tiên — tuy nhiên tín hiệu thị trường đã chiết khấu"),
        "price stability":          ("🦅 Hawkish", "Fed ưu tiên ổn định giá — áp lực giảm ngắn hạn cho vàng"),
        "strong labor market":      ("🦅 Hawkish", "Thị trường lao động mạnh — lý do để không cắt lãi suất"),
        "not cutting":              ("🦅 Hawkish", "Fed công khai từ chối cắt — nhưng thị trường vẫn định giá cắt"),
        # ── Mở rộng (8 mới) ─────────────────────────────────────────────────
        "2 percent":                ("🦅 Hawkish", "Fed nhấn mạnh mục tiêu 2% — chưa hài lòng với lạm phát hiện tại"),
        "still elevated":           ("🦅 Hawkish", "Lạm phát 'vẫn còn cao' — tín hiệu chưa cắt lãi sớm"),
        "further progress":         ("🦅 Hawkish", "Cần thêm bằng chứng trước khi cắt — Fed chưa tự tin"),
        "not the time":             ("🦅 Hawkish", "Chưa đến lúc cắt lãi — tín hiệu hawkish rõ ràng"),
        "bumpy path":               ("🦅 Hawkish", "Con đường về target lạm phát gập ghềnh — Fed thận trọng kéo dài"),
        "resilient economy":        ("🦅 Hawkish", "Kinh tế mạnh → Fed không có lý do vội cắt lãi"),
        "quantitative tightening":  ("🦅 Hawkish", "QT đang chạy — thu hẹp bảng cân đối, hút tiền ra khỏi hệ thống"),
        "upside risk":              ("🦅 Hawkish", "Rủi ro lạm phát tăng thêm — Fed khó cắt giảm lãi suất"),
    }
    DOVE_PHRASES = {
        # ── Cốt lõi (10 cũ) ─────────────────────────────────────────────────
        "restrictive territory": ("🕊️ Dovish", "Lãi suất đang quá cao — Fed chuẩn bị cắt giảm sớm"),
        "balanced risks":        ("🕊️ Dovish", "Rủi ro cân bằng — pivot sắp đến, bullish cho vàng"),
        "financial stability":   ("🕊️ Dovish", "⚠️ Cảnh báo: Có gì đó đang gãy trong hệ thống!"),
        "rate cuts":             ("🕊️ Dovish", "Fed chính thức nói tới cắt lãi suất — very bullish vàng"),
        "easing":                ("🕊️ Dovish", "Tín hiệu nới lỏng — QE/cắt lãi suất đang được thảo luận"),
        "labor market cooling":  ("🕊️ Dovish", "Thị trường lao động hạ nhiệt — tạo điều kiện cắt lãi suất"),
        "transitory":            ("⚠️ Sai lầm lịch sử", "Fed đã sai về 'transitory' năm 2021 — cảnh báo rủi ro"),
        "on track":              ("🕊️ Dovish", "Fed hài lòng với tiến trình — pivot gần hơn"),
        "progress":              ("🕊️ Dovish", "Fed nhận thấy tiến bộ — tín hiệu hawkish đang suy yếu"),
        "appropriate":           ("🕊️ Dovish", "Fed cho rằng chính sách phù hợp — cắt giảm sắp tới"),
        # ── Mở rộng (8 mới) ─────────────────────────────────────────────────
        "cooling inflation":     ("🕊️ Dovish", "Lạm phát đang hạ nhiệt — tạo điều kiện thuận lợi cho cắt lãi"),
        "disinflation":          ("🕊️ Dovish", "Xu hướng giảm phát đang diễn ra — Fed có lý do nới lỏng"),
        "full employment":       ("🕊️ Dovish", "Việc làm đầy đủ đạt được — Fed có thể chuyển sang bảo vệ tăng trưởng"),
        "downside risk":         ("🕊️ Dovish", "Rủi ro tăng trưởng giảm — nghiêng về cắt lãi để bảo vệ kinh tế"),
        "soft landing":          ("🕊️ Dovish", "Hạ cánh mềm — Fed tự tin giảm lãi suất mà không gây suy thoái"),
        "insurance cut":         ("🕊️ Dovish", "Cắt lãi phòng ngừa — Fed hành động chủ động trước khi kinh tế yếu"),
        "accommodation":         ("🕊️ Dovish", "Chính sách nới lỏng/hỗ trợ — Fed đang chuyển sang dễ tiền tệ"),
        "gradual":               ("🕊️ Dovish", "Cắt lãi từ từ, thận trọng — vẫn là hướng dovish rõ ràng"),
    }

    # ── Negation prefixes — phủ định dove phrase = tín hiệu hawkish ──────
    _NEGATIONS = ("not ", "no ", "never ", "don't ", "doesn't ", "won't ", "haven't ", "without ")

    detected = []
    hawk_score = 0
    dove_score = 0

    all_text = " ".join(item.get("title", "").lower() for item in news_items)

    for phrase, (label, meaning) in HAWK_PHRASES.items():
        if phrase in all_text:
            detected.append({"phrase": phrase, "label": label, "meaning": meaning, "type": "hawk"})
            hawk_score += 1

    for phrase, (label, meaning) in DOVE_PHRASES.items():
        # Kiểm tra phủ định trước: "not cutting", "won't ease", v.v.
        negated = any((neg + phrase) in all_text for neg in _NEGATIONS)
        if negated:
            detected.append({
                "phrase":  f"NOT {phrase}",
                "label":   "🦅 Hawkish (phủ định)",
                "meaning": f"Fed phủ nhận '{phrase}' — tín hiệu hawkish ẩn",
                "type":    "hawk",
            })
            hawk_score += 1
        elif phrase in all_text:
            detected.append({"phrase": phrase, "label": label, "meaning": meaning, "type": "dove"})
            dove_score += 1

    net = dove_score - hawk_score
    score = max(-3, min(3, net))

    if score >= 2:    label_out, color = "Dovish mạnh 🕊️", "#3fb950"
    elif score == 1:  label_out, color = "Hơi Dovish",     "#76c3a0"
    elif score == -1: label_out, color = "Hơi Hawkish",    "#ff7b54"
    elif score <= -2: label_out, color = "Hawkish mạnh 🦅","#f85149"
    else:             label_out, color = "Trung lập",      "#8b949e"

    return {
        "score":    score,
        "label":    label_out,
        "color":    color,
        "detected": detected,
        "hawk":     hawk_score,
        "dove":     dove_score,
    }


def fetch_fomc_calendar() -> list:
    """
    Lịch họp FOMC 2025-2026 (hardcoded — công bố trước 1 năm).
    Mỗi cuộc họp kéo dài 2 ngày; ngày công bố quyết định = ngày thứ 2.
    """
    from datetime import date
    meetings = [
        # 2025
        {"date": date(2025, 1, 29),  "label": "Jan 2025",  "status": "done"},
        {"date": date(2025, 3, 19),  "label": "Mar 2025",  "status": "done"},
        {"date": date(2025, 5, 7),   "label": "May 2025",  "status": "done"},
        {"date": date(2025, 6, 18),  "label": "Jun 2025",  "status": "done"},
        {"date": date(2025, 7, 30),  "label": "Jul 2025",  "status": "upcoming"},
        {"date": date(2025, 9, 17),  "label": "Sep 2025",  "status": "upcoming"},
        {"date": date(2025, 11, 5),  "label": "Nov 2025",  "status": "upcoming"},
        {"date": date(2025, 12, 17), "label": "Dec 2025",  "status": "upcoming"},
        # 2026
        {"date": date(2026, 1, 28),  "label": "Jan 2026",  "status": "upcoming"},
        {"date": date(2026, 3, 18),  "label": "Mar 2026",  "status": "upcoming"},
        {"date": date(2026, 4, 29),  "label": "Apr 2026",  "status": "upcoming"},
        {"date": date(2026, 6, 17),  "label": "Jun 2026",  "status": "upcoming"},
        {"date": date(2026, 7, 29),  "label": "Jul 2026",  "status": "upcoming"},
        {"date": date(2026, 9, 16),  "label": "Sep 2026",  "status": "upcoming"},
        {"date": date(2026, 11, 4),  "label": "Nov 2026",  "status": "upcoming"},
        {"date": date(2026, 12, 16), "label": "Dec 2026",  "status": "upcoming"},
    ]
    today = datetime.now().date()
    for m in meetings:
        delta = (m["date"] - today).days
        m["days_away"] = delta
        if delta < 0:
            m["status"] = "done"
        elif delta == 0:
            m["status"] = "today"
        elif delta <= 14:
            m["status"] = "soon"
        else:
            m["status"] = "upcoming"
    return meetings


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fed_funds_futures() -> dict:
    """
    Ước tính xác suất cắt lãi suất từ thị trường qua 13-Week T-Bill yield (^IRX).
    Công thức: Implied Fed Funds Rate ≈ 100 - IRX_price.
    So sánh với FEDFUNDS rate hiện tại từ FRED để tính kỳ vọng thị trường.
    """
    try:
        irx = yf.Ticker("^IRX")
        irx_hist = irx.history(period="5d")
        if irx_hist.empty:
            return {"ok": False}
        irx_val = float(irx_hist["Close"].iloc[-1])
        # IRX là yield %, implied fed funds ≈ irx_val
        # Kỳ vọng thị trường: nếu IRX < FEDFUNDS thì market đang price in cắt
        fed_data = fetch_fred_rates()
        if "fedfunds" in fed_data:
            fedfunds = float(fed_data["fedfunds"].iloc[-1])
            diff = fedfunds - irx_val
            if diff > 0.5:
                market_view = "Thị trường price in CẮT lãi suất mạnh"
                color = "#3fb950"
                score = 2
            elif diff > 0.15:
                market_view = "Thị trường price in cắt lãi suất nhẹ"
                color = "#76c3a0"
                score = 1
            elif diff < -0.15:
                market_view = "Thị trường price in TĂNG lãi suất"
                color = "#f85149"
                score = -2
            else:
                market_view = "Thị trường kỳ vọng giữ nguyên lãi suất"
                color = "#8b949e"
                score = 0
            return {
                "ok": True,
                "irx": irx_val,
                "fedfunds": fedfunds,
                "diff": diff,
                "market_view": market_view,
                "color": color,
                "score": score,
            }
        return {"ok": False}
    except Exception:
        return {"ok": False}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gold_jpy_signal() -> dict:
    """
    Gold vs JPY Safe Haven Signal.
    Khi JPY tăng (USDJPY giảm) + Gold tăng → cả 2 cùng tăng → Safe Haven mode thực sự.
    Khi JPY tăng nhưng Gold giảm → risk-off selective, vàng kém hấp dẫn hơn Yen.
    """
    try:
        xau  = yf.Ticker("GC=F").history(period="30d")["Close"]
        usdjpy = yf.Ticker("USDJPY=X").history(period="30d")["Close"]
        if xau.empty or usdjpy.empty or len(xau) < 5 or len(usdjpy) < 5:
            return {"ok": False}

        xau_ret    = (xau.iloc[-1] / xau.iloc[-5] - 1) * 100
        usdjpy_ret = (usdjpy.iloc[-1] / usdjpy.iloc[-5] - 1) * 100
        jpy_ret    = -usdjpy_ret  # JPY tăng khi USDJPY giảm

        if xau_ret > 1 and jpy_ret > 0.5:
            signal  = "✅ SAFE HAVEN CONFIRMED — Gold + JPY cùng tăng"
            detail  = "Cả vàng và yên đều được mua vào. Rủi ro thị trường thực sự cao."
            color   = "#3fb950"
            score   = 2
            bullish = True
        elif xau_ret > 1 and jpy_ret < -0.5:
            signal  = "⚡ GOLD ONLY — Vàng tăng riêng, JPY yếu"
            detail  = "Vàng tăng không phải vì safe haven mà vì USD yếu hoặc lý do riêng. Ít bền vững hơn."
            color   = "#76c3a0"
            score   = 1
            bullish = True
        elif xau_ret < -1 and jpy_ret > 0.5:
            signal  = "⚠️ JPY ONLY — Yên tăng, vàng giảm"
            detail  = "Thị trường chọn Yên thay vì Vàng. Tín hiệu yếu cho XAU ngắn hạn."
            color   = "#ff7b54"
            score   = -1
            bullish = False
        elif xau_ret < -1 and jpy_ret < -0.5:
            signal  = "🔴 RISK-ON — Cả Gold và JPY đều giảm"
            detail  = "Thị trường đang risk-on. Vốn chảy sang cổ phiếu/crypto. Bearish ngắn hạn cho vàng."
            color   = "#f85149"
            score   = -2
            bullish = False
        else:
            signal  = "➡️ NEUTRAL — Tín hiệu chưa rõ ràng"
            detail  = "Cả vàng và Yên đều trong biên độ bình thường. Chờ xác nhận."
            color   = "#8b949e"
            score   = 0
            bullish = None

        return {
            "ok": True,
            "signal": signal, "detail": detail,
            "color": color, "score": score, "bullish": bullish,
            "xau_ret": round(xau_ret, 2), "jpy_ret": round(jpy_ret, 2),
            "xau_price": round(float(xau.iloc[-1]), 1),
            "usdjpy":    round(float(usdjpy.iloc[-1]), 2),
        }
    except Exception:
        return {"ok": False}


@st.cache_data(ttl=3600, show_spinner=False)
def calc_expert_max_pain() -> dict:
    """
    Tính Max Pain từ options chain GLD (Gold ETF).
    Max Pain = strike price nơi tổng giá trị options hết hạn ít giá trị nhất cho buyer.
    Market maker sẽ cố đưa giá về vùng này trước OpEx.
    """
    try:
        gld = yf.Ticker("GLD")
        exps = gld.options
        if not exps:
            return {"ok": False}
        # Lấy expiry gần nhất
        exp = exps[0]
        chain = gld.option_chain(exp)
        calls = chain.calls[["strike", "openInterest"]].copy()
        puts  = chain.puts[["strike", "openInterest"]].copy()
        calls["openInterest"] = pd.to_numeric(calls["openInterest"], errors="coerce").fillna(0)
        puts["openInterest"]  = pd.to_numeric(puts["openInterest"],  errors="coerce").fillna(0)

        strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))
        if len(strikes) < 3:
            return {"ok": False}

        pain = {}
        calls_dict = dict(zip(calls["strike"], calls["openInterest"]))
        puts_dict  = dict(zip(puts["strike"],  puts["openInterest"]))

        for s in strikes:
            call_pain = sum(
                max(0, s - k) * calls_dict.get(k, 0) for k in strikes if k < s
            )
            put_pain = sum(
                max(0, k - s) * puts_dict.get(k, 0) for k in strikes if k > s
            )
            pain[s] = call_pain + put_pain

        max_pain_strike = min(pain, key=pain.get)
        cur_gld = float(gld.history(period="1d")["Close"].iloc[-1])
        # GLD ≈ Gold/10
        gold_equiv = max_pain_strike * 10
        cur_gold   = cur_gld * 10

        diff_pct = (gold_equiv - cur_gold) / cur_gold * 100

        return {
            "ok":           True,
            "expiry":       exp,
            "max_pain_gld": round(max_pain_strike, 1),
            "gold_equiv":   round(gold_equiv, 0),
            "cur_gld":      round(cur_gld, 2),
            "cur_gold":     round(cur_gold, 0),
            "diff_pct":     round(diff_pct, 1),
        }
    except Exception:
        return {"ok": False}


@st.cache_data(ttl=1800, show_spinner=False)
def calc_summary_forecast() -> dict:
    """
    Dự báo ngắn gọn cho 6 tài sản × 7 kỳ thời gian.
    Dựa trên: RSI, MA20/50/200, momentum, trend alignment, structural bias.
    Returns: {asset_key: {days: {dir, color, score}}}
    """
    TICKERS = {
        "XAU":    "GC=F",
        "XAG":    "SI=F",
        "HG":     "HG=F",
        "CL":     "CL=F",
        "USDVND": "USDVND=X",
        "BTC":    "BTC-USD",
    }
    PERIODS = [1, 3, 7, 14, 21, 30, 60, 90, 180, 270, 365, 730, 1095, 1825]

    result = {}

    for ak, ticker in TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period="200d")["Close"].dropna()
            if len(hist) < 20:
                result[ak] = {p: {"dir": "N/A", "color": "#8b949e", "score": 0} for p in PERIODS}
                continue

            cur   = float(hist.iloc[-1])
            ma20  = float(hist.rolling(20).mean().iloc[-1])
            ma50  = float(hist.rolling(min(50,  len(hist))).mean().iloc[-1])
            ma200 = float(hist.rolling(min(200, len(hist))).mean().iloc[-1])

            # RSI-14
            delta = hist.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi   = float((100 - 100 / (1 + rs)).dropna().iloc[-1]) if not rs.dropna().empty else 50.0

            # Momentum
            mom_1w = (cur / float(hist.iloc[-5])  - 1) * 100 if len(hist) >= 5  else 0.0
            mom_1m = (cur / float(hist.iloc[-20]) - 1) * 100 if len(hist) >= 20 else 0.0
            mom_3m = (cur / float(hist.iloc[-60]) - 1) * 100 if len(hist) >= 60 else 0.0
            mom_6m = (cur / float(hist.iloc[-120])- 1) * 100 if len(hist) >= 120 else mom_3m
            mom_1y = (cur / float(hist.iloc[-0])  - 1) * 100  # fallback same period

            asset_result = {}
            for days in PERIODS:
                score = 0

                # ── Trend alignment (MA stack) ──────────────────────────────
                if cur > ma20:  score += 1
                if cur > ma50:  score += 1
                if cur > ma200: score += 1
                if ma20 > ma50: score += 1

                # ── RSI (mainly short-term) ─────────────────────────────────
                if days <= 7:
                    if rsi < 30:   score += 2
                    elif rsi > 70: score -= 2
                    elif rsi < 40: score += 1
                    elif rsi > 60: score -= 1
                elif days <= 30:
                    if rsi < 35:   score += 1
                    elif rsi > 65: score -= 1

                # ── Momentum weighted by timeframe ──────────────────────────
                if days <= 3:
                    score += 2 if mom_1w > 2 else 1 if mom_1w > 0.5 else -1 if mom_1w < -0.5 else -2 if mom_1w < -2 else 0
                elif days <= 7:
                    score += 1 if mom_1w > 1 else -1 if mom_1w < -1 else 0
                elif days <= 30:
                    score += 1 if mom_1m > 2 else -1 if mom_1m < -2 else 0
                elif days <= 90:
                    score += 1 if mom_3m > 5 else -1 if mom_3m < -5 else 0
                else:
                    score += 1 if mom_6m > 8 else -1 if mom_6m < -8 else 0

                # ── Structural long-term bias ───────────────────────────────
                if days >= 90:
                    if ak == "XAU":
                        score += 2   # de-dollarization + CB buying + debt trap
                    elif ak == "XAG":
                        score += 1   # industrial + monetary dual role
                    elif ak == "BTC":
                        score += 1   # scarcity + halving cycle
                    elif ak == "CL":
                        score -= 1   # energy transition headwind long-term
                if days >= 180:
                    if ak == "XAU":
                        score += 1   # extra structural premium
                    elif ak == "USDVND":
                        score += 1   # long-term VND depreciation trend
                if days >= 730:
                    if ak == "XAU":
                        score += 1   # 2Y+: macro debt cycle, CB accumulation compounding
                    elif ak == "BTC":
                        score += 1   # 2Y+: post-halving bull phase, institutional adoption
                    elif ak == "CL":
                        score -= 1   # 2Y+: EV + renewables accelerating
                if days >= 1095:
                    if ak == "XAU":
                        score += 1   # 3Y+: currency debasement, de-dollarization deepening
                    elif ak == "XAG":
                        score += 1   # 3Y+: solar/green energy industrial demand surge
                    elif ak == "HG":
                        score += 1   # 3Y+: EV + grid infra copper supercycle

                # ── Convert score → direction ───────────────────────────────
                if score >= 5:
                    dir_out, color = "🚀 Tăng mạnh",  "#3fb950"
                elif score >= 2:
                    dir_out, color = "⬆️ Tăng",        "#76c3a0"
                elif score >= 0:
                    dir_out, color = "↗️ Sideway+",    "#ffa657"
                elif score >= -2:
                    dir_out, color = "↘️ Sideway−",    "#ff7b54"
                elif score >= -4:
                    dir_out, color = "⬇️ Giảm",        "#f85149"
                else:
                    dir_out, color = "💣 Giảm mạnh",  "#da3633"

                asset_result[days] = {"dir": dir_out, "color": color, "score": score}

            result[ak] = asset_result

        except Exception:
            result[ak] = {p: {"dir": "N/A", "color": "#8b949e", "score": 0} for p in PERIODS}

    return result


@st.cache_data(ttl=1800, show_spinner=False)
def generate_expert_narrative(
    news_titles: tuple,
    hawk_phrases: tuple,
    dove_phrases: tuple,
    fedspeak_score: int,
    fedspeak_label: str,
    cot_net: int,
    cot_rank: int,
    cot_chg_1w: int,
    futures_score: int,
    hawk_composite: int,
    hawkish_count: int,
    total_expert_score: int,
    xau_price: float,
    walcl_chg: float,
    yield2y_val: float,
    yield2y_chg: float,
    price_1w_chg: float,
    price_1m_chg: float,
    price_3m_chg: float,
) -> str:
    """Gọi Gemini AI (REST API trực tiếp) để tổng hợp phân tích chuyên gia."""
    import requests as _req

    try:
        _api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return "⚠️ Chưa cấu hình GOOGLE_API_KEY trong Streamlit Secrets (Settings → Secrets)."

    news_str  = "\n".join(f"  • {t}" for t in news_titles[:6]) or "  • Không có tin tức mới"
    hawk_str  = "\n".join(f"  • {p}" for p in hawk_phrases[:5]) or "  • Không phát hiện"
    dove_str  = "\n".join(f"  • {p}" for p in dove_phrases[:5]) or "  • Không phát hiện"

    hawk_dir = ("DIỀU HÂU mạnh 🦅🦅" if fedspeak_score <= -2 else
                "Hơi diều hâu 🦅"     if fedspeak_score < 0  else
                "BỒ CÂU mạnh 🕊️🕊️"   if fedspeak_score >= 2 else
                "Hơi bồ câu 🕊️"       if fedspeak_score > 0  else "Trung lập ↔️")
    exp_dir  = ("TÍCH CỰC — BULLISH 🟢" if total_expert_score >= 2 else
                "TIÊU CỰC — BEARISH 🔴" if total_expert_score <= -2 else "TRUNG TÍNH ⚪")
    walcl_note = ("QT đang thu hẹp bảng cân đối — rút thanh khoản" if walcl_chg < -3 else
                  "QE / bơm thanh khoản"                             if walcl_chg > 3  else
                  "Ổn định")
    cot_trend = "cá mập đang MUA" if cot_chg_1w > 0 else "cá mập đang BÁN"

    prompt = f"""Bạn là chuyên gia phân tích vàng XAU/USD cấp độ hedge fund, 20 năm kinh nghiệm thực chiến. Nhiệm vụ: đọc toàn bộ dữ liệu bên dưới và đưa ra PHÂN TÍCH TỔNG HỢP quyết đoán bằng tiếng Việt.

══════ GIÁ & MOMENTUM LỊCH SỬ ══════
Giá XAU hiện tại: ${xau_price:,.2f}
Thay đổi: 1 tuần {price_1w_chg:+.2f}% | 1 tháng {price_1m_chg:+.2f}% | 3 tháng {price_3m_chg:+.2f}%

══════ FED & CHÍNH SÁCH TIỀN TỆ ══════
Tin FED mới nhất:
{news_str}
Fedspeak tone: {hawk_dir} (score {fedspeak_score:+d}/3)
Phrases HAWKISH phát hiện: {hawk_str}
Phrases DOVISH phát hiện: {dove_str}
Hawk-o-meter: {hawkish_count}/5 tín hiệu hawkish (composite {hawk_composite:+d}/5)
WALCL Fed Balance Sheet: {walcl_chg:+.1f}% YoY — {walcl_note}
Yield 2Y: {yield2y_val:.2f}% (thay đổi 20d: {yield2y_chg:+.2f}%)
Fed Funds Futures: score {futures_score:+d}

══════ ĐỊNH VỊ CÁ MẬP (CFTC COT) ══════
Net position XAU: {cot_net:,} hợp đồng | Percentile 52W: {cot_rank}%
Thay đổi 1 tuần: {cot_chg_1w:+,} hợp đồng → {cot_trend}

══════ TỔNG HỢP HỆ THỐNG ══════
Expert Score: {total_expert_score:+d}/8 → {exp_dir}

══════ YÊU CẦU ══════
Viết theo cấu trúc sau, DÙNG số liệu thực tế ở trên, KHÔNG viết chung chung:

## 🎯 VERDICT
[Một câu dứt khoát: TĂNG / GIẢM / ĐI NGANG trong ngắn hạn (1-2 tuần) và trung hạn (1-3 tháng). Lý do chính trong một câu.]

## 🏦 FED ĐANG LÀM GÌ
[Giải mã ngôn ngữ FED từ tin tức thực tế trên — họ đang thực sự muốn gì, tác động đến vàng ra sao. 2-3 câu.]

## 🐋 CÁ MẬP ĐANG ĐẶT CỬA THẾ NÀO
[Phân tích COT: vị thế hiện tại nói lên điều gì, {cot_rank}% percentile có ý nghĩa gì, xu hướng đang thay đổi ra sao. 2-3 câu.]

## ⚡ TÍN HIỆU MÂU THUẪN HAY ĐỒNG THUẬN
[FED và cá mập đang cùng chiều hay ngược chiều? Điều đó quan trọng như thế nào? 2 câu.]

## 🎯 HÀNH ĐỘNG CỤ THỂ
[Vùng giá tham khảo để vào lệnh, mức stop loss logic, cỡ vị thế. Thực chiến, cụ thể. 2-3 câu.]

Phong cách: Briefing sáng hedge fund — sắc bén, số liệu thực, không sáo rỗng."""

    _payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.65, "maxOutputTokens": 1200},
    }

    # Bước 1: Lấy danh sách model thực tế từ API (không đoán mò tên)
    _available_models = []
    try:
        _list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={_api_key}&pageSize=50"
        _lr = _req.get(_list_url, timeout=10)
        if _lr.status_code == 200:
            for _m in _lr.json().get("models", []):
                _mname = _m.get("name", "").replace("models/", "")
                _methods = _m.get("supportedGenerationMethods", [])
                if "generateContent" in _methods and _mname:
                    _available_models.append(_mname)
    except Exception:
        pass

    # Bước 2: Ưu tiên model flash nhỏ (ít quota nhất), lọc bỏ preview/exp/vision nếu có
    def _model_priority(name):
        if "flash-lite" in name or "flash-8b" in name:
            return 0
        if "flash" in name and "pro" not in name:
            return 1
        if "pro" in name:
            return 2
        return 3

    if _available_models:
        _available_models.sort(key=_model_priority)
    else:
        # Fallback nếu ListModels thất bại
        _available_models = [
            "gemini-2.0-flash-lite", "gemini-2.0-flash",
            "gemini-1.5-flash-8b", "gemini-1.5-flash",
        ]

    _all_errs = [f"Models khả dụng: {_available_models[:6]}"]
    for _mn in _available_models[:6]:  # thử tối đa 6 model
        try:
            _url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{_mn}:generateContent?key={_api_key}"
            )
            _r = _req.post(_url, json=_payload, timeout=30)
            if _r.status_code == 200:
                _data = _r.json()
                _candidates = _data.get("candidates", [])
                if _candidates:
                    return _candidates[0]["content"]["parts"][0]["text"]
                else:
                    _all_errs.append(f"{_mn}: OK nhưng không có candidates")
            elif _r.status_code == 429:
                _all_errs.append(f"{_mn}: 429 quota — thử model tiếp theo")
            else:
                _all_errs.append(f"{_mn}: HTTP {_r.status_code} — {_r.text[:150]}")
        except Exception as _e:
            _all_errs.append(f"{_mn}: {_e}")
            continue
    return "⚠️ Lỗi Gemini API:\n" + "\n".join(f"  • {e}" for e in _all_errs)


def render_expert_tab(macro: dict, fred_data: dict):
    """
    🧠 Tab Phân Tích Chuyên Gia — 17 modules · 4 blocks.
    Hedge fund-quality expert lens: Fed Intel + Market Structure + Smart Money + Verdict.
    """

    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px 0;'>
        <h2 style='color:#FFD700;font-size:1.6rem;font-weight:700;letter-spacing:1px;'>
            🧠 PHÂN TÍCH CHUYÊN GIA
        </h2>
        <p style='color:#8b949e;font-size:0.88rem;'>
        Góc nhìn của hedge fund · Cá mập vs Đám đông · FED Intelligence · Smart Money Flow
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── Fetch tất cả data ──────────────────────────────────────────────────────
    with st.spinner("📡 Đang tải dữ liệu chuyên gia..."):
        fed_news     = fetch_fed_news()
        fedspeak     = analyze_fedspeak(fed_news)
        fomc_cal     = fetch_fomc_calendar()
        futures_data = fetch_fed_funds_futures()
        gold_jpy     = fetch_gold_jpy_signal()
        max_pain     = calc_expert_max_pain()
        cot_xau      = fetch_cot_data("XAU")

    # ── Tính sơ bộ Expert Score + Hawk metrics (dùng cho AI narrative ở đầu) ─
    _ai_fed_es  = fedspeak.get("score", 0)
    _ai_fut_es  = futures_data.get("score", 0) if futures_data.get("ok") else 0
    _ai_cot_es  = cot_xau.get("score", 0)      if cot_xau.get("ok")      else 0
    _ai_jpy_es  = gold_jpy.get("score", 0)     if gold_jpy.get("ok")     else 0
    _ai_prelim  = max(-8, min(8, _ai_fed_es + _ai_fut_es + _ai_cot_es + _ai_jpy_es))

    _ai_hk_fed    = -1 if _ai_fut_es < -1 else (1 if _ai_fut_es > 1 else 0)
    _ai_hk_speech = fedspeak.get("score", 0)
    _ai_walcl_ok  = "fed_balance" in fred_data and len(fred_data["fed_balance"]) >= 52
    _ai_wb_chg    = ((float(fred_data["fed_balance"].iloc[-1]) /
                      float(fred_data["fed_balance"].iloc[-52]) - 1) * 100
                     if _ai_walcl_ok else 0.0)
    _ai_hk_walcl  = -1 if _ai_wb_chg < -5 else 0
    _ai_y2y_s     = fred_data["yield2y"].dropna() if "yield2y" in fred_data else None
    _ai_y2y_val   = float(_ai_y2y_s.iloc[-1]) if _ai_y2y_s is not None and len(_ai_y2y_s) >= 1 else 0.0
    _ai_y2y_rise  = (float(_ai_y2y_s.iloc[-1]) - float(_ai_y2y_s.iloc[-20])
                     if _ai_y2y_s is not None and len(_ai_y2y_s) >= 20 else 0.0)
    _ai_hk_yield  = -1 if _ai_y2y_rise > 0.15 else 0
    _ai_cot_rank  = int(cot_xau.get("pct_rank", 50)) if cot_xau.get("ok") else 50
    _ai_cot_chg1w = int(cot_xau.get("chg_1w",   0))  if cot_xau.get("ok") else 0
    _ai_hk_cot    = (-1 if _ai_cot_rank > 80 and _ai_cot_chg1w < 0
                     else (1 if _ai_cot_rank < 20 and _ai_cot_chg1w > 0 else 0))
    _ai_hawk_comp = max(-5, min(5, _ai_hk_fed + _ai_hk_speech + _ai_hk_walcl
                                   + _ai_hk_yield + _ai_hk_cot))
    _ai_hawk_cnt  = sum(1 if s < 0 else 0
                        for s in [_ai_hk_fed, _ai_hk_speech, _ai_hk_walcl,
                                  _ai_hk_yield, _ai_hk_cot])

    # Giá XAU + momentum lịch sử
    try:
        _ai_px, _ = fetch_gold()
        _ai_cur   = float(_ai_px.iloc[-1])
        _ai_p1w   = ((_ai_cur / float(_ai_px.iloc[-5])  - 1) * 100) if len(_ai_px) >= 5  else 0.0
        _ai_p1m   = ((_ai_cur / float(_ai_px.iloc[-20]) - 1) * 100) if len(_ai_px) >= 20 else 0.0
        _ai_p3m   = ((_ai_cur / float(_ai_px.iloc[-60]) - 1) * 100) if len(_ai_px) >= 60 else 0.0
    except Exception:
        _ai_cur = _ai_p1w = _ai_p1m = _ai_p3m = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # AI EXPERT NARRATIVE — Hero section đầu trang
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 🧠 Phân Tích Chuyên Gia AI")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.84rem;'>"
        "Gemini AI tổng hợp tin FED, định vị cá mập, diều hâu/bồ câu và momentum — "
        "<b style='color:#FFD700;'>đưa ra verdict TĂNG/GIẢM/ĐI NGANG như hedge fund analyst</b>. "
        "Cache 30 phút.</p>",
        unsafe_allow_html=True,
    )
    with st.spinner("🧠 Gemini AI đang phân tích toàn bộ tín hiệu..."):
        _ai_text = generate_expert_narrative(
            news_titles        = tuple(item.get("title", "") for item in fed_news[:6]),
            hawk_phrases       = tuple(d["phrase"] for d in fedspeak.get("detected", [])
                                       if d.get("type") == "hawk"),
            dove_phrases       = tuple(d["phrase"] for d in fedspeak.get("detected", [])
                                       if d.get("type") == "dove"),
            fedspeak_score     = int(_ai_fed_es),
            fedspeak_label     = fedspeak.get("label", ""),
            cot_net            = int(cot_xau.get("net", 0)) if cot_xau.get("ok") else 0,
            cot_rank           = _ai_cot_rank,
            cot_chg_1w         = _ai_cot_chg1w,
            futures_score      = int(_ai_fut_es),
            hawk_composite     = int(_ai_hawk_comp),
            hawkish_count      = int(_ai_hawk_cnt),
            total_expert_score = int(_ai_prelim),
            xau_price          = round(_ai_cur, 2),
            walcl_chg          = round(_ai_wb_chg, 2),
            yield2y_val        = round(_ai_y2y_val, 3),
            yield2y_chg        = round(_ai_y2y_rise, 3),
            price_1w_chg       = round(_ai_p1w, 2),
            price_1m_chg       = round(_ai_p1m, 2),
            price_3m_chg       = round(_ai_p3m, 2),
        )
    st.markdown(
        f"<div style='background:#0d1117;border:1px solid #30363d;border-radius:12px;"
        f"padding:22px 26px;line-height:1.9;font-size:0.88rem;color:#e6edf3;'>"
        f"{_ai_text.replace(chr(10), '<br>')}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#8b949e;font-size:0.73rem;margin-top:4px;'>"
        "⚠️ Phân tích AI mang tính tham khảo — không phải khuyến nghị đầu tư. "
        "Powered by Google Gemini 1.5 Flash.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 1: FED INTELLIGENCE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 🏦 BLOCK 1: FED INTELLIGENCE")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem;'>"
        "FED là con cá mập lớn nhất thị trường. Họ kiểm soát thanh khoản toàn cầu, "
        "nhưng ngôn ngữ của họ thường <b style='color:#FFD700;'>nói một đằng làm một nẻo</b>. "
        "Đây là bộ công cụ giải mã Fed.</p>",
        unsafe_allow_html=True,
    )

    b1c1, b1c2 = st.columns([3, 2])

    # Module 1: FED News Feed
    with b1c1:
        st.markdown("### 📰 Module 1: FED News Feed (Mới nhất · Dịch tự động 🇻🇳)")
        if fed_news:
            news_slice  = fed_news[:8]
            raw_titles  = tuple(item.get("title", "") for item in news_slice)
            vi_titles   = translate_to_vi_batch(raw_titles)

            for item, vi_title in zip(news_slice, vi_titles):
                date_str  = item.get("date", "")[:16] if item.get("date") else ""
                en_title  = item.get("title", "")
                # Đánh màu dựa trên từ khóa tiếng Anh gốc (chính xác hơn)
                t_lower = en_title.lower()
                if any(w in t_lower for w in ["rate cut","easing","pivot","dovish","balanced"]):
                    dot_color = "#3fb950"
                elif any(w in t_lower for w in ["hike","hawkish","higher","restrict","inflation"]):
                    dot_color = "#f85149"
                else:
                    dot_color = "#8b949e"
                # Nếu dịch thất bại (trả về y chang tiếng Anh), không hiển thị 2 dòng giống nhau
                show_en = (vi_title.strip().lower() != en_title.strip().lower())
                en_row  = (
                    f"<span style='color:#484f58;font-size:0.70rem;font-style:italic;'>"
                    f"{en_title}</span><br>"
                ) if show_en else ""
                st.markdown(
                    f"<div style='border-left:3px solid {dot_color};padding:5px 10px;margin:4px 0;"
                    f"background:#161b22;border-radius:0 6px 6px 0;'>"
                    f"<span style='color:{dot_color};font-size:0.70rem;'>{date_str}</span><br>"
                    f"<span style='color:#e6edf3;font-size:0.83rem;font-weight:600;'>{vi_title}</span><br>"
                    f"{en_row}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("📡 Không tải được FED RSS feed. Kiểm tra kết nối.")

    # Module 2: Fedspeak Decoder
    with b1c2:
        st.markdown("### 🔍 Module 2: Fedspeak Decoder")
        st.markdown(
            f"<div style='background:{hex_rgba(fedspeak['color'], 0.12)};border:1px solid "
            f"{hex_rgba(fedspeak['color'], 0.4)};border-radius:10px;padding:14px;'>"
            f"<div style='color:{fedspeak['color']};font-size:1.3rem;font-weight:700;'>"
            f"{fedspeak['label']}</div>"
            f"<div style='color:#8b949e;font-size:0.8rem;margin-top:4px;'>"
            f"🦅 Hawkish signals: {fedspeak['hawk']} &nbsp;·&nbsp; "
            f"🕊️ Dovish signals: {fedspeak['dove']}</div></div>",
            unsafe_allow_html=True,
        )
        if fedspeak["detected"]:
            st.markdown("**Tín hiệu phát hiện:**")
            for sig in fedspeak["detected"][:6]:
                t_color = "#3fb950" if sig["type"] == "dove" else "#f85149"
                st.markdown(
                    f"<div style='margin:3px 0;padding:6px 10px;background:#161b22;"
                    f"border-radius:6px;border-left:3px solid {t_color};font-size:0.8rem;'>"
                    f"<b style='color:{t_color};'>{sig['label']}</b> — "
                    f"<i style='color:#8b949e;'>\"{sig['phrase']}\"</i><br>"
                    f"<span style='color:#e6edf3;'>{sig['meaning']}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='color:#8b949e;font-size:0.85rem;padding:8px;'>"
                "⚠️ Không tìm thấy từ khóa Fedspeak trong 15 tin tức mới nhất. "
                "Fed đang im lặng hoặc dữ liệu chưa cập nhật.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Module 3: FED nói vs Thị trường
        st.markdown("### 📊 Module 3: FED nói vs Thị trường")
        if futures_data.get("ok"):
            fd = futures_data
            st.markdown(
                f"<div style='background:{hex_rgba(fd['color'], 0.12)};border:1px solid "
                f"{hex_rgba(fd['color'], 0.4)};border-radius:8px;padding:10px;'>"
                f"<b style='color:{fd['color']};'>{fd['market_view']}</b><br>"
                f"<span style='color:#8b949e;font-size:0.78rem;'>"
                f"Fed Funds thực tế: <b style='color:#e6edf3;'>{fd['fedfunds']:.2f}%</b> &nbsp;·&nbsp; "
                f"T-Bill 3M (market implied): <b style='color:#e6edf3;'>{fd['irx']:.2f}%</b><br>"
                f"Kỳ vọng thị trường: <b style='color:{fd['color']};'>{'Cắt ' if fd['diff']>0 else 'Tăng '}"
                f"{abs(fd['diff']):.2f}%</b></span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<span style='color:#8b949e;font-size:0.75rem;'>"
                "⚡ Khi Fed nói 'higher for longer' nhưng thị trường price in cuts → "
                "thị trường thường đúng 6-12 tháng trước.</span>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Không tải được dữ liệu CME futures.")

    st.markdown("---")

    b1c3, b1c4 = st.columns([1, 2])

    # Module 4: Dissent Tracker
    with b1c3:
        st.markdown("### 🗳️ Module 4: Dissent Tracker")
        st.markdown(
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;'>"
            "<p style='color:#8b949e;font-size:0.82rem;'>FOMC có 12 thành viên bỏ phiếu. "
            "Dissent (bất đồng) là tín hiệu sớm về thay đổi chính sách.</p>"
            "<table style='width:100%;font-size:0.8rem;color:#e6edf3;'>"
            "<tr><td>🟢 0 dissent</td><td style='color:#3fb950;'>Đồng thuận — chính sách ổn định</td></tr>"
            "<tr><td>🟡 1–2 dissent</td><td style='color:#ffa657;'>Bất đồng nhỏ — pivot đang được tranh luận</td></tr>"
            "<tr><td>🔴 3+ dissent</td><td style='color:#f85149;'>Bất đồng lớn — thay đổi chính sách gần</td></tr>"
            "</table>"
            "<hr style='border-color:#30363d;margin:8px 0;'>"
            "<p style='color:#ffa657;font-size:0.82rem;'>📌 Lưu ý: Dissent data cần "
            "đăng ký FRED API. Theo dõi trực tiếp tại "
            "<b>federalreserve.gov/monetarypolicy/fomccalendars.htm</b></p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Module 5: FED Hidden Agenda
    with b1c4:
        st.markdown("### 🕵️ Module 5: FED Hidden Agenda — $36T Debt Trap")
        # Fetch M2 và debt context từ FRED data
        m2_trend  = ""
        debt_note = ""
        if "m2" in fred_data:
            m2s = fred_data["m2"].dropna()
            if len(m2s) >= 12:
                m2_1y_chg = (m2s.iloc[-1] / m2s.iloc[-12] - 1) * 100
                m2_trend  = f"M2 thay đổi 12 tháng: **{m2_1y_chg:+.1f}%**"
        if "fedfunds" in fred_data:
            ff = float(fred_data["fedfunds"].iloc[-1])
            debt_note = f"Với lãi suất {ff:.2f}%, chi phí nợ quốc gia Mỹ ~$36T ≈ **${36000*ff/100:.0f}B/năm**"

        st.markdown(
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;'>"
            "<p style='color:#FFD700;font-weight:700;font-size:0.92rem;'>🧠 Luận điểm: FED bị bẫy bởi nợ công</p>"
            "<ul style='color:#e6edf3;font-size:0.83rem;line-height:1.7;margin:0;padding-left:18px;'>"
            "<li>Mỹ có <b style='color:#f85149;'>$36 nghìn tỷ</b> nợ công. Lãi suất cao = chi phí lãi vay khổng lồ.</li>"
            "<li>Fed <b>không thể</b> giữ lãi suất cao mãi mãi — áp lực chính trị + tài khóa quá lớn.</li>"
            "<li>Khi Fed buộc phải cắt lãi suất (dù lạm phát chưa về 2%) → <b style='color:#3fb950;'>Gold tăng mạnh</b>.</li>"
            "<li>Yield Curve Control (YCC) như Nhật Bản là kịch bản cuối cùng → hyperinflationary cho vàng.</li>"
            "<li>BRICS dedollarization + Central Bank buying tạo <b>sàn cầu</b> dài hạn cho gold.</li>"
            "</ul>"
            f"<hr style='border-color:#30363d;'>"
            f"<p style='color:#8b949e;font-size:0.8rem;'>{m2_trend}</p>"
            f"<p style='color:#8b949e;font-size:0.8rem;'>{debt_note}</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Module 6: FOMC Calendar
    st.markdown("### 📅 Module 6: FOMC Calendar 2025–2026")
    today_date = datetime.now().date()
    next_mtg   = next((m for m in fomc_cal if m["status"] in ("upcoming","soon","today")), None)

    if next_mtg:
        days_str = (
            "HÔM NAY!" if next_mtg["days_away"] == 0
            else f"còn {next_mtg['days_away']} ngày"
        )
        alert_color = "#f85149" if next_mtg["days_away"] <= 7 else "#ffa657" if next_mtg["days_away"] <= 30 else "#3fb950"
        st.markdown(
            f"<div style='background:{hex_rgba(alert_color,0.12)};border:1px solid "
            f"{hex_rgba(alert_color,0.4)};border-radius:8px;padding:10px;margin-bottom:8px;'>"
            f"<b style='color:{alert_color};'>⏰ Cuộc họp FOMC tiếp theo: {next_mtg['label']} "
            f"({next_mtg['date'].strftime('%d/%m/%Y')}) — {days_str}</b></div>",
            unsafe_allow_html=True,
        )

    cal_cols = st.columns(8)
    for i, m in enumerate(fomc_cal):
        col = cal_cols[i % 8]
        if m["status"] == "done":
            bg, txt = "#1c2128", "#484f58"
        elif m["status"] == "today":
            bg, txt = hex_rgba("#FFD700", 0.2), "#FFD700"
        elif m["status"] == "soon":
            bg, txt = hex_rgba("#f85149", 0.15), "#f85149"
        else:
            bg, txt = hex_rgba("#3fb950", 0.10), "#3fb950"
        col.markdown(
            f"<div style='background:{bg};border-radius:6px;padding:6px 4px;text-align:center;"
            f"margin:2px 0;font-size:0.72rem;color:{txt};font-weight:600;'>"
            f"{m['label']}<br>"
            f"<span style='font-size:0.62rem;font-weight:400;'>{m['date'].strftime('%d/%m')}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 2: MARKET STRUCTURE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 🏗️ BLOCK 2: MARKET STRUCTURE")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem;'>"
        "Cá mập (market maker + hedge fund lớn) cần <b style='color:#FFD700;'>tạo thanh khoản</b> "
        "trước khi di chuyển thị trường. Hiểu cấu trúc thị trường = không bị bẫy.</p>",
        unsafe_allow_html=True,
    )

    b2c1, b2c2 = st.columns(2)

    # Module 7: Narrative Cycle
    with b2c1:
        st.markdown("### 🎭 Module 7: Narrative Cycle")
        # Phân tích narrative dựa trên macro signals
        try:
            gold_90d = yf.Ticker("GC=F").history(period="90d")["Close"]
            if len(gold_90d) >= 20:
                g_cur  = float(gold_90d.iloc[-1])
                g_ma20 = float(gold_90d.rolling(20).mean().iloc[-1])
                g_ma50 = float(gold_90d.rolling(50).mean().iloc[-1]) if len(gold_90d) >= 50 else g_ma20
                g_chg  = (g_cur / float(gold_90d.iloc[-20]) - 1) * 100

                if g_cur > g_ma20 > g_ma50 and g_chg > 5:
                    narrative = "🚀 BULL MOMENTUM — Đà tăng mạnh"
                    narr_detail = (
                        "Giá vàng đang trong uptrend rõ ràng. Narrative thị trường đang tập trung vào "
                        "các yếu tố bullish (lạm phát, nợ công, địa chính trị). "
                        "**Rủi ro:** đám đông đang bullish — cá mập có thể shake out trước khi tăng tiếp."
                    )
                    narr_color = "#3fb950"
                elif g_cur < g_ma20 and g_chg < -3:
                    narrative = "🩸 BEAR PRESSURE — Áp lực giảm"
                    narr_detail = (
                        "Giá vàng đang chịu áp lực. Narrative tập trung vào dollar mạnh, lãi suất cao. "
                        "**Cơ hội:** cá mập thường mua vào khi đám đông bán hoảng loạn."
                    )
                    narr_color = "#f85149"
                elif g_cur > g_ma20 and abs(g_chg) < 2:
                    narrative = "🔄 ACCUMULATION — Tích lũy ngầm"
                    narr_detail = (
                        "Vàng đi ngang nhưng trên MA20. Đây thường là giai đoạn cá mập tích lũy âm thầm "
                        "trước khi breakout. Volume thấp + sideways = setup tốt cho long."
                    )
                    narr_color = "#ffa657"
                else:
                    narrative = "➡️ TRANSITION — Đang chuyển pha"
                    narr_detail = (
                        "Thị trường đang trong giai đoạn chuyển tiếp. Chưa đủ bằng chứng xác định "
                        "xu hướng. Giảm size, chờ tín hiệu xác nhận."
                    )
                    narr_color = "#8b949e"
            else:
                narrative = "N/A"
                narr_detail = "Không đủ dữ liệu."
                narr_color = "#8b949e"
        except Exception:
            narrative = "N/A"
            narr_detail = "Lỗi kết nối dữ liệu."
            narr_color = "#8b949e"

        st.markdown(
            f"<div style='background:{hex_rgba(narr_color,0.12)};border:1px solid "
            f"{hex_rgba(narr_color,0.4)};border-radius:10px;padding:14px;'>"
            f"<b style='color:{narr_color};font-size:1.05rem;'>{narrative}</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown(narr_detail)

        st.markdown("---")
        # Module 9: Options Max Pain
        st.markdown("### 💀 Module 9: Options Max Pain (GLD OpEx)")
        if max_pain.get("ok"):
            mp = max_pain
            diff_color = "#3fb950" if mp["diff_pct"] > 0 else "#f85149"
            st.markdown(
                f"<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;'>"
                f"<b style='color:#FFD700;'>Expiry: {mp['expiry']}</b><br>"
                f"<table style='width:100%;font-size:0.83rem;margin-top:6px;'>"
                f"<tr><td style='color:#8b949e;'>Max Pain GLD</td>"
                f"<td style='color:#e6edf3;font-weight:700;'>${mp['max_pain_gld']}</td></tr>"
                f"<tr><td style='color:#8b949e;'>Gold equiv (~GLD×10)</td>"
                f"<td style='color:#e6edf3;font-weight:700;'>${mp['gold_equiv']:,.0f}</td></tr>"
                f"<tr><td style='color:#8b949e;'>Gold hiện tại (GC=F×10)</td>"
                f"<td style='color:#e6edf3;'>${mp['cur_gold']:,.0f}</td></tr>"
                f"<tr><td style='color:#8b949e;'>Khoảng cách đến Max Pain</td>"
                f"<td style='color:{diff_color};font-weight:700;'>{mp['diff_pct']:+.1f}%</td></tr>"
                f"</table>"
                f"<hr style='border-color:#30363d;margin:8px 0;'>"
                f"<span style='color:#8b949e;font-size:0.76rem;'>"
                f"⚡ Market maker thường kéo giá về Max Pain trước ngày đáo hạn options. "
                f"{'Pressure tăng lên ${:,.0f}'.format(mp['gold_equiv']) if mp['diff_pct'] > 0 else 'Pressure giảm về ${:,.0f}'.format(mp['gold_equiv'])}"
                f"</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Không tải được options chain GLD.")

    # Module 8: Liquidity Stop Hunt + Module 10: Crowd Trap
    with b2c2:
        st.markdown("### 🎯 Module 8: Liquidity & Stop Hunt Zones")
        try:
            gc_hist = yf.Ticker("GC=F").history(period="30d")
            if len(gc_hist) >= 10:
                gc_h  = gc_hist["High"]
                gc_l  = gc_hist["Low"]
                gc_c  = gc_hist["Close"]
                cur_g = float(gc_c.iloc[-1])

                r1 = round(float(gc_h.iloc[-20:].max()), 0) if len(gc_h) >= 20 else round(float(gc_h.max()), 0)
                r2 = round(float(gc_h.iloc[-5:].max()), 0)
                s1 = round(float(gc_l.iloc[-20:].min()), 0) if len(gc_l) >= 20 else round(float(gc_l.min()), 0)
                s2 = round(float(gc_l.iloc[-5:].min()), 0)

                st.markdown(
                    "<div style='background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;'>"
                    f"<p style='color:#8b949e;font-size:0.8rem;margin:0 0 8px 0;'>Giá hiện tại GC=F: "
                    f"<b style='color:#e6edf3;'>${cur_g:,.0f}</b></p>"
                    "<table style='width:100%;font-size:0.83rem;'>"
                    f"<tr><td style='color:#f85149;'>🔴 Stop Hunt cao (20D high)</td>"
                    f"<td style='color:#f85149;font-weight:700;'>${r1:,.0f}</td></tr>"
                    f"<tr><td style='color:#ff7b54;'>🟠 Kháng cự 5D high</td>"
                    f"<td style='color:#ff7b54;font-weight:700;'>${r2:,.0f}</td></tr>"
                    f"<tr><td style='color:#76c3a0;'>🟢 Hỗ trợ 5D low</td>"
                    f"<td style='color:#76c3a0;font-weight:700;'>${s2:,.0f}</td></tr>"
                    f"<tr><td style='color:#3fb950;'>🟩 Stop Hunt thấp (20D low)</td>"
                    f"<td style='color:#3fb950;font-weight:700;'>${s1:,.0f}</td></tr>"
                    "</table>"
                    "<hr style='border-color:#30363d;margin:8px 0;'>"
                    "<p style='color:#8b949e;font-size:0.77rem;'>"
                    "⚡ Cá mập thường bẫy stop loss tại đỉnh/đáy 20 ngày trước khi đảo chiều. "
                    "Break giả (False breakout) trên/dưới các mức này là tín hiệu entry tốt.</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("Không đủ dữ liệu lịch sử.")
        except Exception:
            st.info("Lỗi tải dữ liệu giá vàng.")

        st.markdown("---")
        # Module 10: Crowd Trap Detector
        st.markdown("### 🐑 Module 10: Crowd Trap Detector")
        # Dùng fedspeak score + cot data để detect consensus danger
        crowd_signals = []
        crowd_score = 0

        if fedspeak["score"] >= 2:
            crowd_signals.append(("⚠️", "Toàn bộ trader đang bullish theo Dovish Fed — đây thường là lúc cá mập bán", "#ffa657"))
            crowd_score -= 1
        if fedspeak["score"] <= -2:
            crowd_signals.append(("✅", "Tâm lý đám đông đang panic về Hawkish — cơ hội mua khi máu chảy", "#3fb950"))
            crowd_score += 1

        if cot_xau.get("ok"):
            cot_p = cot_xau.get("net_pct", 50)
            if cot_p > 80:
                crowd_signals.append(("⚠️", f"COT: Managed Money đang NET LONG cực đoan ({cot_p:.0f}th percentile) — rủi ro reversal", "#ffa657"))
                crowd_score -= 1
            elif cot_p < 20:
                crowd_signals.append(("✅", f"COT: Smart Money đang NET SHORT bất thường ({cot_p:.0f}th percentile) — accumulation phase", "#3fb950"))
                crowd_score += 1

        if max_pain.get("ok"):
            if max_pain["diff_pct"] < -2:
                crowd_signals.append(("⚡", f"Options: Đám đông đang long nhưng Max Pain thấp hơn {abs(max_pain['diff_pct']):.1f}% — gravitational pull giảm", "#ffa657"))

        if not crowd_signals:
            crowd_signals.append(("➡️", "Chưa phát hiện bẫy đám đông rõ ràng — thị trường đang cân bằng", "#8b949e"))

        for ico, txt, col in crowd_signals:
            st.markdown(
                f"<div style='background:{hex_rgba(col,0.10)};border-left:3px solid {col};"
                f"border-radius:0 6px 6px 0;padding:8px 12px;margin:4px 0;font-size:0.83rem;"
                f"color:#e6edf3;'>{ico} {txt}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 3: SMART MONEY
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 💼 BLOCK 3: SMART MONEY TRACKING")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem;'>"
        "Smart money = hedge fund lớn + ngân hàng trung ương + whale institutional. "
        "Theo dấu họ thay vì theo đám đông retail.</p>",
        unsafe_allow_html=True,
    )

    b3c1, b3c2 = st.columns(2)

    # Module 11: COT Pro vs Retail
    with b3c1:
        st.markdown("### 📊 Module 11: COT — Pro vs Retail")
        if cot_xau.get("ok"):
            cot = cot_xau
            mm_net    = cot.get("net", 0)
            mm_chg    = cot.get("chg4w", 0)
            mm_pct    = cot.get("net_pct", 50)
            cot_label = cot.get("label", "N/A")
            cot_color = cot.get("color", "#8b949e")

            st.markdown(
                f"<div style='background:{hex_rgba(cot_color,0.12)};border:1px solid "
                f"{hex_rgba(cot_color,0.4)};border-radius:10px;padding:14px;'>"
                f"<b style='color:{cot_color};font-size:1.05rem;'>{cot_label}</b><br>"
                f"<table style='width:100%;font-size:0.82rem;margin-top:8px;'>"
                f"<tr><td style='color:#8b949e;'>Managed Money NET</td>"
                f"<td style='color:#e6edf3;font-weight:700;'>{mm_net:+,d} hợp đồng</td></tr>"
                f"<tr><td style='color:#8b949e;'>Thay đổi 4 tuần</td>"
                f"<td style='color:{cot_color};'>{mm_chg:+,d}</td></tr>"
                f"<tr><td style='color:#8b949e;'>Percentile lịch sử</td>"
                f"<td style='color:{cot_color};'>{mm_pct:.0f}th</td></tr>"
                f"</table>"
                f"<hr style='border-color:#30363d;margin:8px 0;'>"
                f"<span style='color:#8b949e;font-size:0.77rem;'>"
                f"CFTC COT Report — Managed Money = hedge fund chuyên nghiệp. "
                f"Khi họ NET LONG cực đoan → thị trường dễ reversal. "
                f"Khi NET SHORT → smart money đang tích lũy.</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Không tải được COT data từ CFTC.")

        st.markdown("---")
        # Module 13: Gold vs JPY
        st.markdown("### 🇯🇵 Module 13: Gold vs JPY Safe Haven")
        if gold_jpy.get("ok"):
            gj = gold_jpy
            xau_clr = "#3fb950" if gj["xau_ret"] > 0 else "#f85149"
            jpy_clr = "#3fb950" if gj["jpy_ret"] > 0 else "#f85149"
            st.markdown(
                f"<div style='background:{hex_rgba(gj['color'],0.12)};border:1px solid "
                f"{hex_rgba(gj['color'],0.4)};border-radius:10px;padding:14px;'>"
                f"<b style='color:{gj['color']};font-size:1.0rem;'>{gj['signal']}</b><br>"
                f"<p style='color:#e6edf3;font-size:0.83rem;margin:6px 0;'>{gj['detail']}</p>"
                f"<table style='width:100%;font-size:0.8rem;'>"
                f"<tr><td style='color:#8b949e;'>Gold 5D change</td>"
                f"<td style='color:{xau_clr};'>{gj['xau_ret']:+.2f}%</td></tr>"
                f"<tr><td style='color:#8b949e;'>JPY 5D change (vs USD)</td>"
                f"<td style='color:{jpy_clr};'>{gj['jpy_ret']:+.2f}%</td></tr>"
                f"<tr><td style='color:#8b949e;'>USD/JPY hiện tại</td>"
                f"<td style='color:#e6edf3;'>{gj['usdjpy']:.2f}</td></tr>"
                f"</table></div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Không tải được dữ liệu USDJPY.")

    with b3c2:
        # Module 12: Central Bank Buying
        st.markdown("### 🏛️ Module 12: Central Bank Buying")
        # Dùng FRED data để infer CB demand
        m2_info  = ""
        cb_score = 0
        if "m2" in fred_data:
            m2s = fred_data["m2"].dropna()
            if len(m2s) >= 12:
                m2_1y_chg = (m2s.iloc[-1] / m2s.iloc[-12] - 1) * 100
                if m2_1y_chg > 5:
                    m2_info = f"M2 tăng {m2_1y_chg:.1f}% YoY — cung tiền mở rộng, central bank nới lỏng"
                    cb_score += 1
                elif m2_1y_chg < 0:
                    m2_info = f"M2 giảm {m2_1y_chg:.1f}% YoY — cung tiền co lại, thắt chặt tiền tệ"
                    cb_score -= 1
                else:
                    m2_info = f"M2 tăng {m2_1y_chg:.1f}% YoY — cung tiền tăng nhẹ, trung tính"

        st.markdown(
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;'>"
            "<p style='color:#FFD700;font-weight:700;'>🏦 Central Bank Gold Demand</p>"
            "<ul style='color:#e6edf3;font-size:0.82rem;line-height:1.7;margin:0;padding-left:18px;'>"
            "<li>WGC 2024: Ngân hàng TW toàn cầu mua <b style='color:#3fb950;'>1,045 tấn vàng</b> — năm thứ 3 liên tiếp >1000 tấn</li>"
            "<li>Trung Quốc (PBoC): Tiếp tục tăng dự trữ vàng, giảm trái phiếu Mỹ</li>"
            "<li>Nga, Ấn Độ, Ba Lan, Thổ Nhĩ Kỳ: Mua vào liên tục 2022–2025</li>"
            "<li>BRICS: Thảo luận về đồng tiền chung có hậu thuẫn vàng</li>"
            "<li>Mục tiêu dài hạn: Thoát phụ thuộc đồng USD → <b>structural bull case</b></li>"
            "</ul>"
            f"<hr style='border-color:#30363d;margin:8px 0;'>"
            f"<p style='color:#8b949e;font-size:0.8rem;'>{m2_info}</p>"
            "<p style='color:#3fb950;font-size:0.82rem;font-weight:700;'>"
            "📌 Tổng kết: Central bank buying tạo sàn cầu bền vững 3–5 năm cho vàng.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        # Module 14: Calendar Risk
        st.markdown("### 📆 Module 14: Calendar Risk — Sự kiện quan trọng")
        today_date2 = datetime.now().date()

        risk_events = []
        # FOMC events
        for m in fomc_cal:
            if 0 <= m["days_away"] <= 30:
                risk_events.append({
                    "name":  f"FOMC {m['label']}",
                    "date":  m["date"],
                    "days":  m["days_away"],
                    "type":  "fomc",
                    "color": "#ffa657",
                    "impact": "High — Fed có thể thay đổi lãi suất hoặc forward guidance",
                })

        # Options expiry (3rd Friday của tháng)
        import calendar as _cal
        cur_month = today_date2.month
        cur_year  = today_date2.year
        for _ in range(3):
            # Tính 3rd Friday
            cal_obj = _cal.monthcalendar(cur_year, cur_month)
            fridays = [week[4] for week in cal_obj if week[4] != 0]
            if len(fridays) >= 3:
                opex_day = fridays[2]
                opex_date = datetime(cur_year, cur_month, opex_day).date()
                days_to_opex = (opex_date - today_date2).days
                if 0 <= days_to_opex <= 30:
                    risk_events.append({
                        "name":  f"Options Expiry (3rd Fri)",
                        "date":  opex_date,
                        "days":  days_to_opex,
                        "type":  "opex",
                        "color": "#76c3a0",
                        "impact": "Medium — Max Pain gravity, potential stop hunt",
                    })
            if cur_month == 12:
                cur_month, cur_year = 1, cur_year + 1
            else:
                cur_month += 1

        risk_events.sort(key=lambda x: x["days"])

        if risk_events:
            for ev in risk_events[:5]:
                ev_color = ev["color"]
                days_str = "HÔM NAY" if ev["days"] == 0 else f"{ev['days']} ngày nữa"
                st.markdown(
                    f"<div style='background:{hex_rgba(ev_color,0.10)};border-left:3px solid {ev_color};"
                    f"border-radius:0 8px 8px 0;padding:8px 12px;margin:4px 0;'>"
                    f"<b style='color:{ev_color};font-size:0.85rem;'>{ev['name']}</b> "
                    f"<span style='color:#8b949e;font-size:0.78rem;'>— {ev['date'].strftime('%d/%m/%Y')} ({days_str})</span><br>"
                    f"<span style='color:#e6edf3;font-size:0.8rem;'>{ev['impact']}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='color:#8b949e;font-size:0.84rem;padding:8px;'>"
                "✅ Không có sự kiện rủi ro cao trong 30 ngày tới.</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 4: EXPERT VERDICT
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 🎯 BLOCK 4: EXPERT VERDICT")

    v1, v2, v3 = st.columns(3)

    # Module 15: Structural Bull Case
    with v1:
        st.markdown("### 🐂 Module 15: Structural Bull Case")
        real_rate_str = ""
        if "real_rate" in fred_data:
            rr = float(fred_data["real_rate"].dropna().iloc[-1])
            real_rate_str = f"Real Rate DFII10: **{rr:+.2f}%** {'— đang hỗ trợ gold' if rr < 0 else '— áp lực ngắn hạn'}"
        breakeven_str = ""
        if "breakeven5y" in fred_data:
            be = float(fred_data["breakeven5y"].dropna().iloc[-1])
            breakeven_str = f"Breakeven 5Y: **{be:.2f}%** {'— kỳ vọng lạm phát cao' if be > 2.5 else ''}"

        st.markdown(
            "<div style='background:#0d2818;border:1px solid #238636;border-radius:10px;padding:14px;'>"
            "<p style='color:#3fb950;font-weight:700;font-size:0.95rem;'>📈 THESIS TĂNG GIÁ DÀI HẠN</p>"
            "<ol style='color:#e6edf3;font-size:0.82rem;line-height:1.8;margin:0;padding-left:18px;'>"
            "<li><b>De-dollarization:</b> BRICS mua vàng thay USD reserves</li>"
            "<li><b>Debt trap:</b> Mỹ không thể trả $36T nợ mà không in tiền</li>"
            "<li><b>CB demand floor:</b> >1,000 tấn/năm tạo sàn cầu bền vững</li>"
            "<li><b>Real rates:</b> Khi Fed cắt → real rates giảm → gold tăng</li>"
            "<li><b>Geopolitical:</b> Ukraine, Trung Đông, Đài Loan → safe haven premium</li>"
            "<li><b>M2 expansion:</b> Dài hạn, cung tiền luôn tăng → purchasing power giảm</li>"
            "</ol>"
            f"<hr style='border-color:#238636;'>"
            f"<p style='color:#8b949e;font-size:0.78rem;'>{real_rate_str}</p>"
            f"<p style='color:#8b949e;font-size:0.78rem;'>{breakeven_str}</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Module 16: Black Swan Scenarios
    with v2:
        st.markdown("### 🦢 Module 16: Black Swan Scenarios")
        st.markdown(
            "<div style='background:#1a1000;border:1px solid #bb8009;border-radius:10px;padding:14px;'>"
            "<p style='color:#ffa657;font-weight:700;font-size:0.95rem;'>⚠️ 3 RỦI RO CHƯA ĐƯỢC ĐỊNH GIÁ</p>"
            "<div style='color:#e6edf3;font-size:0.82rem;line-height:1.7;'>"
            "<p><b style='color:#f85149;'>🔴 Scenario 1: FED Yield Curve Control (YCC)</b><br>"
            "Như Nhật Bản 2016–2022: Fed buộc phải mua trái phiếu không giới hạn để kiểm soát yield. "
            "→ Phá hủy USD, gold có thể x2–x3 trong 12 tháng.</p>"
            "<p><b style='color:#ffa657;'>🟠 Scenario 2: Treasury Market Dislocation</b><br>"
            "Nước ngoài ngừng mua/bán tháo trái phiếu Mỹ (như tháng 3/2020). "
            "→ Fed buộc phải QE khẩn cấp → hyperinflationary cho vàng.</p>"
            "<p><b style='color:#76c3a0;'>🟢 Scenario 3: Digital Gold Reserve</b><br>"
            "Một nước G20 công bố mua vàng/bitcoin làm dự trữ quốc gia. "
            "→ Domino effect → multiple central banks cạnh tranh mua → price shock.</p>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Module 17: Expert Verdict
    with v3:
        st.markdown("### 🏆 Module 17: Expert Verdict")

        # Tổng hợp tất cả signals
        total_expert_score = 0
        score_breakdown    = []

        # Fedspeak
        fed_es = fedspeak.get("score", 0)
        total_expert_score += fed_es
        score_breakdown.append(f"Fedspeak: {fed_es:+d}")

        # Market futures
        if futures_data.get("ok"):
            fut_es = futures_data.get("score", 0)
            total_expert_score += fut_es
            score_breakdown.append(f"Mkt Futures: {fut_es:+d}")

        # COT
        if cot_xau.get("ok"):
            cot_es = cot_xau.get("score", 0)
            total_expert_score += cot_es
            score_breakdown.append(f"COT: {cot_es:+d}")

        # Gold/JPY
        if gold_jpy.get("ok"):
            jpy_es = gold_jpy.get("score", 0)
            total_expert_score += jpy_es
            score_breakdown.append(f"Gold/JPY: {jpy_es:+d}")

        # Narrative
        narr_score = 1 if "BULL" in narrative else -1 if "BEAR" in narrative else 0
        total_expert_score += narr_score
        score_breakdown.append(f"Narrative: {narr_score:+d}")

        # Clamp
        total_expert_score = max(-8, min(8, total_expert_score))

        # ── Hawk-o-meter ─────────────────────────────────────────────────────
        # hawk_score_fed: dùng futures_data (market pricing of Fed hawkishness)
        _fs = futures_data.get("score", 0) if futures_data.get("ok") else 0
        hawk_score_fed    = -1 if _fs < -1 else (1 if _fs > 1 else 0)
        hawk_score_speech = fedspeak.get("score", 0)  # -3..+3 từ analyze_fedspeak
        # WALCL: QT/QE check
        _walcl_ok = "fed_balance" in fred_data and len(fred_data["fed_balance"]) >= 52
        if _walcl_ok:
            _wb = fred_data["fed_balance"]
            _wb_chg = (float(_wb.iloc[-1]) / float(_wb.iloc[-52]) - 1) * 100
        else:
            _wb_chg = 0
        hawk_score_walcl = -1 if _wb_chg < -5 else 0
        # Yield 2Y rising check
        _y2y_ok = "yield2y" in fred_data and len(fred_data["yield2y"]) >= 20
        if _y2y_ok:
            _y2y = fred_data["yield2y"].dropna()
            _y2y_rise = float(_y2y.iloc[-1]) - float(_y2y.iloc[-20]) if len(_y2y) >= 20 else 0
        else:
            _y2y_rise = 0
        hawk_score_yield2y = -1 if _y2y_rise > 0.15 else 0
        # COT: whale đảo chiều bearish
        _cot_rank = cot_xau.get("pct_rank", 50) if cot_xau.get("ok") else 50
        _cot_chg  = cot_xau.get("chg_1w", 0) if cot_xau.get("ok") else 0
        hawk_score_cot = (-1 if _cot_rank > 80 and _cot_chg < 0
                          else (1 if _cot_rank < 20 and _cot_chg > 0 else 0))

        hawk_composite = max(-5, min(5,
            hawk_score_fed + hawk_score_speech + hawk_score_walcl
            + hawk_score_yield2y + hawk_score_cot
        ))
        hawkish_signals_count = sum([
            1 if hawk_score_fed     < 0 else 0,
            1 if hawk_score_speech  < 0 else 0,
            1 if hawk_score_walcl   < 0 else 0,
            1 if hawk_score_yield2y < 0 else 0,
            1 if hawk_score_cot     < 0 else 0,
        ])

        # Cảnh báo khi nhiều tín hiệu cùng chiều hawkish
        if hawkish_signals_count >= 3:
            st.error(f"🚨 CẢNH BÁO DIỀU HÂU: {hawkish_signals_count}/5 tín hiệu đang cùng hướng BEARISH vàng — áp lực giảm giá mạnh!")
        elif hawkish_signals_count == 2:
            st.warning(f"⚠️ CHÚ Ý: {hawkish_signals_count}/5 tín hiệu diều hâu — theo dõi sát thêm 1-2 ngày tới.")

        # Visual Hawk-o-meter bar
        _level_pct = (hawk_composite + 5) / 10 * 100  # -5..+5 → 0%..100%
        _bar_color = ("#f85149" if hawk_composite <= -3 else
                      "#f9a825" if hawk_composite < 0 else
                      "#3fb950" if hawk_composite >= 3 else
                      "#FFD700")
        _hawk_label = ("Rất Hawkish 🦅🦅" if hawk_composite <= -4 else
                       "Hawkish 🦅"       if hawk_composite <= -2 else
                       "Hơi Hawkish"      if hawk_composite < 0 else
                       "Hơi Dovish"       if hawk_composite <= 1 else
                       "Dovish 🕊️"       if hawk_composite <= 3 else
                       "Rất Dovish 🕊️🕊️")
        st.markdown(f"""
<div style='background:#161b22;border-radius:10px;padding:14px 18px;border:1px solid #30363d;margin:8px 0'>
  <div style='display:flex;justify-content:space-between;margin-bottom:6px'>
    <span style='color:#f85149;font-size:0.82rem'>🦅 Diều Hâu (Bearish Gold)</span>
    <b style='color:{_bar_color};font-size:1.0rem'>{_hawk_label}</b>
    <span style='color:#3fb950;font-size:0.82rem'>Bồ Câu (Bullish Gold) 🕊️</span>
  </div>
  <div style='background:#0d1117;border-radius:6px;height:18px;position:relative'>
    <div style='background:{_bar_color};width:{_level_pct:.0f}%;height:100%;border-radius:6px'></div>
    <div style='position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#e6edf3;font-size:0.78rem;font-weight:bold'>Trung lập</div>
  </div>
  <div style='display:flex;justify-content:space-between;margin-top:4px;color:#8b949e;font-size:0.72rem'>
    <span>Mkt Futures: {'✅ Hawk' if hawk_score_fed < 0 else '○'}</span>
    <span>Fedspeak: {'✅ Hawk' if hawk_score_speech < 0 else '○'}</span>
    <span>QT (WALCL): {'✅ Hawk' if hawk_score_walcl < 0 else '○'}</span>
    <span>Yield 2Y↑: {'✅ Hawk' if hawk_score_yield2y < 0 else '○'}</span>
    <span>COT đảo chiều: {'✅ Hawk' if hawk_score_cot < 0 else '○'}</span>
  </div>
</div>
""", unsafe_allow_html=True)

        if total_expert_score >= 4:
            verdict       = "🚀 STRONG BULLISH"
            verdict_color = "#3fb950"
            verdict_text  = (
                "Đa số chỉ báo chuyên gia đều bullish. Fed đang dovish hơn thị trường nhận ra, "
                "smart money đang tích lũy, và cấu trúc thị trường hỗ trợ upside. "
                "**Chiến lược:** Tích lũy dần, giữ qua các pullback."
            )
        elif total_expert_score >= 2:
            verdict       = "📈 BULLISH"
            verdict_color = "#76c3a0"
            verdict_text  = (
                "Phần lớn tín hiệu hướng lên. Có một số rủi ro ngắn hạn nhưng xu hướng trung hạn "
                "vẫn tích cực. **Chiến lược:** Duy trì long position, tightened stop loss."
            )
        elif total_expert_score <= -4:
            verdict       = "🩸 STRONG BEARISH"
            verdict_color = "#f85149"
            verdict_text  = (
                "Fed hawkish hơn thị trường kỳ vọng, smart money đang thoát hàng, "
                "cấu trúc thị trường yếu. **Chiến lược:** Giảm exposure, chờ capitulation."
            )
        elif total_expert_score <= -2:
            verdict       = "📉 BEARISH"
            verdict_color = "#ff7b54"
            verdict_text  = (
                "Áp lực ngắn-trung hạn. Không nên mua thêm, xem xét chốt lời một phần. "
                "**Chiến lược:** Giảm size, đặt stop loss chặt hơn."
            )
        else:
            verdict       = "➡️ NEUTRAL / WAIT"
            verdict_color = "#8b949e"
            verdict_text  = (
                "Tín hiệu chuyên gia chưa rõ ràng. Thị trường đang trong giai đoạn cân bằng. "
                "**Chiến lược:** Giảm size, chờ catalyst rõ ràng trước khi vào lệnh lớn."
            )

        st.markdown(
            f"<div style='background:{hex_rgba(verdict_color,0.15)};border:2px solid "
            f"{hex_rgba(verdict_color,0.5)};border-radius:12px;padding:18px;'>"
            f"<div style='color:{verdict_color};font-size:1.4rem;font-weight:700;"
            f"text-align:center;margin-bottom:10px;'>{verdict}</div>"
            f"<div style='color:#8b949e;font-size:0.75rem;text-align:center;margin-bottom:10px;'>"
            f"Expert Score: {total_expert_score:+d}/8 &nbsp;·&nbsp; "
            + " · ".join(score_breakdown) +
            f"</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(verdict_text)

        st.markdown(
            "<div style='background:#1c2128;border:1px solid #30363d;border-radius:8px;"
            "padding:10px;margin-top:10px;'>"
            "<p style='color:#8b949e;font-size:0.76rem;margin:0;'>"
            "⚠️ <b>Disclaimer chuyên gia:</b> Phân tích này dựa trên dữ liệu công khai và mô hình "
            "định lượng. Thị trường luôn có thể surprise. Không phải khuyến nghị đầu tư. "
            "Quản lý rủi ro là ưu tiên số 1.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # BẢNG DỰ BÁO GIÁ VÀNG XAU THEO KỲ
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 🔮 Dự Báo Giá Vàng (XAU/USD) theo Kỳ")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.84rem;'>"
        "Giá mục tiêu từ mô hình tổng hợp: Holt-Winters · ARIMA · Momentum · Macro Expert Score. "
        "<b style='color:#FFD700;'>Kỳ càng dài → vùng dao động càng rộng</b> — bình thường về mặt thống kê.</p>",
        unsafe_allow_html=True,
    )

    _FORECAST_PERIODS = [1, 3, 7, 14, 21, 30, 60, 90, 180, 270, 365]

    with st.spinner("🔮 Chạy mô hình dự báo giá vàng..."):
        try:
            _xau_px, _ = fetch_gold()
            _cur   = float(_xau_px.iloc[-1])
            _ldate = str(_xau_px.index[-1])

            # Lấy tech score XAU từ bảng kết luận nhanh (đã cached, không tốn thêm request)
            _summary    = calc_summary_forecast()
            _xau_tech   = _summary.get("XAU", {})

            _fc_rows = []
            for _d in _FORECAST_PERIODS:
                try:
                    _tech_score = _xau_tech.get(_d, {}).get("score", 0)

                    # Blend signal theo timeframe:
                    # ≤30d  → 100% technical (MA/RSI/Momentum) — nhất quán bảng kết luận nhanh
                    # 60–90d → 50% tech + 50% expert score
                    # ≥180d → 100% expert score (Fedspeak, COT, Whale, macro)
                    if _d <= 30:
                        _mscore = max(-12, min(12, int(_tech_score * 1.5)))
                        _src    = "Kỹ thuật"
                    elif _d <= 90:
                        _mscore = max(-12, min(12, int((_tech_score + total_expert_score) * 0.75)))
                        _src    = "Kỹ thuật + Chuyên gia"
                    else:
                        _mscore = max(-12, min(12, int(total_expert_score * 1.5)))
                        _src    = "Chuyên gia"

                    _fm, _flo, _fhi = forecast(_xau_px.values, _ldate, _d, _mscore, "XAU", 0.5)
                    _tgt = float(_fm.iloc[-1])
                    _lo  = float(_flo.iloc[-1])
                    _hi  = float(_fhi.iloc[-1])
                    _chg = (_tgt / _cur - 1) * 100
                    _fc_rows.append((_d, _tgt, _lo, _hi, _chg, _src))
                except Exception:
                    pass
        except Exception:
            _fc_rows = []

    if _fc_rows:
        _hdr = (
            "<div style='overflow-x:auto'>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.84rem;'>"
            "<thead><tr style='background:#161b22;color:#8b949e;text-align:center;'>"
            "<th style='padding:8px 12px;text-align:left;border-bottom:1px solid #30363d;'>Kỳ dự báo</th>"
            "<th style='padding:8px;border-bottom:1px solid #30363d;'>Xu hướng</th>"
            "<th style='padding:8px;border-bottom:1px solid #30363d;'>Giá mục tiêu</th>"
            "<th style='padding:8px;border-bottom:1px solid #30363d;'>Thay đổi %</th>"
            "<th style='padding:8px;border-bottom:1px solid #30363d;'>Vùng dao động (90%)</th>"
            "<th style='padding:8px;border-bottom:1px solid #30363d;'>Nguồn tín hiệu</th>"
            "<th style='padding:8px;border-bottom:1px solid #30363d;'>Độ tin cậy</th>"
            "</tr></thead><tbody>"
        )
        _rows_html = ""
        for _i, (_d, _tgt, _lo, _hi, _chg, _src) in enumerate(_fc_rows):
            _lbl   = PERIOD_LABELS.get(_d, f"{_d}d")
            _up    = _chg > 0.3
            _dn    = _chg < -0.3
            _arrow = ("📈 TĂNG" if _up else "📉 GIẢM" if _dn else "➡️ ĐI NGANG")
            _tclr  = ("#3fb950" if _up else "#f85149" if _dn else "#f9a825")
            _chg_s = f"{_chg:+.2f}%"
            _conf  = ("🔵🔵🔵 Cao"       if _d <= 7  else
                      "🔵🔵○ Khá"        if _d <= 30 else
                      "🔵○○ Trung bình"  if _d <= 90 else
                      "○○○ Thấp")
            _row_bg = "#0d1117" if _i % 2 == 0 else "#161b22"
            _rows_html += (
                f"<tr style='background:{_row_bg};text-align:center;'>"
                f"<td style='padding:9px 12px;text-align:left;color:#e6edf3;font-weight:600;border-bottom:1px solid #21262d;'>{_lbl}</td>"
                f"<td style='padding:9px 8px;color:{_tclr};font-weight:700;border-bottom:1px solid #21262d;'>{_arrow}</td>"
                f"<td style='padding:9px 8px;color:{_tclr};font-weight:700;font-size:0.95rem;border-bottom:1px solid #21262d;'>${_tgt:,.0f}</td>"
                f"<td style='padding:9px 8px;color:{_tclr};font-weight:600;border-bottom:1px solid #21262d;'>{_chg_s}</td>"
                f"<td style='padding:9px 8px;color:#8b949e;font-size:0.82rem;border-bottom:1px solid #21262d;'>${_lo:,.0f} – ${_hi:,.0f}</td>"
                f"<td style='padding:9px 8px;color:#8b949e;font-size:0.78rem;border-bottom:1px solid #21262d;'>{_src}</td>"
                f"<td style='padding:9px 8px;color:#8b949e;font-size:0.80rem;border-bottom:1px solid #21262d;'>{_conf}</td>"
                f"</tr>"
            )
        st.markdown(_hdr + _rows_html + "</tbody></table></div>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:#8b949e;font-size:0.76rem;margin-top:6px;'>"
            f"📍 Giá hiện tại: <b style='color:#FFD700;'>${_cur:,.2f}</b> · "
            f"Expert Score: <b>{total_expert_score:+d}/8</b> · "
            f"Nguồn: Kỹ thuật (≤30d) · Kỹ thuật+Chuyên gia (60–90d) · Chuyên gia (≥180d) · "
            f"⚠️ Không phải khuyến nghị đầu tư.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info("⚠️ Không lấy được giá XAU để chạy dự báo.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # DỰ BÁO DÀI HẠN 2–5 NĂM (EXPERT DEEP VIEW)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 🌐 DỰ BÁO DÀI HẠN — 2 Năm · 3 Năm · 5 Năm")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem;'>"
        "Góc nhìn siêu dài hạn của hedge fund vĩ mô. Ở timeframe này, tín hiệu kỹ thuật gần như vô nghĩa — "
        "<b style='color:#FFD700;'>chỉ có cấu trúc vĩ mô và megatrend mới quyết định</b>.</p>",
        unsafe_allow_html=True,
    )

    # Định nghĩa thesis dài hạn cho từng tài sản
    LT_THESIS = {
        "XAU": {
            "name":  "🥇 Vàng",
            "2y":    ("🚀 Tăng mạnh", "#3fb950",
                      "Fed buộc cắt lãi suất → real rate giảm. CB tiếp tục mua >1,000t/năm. "
                      "Nợ Mỹ $36T tạo áp lực phá giá USD mạn tính."),
            "3y":    ("🚀 Tăng mạnh", "#3fb950",
                      "De-dollarization BRICS tăng tốc. Fed có thể phải YCC (Yield Curve Control) "
                      "→ cực kỳ bullish vàng. Mục tiêu $3,500–$4,000/oz theo Goldman Sachs."),
            "5y":    ("🚀 Tăng mạnh", "#3fb950",
                      "Chu kỳ nợ Kondratiev. M2 toàn cầu tăng gấp đôi mỗi 10–12 năm → vàng là "
                      "neo tiền tệ dài hạn. Kịch bản cực đoan: $5,000–$10,000/oz nếu USD mất vị thế."),
        },
        "XAG": {
            "name":  "🥈 Bạc",
            "2y":    ("⬆️ Tăng", "#76c3a0",
                      "Tỷ lệ Gold/Silver ~85 — lịch sử trung bình 60 → Bạc còn dư địa đuổi kịp Vàng. "
                      "Nhu cầu solar panel tăng 15%/năm (bạc là thành phần không thể thay thế)."),
            "3y":    ("🚀 Tăng mạnh", "#3fb950",
                      "Green energy transition: mỗi MW solar cần ~1 tấn bạc. Nhu cầu EV + grid storage. "
                      "Nguồn cung mỏ bạc đang cạn dần — structural deficit từ 2024."),
            "5y":    ("🚀 Tăng mạnh", "#3fb950",
                      "IEA dự báo nhu cầu bạc cho năng lượng tái tạo tăng 3x đến 2030. "
                      "Kết hợp với vai trò tiền tệ khi vàng tăng → bạc có thể outperform vàng."),
        },
        "HG": {
            "name":  "🟤 Đồng",
            "2y":    ("↗️ Sideway+", "#ffa657",
                      "Nhu cầu EV (mỗi xe cần 4x đồng hơn xe xăng) và lưới điện thông minh. "
                      "Tuy nhiên, suy thoái kinh tế Trung Quốc là rủi ro lớn ngắn-trung hạn."),
            "3y":    ("🚀 Tăng mạnh", "#3fb950",
                      "Goldman Sachs: Đồng sẽ vào supercycle 2025–2030. Nguồn cung mỏ mới cần "
                      "7–10 năm để đưa vào sản xuất → structural deficit kéo dài."),
            "5y":    ("🚀 Tăng mạnh", "#3fb950",
                      "Đồng là 'dầu mỏ của thế kỷ 21'. Hạ tầng điện toàn cầu, AI data centers, "
                      "xe điện, điện gió/mặt trời → tất cả đều cần đồng. Mục tiêu $6–$8/lb."),
        },
        "CL": {
            "name":  "🛢️ Dầu WTI",
            "2y":    ("↗️ Sideway+", "#ffa657",
                      "OPEC+ kiểm soát nguồn cung. Nhu cầu toàn cầu vẫn ~100mb/d. "
                      "Rủi ro địa chính trị Trung Đông tạo premium. Dao động $70–$90/barrel."),
            "3y":    ("↘️ Sideway−", "#ff7b54",
                      "EV penetration tăng tốc → đỉnh nhu cầu dầu (peak oil demand) có thể đến 2027–2028. "
                      "Mỹ và Saudi Arabia tăng sản lượng để giữ thị phần trước khi quá muộn."),
            "5y":    ("⬇️ Giảm", "#f85149",
                      "IEA: Nhu cầu dầu đạt đỉnh trước 2030. Năng lượng tái tạo chiếm >50% điện mới. "
                      "Dầu vẫn cần cho hóa dầu/hàng không nhưng áp lực giá dài hạn nghiêng về giảm."),
        },
        "USDVND": {
            "name":  "💵 USD/VND",
            "2y":    ("⬆️ Tăng", "#76c3a0",
                      "VND chịu áp lực từ thâm hụt thương mại và Fed rate differential. "
                      "SBV có thể tiếp tục giữ neo tỷ giá nhưng áp lực điều chỉnh tích lũy."),
            "3y":    ("⬆️ Tăng", "#76c3a0",
                      "Xu hướng dài hạn: VND mất giá ~3–5%/năm so với USD. "
                      "FDI vào Việt Nam giúp cân bằng một phần nhưng không đủ chống đỡ hoàn toàn."),
            "5y":    ("🚀 Tăng mạnh", "#3fb950",
                      "Nếu USD suy yếu do Fed nới lỏng dài hạn: tỷ giá có thể đi ngang hoặc VND "
                      "mạnh lên nhờ xuất khẩu tăng. Nhưng kịch bản cơ sở: USD/VND vẫn tăng chậm."),
        },
        "BTC": {
            "name":  "₿ Bitcoin",
            "2y":    ("🚀 Tăng mạnh", "#3fb950",
                      "Post-halving Apr 2024: lịch sử 3/3 lần halving đều có bull run 12–18 tháng sau. "
                      "ETF Bitcoin spot tạo dòng chảy vốn tổ chức ổn định. Mục tiêu $150k–$200k."),
            "3y":    ("⬆️ Tăng", "#76c3a0",
                      "Chu kỳ 4 năm: sau đỉnh 2025, Bitcoin thường điều chỉnh 70–80% rồi tích lũy. "
                      "Halving tiếp theo 2028 tạo động lực mới. Adoption tiếp tục tăng."),
            "5y":    ("🚀 Tăng mạnh", "#3fb950",
                      "Lightning Network, Layer 2, ETF toàn cầu → Bitcoin tiến tới 'digital gold' status. "
                      "Nếu 1% tài sản toàn cầu ($500T) chuyển sang BTC → giá lý thuyết >$1M/BTC."),
        },
    }

    for ak, thesis in LT_THESIS.items():
        with st.expander(f"{thesis['name']} — Nhận định 2 năm · 3 năm · 5 năm", expanded=(ak == "XAU")):
            c2y, c3y, c5y = st.columns(3)
            for col, key, label in [(c2y, "2y", "2 NĂM TỚI"), (c3y, "3y", "3 NĂM TỚI"), (c5y, "5y", "5 NĂM TỚI")]:
                icon, color, detail = thesis[key]
                col.markdown(
                    f"<div style='background:{hex_rgba(color, 0.12)};border:1px solid "
                    f"{hex_rgba(color, 0.4)};border-radius:10px;padding:12px;height:100%;'>"
                    f"<div style='color:#8b949e;font-size:0.72rem;font-weight:700;margin-bottom:4px;'>{label}</div>"
                    f"<div style='color:{color};font-size:1.0rem;font-weight:700;margin-bottom:8px;'>{icon}</div>"
                    f"<div style='color:#e6edf3;font-size:0.78rem;line-height:1.55;'>{detail}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # BẢNG KẾT LUẬN TỔNG HỢP — 6 TÀI SẢN × 7 KỲ
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📋 KẾT LUẬN NHANH — Tăng hay Giảm?")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.85rem;margin-bottom:12px;'>"
        "Tổng hợp từ MA20/50/200 · RSI · Momentum · Cấu trúc vĩ mô dài hạn · "
        "Bias theo từng tài sản (vàng: structural bull; dầu: energy transition headwind; ...)</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("⚡ Đang tính dự báo đa kỳ..."):
        summary = calc_summary_forecast()

    ASSET_LABELS = {
        "XAU":    "🥇 Vàng",
        "XAG":    "🥈 Bạc",
        "HG":     "🟤 Đồng",
        "CL":     "🛢️ Dầu WTI",
        "USDVND": "💵 USD/VND",
        "BTC":    "₿ Bitcoin",
    }
    PERIOD_GROUPS = [
        {
            "title": "⚡ Ngắn — Trung hạn",
            "cols":  [(1,"Ngày mai"),(3,"3 ngày"),(7,"1 tuần"),
                      (14,"2 tuần"),(21,"3 tuần"),(30,"1 tháng"),
                      (90,"3 tháng"),(180,"6 tháng"),(365,"1 năm")],
        },
        {
            "title": "🌐 Dài hạn (2–5 năm)",
            "cols":  [(730,"2 năm"),(1095,"3 năm"),(1825,"5 năm")],
        },
    ]

    def _render_table(period_cols):
        n = len(period_cols)
        hdr = st.columns([2] + [1] * n)
        hdr[0].markdown(
            "<div style='color:#8b949e;font-size:0.75rem;font-weight:700;padding:4px 0;'>Tài sản</div>",
            unsafe_allow_html=True,
        )
        for i, (_, plabel) in enumerate(period_cols):
            hdr[i + 1].markdown(
                f"<div style='color:#8b949e;font-size:0.72rem;font-weight:700;"
                f"text-align:center;padding:4px 0;'>{plabel}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<hr style='border-color:#30363d;margin:2px 0 4px 0;'>", unsafe_allow_html=True)
        for ak, alabel in ASSET_LABELS.items():
            row = st.columns([2] + [1] * n)
            row[0].markdown(
                f"<div style='color:#e6edf3;font-size:0.83rem;font-weight:700;padding:6px 0;'>{alabel}</div>",
                unsafe_allow_html=True,
            )
            asset_fc = summary.get(ak, {})
            for i, (days, _) in enumerate(period_cols):
                cell    = asset_fc.get(days, {"dir": "N/A", "color": "#8b949e", "score": 0})
                d_color = cell["color"]
                d_text  = cell["dir"]
                row[i + 1].markdown(
                    f"<div style='background:{hex_rgba(d_color, 0.15)};"
                    f"border:1px solid {hex_rgba(d_color, 0.35)};border-radius:6px;"
                    f"padding:5px 2px;text-align:center;font-size:0.70rem;"
                    f"color:{d_color};font-weight:600;line-height:1.3;'>{d_text}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='height:3px;'></div>", unsafe_allow_html=True)

    for grp in PERIOD_GROUPS:
        st.markdown(f"**{grp['title']}**")
        _render_table(grp["cols"])
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px;"
        "margin-top:6px;font-size:0.78rem;color:#8b949e;'>"
        "📌 <b style='color:#e6edf3;'>Cách đọc:</b> "
        "🚀 Tăng mạnh · ⬆️ Tăng · ↗️ Sideway+ · ↘️ Sideway− · ⬇️ Giảm · 💣 Giảm mạnh. "
        "<br>Kỳ ngắn (1–21 ngày): RSI + momentum. "
        "Kỳ trung (1–12 tháng): MA stack + momentum. "
        "Kỳ dài (2–5 năm): cấu trúc vĩ mô — "
        "Vàng: de-dollarization + nợ công; "
        "Bạc: nhu cầu solar/xanh; "
        "Đồng: supercycle EV + lưới điện; "
        "Bitcoin: halving; "
        "Dầu: headwind năng lượng tái tạo."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<p style='color:#484f58;font-size:0.75rem;text-align:center;'>"
        "🧠 Expert Analysis · FED Intelligence + Market Structure + Smart Money + Verdict · "
        "Dữ liệu: FED RSS · CFTC COT · FRED · yfinance · Options Chain (GLD) · "
        "⚠️ Chỉ mang tính tham khảo.</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION HISTORY — GitHub JSON Storage
# ══════════════════════════════════════════════════════════════════════════════

_GH_REPO      = "vinhthan/gold-analysis"
_GH_PRED_FILE = "predictions.json"


def _gh_token() -> str:
    """Lấy GitHub token từ Streamlit secrets (an toàn, không hardcode)."""
    try:
        return st.secrets["GITHUB_TOKEN"]
    except Exception:
        return ""


@st.cache_data(ttl=1800, show_spinner=False)
def load_predictions_from_github() -> list:
    """Tải file predictions.json từ GitHub repo."""
    import requests as _req, json as _json, base64 as _b64
    token = _gh_token()
    if not token:
        return []
    url  = f"https://api.github.com/repos/{_GH_REPO}/contents/{_GH_PRED_FILE}"
    hdrs = {"Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"}
    try:
        r = _req.get(url, headers=hdrs, timeout=15)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        raw = _b64.b64decode(r.json()["content"]).decode("utf-8")
        return _json.loads(raw)
    except Exception:
        return []


def _save_predictions_to_github(predictions: list) -> bool:
    """Ghi file predictions.json lên GitHub (overwrite)."""
    import requests as _req, json as _json, base64 as _b64
    token = _gh_token()
    if not token:
        return False
    url  = f"https://api.github.com/repos/{_GH_REPO}/contents/{_GH_PRED_FILE}"
    hdrs = {"Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"}
    try:
        sha = None
        r = _req.get(url, headers=hdrs, timeout=15)
        if r.status_code == 200:
            sha = r.json().get("sha")
        body = _json.dumps(predictions, ensure_ascii=False, indent=2).encode("utf-8")
        payload = {
            "message":   f"auto: predictions {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "content":   _b64.b64encode(body).decode("ascii"),
            "committer": {"name": "Gold Bot", "email": "bot@gold-analysis.app"},
        }
        if sha:
            payload["sha"] = sha
        r2 = _req.put(url, headers=hdrs, json=payload, timeout=30)
        r2.raise_for_status()
        load_predictions_from_github.clear()
        return True
    except Exception:
        return False


# ── Snapshot collection ────────────────────────────────────────────────────

def _collect_snapshot(asset_key: str, macro: dict) -> dict:
    """
    Thu thập signal snapshot hôm nay cho một asset.
    Reuse @st.cache_data đã có — không tạo request mới nếu đã cached.
    """
    try:
        price, _ = fetch_price(asset_key)
        if price is None or len(price) < 60:
            return {}

        cur   = float(price.iloc[-1])
        ma20  = price.rolling(20).mean()
        ma50  = price.rolling(50).mean()
        ma200 = price.rolling(200).mean()
        rsi_s = calc_rsi(price)
        rsi_v = float(rsi_s.iloc[-1])

        macro_score, _, _, _ = macro_regime(macro, price, asset_key)
        ml_res  = ml_directional_signal(price.values, 30)
        ml_prob = ml_res.get("prob_up", 0.5)   if ml_res.get("ok") else 0.5
        ml_conf = ml_res.get("confidence", 0.0) if ml_res.get("ok") else 0.0

        score = 0
        if cur > float(ma20.iloc[-1]):                       score += 1
        if cur > float(ma50.iloc[-1]):                       score += 1
        if cur > float(ma200.iloc[-1]):                      score += 1
        if float(ma20.iloc[-1]) > float(ma50.iloc[-1]):      score += 1
        if   45 < rsi_v < 70:   score += 1
        elif rsi_v >= 70:       score -= 1
        elif rsi_v <= 30:       score += 1
        score += max(-2, min(2, round(macro_score / 3)))
        if   ml_prob >= 0.65:   score += 2
        elif ml_prob >= 0.55:   score += 1
        elif ml_prob <= 0.35:   score -= 2
        elif ml_prob <= 0.45:   score -= 1

        if   score >= 6:  sig_vi, sig_en = "TĂNG MẠNH",             "UP"
        elif score >= 3:  sig_vi, sig_en = "CÓ XU HƯỚNG TĂNG",      "UP"
        elif score <= -5: sig_vi, sig_en = "GIẢM MẠNH",             "DOWN"
        elif score <= -2: sig_vi, sig_en = "CÓ XU HƯỚNG GIẢM",      "DOWN"
        else:             sig_vi, sig_en = "ĐI NGANG / TRUNG TÍNH",  "NEUTRAL"

        today_str    = datetime.now().strftime("%Y-%m-%d")
        periods_data = {}
        for d in PERIOD_LABELS:
            periods_data[str(d)] = {
                "label":        PERIOD_LABELS[d],
                "target_date":  (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d"),
                "actual_price": None,
                "change_pct":   None,
                "result":       "PENDING",
            }

        return {
            "id":            f"{asset_key}_{today_str}",
            "asset":         asset_key,
            "asset_name":    ASSETS[asset_key]["name"],
            "asset_color":   ASSETS[asset_key]["color"],
            "recorded_date": today_str,
            "signal":        sig_vi,
            "signal_en":     sig_en,
            "score":         score,
            "ml_prob_up":    round(ml_prob, 3),
            "ml_confidence": round(ml_conf, 3),
            "macro_score":   macro_score,
            "price_now":     round(cur, 4),
            "periods":       periods_data,
        }
    except Exception:
        return {}


# ── Verification ───────────────────────────────────────────────────────────

def _verify_predictions(predictions: list) -> tuple:
    """
    Kiểm chứng các dự báo đã đến hạn: điền actual_price + result.
    Trả về (updated_list, changed_bool).
    """
    today        = datetime.now().date()
    changed      = False
    NEUTRAL_BAND = 0.5   # % — ngưỡng "trung tính"

    for rec in predictions:
        asset_key = rec.get("asset", "")
        price_now = rec.get("price_now", 0.0)
        sig_en    = rec.get("signal_en", "NEUTRAL")

        if not price_now or asset_key not in ASSETS:
            continue
        try:
            price_s, _ = fetch_price(asset_key)
        except Exception:
            continue

        for d_str, pdata in rec.get("periods", {}).items():
            if pdata.get("result") != "PENDING":
                continue
            tgt_str = pdata.get("target_date", "")
            try:
                tgt_date = datetime.strptime(tgt_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if tgt_date > today:
                continue   # Chưa đến hạn

            try:
                mask = price_s.index.normalize() >= pd.Timestamp(tgt_date)
                if not mask.any():
                    continue
                actual = float(price_s[mask].iloc[0])
                chg    = (actual - price_now) / price_now * 100

                if   sig_en == "UP":   result = "ĐÚNG" if chg > 0            else "SAI"
                elif sig_en == "DOWN": result = "ĐÚNG" if chg < 0            else "SAI"
                else:                  result = "ĐÚNG" if abs(chg) <= NEUTRAL_BAND else "SAI"

                pdata["actual_price"] = round(actual, 4)
                pdata["change_pct"]   = round(chg, 2)
                pdata["result"]       = result
                changed = True
            except Exception:
                continue

    return predictions, changed


# ── Auto-save daily ────────────────────────────────────────────────────────

def _auto_save_daily(macro: dict):
    """
    Lưu snapshot hôm nay cho mọi asset (1 lần/ngày/session).
    Gọi ở cuối main() sau khi tabs đã render — dùng cached data.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    if st.session_state.get("_pred_saved") == today_str:
        return

    existing    = load_predictions_from_github()
    saved_today = {r["asset"] for r in existing if r.get("recorded_date") == today_str}

    new_snaps = []
    for ak in ASSETS:
        if ak in saved_today:
            continue
        snap = _collect_snapshot(ak, macro)
        if snap:
            new_snaps.append(snap)

    existing, verified = _verify_predictions(existing)

    if new_snaps or verified:
        merged = existing + new_snaps
        if len(merged) > 2000:          # giới hạn tối đa 2000 records
            merged = merged[-2000:]
        _save_predictions_to_github(merged)

    st.session_state["_pred_saved"] = today_str


# ── Backfill historical data ───────────────────────────────────────────────

def _backfill_history(days_back: int) -> tuple:
    """
    Tái tạo dữ liệu lịch sử cho `days_back` ngày qua.
    - Tín hiệu kỹ thuật (RSI, MA, ML): chính xác theo lịch sử (no lookahead).
    - Macro score: dùng giá trị hiện tại (gần đúng).
    Trả về (n_added, n_verified).
    """
    try:
        macro = fetch_macro()
    except Exception:
        return 0, 0

    existing     = load_predictions_from_github()
    existing_ids = {r["id"] for r in existing}

    today      = datetime.now().date()
    new_records = []
    total_ops  = days_back * len(ASSETS)
    done       = 0
    prog       = st.progress(0, text="Đang khởi động...")

    for offset in range(days_back, 0, -1):
        past_date = today - timedelta(days=offset)
        past_str  = past_date.strftime("%Y-%m-%d")

        for asset_key in ASSETS:
            done += 1
            record_id = f"{asset_key}_{past_str}"
            aname     = ASSETS[asset_key]["short"]
            prog.progress(done / total_ops,
                          text=f"Tái tạo: **{aname}** — {past_str} ({done}/{total_ops})")

            if record_id in existing_ids:
                continue   # Đã có, bỏ qua

            try:
                price, _ = fetch_price(asset_key)
                if price is None or len(price) < 60:
                    continue

                # Cắt giá đến đúng ngày đó — không dùng dữ liệu tương lai
                mask     = price.index.normalize() <= pd.Timestamp(past_date)
                price_sl = price[mask]
                if len(price_sl) < 60:
                    continue

                cur   = float(price_sl.iloc[-1])
                ma20  = price_sl.rolling(20).mean()
                ma50  = price_sl.rolling(50).mean()
                ma200 = price_sl.rolling(200).mean()
                rsi_s = calc_rsi(price_sl)
                rsi_v = float(rsi_s.iloc[-1])

                macro_score, _, _, _ = macro_regime(macro, price_sl, asset_key)
                ml_res  = ml_directional_signal(price_sl.values, 30)
                ml_prob = ml_res.get("prob_up", 0.5)    if ml_res.get("ok") else 0.5
                ml_conf = ml_res.get("confidence", 0.0) if ml_res.get("ok") else 0.0

                score = 0
                if cur > float(ma20.iloc[-1]):                    score += 1
                if cur > float(ma50.iloc[-1]):                    score += 1
                if cur > float(ma200.iloc[-1]):                   score += 1
                if float(ma20.iloc[-1]) > float(ma50.iloc[-1]):  score += 1
                if   45 < rsi_v < 70:  score += 1
                elif rsi_v >= 70:      score -= 1
                elif rsi_v <= 30:      score += 1
                score += max(-2, min(2, round(macro_score / 3)))
                if   ml_prob >= 0.65:  score += 2
                elif ml_prob >= 0.55:  score += 1
                elif ml_prob <= 0.35:  score -= 2
                elif ml_prob <= 0.45:  score -= 1

                if   score >= 6:  sig_vi, sig_en = "TĂNG MẠNH",             "UP"
                elif score >= 3:  sig_vi, sig_en = "CÓ XU HƯỚNG TĂNG",      "UP"
                elif score <= -5: sig_vi, sig_en = "GIẢM MẠNH",             "DOWN"
                elif score <= -2: sig_vi, sig_en = "CÓ XU HƯỚNG GIẢM",      "DOWN"
                else:             sig_vi, sig_en = "ĐI NGANG / TRUNG TÍNH",  "NEUTRAL"

                periods_data = {}
                for d in PERIOD_LABELS:
                    target = (past_date + timedelta(days=d)).strftime("%Y-%m-%d")
                    periods_data[str(d)] = {
                        "label":        PERIOD_LABELS[d],
                        "target_date":  target,
                        "actual_price": None,
                        "change_pct":   None,
                        "result":       "PENDING",
                    }

                new_records.append({
                    "id":            record_id,
                    "asset":         asset_key,
                    "asset_name":    ASSETS[asset_key]["name"],
                    "asset_color":   ASSETS[asset_key]["color"],
                    "recorded_date": past_str,
                    "signal":        sig_vi,
                    "signal_en":     sig_en,
                    "score":         score,
                    "ml_prob_up":    round(ml_prob, 3),
                    "ml_confidence": round(ml_conf, 3),
                    "macro_score":   macro_score,
                    "price_now":     round(cur, 4),
                    "periods":       periods_data,
                    "backfilled":    True,
                })
            except Exception:
                continue

    prog.progress(1.0, text="Đang kiểm chứng kết quả đã đáo hạn...")

    if not new_records:
        prog.empty()
        return 0, 0

    merged          = existing + new_records
    merged, changed = _verify_predictions(merged)
    n_verified      = sum(
        1 for r in merged
        for p in r.get("periods", {}).values()
        if p.get("result") != "PENDING"
    ) if changed else 0

    if len(merged) > 2000:
        merged = merged[-2000:]
    _save_predictions_to_github(merged)
    prog.empty()
    return len(new_records), n_verified


# ── Backfill UI section (dùng lại ở 2 nơi) ────────────────────────────────

def _render_backfill_section():
    st.markdown("**Tái tạo tín hiệu dự báo từ dữ liệu giá lịch sử**")
    st.caption(
        "Tín hiệu kỹ thuật (RSI, MA, ML) chính xác theo lịch sử. "
        "Macro score dùng giá trị hiện tại (gần đúng ~80%). "
        "Kỳ hạn ngắn (1d, 3d, 7d) sẽ được kiểm chứng kết quả ngay."
    )
    bf1, bf2 = st.columns([2, 4])
    with bf1:
        days_back = st.selectbox(
            "Số ngày muốn tái tạo",
            [7, 14, 30],
            format_func=lambda x: f"{x} ngày qua  (~{x * len(ASSETS)} records)",
            key="backfill_days",
        )
    with bf2:
        st.markdown("")   # spacer
        if st.button("🔄 Bắt đầu tái tạo", type="primary", key="do_backfill"):
            n_added, n_verified = _backfill_history(days_back)
            if n_added > 0:
                st.success(
                    f"Đã thêm **{n_added}** records · "
                    f"Kiểm chứng được **{n_verified}** kết quả đã đáo hạn!"
                )
                st.rerun()
            else:
                st.info("Tất cả records trong khoảng này đã tồn tại, không có gì mới.")


# ── History tab renderer ───────────────────────────────────────────────────

def render_history_tab():
    """Tab 📋 Lịch Sử & Thống Kê Độ Chính Xác Dự Báo."""
    st.markdown("### 📋 Lịch Sử & Thống Kê Độ Chính Xác Dự Báo")
    st.caption(
        "Mỗi ngày app tự động lưu tín hiệu phân tích cho 6 tài sản. "
        "Khi đến ngày đáo hạn, giá thực tế được đối chiếu tự động để tính đúng/sai."
    )

    with st.spinner("📂 Đang tải lịch sử từ GitHub..."):
        predictions = load_predictions_from_github()

    if not predictions:
        st.info("⏳ Chưa có dữ liệu tự động. Dùng tính năng bên dưới để tái tạo dữ liệu lịch sử.")
        with st.expander("ℹ️ Cách hoạt động"):
            st.markdown("""
- **Lưu tự động**: Mỗi ngày, app thu thập tín hiệu (MUA / BÁN / TRUNG TÍNH) cho 6 tài sản và lưu vào GitHub.
- **Kiểm chứng tự động**: Khi đến ngày đáo hạn, app lấy giá thực tế và đánh giá đúng/sai.
- **Tiêu chí đúng**: Signal MUA → giá thực tế **tăng**; BÁN → **giảm**; Trung tính → dao động **< 0.5%**.
- **Kỳ hạn theo dõi**: 12 mốc từ Ngày mai đến 5 năm tới.
""")
        _render_backfill_section()
        return

    # ── Flatten records ──────────────────────────────────────────────────
    rows = []
    for rec in predictions:
        for d_str, pdata in rec.get("periods", {}).items():
            try:
                d_int = int(d_str)
            except ValueError:
                continue
            rows.append({
                "asset":          rec["asset"],
                "Tài sản":        rec.get("asset_name", rec["asset"]),
                "asset_color":    rec.get("asset_color", "#FFD700"),
                "Ngày ghi":       rec.get("recorded_date", ""),
                "Tín hiệu":       rec.get("signal", ""),
                "signal_en":      rec.get("signal_en", "NEUTRAL"),
                "ML Prob %":      round(rec.get("ml_prob_up", 0.5) * 100, 1),
                "Giá ghi nhận":   rec.get("price_now", 0),
                "period_days":    d_int,
                "Kỳ hạn":         pdata.get("label", d_str),
                "Ngày đáo hạn":   pdata.get("target_date", ""),
                "Giá thực tế":    pdata.get("actual_price"),
                "Thay đổi %":     pdata.get("change_pct"),
                "Kết quả":        pdata.get("result", "PENDING"),
            })

    if not rows:
        st.info("⏳ Chưa có dữ liệu chi tiết.")
        return

    df_all = pd.DataFrame(rows)

    # ── Filters ──────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 2, 2])
    with fc1:
        asset_opts = ["Tất cả"] + sorted(df_all["asset"].unique().tolist())
        sel_asset  = st.selectbox(
            "🪙 Tài sản", asset_opts,
            format_func=lambda x: ("Tất cả" if x == "Tất cả"
                                   else ASSETS.get(x, {}).get("short", x))
        )
    with fc2:
        p_days = sorted(df_all["period_days"].unique().tolist())
        p_map  = {0: "Tất cả", **{d: PERIOD_LABELS.get(d, str(d)) for d in p_days}}
        sel_period = st.selectbox("📅 Kỳ hạn", [0] + p_days,
                                   format_func=lambda x: p_map.get(x, str(x)))
    with fc3:
        r_opts = {"all": "Tất cả", "ĐÚNG": "✅ Đúng",
                  "SAI": "❌ Sai", "PENDING": "⏳ Đang chờ"}
        sel_result = st.selectbox("🎯 Kết quả", list(r_opts.keys()),
                                   format_func=lambda x: r_opts[x])

    df = df_all.copy()
    if sel_asset != "Tất cả":
        df = df[df["asset"] == sel_asset]
    if sel_period != 0:
        df = df[df["period_days"] == sel_period]
    if sel_result != "all":
        df = df[df["Kết quả"] == sel_result]

    # ── Summary metrics ────────────────────────────────────────────────
    df_done   = df[df["Kết quả"].isin(["ĐÚNG", "SAI"])]
    n_total   = len(df)
    n_correct = int((df["Kết quả"] == "ĐÚNG").sum())
    n_wrong   = int((df["Kết quả"] == "SAI").sum())
    n_pending = int((df["Kết quả"] == "PENDING").sum())
    accuracy  = n_correct / len(df_done) * 100 if len(df_done) > 0 else 0.0

    st.markdown("---")
    cm1, cm2, cm3, cm4, cm5 = st.columns(5)
    cm1.metric("📊 Tổng dự báo",   f"{n_total:,}")
    cm2.metric("✅ Đúng",          f"{n_correct:,}")
    cm3.metric("❌ Sai",           f"{n_wrong:,}")
    cm4.metric("⏳ Đang chờ",     f"{n_pending:,}")
    cm5.metric("🎯 Độ chính xác", f"{accuracy:.1f}%",
               delta="Chỉ tính đã đáo hạn", delta_color="off")
    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────
    if len(df_done) >= 3:
        ch1, ch2 = st.columns(2)

        # Chart 1: Accuracy by period
        with ch1:
            st.markdown("#### 📈 Độ chính xác theo kỳ hạn")
            pa_rows = []
            for d in sorted(p_days):
                sub = df_done[df_done["period_days"] == d]
                if len(sub) == 0:
                    continue
                acc = (sub["Kết quả"] == "ĐÚNG").sum() / len(sub) * 100
                pa_rows.append({"lbl": PERIOD_LABELS.get(d, str(d)), "acc": acc, "n": len(sub)})
            if pa_rows:
                bar_colors = [
                    "#3fb950" if r["acc"] >= 60 else
                    "#f9a825" if r["acc"] >= 50 else "#f85149"
                    for r in pa_rows
                ]
                fig1 = go.Figure(go.Bar(
                    x=[r["lbl"] for r in pa_rows],
                    y=[r["acc"] for r in pa_rows],
                    text=[f"{r['acc']:.0f}%<br>n={r['n']}" for r in pa_rows],
                    textposition="outside",
                    marker_color=bar_colors,
                    marker_line_width=0,
                ))
                fig1.add_hline(y=50, line_dash="dash", line_color="#FFD700",
                               annotation_text="50% ngẫu nhiên",
                               annotation_position="top right")
                fig1.update_layout(
                    template="plotly_dark", plot_bgcolor="#0d1117",
                    paper_bgcolor="#0d1117", height=340,
                    yaxis=dict(range=[0, 118], title="Tỷ lệ đúng (%)"),
                    xaxis_title="Kỳ hạn", margin=dict(t=30, b=20, l=40, r=20),
                )
                st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Accuracy by asset
        with ch2:
            st.markdown("#### 🪙 Độ chính xác theo tài sản")
            aa_rows = []
            for ak, ainfo in ASSETS.items():
                sub = df_done[df_done["asset"] == ak]
                if len(sub) == 0:
                    continue
                acc = (sub["Kết quả"] == "ĐÚNG").sum() / len(sub) * 100
                aa_rows.append({
                    "name":  ainfo["short"],
                    "acc":   acc,
                    "n":     len(sub),
                    "color": ainfo["color"],
                })
            if aa_rows:
                fig2 = go.Figure(go.Bar(
                    x=[r["name"]  for r in aa_rows],
                    y=[r["acc"]   for r in aa_rows],
                    text=[f"{r['acc']:.0f}%<br>n={r['n']}" for r in aa_rows],
                    textposition="outside",
                    marker_color=[r["color"] for r in aa_rows],
                    marker_line_width=0,
                ))
                fig2.add_hline(y=50, line_dash="dash", line_color="#FFD700",
                               annotation_text="50% ngẫu nhiên",
                               annotation_position="top right")
                fig2.update_layout(
                    template="plotly_dark", plot_bgcolor="#0d1117",
                    paper_bgcolor="#0d1117", height=340,
                    yaxis=dict(range=[0, 118], title="Tỷ lệ đúng (%)"),
                    xaxis_title="Tài sản", margin=dict(t=30, b=20, l=40, r=20),
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Chart 3: Accuracy trend over time
        st.markdown("#### 📅 Xu hướng độ chính xác theo thời gian")
        df_tl = (
            df_done.groupby("Ngày ghi")
            .apply(lambda g: (g["Kết quả"] == "ĐÚNG").sum() / len(g) * 100)
            .reset_index(name="acc")
            .sort_values("Ngày ghi")
        )
        if len(df_tl) >= 2:
            dot_colors = ["#3fb950" if v >= 50 else "#f85149" for v in df_tl["acc"]]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=df_tl["Ngày ghi"], y=df_tl["acc"],
                mode="lines+markers",
                line=dict(color="#FFD700", width=2),
                marker=dict(size=9, color=dot_colors,
                            line=dict(color="#0d1117", width=1)),
                name="Accuracy / ngày",
            ))
            fig3.add_hline(y=50, line_dash="dash", line_color="#8b949e",
                           annotation_text="50% ngưỡng ngẫu nhiên",
                           annotation_position="bottom right")
            fig3.update_layout(
                template="plotly_dark", plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117", height=260,
                yaxis=dict(range=[0, 110], title="Tỷ lệ đúng (%)"),
                margin=dict(t=15, b=15, l=40, r=20),
            )
            st.plotly_chart(fig3, use_container_width=True)

    elif len(df_done) > 0:
        st.info(f"ℹ️ Cần ít nhất 3 kết quả đã đáo hạn để vẽ biểu đồ. Hiện có: **{len(df_done)}**.")

    # ── Detail Table ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📝 Bảng chi tiết dự báo")

    df_sorted = df.sort_values(
        ["Ngày ghi", "asset", "period_days"], ascending=[False, True, True]
    )

    def _fp(x):
        if x is None:
            return "⏳"
        try:
            return f"{float(x):,.4f}"
        except Exception:
            return str(x)

    def _fpct(x):
        if x is None:
            return "⏳"
        try:
            return f"{float(x):+.2f}%"
        except Exception:
            return str(x)

    DISPLAY_COLS = [
        "Ngày ghi", "Tài sản", "Kỳ hạn", "Tín hiệu",
        "ML Prob %", "Giá ghi nhận", "Ngày đáo hạn",
        "Giá thực tế", "Thay đổi %", "Kết quả",
    ]
    disp = df_sorted[DISPLAY_COLS].copy()
    disp["Giá ghi nhận"] = disp["Giá ghi nhận"].apply(_fp)
    disp["Giá thực tế"]  = disp["Giá thực tế"].apply(_fp)
    disp["Thay đổi %"]   = disp["Thay đổi %"].apply(_fpct)

    def _style_row(row):
        r = row["Kết quả"]
        if r == "ĐÚNG":
            base = "background-color:#071f12; color:#3fb950"
            bold = base + "; font-weight:bold"
            return [base] * (len(row) - 1) + [bold]
        elif r == "SAI":
            base = "background-color:#2d0f0f; color:#f85149"
            bold = base + "; font-weight:bold"
            return [base] * (len(row) - 1) + [bold]
        else:
            return ["color:#8b949e; font-style:italic"] * len(row)

    styled = disp.style.apply(_style_row, axis=1)
    st.dataframe(styled, use_container_width=True, height=520)

    # ── Action buttons ────────────────────────────────────────────────
    st.markdown("---")
    ba1, ba2, _ = st.columns([2, 2, 5])
    with ba1:
        if st.button("🔄 Kiểm chứng kết quả ngay", use_container_width=True):
            with st.spinner("Đang kiểm chứng..."):
                all_p = load_predictions_from_github()
                upd, chg = _verify_predictions(all_p)
                if chg:
                    _save_predictions_to_github(upd)
                    st.success("Đã cập nhật kết quả dự báo đã đáo hạn!")
                    st.rerun()
                else:
                    st.info("Không có kết quả mới cần cập nhật.")
    with ba2:
        csv_bytes = df_sorted[DISPLAY_COLS].to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "📥 Xuất CSV",
            data=csv_bytes,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Backfill section ──────────────────────────────────────────────
    with st.expander("🕐 Tái tạo dữ liệu lịch sử (backfill)"):
        _render_backfill_section()


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-REBOOT
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_boot_time():
    """Lưu thời điểm khởi động — chỉ reset khi cache bị xoá hoàn toàn."""
    return {"ts": datetime.now()}

def check_auto_reboot(interval_hours: float = 3.0):
    """Tự động reboot sau mỗi interval_hours bằng cách xoá toàn bộ cache."""
    boot = _get_boot_time()
    elapsed = (datetime.now() - boot["ts"]).total_seconds()
    interval_sec = interval_hours * 3600

    if elapsed >= interval_sec:
        st.cache_data.clear()
        st.cache_resource.clear()   # xoá luôn _get_boot_time → timer reset
        st.rerun()

    remaining = interval_sec - elapsed
    h, r = divmod(int(remaining), 3600)
    m = r // 60
    next_reboot = datetime.now() + timedelta(seconds=remaining)
    st.sidebar.caption(
        f"🔄 Auto-reboot sau: **{h}g{m:02d}p** ({next_reboot.strftime('%H:%M')})"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    check_auto_reboot(interval_hours=3.0)

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
        try:
            macro = fetch_macro()
        except Exception as _e:
            st.error(f"⚠️ Không tải được dữ liệu vĩ mô: {_e}")
            st.info("Nhấn 🔄 Làm mới để thử lại.")
            st.stop()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_keys   = list(ASSETS.keys())
    tab_labels = [ASSETS[k]["tab"] for k in tab_keys]
    all_tab_labels = tab_labels + ["🧠 Chuyên Gia", "📋 Lịch Sử"]
    tabs = st.tabs(all_tab_labels)

    for tab, asset_key in zip(tabs[:-2], tab_keys):
        with tab:
            try:
                render_asset_tab(asset_key, macro, forecast_days)
            except Exception as _e:
                st.error(f"⚠️ Lỗi tải tab {ASSETS[asset_key]['short']}: {_e}")
                st.info("Nhấn 🔄 **Làm mới** hoặc reload trang để thử lại.")
                with st.expander("Chi tiết lỗi (để báo cáo bug)"):
                    st.code(str(_e))

    with tabs[-2]:
        try:
            fred_data = fetch_fred_rates()
            render_expert_tab(macro, fred_data)
        except Exception as _e:
            st.error(f"⚠️ Lỗi tab Chuyên Gia: {_e}")
            st.info("Nhấn 🔄 **Làm mới** hoặc reload trang để thử lại.")

    with tabs[-1]:
        try:
            render_history_tab()
        except Exception as _e:
            st.error(f"⚠️ Lỗi tab Lịch Sử: {_e}")
            st.info("Nhấn 🔄 **Làm mới** hoặc reload trang để thử lại.")

    # ── Auto-save daily snapshot (sau khi tabs render) ────────────────────────
    try:
        _auto_save_daily(macro)
    except Exception:
        pass   # silent — không block UI


if __name__ == "__main__":
    main()
