#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOLD PRICE TREND ANALYSIS — Streamlit Web App
Phân tích & Dự báo xu hướng giá vàng thế giới (XAU/USD)
Deploy: Streamlit Cloud  |  Xem trên mọi thiết bị kể cả Android
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
from sklearn.linear_model import LinearRegression

# ─────────────────────── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Gold Trend Analysis",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0d1117; }
    section[data-testid="stSidebar"] { background-color: #161b22; }

    /* Header */
    .gold-header {
        text-align: center;
        padding: 18px 0 6px 0;
    }
    .gold-header h1 {
        color: #FFD700;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    .gold-header p {
        color: #8b949e;
        font-size: 0.88rem;
        margin: 4px 0 0 0;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.82rem !important; }
    [data-testid="stMetricValue"]  { color: #e6edf3 !important; font-size: 1.5rem !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"]  { font-size: 0.95rem !important; }

    /* Buttons */
    .stButton > button {
        font-weight: 700;
        font-size: 1rem;
        border-radius: 8px;
        border: none;
        padding: 10px 0;
        width: 100%;
        transition: opacity 0.15s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Signal box */
    .signal-box {
        border-radius: 10px;
        padding: 12px 18px;
        margin: 10px 0;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* Radio label color */
    .stRadio label { color: #e6edf3 !important; }

    /* Divider */
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────── Helpers ─────────────────────────────────────────────

def hex_rgba(hex_c: str, alpha: float) -> str:
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────── Data fetching ───────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gold() -> tuple[pd.Series, str]:
    for ticker in ("GC=F", "GLD", "IAU"):
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
    raise RuntimeError(
        "Không tải được dữ liệu.\nKiểm tra kết nối Internet và thử lại."
    )


# ─────────────────────── Indicators ──────────────────────────────────────────

def calc_rsi(s: pd.Series, period: int = 14) -> pd.Series:
    d    = s.diff()
    gain = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


# ─────────────────────── Forecasting ─────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def forecast(_price_values: np.ndarray, last_date_str: str, days: int):
    price = pd.Series(_price_values)
    train = price.tail(min(365, len(price)))

    # Model A: Holt-Winters
    hw_vals = None
    try:
        hw = ExponentialSmoothing(
            train, trend="add", damped_trend=True, seasonal=None
        ).fit(optimized=True, use_brute=False)
        hw_vals = hw.forecast(days).values
    except Exception:
        pass

    # Model B: log-return regression (last 90 days, damped)
    log_p = np.log(train.values)
    w     = min(90, len(log_p))
    slope = LinearRegression().fit(
        np.arange(w).reshape(-1, 1), log_p[-w:]
    ).coef_[0] * 0.55
    lr_vals = np.exp(log_p[-1] + slope * np.arange(1, days + 1))

    combined = 0.58 * hw_vals + 0.42 * lr_vals if hw_vals is not None else lr_vals

    # Confidence interval ~90%
    vol    = train.pct_change().std()
    spread = combined * vol * np.sqrt(np.arange(1, days + 1)) * 1.65

    fut = pd.bdate_range(start=pd.Timestamp(last_date_str) + timedelta(days=1), periods=days)
    n   = min(len(fut), len(combined))
    idx = fut[:n]

    return (
        pd.Series(combined[:n], index=idx),
        pd.Series((combined - spread)[:n], index=idx),
        pd.Series((combined + spread)[:n], index=idx),
    )


# ─────────────────────── Signal scoring ──────────────────────────────────────

def compute_signal(price, ma20, ma50, ma200, rsi, fc_mean):
    cur  = float(price.iloc[-1])
    r    = float(rsi.iloc[-1])
    fc_e = float(fc_mean.iloc[-1])
    chg  = (fc_e - cur) / cur * 100

    score = 0
    notes = []

    if cur > float(ma20.iloc[-1]):  score += 1
    if cur > float(ma50.iloc[-1]):  score += 1
    if cur > float(ma200.iloc[-1]):
        score += 1; notes.append("✅ Trên MA200 — xu hướng dài hạn **tăng**")
    else:
        notes.append("⚠️ Dưới MA200 — xu hướng dài hạn chưa rõ")

    if float(ma20.iloc[-1]) > float(ma50.iloc[-1]):
        score += 1; notes.append("✅ MA20 > MA50 — tín hiệu tăng ngắn hạn")
    else:
        notes.append("⚠️ MA20 < MA50 — áp lực giảm ngắn hạn")

    if 45 < r < 70:
        score += 1; notes.append(f"✅ RSI {r:.0f} — động lực tăng ổn định")
    elif r >= 70:
        score -= 1; notes.append(f"⚠️ RSI {r:.0f} — vùng **quá mua**, cẩn thận")
    elif r <= 30:
        score += 1; notes.append(f"✅ RSI {r:.0f} — vùng **quá bán**, có thể hồi phục")
    else:
        notes.append(f"➡️ RSI {r:.0f} — trung tính")

    if chg >= 3:
        score += 2; notes.append(f"✅ Dự báo tăng **{chg:.1f}%** — tích cực")
    elif chg >= 0.5:
        score += 1; notes.append(f"✅ Dự báo tăng nhẹ **{chg:.1f}%**")
    elif chg <= -3:
        score -= 2; notes.append(f"🔴 Dự báo giảm **{abs(chg):.1f}%** — tiêu cực")
    elif chg < -0.5:
        score -= 1; notes.append(f"⚠️ Dự báo giảm nhẹ **{abs(chg):.1f}%**")
    else:
        notes.append(f"➡️ Dự báo biến động nhẹ **{chg:+.1f}%**")

    if score >= 4:
        return "TĂNG MẠNH", "#3fb950", "🟢", notes
    elif score >= 2:
        return "CÓ XU HƯỚNG TĂNG", "#76ff03", "🟡", notes
    elif score <= -3:
        return "GIẢM MẠNH", "#f85149", "🔴", notes
    elif score <= -1:
        return "CÓ XU HƯỚNG GIẢM", "#ff7b54", "🟠", notes
    else:
        return "ĐI NGANG / TRUNG TÍNH", "#FFD700", "⚪", notes


# ─────────────────────── Chart ────────────────────────────────────────────────

def build_chart(p, m20, m50, m200, bbu, bbl, hi52, lo52,
                fc_mean, fc_lo, fc_hi, rsi_s,
                sig_color: str, forecast_days: int) -> go.Figure:

    H   = 150  # show last ~5 months history
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.68, 0.32],
        shared_xaxes=True,
        vertical_spacing=0.04,
    )

    def s(series): return series.tail(H)

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=s(bbu).index, y=s(bbu),
        name="BB Upper", line=dict(color="#79c0ff", width=0.5),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=s(bbl).index, y=s(bbl),
        name="Bollinger Bands",
        line=dict(color="#79c0ff", width=0.5),
        fill="tonexty",
        fillcolor="rgba(121,192,255,0.06)",
    ), row=1, col=1)

    # ── 52-week levels ───────────────────────────────────────────────────────
    fig.add_hline(y=hi52, line_dash="dot", line_color="#3fb950",
                  line_width=0.9, opacity=0.5,
                  annotation_text=f"52W High  ${hi52:,.0f}",
                  annotation_font_color="#3fb950", annotation_font_size=10,
                  row=1, col=1)
    fig.add_hline(y=lo52, line_dash="dot", line_color="#f85149",
                  line_width=0.9, opacity=0.5,
                  annotation_text=f"52W Low  ${lo52:,.0f}",
                  annotation_font_color="#f85149", annotation_font_size=10,
                  annotation_position="bottom right",
                  row=1, col=1)

    # ── MAs ─────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=s(m200).index, y=s(m200), name="MA200",
        line=dict(color="#ff7b54", width=1.1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=s(m50).index, y=s(m50), name="MA50",
        line=dict(color="#bc8cff", width=1.2, dash="dash"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=s(m20).index, y=s(m20), name="MA20",
        line=dict(color="#f9a825", width=1.2, dash="dash"),
    ), row=1, col=1)

    # ── Forecast CI band ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=fc_hi.index, y=fc_hi,
        name="CI Upper", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fc_lo.index, y=fc_lo,
        name="Vùng dự báo (90% CI)",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=hex_rgba(sig_color, 0.16),
    ), row=1, col=1)

    # ── Historical price ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=s(p).index, y=s(p),
        name="Giá Vàng (XAU/USD)",
        line=dict(color="#c9d1d9", width=2.2),
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Giá: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    # ── Forecast line ────────────────────────────────────────────────────────
    month_lbl = f"{forecast_days // 30} tháng"
    fig.add_trace(go.Scatter(
        x=fc_mean.index, y=fc_mean,
        name=f"Dự báo {month_lbl}",
        line=dict(color=sig_color, width=2.6),
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Dự báo: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    # TODAY divider
    fig.add_vline(
        x=p.index[-1].timestamp() * 1000,
        line_dash="dash", line_color="rgba(255,255,255,0.18)", line_width=1,
    )

    # Current price dot
    fig.add_trace(go.Scatter(
        x=[p.index[-1]], y=[float(p.iloc[-1])],
        mode="markers+text",
        marker=dict(color="#FFD700", size=10, symbol="circle"),
        text=[f"  Hôm nay<br>  ${float(p.iloc[-1]):,.0f}"],
        textposition="middle right",
        textfont=dict(color="#FFD700", size=11),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # Forecast end dot
    fig.add_trace(go.Scatter(
        x=[fc_mean.index[-1]], y=[float(fc_mean.iloc[-1])],
        mode="markers+text",
        marker=dict(color=sig_color, size=9, symbol="diamond"),
        text=[f"  ${float(fc_mean.iloc[-1]):,.0f}"],
        textposition="middle right",
        textfont=dict(color=sig_color, size=11, family="Arial Black"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # ── RSI ──────────────────────────────────────────────────────────────────
    r = rsi_s.tail(H)
    fig.add_trace(go.Scatter(
        x=r.index, y=r, name="RSI (14)",
        line=dict(color="#ce93d8", width=1.5),
        hovertemplate="RSI: %{y:.1f}<extra></extra>",
    ), row=2, col=1)

    # RSI fill zones
    fig.add_hrect(y0=70, y1=90, fillcolor="rgba(248,81,73,0.07)",
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=10, y1=30, fillcolor="rgba(63,185,80,0.07)",
                  line_width=0, row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#f85149",
                  line_width=0.9, opacity=0.6,
                  annotation_text="Quá mua",
                  annotation_font_color="#f85149", annotation_font_size=9,
                  row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#3fb950",
                  line_width=0.9, opacity=0.6,
                  annotation_text="Quá bán",
                  annotation_font_color="#3fb950", annotation_font_size=9,
                  annotation_position="bottom right",
                  row=2, col=1)
    fig.add_hline(y=50, line_color="rgba(255,255,255,0.1)",
                  line_width=0.5, row=2, col=1)

    # RSI current dot
    fig.add_trace(go.Scatter(
        x=[r.index[-1]], y=[float(r.iloc[-1])],
        mode="markers",
        marker=dict(color="#ce93d8", size=7),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", size=11, family="Segoe UI, Arial"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d",
                        font_color="#e6edf3", font_size=11),
        legend=dict(
            bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
            font=dict(color="#c9d1d9", size=10),
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0,
        ),
        margin=dict(l=10, r=80, t=10, b=10),
        height=560,
        xaxis_rangeslider_visible=False,
        dragmode="pan",
    )

    grid_style = dict(gridcolor="rgba(255,255,255,0.05)", showline=False,
                      zerolinecolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(**grid_style)
    fig.update_yaxes(**grid_style)

    fig.update_yaxes(title_text="USD / troy oz", row=1, col=1,
                     title_font_size=10)
    fig.update_yaxes(title_text="RSI", row=2, col=1,
                     range=[10, 90], title_font_size=10)
    fig.update_xaxes(tickformat="%d/%m/%y", tickangle=-30, row=2, col=1)

    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="gold-header">
        <h1>🥇 GOLD PRICE TREND ANALYSIS</h1>
        <p>Phân tích kỹ thuật + Dự báo xu hướng giá vàng thế giới (XAU/USD)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Controls ─────────────────────────────────────────────────────────────
    col_r, col_b, col_space = st.columns([2, 1, 4])
    with col_r:
        forecast_days = st.radio(
            "Kỳ dự báo:",
            options=[30, 60],
            format_func=lambda x: f"{'1 tháng' if x == 30 else '2 tháng'} tới",
            horizontal=True,
            label_visibility="collapsed",
        )
    with col_b:
        refresh = st.button("🔄 Làm mới", use_container_width=True)

    if refresh:
        st.cache_data.clear()

    # ── Fetch data ────────────────────────────────────────────────────────────
    with st.spinner("📡 Đang tải dữ liệu giá vàng..."):
        try:
            price, ticker = fetch_gold()
        except RuntimeError as e:
            st.error(str(e))
            return

    # ── Compute indicators ────────────────────────────────────────────────────
    ma20  = price.rolling(20).mean()
    ma50  = price.rolling(50).mean()
    ma200 = price.rolling(200).mean()
    rsi   = calc_rsi(price)
    bb_mid = price.rolling(20).mean()
    bb_std = price.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    hi52   = float(price.tail(252).max())
    lo52   = float(price.tail(252).min())

    # ── Forecast ──────────────────────────────────────────────────────────────
    with st.spinner("🔮 Đang chạy mô hình dự báo..."):
        fc_mean, fc_lo_s, fc_hi_s = forecast(
            price.values, str(price.index[-1]), forecast_days
        )

    # ── Signal ────────────────────────────────────────────────────────────────
    signal, sig_color, sig_icon, notes = compute_signal(
        price, ma20, ma50, ma200, rsi, fc_mean
    )

    cur  = float(price.iloc[-1])
    fc_e = float(fc_mean.iloc[-1])
    chg  = (fc_e - cur) / cur * 100
    sign = "+" if chg >= 0 else ""

    # ── Metrics row ───────────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("💰 Giá hiện tại",   f"${cur:,.0f}",
               f"Nguồn: {ticker}")
    mc2.metric(f"📅 Dự báo ({forecast_days//30} tháng)",
               f"${fc_e:,.0f}",
               f"{sign}{chg:.1f}%",
               delta_color="normal")
    mc3.metric("📊 52W High / Low",
               f"${hi52:,.0f}",
               f"Low: ${lo52:,.0f}")
    mc4.metric("📈 RSI (14)",
               f"{float(rsi.iloc[-1]):.1f}",
               "Quá mua" if float(rsi.iloc[-1]) > 70
               else ("Quá bán" if float(rsi.iloc[-1]) < 30 else "Bình thường"),
               delta_color="off")

    st.markdown("---")

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = build_chart(
        price, ma20, ma50, ma200, bb_up, bb_lo,
        hi52, lo52,
        fc_mean, fc_lo_s, fc_hi_s, rsi,
        sig_color, forecast_days,
    )

    month_lbl = f"{forecast_days // 30} tháng"
    st.markdown(
        f"#### Biểu đồ Giá Vàng  ·  Dự báo xu hướng {month_lbl} tới  ·  "
        f"Cập nhật: {datetime.now():%H:%M  %d/%m/%Y}"
    )
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
        <b style="color:{sig_color};font-size:1.05rem;">{sig_icon} NHẬN ĐỊNH: {signal}</b><br>
        <span style="color:#8b949e;font-size:0.85rem;">
        Giá hiện tại: <b style="color:#e6edf3;">${cur:,.0f}</b> &nbsp;→&nbsp;
        Dự báo cuối kỳ: <b style="color:{sig_color};">${fc_e:,.0f}</b>
        ({sign}{chg:.1f}%)
        </span>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("**Phân tích chi tiết:**")
    for note in notes:
        st.markdown(f"- {note}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<p style='color:#484f58;font-size:0.78rem;text-align:center;'>"
        f"Nguồn dữ liệu: Yahoo Finance ({ticker})  ·  "
        f"Mô hình: Holt-Winters + Linear Regression (ensemble)  ·  "
        f"Chỉ mang tính tham khảo, không phải khuyến nghị đầu tư.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
