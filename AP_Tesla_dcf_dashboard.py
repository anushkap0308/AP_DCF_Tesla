import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Tesla DCF Model")

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #080c10; color: #c9d1d9; }
.stApp { background: #080c10; }
[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e2a38; }
[data-testid="stSidebar"] * { color: #8b949e !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; }
[data-testid="stSidebar"] .stSlider > label, [data-testid="stSidebar"] .stNumberInput > label { color: #58a6ff !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; }
h1 { font-family: 'IBM Plex Mono', monospace !important; color: #f0f6fc !important; font-size: 1.6rem !important; font-weight: 700 !important; border-bottom: 1px solid #21262d; padding-bottom: 0.5rem; margin-bottom: 0.2rem !important; }
h2 { font-family: 'IBM Plex Mono', monospace !important; color: #58a6ff !important; font-size: 0.75rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.12em; margin-top: 1.8rem !important; margin-bottom: 0.6rem !important; border-left: 2px solid #58a6ff; padding-left: 8px; }
[data-testid="metric-container"] { background: #0d1117; border: 1px solid #1e2a38; border-radius: 4px; padding: 16px 20px !important; position: relative; overflow: hidden; }
[data-testid="metric-container"]::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #58a6ff, #3fb950); }
[data-testid="metric-container"] label { font-family: 'IBM Plex Mono', monospace !important; font-size: 10px !important; color: #8b949e !important; text-transform: uppercase; letter-spacing: 0.1em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 1.4rem !important; color: #f0f6fc !important; font-weight: 700 !important; }
[data-testid="stDataFrame"] { border: 1px solid #1e2a38 !important; border-radius: 4px; overflow: hidden; }
.stTabs [data-baseweb="tab-list"] { background: #0d1117; border-bottom: 1px solid #1e2a38; }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; color: #8b949e !important; text-transform: uppercase; letter-spacing: 0.08em; }
.stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }
hr { border-color: #1e2a38 !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHART CONSTANTS
# ─────────────────────────────────────────────
plt.style.use('dark_background')
CHART_BG    = '#0d1117'
CHART_FG    = '#1e2a38'
ACCENT_BLUE = '#58a6ff'
ACCENT_GRN  = '#3fb950'
ACCENT_ORG  = '#d29922'
ACCENT_RED  = '#f85149'
TEXT_DIM    = '#8b949e'
TEXT_BRIGHT = '#f0f6fc'

def style_ax(ax, title=''):
    ax.set_facecolor(CHART_BG)
    ax.figure.set_facecolor(CHART_BG)
    ax.spines['bottom'].set_color(CHART_FG)
    ax.spines['left'].set_color(CHART_FG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.yaxis.label.set_color(TEXT_DIM)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_size(8)
    ax.xaxis.label.set_size(8)
    if title:
        ax.set_title(title, color=TEXT_BRIGHT, fontsize=9,
                     fontweight='600', fontfamily='monospace', pad=10)
    ax.grid(axis='y', color=CHART_FG, linewidth=0.5, alpha=0.6)
    ax.grid(axis='x', visible=False)

TABLE_STYLES = [
    {"selector": "thead th", "props": [
        ("background-color", "#161b22"), ("color", "#58a6ff"),
        ("font-family", "IBM Plex Mono, monospace"), ("font-size", "10px"),
        ("text-transform", "uppercase"), ("letter-spacing", "0.08em"),
        ("border-bottom", "1px solid #1e2a38"),
    ]},
    {"selector": "tbody td", "props": [
        ("background-color", "#0d1117"), ("color", "#c9d1d9"),
        ("font-family", "IBM Plex Mono, monospace"), ("font-size", "12px"),
        ("border-bottom", "1px solid #161b22"),
    ]},
]

# ════════════════════════════════════════════
# DATA — EXACTLY FROM YOUR SPREADSHEET
# Historical: 2021–2025 (index 0–4)
# Projected:  2026–2030 (index 5–9)
# All 10 values covering FY2021–FY2030E
# ════════════════════════════════════════════

# Year axis: 2021 → 2030
YRS_10 = np.arange(2021, 2031)    # [2021,2022,...,2030]

# ── HISTORICAL DATA (from Image 1 / Historical Data tab) ──────────────────
# Revenue segments (historical 2021–2025, projected 2026–2030 from Image 2)
SVC_REV    = np.array([3802.00,  6091.00,  8319.00,  10534.00, 12530.00,
                        13783.00, 15161.30, 16374.20, 17684.14, 19098.87])
ENERGY_REV = np.array([2789.00,  3909.00,  6035.00,  10086.00, 12771.00,
                        14686.65, 16889.64, 18916.41, 20808.05, 22888.85])
AUTO_REV   = np.array([47232.00, 71462.00, 82419.00, 77070.00, 69526.00,
                        77556.25, 85125.74, 91765.55, 98115.72, 102962.64])
TOTAL_REV  = np.array([53823.00, 81462.00, 96773.00, 97690.00, 94827.00,
                        106025.90, 117176.69, 127056.16, 136607.91, 144950.37])

# Gross margin (%) — from historical sheet rows 40 & assumptions
GROSS_MARGIN = np.array([25.30, 25.60, 18.20, 17.90, 18.00,
                          18.00, 19.00, 20.00, 20.00, 21.00])

# Operating margin (%) — historical + projected
OP_MARGIN  = np.array([12.10, 16.80,  9.20,  7.20,  4.60,
                         6.50,  8.00, 10.00, 11.00, 12.00])

# EBIT — from operating model (Image 2, rows 26)
EBIT = np.array([6512.58, 13685.62, 8903.12, 7033.68, 4362.04,
                  6891.68,  9374.14, 12705.62, 15026.87, 17394.04])

# Tax rate row (Image 2 row 26) — matches historical tax rates
TAX_RATE   = np.array([8.25, 8.25, 65.21, 20.43, 26.96,
                        26.96, 26.96, 26.96, 26.96, 26.96])

# NOPAT (Image 2 row 27)
NOPAT = np.array([12556.37, 12556.37, 3097.75, 5596.43, 3185.99,
                   5168.76,  7030.60,  9529.21, 11270.15, 13045.53])
# NOTE: 2021 NOPAT from model = 12556.37 (matches Image 2 row 27 col B)
# 2021 historical NOPAT corrected to match spreadsheet exactly:
NOPAT[0] = 12556.37

# Reinvestment components
DA    = np.array([3747.00,  3747.00,  4667.00,  5368.00,  6148.00,
                   6762.87,  7371.45,  7961.17,  8518.45,  9029.56])
CAPEX = np.array([7158.00,  7158.00,  8898.00, 11339.00,  8527.00,
                   9542.33, 10545.90, 10164.49,  9562.55, 10146.53])
DWC   = np.array([2891.00,  2891.00,  2167.00,  1258.00,  -364.00,
                  -2128.85,   154.41,   136.80,   132.26,   115.52])

# Fix 2021 values from historical sheet (Image 1)
DA[0]    = 3747.00
CAPEX[0] = 7158.00
DWC[0]   = 2891.00

REINVEST = np.array([13796.00, 13796.00, 15732.00, 17965.00, 14311.00,
                      14176.28, 18071.76, 18262.46, 18213.27, 19291.60])

# FCF — from Image 2 FREE CASH FLOW section (row 43 FCF)
# Historical: 2021=6254.37, 2022=-3300.25, 2023=-1632.57, 2024=1170.99, 2025=1170.99
# Projected:  2026=6241.00, 2027=6045.28, 2028=10365.49, 2029=13850.50, 2030=16161.56
FCF_10 = np.array([6254.37, -3300.25, -1632.57,  1170.99,  1170.99,
                    6241.00,  6045.28, 10365.49, 13850.50, 16161.56])

# ── VALUATION — EXACTLY FROM YOUR MODEL (Image 2) ─────────────────────────
# PV(FCF) for projected years 2026–2030 + one extra (your model discounts 6 years)
# From Image 2 row 48: PV(FCF) values
# Columns F(2026) G(2027) H(2028) I(2029) J(2030) K(2030 or TV col)
# Your model: 2988.81, 4140.33, -2184.73, -1080.74, 775.18, 4131.48 ... 
# Correct PV(FCF) from spreadsheet row 48:
PV_FCF_6 = np.array([2988.81, 4140.33, 4001.91, 6861.84, 9168.88, 10698.78])
FCF_PROJ = np.array([6241.00, 6045.28, 10365.49, 13850.50, 16161.56, 16161.56])
YRS_PROJ_6 = np.array([2025, 2026, 2027, 2028, 2029, 2030])

# Exact valuation outputs from your model
TV          = 365391.71
PV_TV       = 241885.35
PV_FCF_SUM  = float(np.sum(PV_FCF_6))   # = 37,861M
EV_BASE     = 281387.08
NET_DEBT    = -35683.00    # negative = net cash position
EQUITY_BASE = 317070.08
SHARES      = 3539.00      # from Image 2 row 51: 3,539M (corrected from 3540)
PRICE_BASE  = 89.59

# ─────────────────────────────────────────────
# SIDEBAR — SENSITIVITY CONTROLS ONLY
# ─────────────────────────────────────────────
st.sidebar.markdown(
    '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;'
    'color:#f0f6fc;font-weight:700;letter-spacing:0.05em;margin-bottom:4px;">'
    '// SENSITIVITY CONTROLS</div>'
    '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;'
    'color:#484f58;margin-bottom:14px;">Drives Sensitivity tab only.<br>'
    'Base model always shows exact data.</div>',
    unsafe_allow_html=True
)
s_wacc = st.sidebar.slider("WACC (%)",               6.0, 15.0,  8.6)
s_tg   = st.sidebar.slider("Terminal Growth (%)",    1.0,  5.0,  4.0)
s_om   = st.sidebar.slider("Operating Margin (%)",   4.0, 25.0, 12.0)
s_tax  = st.sidebar.slider("Tax Rate (%)",          10.0, 40.0, 26.96)
s_ri   = st.sidebar.slider("Reinvestment Rate (%)", 10.0,100.0, 60.0)
s_nd   = st.sidebar.number_input("Net Debt ($M)",           value=-35683)
s_sh   = st.sidebar.number_input("Shares Outstanding (M)",  value=3539)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;'
    'color:#484f58;line-height:1.9;">'
    '📌 YOUR BASE MODEL<br>'
    'EV &nbsp;&nbsp;&nbsp;: $281,387M<br>'
    'Equity: $317,070M<br>'
    'Price &nbsp;: $89.59<br>'
    'Shares: 3,539M<br>'
    'Cash &nbsp;: $35,683M<br>'
    'WACC &nbsp;: 8.60%<br>'
    'g &nbsp;&nbsp;&nbsp;&nbsp;: 4.00%'
    '</div>', unsafe_allow_html=True
)

# Sensitivity model — driven by sliders, uses projected years 2026-2030
rev_s    = TOTAL_REV[5:]                              # 2026-2030, shape (5,)
fcf_s    = rev_s * (s_om/100) * (1-s_tax/100) * (1-s_ri/100)
disc_s   = np.array([(1/(1+s_wacc/100)**(i+1)) for i in range(5)])
if s_wacc > s_tg:
    tv_s     = fcf_s[-1] * (1+s_tg/100) / ((s_wacc-s_tg)/100)
    pv_tv_s  = tv_s * disc_s[-1]
else:
    pv_tv_s  = 0
ev_s     = float(np.sum(fcf_s * disc_s)) + pv_tv_s
eq_s     = ev_s - s_nd
sp_s     = eq_s / s_sh

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
c1, c2 = st.columns([5, 1])
with c1:
    st.title(" TESLA:DCF VALUATION MODEL")
    st.markdown(
        '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
        'color:#8b949e;letter-spacing:0.08em;">'
        'TSLA · NASDAQ · FY2021–FY2030E · USD MILLIONS</span>',
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        '<div style="text-align:right;margin-top:14px;">'
        '<span style="background:#161b22;border:1px solid #3fb950;color:#3fb950;'
        'font-family:\'IBM Plex Mono\',monospace;font-size:10px;padding:4px 10px;'
        'border-radius:2px;letter-spacing:0.1em;">LIVE MODEL</span></div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# KPI CARDS — EXACT BASE VALUES FROM YOUR MODEL
# ─────────────────────────────────────────────
st.markdown("## Base Valuation")
c = st.columns(5)
c[0].metric("Enterprise Value",   "$281,387M",  "PV(FCF) + PV(TV)")
c[1].metric("Equity Value",       "$317,070M",  "EV + Net Cash $35,683M")
c[2].metric("Share Price (DCF)",  "$89.59",     "$317,070M ÷ 3,539M shares")
c[3].metric("PV Terminal Value",  "$241,885M",  f"TV = $365,392M · {PV_TV/EV_BASE*100:.1f}% of EV")
c[4].metric("PV of FCFs",         f"${PV_FCF_SUM:,.0f}M", "Sum 2025–2030")

st.markdown("")
c2r = st.columns(5)
c2r[0].metric("WACC",            "8.60%",       "g = 4.00%")
c2r[1].metric("FY2024 Revenue",  "$97,690M",    "FY2023: $96,773M")
c2r[2].metric("FY2030E Revenue", "$144,950M",   "CAGR ~8.9% from 2021")
c2r[3].metric("FY2024 FCF",      "$1,171M",     "Recovered from -$1,633M")
c2r[4].metric("FY2030E FCF",     "$16,162M",    "Peak projected FCF")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  REVENUE MODEL  ", "  CASH FLOW ANALYSIS  ",
    "  DCF WATERFALL  ", "  SENSITIVITY  "
])

# ════════════════════════════════════════════
# TAB 1 — REVENUE MODEL
# ════════════════════════════════════════════
with tab1:
    st.markdown("## Revenue Breakdown (FY2021–FY2030E)")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.set_facecolor(CHART_BG)
    fig.subplots_adjust(wspace=0.38)

    # Chart 1 — Stacked Revenue
    ax = axes[0]
    ax.bar(YRS_10, AUTO_REV/1000,   0.6, label='Automotive', color=ACCENT_BLUE, alpha=0.9)
    ax.bar(YRS_10, SVC_REV/1000,    0.6, label='Services',   color=ACCENT_GRN,  alpha=0.9,
           bottom=AUTO_REV/1000)
    ax.bar(YRS_10, ENERGY_REV/1000, 0.6, label='Energy',     color=ACCENT_ORG,  alpha=0.9,
           bottom=(AUTO_REV+SVC_REV)/1000)
    # Divider after 2025 (last historical year)
    ax.axvline(2025.5, color='#ff7b72', linewidth=1.2, linestyle='--', alpha=0.8)
    ax.text(2025.65, 50, 'PROJ →', color='#ff7b72', fontsize=7, fontfamily='monospace')
    ax.set_ylabel('Revenue ($B)', fontsize=8)
    ax.legend(fontsize=7, facecolor=CHART_BG, edgecolor=CHART_FG,
              labelcolor=TEXT_DIM, framealpha=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}B'))
    style_ax(ax, 'REVENUE MIX BY SEGMENT')

    # Chart 2 — Operating Margin
    ax2 = axes[1]
    colors_om = [ACCENT_BLUE if y <= 2025 else ACCENT_GRN for y in YRS_10]
    bars_om = ax2.bar(YRS_10, OP_MARGIN, 0.6, color=colors_om, alpha=0.85)
    for b, v in zip(bars_om, OP_MARGIN):
        ax2.text(b.get_x()+b.get_width()/2, v+0.15, f'{v:.1f}%',
                 ha='center', fontsize=6.5, color=TEXT_DIM, fontfamily='monospace')
    ax2.axvline(2025.5, color='#ff7b72', linewidth=1.2, linestyle='--', alpha=0.8)
    ax2.set_ylabel('Operating Margin (%)', fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    style_ax(ax2, 'OPERATING MARGIN TREND')

    # Chart 3 — YoY Growth (9 values: diff of 10)
    yoy     = np.diff(TOTAL_REV) / TOTAL_REV[:-1] * 100   # shape (9,)
    yrs_yoy = YRS_10[1:]                                    # 2022–2030
    c_yoy   = [ACCENT_BLUE if y <= 2025 else ACCENT_GRN for y in yrs_yoy]
    ax3 = axes[2]
    ax3.bar(yrs_yoy, yoy, 0.6, color=c_yoy, alpha=0.85)
    ax3.axhline(0, color=ACCENT_RED, linewidth=0.8, alpha=0.5)
    ax3.axvline(2025.5, color='#ff7b72', linewidth=1.2, linestyle='--', alpha=0.8)
    ax3.set_ylabel('YoY Growth (%)', fontsize=8)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    style_ax(ax3, 'REVENUE GROWTH YoY')

    st.pyplot(fig)
    plt.close(fig)

    # Revenue Table
    st.markdown("## Revenue Model Table")
    rev_df = pd.DataFrame({
        "Year":                [f"{y}E" if y >= 2026 else str(y) for y in YRS_10],
        "Automotive ($M)":     AUTO_REV.round(0),
        "Services ($M)":       SVC_REV.round(0),
        "Energy ($M)":         ENERGY_REV.round(0),
        "Total Revenue ($M)":  TOTAL_REV.round(0),
        "EBIT ($M)":           EBIT.round(0),
        "Op. Margin (%)":      OP_MARGIN,
    })
    st.dataframe(
        rev_df.style
        .format({"Automotive ($M)": "{:,.0f}", "Services ($M)": "{:,.0f}",
                 "Energy ($M)": "{:,.0f}", "Total Revenue ($M)": "{:,.0f}",
                 "EBIT ($M)": "{:,.0f}", "Op. Margin (%)": "{:.2f}%"})
        .set_table_styles(TABLE_STYLES),
        use_container_width=True, hide_index=True
    )

    # Assumption table
    st.markdown("## Key Assumptions")
    assume_df = pd.DataFrame({
        "Parameter": ["Delivery Growth Rate", "Service Rev Growth",
                       "Energy Rev Growth", "CapEx % of Revenue",
                       "D&A Growth Rate", "Tax Rate (proj.)", "WACC", "Terminal Growth"],
        "FY2026E": ["12%","10%","15%","9%","9%","26.96%","8.60%","4.00%"],
        "FY2027E": ["10%","8%", "12%","8%","8%","26.96%","8.60%","4.00%"],
        "FY2028E": ["8%", "8%", "10%","7%","7%","26.96%","8.60%","4.00%"],
        "FY2029E": ["6%", "8%", "10%","7%","6%","26.96%","8.60%","4.00%"],
        "FY2030E": ["6%", "8%", "10%","7%","6%","26.96%","8.60%","4.00%"],
    })
    st.dataframe(assume_df.style.set_table_styles(TABLE_STYLES),
                 use_container_width=True, hide_index=True)

# ════════════════════════════════════════════
# TAB 2 — CASH FLOW ANALYSIS
# ════════════════════════════════════════════
with tab2:
    st.markdown("## Free Cash Flow — FY2021 to FY2030E")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4.5))
    fig2.set_facecolor(CHART_BG)
    fig2.subplots_adjust(wspace=0.38)

    # Chart 1 — FCF bars (10 values, 10 years)
    ax = axes2[0]
    c_fcf = [ACCENT_GRN if v >= 0 else ACCENT_RED for v in FCF_10]
    bars  = ax.bar(YRS_10, FCF_10/1000, 0.6, color=c_fcf, alpha=0.88, zorder=3)
    ax.axhline(0, color=TEXT_DIM, linewidth=0.8, alpha=0.5)
    ax.axvline(2025.5, color='#ff7b72', linewidth=1.2, linestyle='--', alpha=0.8)
    ax.set_ylabel('FCF ($B)', fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.1f}B'))
    for bar, val in zip(bars, FCF_10):
        offset = 0.25 if val >= 0 else -0.55
        ax.text(bar.get_x()+bar.get_width()/2, val/1000+offset,
                f'${val/1000:.1f}B', ha='center', va='bottom',
                fontsize=6.5, color=TEXT_DIM, fontfamily='monospace')
    style_ax(ax, 'FREE CASH FLOW 2021–2030E')

    # Chart 2 — NOPAT vs Reinvestment vs FCF (projected years 2026-2030)
    PROJ5_YRS   = YRS_10[5:]        # 2026–2030, shape(5,)
    nopat_p5    = NOPAT[5:]
    reinvest_p5 = REINVEST[5:]
    fcf_p5      = FCF_10[5:]

    ax2 = axes2[1]
    ax2.bar(PROJ5_YRS, nopat_p5/1000,    0.5, label='NOPAT',        color=ACCENT_BLUE, alpha=0.85)
    ax2.bar(PROJ5_YRS, reinvest_p5/1000, 0.5, label='Reinvestment', color=ACCENT_ORG,  alpha=0.75)
    ax2r = ax2.twinx()
    ax2r.plot(PROJ5_YRS, fcf_p5/1000, color=ACCENT_GRN,
              linewidth=2.2, marker='o', markersize=6, label='FCF', zorder=5)
    ax2r.set_ylabel('FCF ($B)', fontsize=8, color=ACCENT_GRN)
    ax2r.tick_params(colors=ACCENT_GRN, labelsize=8)
    ax2r.spines['right'].set_color(ACCENT_GRN)
    ax2r.spines['top'].set_visible(False)
    ax2r.spines['left'].set_visible(False)
    ax2r.spines['bottom'].set_visible(False)
    ax2r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.1f}B'))
    ax2.set_ylabel('$B', fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.1f}B'))
    ax2.legend(fontsize=7, facecolor=CHART_BG, edgecolor=CHART_FG, labelcolor=TEXT_DIM)
    style_ax(ax2, 'NOPAT vs REINVESTMENT vs FCF (2026E–2030E)')

    st.pyplot(fig2)
    plt.close(fig2)

    # Full Model Table — exact data
    st.markdown("## Full Financial Model — Exact Data")
    full_df = pd.DataFrame({
        "Year":              [f"{y}E" if y >= 2026 else str(y) for y in YRS_10],
        "Revenue ($M)":      TOTAL_REV,
        "EBIT ($M)":         EBIT,
        "NOPAT ($M)":        NOPAT,
        "D&A ($M)":          DA,
        "CapEx ($M)":        CAPEX,
        "ΔWC ($M)":          DWC,
        "Reinvestment ($M)": REINVEST,
        "FCF ($M)":          FCF_10,
    })

    def color_val(v):
        if not isinstance(v, (int, float, np.floating)): return ''
        if v < 0:     return 'color: #f85149'
        if v > 10000: return 'color: #3fb950'
        return 'color: #c9d1d9'

    st.dataframe(
        full_df.style
        .format({col: "{:,.0f}" for col in full_df.columns if col != "Year"})
        .map(color_val)
        .set_table_styles(TABLE_STYLES),
        use_container_width=True, hide_index=True
    )

# ════════════════════════════════════════════
# TAB 3 — DCF WATERFALL
# ════════════════════════════════════════════
with tab3:
    st.markdown("## DCF Valuation Waterfall")

    cw, ct = st.columns([3, 2])

    with cw:
        fig3, ax = plt.subplots(figsize=(9, 5.5))
        fig3.set_facecolor(CHART_BG)
        ax.set_facecolor(CHART_BG)

        pf  = PV_FCF_SUM   # ~37,860M
        ptv = PV_TV         # 241,885M
        ev  = EV_BASE       # 281,387M
        nc  = abs(NET_DEBT) # 35,683M  (net cash, added)
        eq  = EQUITY_BASE   # 317,070M

        wf_lbls = ['PV(FCF)\n2025–30', 'PV(TV)', 'Enterprise\nValue',
                   '+ Net Cash', 'Equity\nValue']
        wf_h    = [pf,  ptv, ev, nc,  eq ]
        wf_bot  = [0,   pf,  0,  ev,  0  ]
        wf_col  = [ACCENT_BLUE, ACCENT_GRN, '#c9d1d9', ACCENT_ORG, '#d2a8ff']
        wf_solid= [True, True, False, True, False]

        for i, (lbl, h, bot, col_b, solid) in enumerate(
                zip(wf_lbls, wf_h, wf_bot, wf_col, wf_solid)):
            if solid:
                ax.bar(i, h, 0.55, bottom=bot, color=col_b, alpha=0.85, zorder=3)
            else:
                ax.bar(i, h, 0.55, color=col_b, alpha=0.18,
                       edgecolor=col_b, linewidth=2.5, zorder=3)
            ty = (bot + h/2) if solid else h/2
            ax.text(i, ty, f'${h/1000:,.0f}B',
                    ha='center', va='center', fontsize=9.5, fontweight='bold',
                    color=TEXT_BRIGHT, fontfamily='monospace', zorder=4)

        # Connector dashes
        for (a_i, b_i) in [(0,1),(1,2),(3,4)]:
            y_conn = wf_bot[a_i] + wf_h[a_i]
            ax.plot([a_i+0.28, b_i-0.28], [y_conn, y_conn],
                    '--', color=TEXT_DIM, linewidth=0.9, alpha=0.5)

        ax.set_xticks(range(5))
        ax.set_xticklabels(wf_lbls, fontsize=8.5, color=TEXT_DIM, fontfamily='monospace')
        ax.set_xlim(-0.5, 4.5)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}B'))
        style_ax(ax, 'VALUATION WATERFALL — YOUR EXACT MODEL')
        st.pyplot(fig3)
        plt.close(fig3)

    with ct:
        st.markdown("## PV(FCF) Breakdown")

        pv_df = pd.DataFrame({
            "Year":        ["2025","2026E","2027E","2028E","2029E","2030E"],
            "FCF ($M)":    FCF_PROJ,
            "PV(FCF)($M)": PV_FCF_6,
        })
        st.dataframe(
            pv_df.style
            .format({"FCF ($M)": "{:,.0f}", "PV(FCF)($M)": "{:,.0f}"})
            .set_table_styles(TABLE_STYLES),
            use_container_width=True, hide_index=True
        )

        # Valuation summary card
        st.markdown(f"""
<div style="background:#0d1117;border:1px solid #1e2a38;border-radius:4px;
padding:18px;margin-top:14px;font-family:'IBM Plex Mono',monospace;">
<div style="color:#58a6ff;font-size:10px;text-transform:uppercase;
letter-spacing:0.12em;margin-bottom:12px;">// VALUATION BRIDGE</div>
<table style="width:100%;border-collapse:collapse;font-size:12px;">
<tr>
  <td style="color:#8b949e;padding:5px 0;">Σ PV(FCF) 2025–30</td>
  <td style="color:#58a6ff;text-align:right;">${PV_FCF_SUM:,.0f}M</td>
</tr>
<tr>
  <td style="color:#8b949e;padding:5px 0;">PV(Terminal Value)</td>
  <td style="color:#3fb950;text-align:right;">${PV_TV:,.0f}M</td>
</tr>
<tr style="border-top:1px solid #21262d;">
  <td style="color:#c9d1d9;padding:6px 0;font-weight:700;">Enterprise Value</td>
  <td style="color:#f0f6fc;text-align:right;font-weight:700;">${EV_BASE:,.0f}M</td>
</tr>
<tr>
  <td style="color:#8b949e;padding:5px 0;">Net Cash (added)</td>
  <td style="color:#d29922;text-align:right;">+$35,683M</td>
</tr>
<tr style="border-top:1px solid #21262d;">
  <td style="color:#c9d1d9;padding:6px 0;font-weight:700;">Equity Value</td>
  <td style="color:#f0f6fc;text-align:right;font-weight:700;">${EQUITY_BASE:,.0f}M</td>
</tr>
<tr>
  <td style="color:#8b949e;padding:5px 0;">÷ Shares Outstanding</td>
  <td style="color:#8b949e;text-align:right;">3,539M</td>
</tr>
<tr style="border-top:2px solid #58a6ff;margin-top:4px;">
  <td style="color:#58a6ff;padding:8px 0 4px;font-weight:700;font-size:13px;">
    Implied Share Price</td>
  <td style="color:#3fb950;text-align:right;font-weight:700;font-size:20px;">
    $89.59</td>
</tr>
</table>
<div style="margin-top:10px;padding-top:8px;border-top:1px solid #161b22;
color:#484f58;font-size:9px;line-height:1.7;">
WACC: 8.60% · g: 4.00% · TV: $365,392M · Tax: 26.96%<br>
TV as % of EV: {PV_TV/EV_BASE*100:.1f}%
</div>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 4 — SENSITIVITY
# ════════════════════════════════════════════
with tab4:
    st.markdown("## Sensitivity Analysis")

    # Slider status bar
    st.markdown(
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
        f'color:#8b949e;background:#0d1117;border:1px solid #1e2a38;'
        f'padding:10px 14px;border-radius:3px;margin-bottom:14px;">'
        f'Slider inputs → '
        f'WACC: <span style="color:#58a6ff">{s_wacc:.1f}%</span> · '
        f'g: <span style="color:#58a6ff">{s_tg:.1f}%</span> · '
        f'Margin: <span style="color:#58a6ff">{s_om:.1f}%</span> · '
        f'Tax: <span style="color:#58a6ff">{s_tax:.2f}%</span> · '
        f'Reinv: <span style="color:#58a6ff">{s_ri:.0f}%</span> · '
        f'Implied Price: <span style="color:#3fb950;font-weight:700;">'
        f'${sp_s:,.2f}</span> vs Base: '
        f'<span style="color:#d29922">$89.59</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    cs1, cs2 = st.columns(2)

    # Heatmap 1: WACC × Terminal Growth
    wacc_rng = np.arange(6.0, 13.0, 0.5)
    g_rng    = np.arange(2.0, 6.0, 0.5)

    h1 = []
    for w in wacc_rng:
        row = []
        for g in g_rng:
            if w <= g:
                row.append(np.nan)
                continue
            fc = rev_s * (s_om/100) * (1-s_tax/100) * (1-s_ri/100)
            d  = np.array([(1/(1+w/100)**(i+1)) for i in range(len(fc))])
            tv = fc[-1] * (1+g/100) / ((w-g)/100)
            ev_h = float(np.sum(fc*d)) + tv*d[-1]
            row.append(round((ev_h - s_nd)/s_sh, 1))
        h1.append(row)

    df_h1 = pd.DataFrame(h1,
        index=[f"{w:.1f}%" for w in wacc_rng],
        columns=[f"{g:.1f}%" for g in g_rng])

    with cs1:
        fig_h1, ax_h1 = plt.subplots(figsize=(7, 6))
        fig_h1.set_facecolor(CHART_BG)
        ax_h1.set_facecolor(CHART_BG)
        sns.heatmap(df_h1, annot=True, fmt=".0f", ax=ax_h1,
                    cmap=sns.diverging_palette(15, 145, s=80, l=40, as_cmap=True),
                    linewidths=0.4, linecolor='#0d1117',
                    annot_kws={"size": 8, "family": "IBM Plex Mono", "color": TEXT_BRIGHT},
                    cbar_kws={"shrink": 0.8})
        ax_h1.set_title("Share Price ($) — WACC (rows) × g (cols)",
                         color=TEXT_BRIGHT, fontsize=8.5, fontfamily='monospace', pad=10)
        ax_h1.set_xlabel("Terminal Growth Rate g", color=TEXT_DIM, fontsize=8)
        ax_h1.set_ylabel("WACC", color=TEXT_DIM, fontsize=8)
        ax_h1.tick_params(colors=TEXT_DIM, labelsize=7.5)
        ax_h1.figure.axes[-1].tick_params(colors=TEXT_DIM, labelsize=7)
        st.pyplot(fig_h1)
        plt.close(fig_h1)

    # Heatmap 2: Operating Margin × WACC
    om_rng    = np.arange(4.0, 18.0, 2.0)
    wacc_rng2 = np.arange(7.0, 13.0, 0.5)

    h2 = []
    for om in om_rng:
        row = []
        for w in wacc_rng2:
            if w <= s_tg:
                row.append(np.nan)
                continue
            fc = rev_s * (om/100) * (1-s_tax/100) * (1-s_ri/100)
            d  = np.array([(1/(1+w/100)**(i+1)) for i in range(len(fc))])
            tv = fc[-1] * (1+s_tg/100) / ((w-s_tg)/100)
            ev_h = float(np.sum(fc*d)) + tv*d[-1]
            row.append(round((ev_h - s_nd)/s_sh, 1))
        h2.append(row)

    df_h2 = pd.DataFrame(h2,
        index=[f"{o:.0f}%" for o in om_rng],
        columns=[f"{w:.1f}%" for w in wacc_rng2])

    with cs2:
        fig_h2, ax_h2 = plt.subplots(figsize=(7, 6))
        fig_h2.set_facecolor(CHART_BG)
        ax_h2.set_facecolor(CHART_BG)
        sns.heatmap(df_h2, annot=True, fmt=".0f", ax=ax_h2,
                    cmap=sns.diverging_palette(230, 20, s=80, l=40, as_cmap=True),
                    linewidths=0.4, linecolor='#0d1117',
                    annot_kws={"size": 8, "family": "IBM Plex Mono", "color": TEXT_BRIGHT},
                    cbar_kws={"shrink": 0.8})
        ax_h2.set_title("Share Price ($) — Op. Margin (rows) × WACC (cols)",
                          color=TEXT_BRIGHT, fontsize=8.5, fontfamily='monospace', pad=10)
        ax_h2.set_xlabel("WACC", color=TEXT_DIM, fontsize=8)
        ax_h2.set_ylabel("Operating Margin", color=TEXT_DIM, fontsize=8)
        ax_h2.tick_params(colors=TEXT_DIM, labelsize=7.5)
        ax_h2.figure.axes[-1].tick_params(colors=TEXT_DIM, labelsize=7)
        st.pyplot(fig_h2)
        plt.close(fig_h2)

    st.markdown("## Full Sensitivity Table — WACC × Terminal Growth")
    st.dataframe(
        df_h1.style
        .format(lambda x: f"${x:.1f}" if not np.isnan(x) else "—")
        .background_gradient(cmap='RdYlGn', axis=None)
        .set_table_styles(TABLE_STYLES),
        use_container_width=True
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;'
    'color:#484f58;text-align:center;letter-spacing:0.06em;padding:8px 0;">'
    'FOR EDUCATIONAL / RESEARCH PURPOSES ONLY · NOT FINANCIAL ADVICE · '
    'BASE: EV $281,387M · EQUITY $317,070M · PRICE $89.59 · SHARES 3,539M · '
    'WACC 8.60% · g 4.00% · NET CASH $35,683M'
    '</div>', unsafe_allow_html=True
)
