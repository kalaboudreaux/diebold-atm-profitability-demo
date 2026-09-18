"""
ATM Profitability Intelligence Platform
Diebold Nixdorf × Snowflake — POC Demonstration
Prepared for: Tyler Wise (Finance), Philip Mannering (Aimpoint Digital)
Built by: Kala Boudreaux & Jordan Ude, Snowflake Account Team
"""

import random
import time
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ATM Profitability & Operational Excellence | Diebold Nixdorf",
    page_icon="🏧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #00447C 0%, #002055 100%);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #dce8f5 !important; }
[data-testid="stSidebar"] hr { border-color: #1a5a9a; }

.hero-banner {
    background: linear-gradient(135deg, #00447C 0%, #002055 100%);
    padding: 1.4rem 2rem;
    border-radius: 8px;
    margin-bottom: 1.2rem;
}
.hero-title { color: white; font-size: 1.6rem; font-weight: 700; margin: 0; }
.hero-sub { color: #9BC8E8; font-size: 0.9rem; margin-top: 0.3rem; }

.state-card { border: 1px solid #e0e0e0; border-radius: 6px; padding: 1rem 1.2rem; }
.current-state { border-top: 4px solid #FF6B00; }
.future-state  { border-top: 4px solid #00A651; }

.insight-box {
    background: #EBF5FB;
    border-left: 4px solid #00447C;
    padding: 0.8rem 1rem;
    border-radius: 0 4px 4px 0;
    margin: 0.6rem 0;
}
.align-note { color: #555; font-size: 0.85rem; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA — MARKETS, BANK CUSTOMERS, SITE METADATA
# ─────────────────────────────────────────────────────────────────────────────

REGIONS: dict[str, list[tuple]] = {
    "Northeast": [
        ("New York, NY",      40.713, -74.006, 1.28),
        ("Boston, MA",        42.360, -71.059, 1.16),
        ("Philadelphia, PA",  39.953, -75.165, 1.06),
        ("Newark, NJ",        40.736, -74.172, 1.00),
        ("Hartford, CT",      41.766, -72.685, 0.94),
    ],
    "Southeast": [
        ("Atlanta, GA",   33.749, -84.388, 1.21),
        ("Miami, FL",     25.762, -80.192, 1.19),
        ("Charlotte, NC", 35.227, -80.843, 1.07),
        ("Nashville, TN", 36.163, -86.782, 1.10),
        ("Tampa, FL",     27.951, -82.457, 0.99),
        ("Orlando, FL",   28.538, -81.379, 1.13),
    ],
    "Midwest": [
        ("Chicago, IL",      41.878, -87.630, 1.17),
        ("Detroit, MI",      42.331, -83.046, 0.89),
        ("Cleveland, OH",    41.499, -81.694, 0.87),
        ("Columbus, OH",     39.961, -82.999, 0.94),
        ("Indianapolis, IN", 39.768, -86.158, 0.91),
        ("Minneapolis, MN",  44.978, -93.265, 1.04),
    ],
    "Southwest": [
        ("Dallas, TX",     32.777, -96.797, 1.19),
        ("Houston, TX",    29.760, -95.370, 1.23),
        ("Phoenix, AZ",    33.448, -112.074, 1.11),
        ("San Antonio, TX",29.424, -98.494,  1.04),
        ("Austin, TX",     30.267, -97.743,  1.26),
        ("Denver, CO",     39.739, -104.990, 1.09),
    ],
    "West Coast": [
        ("Los Angeles, CA",  34.052, -118.244, 1.31),
        ("San Francisco, CA",37.775, -122.419, 1.37),
        ("Seattle, WA",      47.606, -122.332, 1.22),
        ("Portland, OR",     45.505, -122.675, 1.06),
        ("San Diego, CA",    32.716, -117.161, 1.14),
    ],
}

BANK_CUSTOMERS = [
    ("Bank of America", 22),
    ("JPMorgan Chase",  18),
    ("Wells Fargo",     15),
    ("Citibank",        10),
    ("US Bancorp",       8),
    ("Truist Financial", 7),
    ("PNC Financial",    6),
    ("Capital One",      5),
    ("TD Bank",          4),
    ("Regions Bank",     3),
    ("KeyBank",          2),
]

LOCATION_TYPES = [
    ("Bank Branch",         1.10),
    ("Retail – Grocery",    1.22),
    ("Retail – Convenience",1.00),
    ("Transit Hub",         1.16),
    ("Hospital/Medical",    0.94),
    ("University Campus",   1.06),
    ("Hotel/Casino",        1.28),
    ("Corporate Campus",    0.84),
]

ATM_TYPES = [
    ("DN Series Recycler",      1.16),
    ("DN Series Cash Dispenser",1.00),
    ("Vynamic TCR",             1.13),
    ("Legacy Cash Dispenser",   0.87),
    ("Drive-Up ATM",            1.05),
]


@st.cache_data
def generate_sites() -> pd.DataFrame:
    rows = []
    sid = 1
    for region, markets in REGIONS.items():
        for city, lat, lon, mkt_factor in markets:
            n = random.randint(7, 20)
            for _ in range(n):
                bank = random.choices(
                    [b[0] for b in BANK_CUSTOMERS],
                    weights=[b[1] for b in BANK_CUSTOMERS],
                )[0]
                loc_type, loc_f = random.choices(LOCATION_TYPES)[0]
                atm_type, atm_f = random.choices(ATM_TYPES, weights=[3, 2, 2, 2, 1])[0]
                pf = round(mkt_factor * loc_f * atm_f * np.random.uniform(0.87, 1.13), 3)
                status = random.choices(
                    ["Active", "Under Maintenance", "Low Performer"],
                    weights=[84, 9, 7],
                )[0]
                rows.append({
                    "site_id":             f"DBD-{sid:04d}",
                    "city":                city,
                    "region":              region,
                    "lat":                 lat + np.random.uniform(-0.18, 0.18),
                    "lon":                 lon + np.random.uniform(-0.18, 0.18),
                    "bank_customer":       bank,
                    "location_type":       loc_type,
                    "atm_type":            atm_type,
                    "install_year":        random.randint(2017, 2024),
                    "profitability_factor":pf,
                    "status":              status,
                })
                sid += 1
    return pd.DataFrame(rows)


@st.cache_data
def generate_monthly_pl(sites: pd.DataFrame) -> pd.DataFrame:
    today = date.today()
    rows = []
    for _, s in sites.iterrows():
        pf = s["profitability_factor"]
        for mo in range(17, -1, -1):
            mdate = (today.replace(day=1) - timedelta(days=mo * 30)).replace(day=1)
            seasonal = 1.0 + 0.07 * np.sin((mdate.month - 3) * np.pi / 6)

            txns = max(50, int(np.random.normal(880 * pf * seasonal, 75)))
            t_rev = round(txns * np.random.uniform(0.92, 1.22), 0)
            t_sur = round(txns * np.random.uniform(0.21, 0.33), 0)
            t_mgd = round(np.random.uniform(155, 325), 0)
            total_rev = t_rev + t_sur + t_mgd

            c_cash    = round(max(0, np.random.normal(298, 38)), 0)
            c_maint   = round(max(0, np.random.normal(198, 30)), 0)
            c_armored = round(max(0, np.random.normal(168, 22)), 0)
            c_rent    = round(max(0, np.random.normal(290, 88) / pf), 0)
            c_network = round(np.random.uniform(58, 74), 0)
            total_cost = c_cash + c_maint + c_armored + c_rent + c_network

            net    = round(total_rev - total_cost, 0)
            margin = round(net / total_rev * 100, 1) if total_rev > 0 else 0.0
            uptime = round(min(99.9, max(88.0, np.random.normal(97.5, 1.3))), 1)

            # Service-call rate: older machines and lower profitability-factor sites
            # tend to require more physical service visits — this is what drives the
            # "price problem vs. cost problem" diagnostic below.
            age = max(0, mdate.year - s["install_year"])
            base_calls = 7.5 + age * 0.55 + max(0, (1.05 - pf)) * 6
            service_calls = max(1, int(round(np.random.normal(base_calls, 1.6))))

            rows.append({
                "site_id":          s["site_id"],
                "month_date":       mdate,
                "month_str":        mdate.strftime("%Y-%m"),
                "txn_count":        txns,
                "rev_txn_fees":     round(total_rev * 0.61, 0),
                "rev_surcharge":    round(total_rev * 0.22, 0),
                "rev_managed_svc":  round(total_rev * 0.17, 0),
                "total_revenue":    total_rev,
                "cost_cash_mgmt":   c_cash,
                "cost_maintenance": c_maint,
                "cost_armored_car": c_armored,
                "cost_rent":        c_rent,
                "cost_network":     c_network,
                "total_cost":       total_cost,
                "net_income":       net,
                "margin_pct":       margin,
                "uptime_pct":       uptime,
                "service_calls":    service_calls,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & MERGE DATA
# ─────────────────────────────────────────────────────────────────────────────

sites_df   = generate_sites()
monthly_df = generate_monthly_pl(sites_df)

monthly_full = monthly_df.merge(
    sites_df[[
        "site_id", "bank_customer", "region", "city",
        "atm_type", "location_type", "status", "profitability_factor",
        "install_year",
    ]],
    on="site_id",
)

TODAY              = date.today()
CURR_MONTH_STR     = TODAY.replace(day=1).strftime("%Y-%m")
PREV_MONTH_STR     = (TODAY.replace(day=1) - timedelta(days=1)).replace(day=1).strftime("%Y-%m")
CURR_MONTH_DISPLAY = TODAY.replace(day=1).strftime("%B %Y")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏧 ATM Profitability\n**Intelligence Platform**")
    st.markdown("---")

    st.markdown("**Portfolio Filters**")
    all_regions = ["All Regions"] + sorted(sites_df["region"].unique().tolist())
    sel_region  = st.selectbox("Region", all_regions)

    all_banks = ["All Banks"] + sorted(sites_df["bank_customer"].unique().tolist())
    sel_bank  = st.selectbox("Bank Customer", all_banks)

    all_statuses = ["All Statuses"] + sorted(sites_df["status"].unique().tolist())
    sel_status   = st.selectbox("Site Status", all_statuses)

    st.markdown("---")
    st.markdown("**Portfolio Stats**")
    total_sites  = len(sites_df)
    active_sites = len(sites_df[sites_df["status"] == "Active"])
    st.metric("Total ATM Sites",  f"{total_sites:,}")
    st.metric("Active Sites",     f"{active_sites:,}",
              delta=f"{active_sites / total_sites * 100:.0f}% active")

    st.markdown("---")
    st.caption(f"📊 Illustrative demo data · {CURR_MONTH_DISPLAY}")
    st.caption("Prepared for Frank Baur (EVP & COO) · Tyler Wise (Finance)")
    st.caption("Kala Boudreaux | Jordan Ude — Snowflake Account Team")


# ─────────────────────────────────────────────────────────────────────────────
# FILTER HELPER
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if sel_region != "All Regions":
        df = df[df["region"] == sel_region]
    if sel_bank != "All Banks":
        df = df[df["bank_customer"] == sel_bank]
    if sel_status != "All Statuses":
        df = df[df["status"] == sel_status]
    return df


filtered_sites   = apply_filters(sites_df.copy())
filtered_monthly = apply_filters(monthly_full.copy())
curr_f = filtered_monthly[filtered_monthly["month_str"] == CURR_MONTH_STR]
prev_f = filtered_monthly[filtered_monthly["month_str"] == PREV_MONTH_STR]

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🏧 ATM Profitability &amp; Operational Excellence Platform</div>
  <div class="hero-sub">
    Diebold Nixdorf × Snowflake &nbsp;·&nbsp; Powered by Cortex AI
    &nbsp;·&nbsp; Source → Make → Deliver, quantified at the site level
    &nbsp;·&nbsp; Prepared for Frank Baur (EVP &amp; COO) and Tyler Wise (Finance)
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Executive Overview",
    "🏧 Site P&L",
    "🏦 Customer Analytics",
    "🎯 Prescriptive Diagnostics",
    "🧭 Operational Excellence",
    "🗺️ Geographic View",
    "🤖 AI Insights",
    "⚙️ Architecture & Roadmap",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

with tab1:

    # ── KPIs ──────────────────────────────────────────────────────────────
    total_rev  = curr_f["total_revenue"].sum()
    prev_rev   = prev_f["total_revenue"].sum()
    total_net  = curr_f["net_income"].sum()
    prev_net   = prev_f["net_income"].sum()
    avg_margin = curr_f["margin_pct"].mean() if len(curr_f) > 0 else 0.0
    avg_uptime = curr_f["uptime_pct"].mean() if len(curr_f) > 0 else 0.0
    total_txns = curr_f["txn_count"].sum()
    n_sites    = curr_f["site_id"].nunique()

    rev_delta = (total_rev - prev_rev) / prev_rev * 100 if prev_rev > 0 else 0
    net_delta = (total_net - prev_net) / prev_net * 100 if prev_net > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Portfolio Revenue (MTD)",  f"${total_rev / 1e6:.2f}M",
              f"{rev_delta:+.1f}% vs prior month")
    k2.metric("Net Income (MTD)",         f"${total_net / 1e6:.2f}M",
              f"{net_delta:+.1f}% vs prior month")
    k3.metric("Avg Net Margin",           f"{avg_margin:.1f}%")
    k4.metric("Avg ATM Uptime",           f"{avg_uptime:.1f}%",
              "Target: ≥97.5%")
    k5.metric("Transactions (MTD)",       f"{total_txns:,}",
              f"{n_sites:,} sites")

    st.markdown("---")

    c_l, c_r = st.columns([3, 2])

    with c_l:
        st.markdown("**Monthly Revenue & Net Income — 18-Month Trend**")
        trend = (
            filtered_monthly
            .groupby("month_str")
            .agg(total_revenue=("total_revenue", "sum"),
                 net_income=("net_income", "sum"))
            .reset_index()
            .sort_values("month_str")
        )
        trend_m = trend.melt(
            id_vars="month_str",
            value_vars=["total_revenue", "net_income"],
            var_name="metric", value_name="amount",
        )
        trend_m["amount_M"] = trend_m["amount"] / 1e6
        trend_m["metric"]   = trend_m["metric"].map(
            {"total_revenue": "Revenue", "net_income": "Net Income"}
        )
        chart = (
            alt.Chart(trend_m)
            .mark_line(point=True)
            .encode(
                x=alt.X("month_str:O", title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("amount_M:Q", title="$M", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "metric:N",
                    scale=alt.Scale(
                        domain=["Revenue", "Net Income"],
                        range=["#00447C", "#00A651"],
                    ),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("month_str:O", title="Month"),
                    alt.Tooltip("metric:N"),
                    alt.Tooltip("amount_M:Q", title="Amount ($M)", format=".2f"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

    with c_r:
        st.markdown("**Cost Structure (All Months)**")
        cost_agg = filtered_monthly[[
            "cost_cash_mgmt", "cost_maintenance",
            "cost_armored_car", "cost_rent", "cost_network",
        ]].sum()
        cost_df = pd.DataFrame({
            "category": [
                "Cash Management", "Maintenance", "Armored Car",
                "Rent / Real Estate", "Network / Comms",
            ],
            "amount_M": cost_agg.values / 1e6,
        })
        pie = (
            alt.Chart(cost_df)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("amount_M:Q"),
                color=alt.Color(
                    "category:N",
                    scale=alt.Scale(scheme="blues"),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("category:N", title="Cost Category"),
                    alt.Tooltip("amount_M:Q", title="Total ($M)", format=".2f"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(pie, use_container_width=True)

    st.markdown("---")

    # ── Current vs Future State ────────────────────────────────────────────
    st.markdown("**Current State vs. Snowflake Future State**")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
<div class="state-card current-state">
<h4>⚠️ Current State — Alteryx + Power BI</h4>
<ul>
<li>📅 <strong>Monthly reporting only</strong> — no intra-month P&L visibility</li>
<li>📁 <strong>Manual file downloads</strong> — hours of engineering each refresh</li>
<li>🗺️ <strong>Cost-center geography level only</strong> — no individual site drill</li>
<li>❌ <strong>No customer cross-cut</strong> — can't see BofA's nationwide ATM P&L</li>
<li>📊 <strong>Static reports</strong> — Finance can't run ad-hoc queries without IT</li>
<li>🏗️ <strong>8–9 month build</strong> by Aimpoint — brittle, limited scalability</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
<div class="state-card future-state">
<h4>✅ Future State — Snowflake + Cortex AI</h4>
<ul>
<li>⚡ <strong>Real-time / daily P&L</strong> — intra-month visibility, automated pipelines</li>
<li>🤖 <strong>Zero manual downloads</strong> — Snowpipe Streaming + Dynamic Tables</li>
<li>📍 <strong>Every ATM site</strong> — granular P&L at the location level</li>
<li>🏦 <strong>Customer cross-cut P&L</strong> — BofA, Chase, WF nationwide in one click</li>
<li>💬 <strong>Natural language queries</strong> — Finance asks questions in plain English</li>
<li>🗺️ <strong>Competitive market intel</strong> — zip/county data via Snowflake Marketplace</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Investor Day Alignment ─────────────────────────────────────────────
    st.markdown("**Diebold Nixdorf 2025 Investor Day — Financial Targets Alignment**")

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Banking Operating Profit",   "$507M",       "+12% YoY (2024→2025)")
    i2.metric("Adjusted EBITDA",            "$485M",       "2025 actuals")
    i3.metric("Free Cash Flow",             "$239M",       "Doubled year-over-year")
    i4.metric("Revenue Growth Target",      "Mid-single %","Banking segment by 2027")

    st.markdown("""
<div class="insight-box">
<strong>🎯 Strategic Alignment:</strong> Diebold Nixdorf's 2025 Investor Day Growth Acceleration Plan targets
<strong>double-digit adjusted EBITDA growth</strong> and <strong>60%+ free cash flow conversion by 2027</strong>.
Site-level ATM profitability analytics directly enables cost optimization, underperformer triage,
and data-driven capital allocation decisions — core levers for these stated financial targets.
Serving 80%+ of the world's top 100 financial institutions, even marginal margin improvement
across the installed ATM base represents material financial impact.
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — SITE P&L
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Site-Level P&L Explorer")
    st.caption(
        "Individual ATM profitability — the primary gap in the current Alteryx/Power BI solution. "
        f"Showing {CURR_MONTH_DISPLAY} data."
    )

    site_summary = (
        curr_f.groupby("site_id")
        .agg(
            revenue=("total_revenue", "sum"),
            cost=("total_cost", "sum"),
            net_income=("net_income", "sum"),
            margin_pct=("margin_pct", "mean"),
            txn_count=("txn_count", "sum"),
            uptime=("uptime_pct", "mean"),
        )
        .reset_index()
        .merge(
            sites_df[["site_id", "city", "region", "bank_customer",
                       "location_type", "atm_type", "status"]],
            on="site_id",
        )
        .sort_values("net_income", ascending=False)
    )

    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown("**Top 20 & Bottom 20 Sites by Net Income (MTD)**")
        top20 = site_summary.head(20).assign(label="Top Performer")
        bot20 = site_summary.tail(20).assign(label="Bottom Performer")
        tb    = pd.concat([top20, bot20])

        bar = (
            alt.Chart(tb)
            .mark_bar()
            .encode(
                x=alt.X("net_income:Q", title="Net Income ($)",
                         axis=alt.Axis(format="$,.0f")),
                y=alt.Y("site_id:N", sort="-x", title=None,
                         axis=alt.Axis(labelFontSize=9)),
                color=alt.Color(
                    "label:N",
                    scale=alt.Scale(
                        domain=["Top Performer", "Bottom Performer"],
                        range=["#00447C", "#FF6B00"],
                    ),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("site_id:N",        title="Site"),
                    alt.Tooltip("city:N",            title="City"),
                    alt.Tooltip("bank_customer:N",   title="Bank"),
                    alt.Tooltip("net_income:Q",      title="Net Income ($)", format="$,.0f"),
                    alt.Tooltip("margin_pct:Q",      title="Margin %", format=".1f"),
                ],
            )
            .properties(height=520)
        )
        st.altair_chart(bar, use_container_width=True)

    with c_right:
        st.markdown("**Margin Distribution**")
        hist = (
            alt.Chart(site_summary)
            .mark_bar(color="#00447C", opacity=0.78)
            .encode(
                x=alt.X("margin_pct:Q", bin=alt.Bin(step=3), title="Net Margin (%)"),
                y=alt.Y("count():Q", title="# Sites"),
                tooltip=[
                    alt.Tooltip("margin_pct:Q", title="Margin %", bin=True),
                    alt.Tooltip("count():Q", title="# Sites"),
                ],
            )
            .properties(height=210)
        )
        st.altair_chart(hist, use_container_width=True)

        st.markdown("**Revenue by Location Type**")
        loc_rev = (
            curr_f.groupby("location_type")["total_revenue"]
            .sum()
            .reset_index()
            .rename(columns={"total_revenue": "rev_K"})
        )
        loc_rev["rev_K"] /= 1_000
        loc_rev = loc_rev.sort_values("rev_K")

        loc_bar = (
            alt.Chart(loc_rev)
            .mark_bar(color="#00A651")
            .encode(
                x=alt.X("rev_K:Q", title="Revenue ($K)"),
                y=alt.Y("location_type:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("location_type:N", title="Location"),
                    alt.Tooltip("rev_K:Q", title="Revenue ($K)", format=",.1f"),
                ],
            )
            .properties(height=230)
        )
        st.altair_chart(loc_bar, use_container_width=True)

    st.markdown("**Full Site P&L — searchable table**")
    display = site_summary[[
        "site_id", "city", "region", "bank_customer", "location_type",
        "revenue", "cost", "net_income", "margin_pct", "txn_count", "uptime", "status",
    ]].rename(columns={
        "site_id":       "Site ID",
        "city":          "City",
        "region":        "Region",
        "bank_customer": "Bank Customer",
        "location_type": "Location Type",
        "revenue":       "Revenue ($)",
        "cost":          "Cost ($)",
        "net_income":    "Net Income ($)",
        "margin_pct":    "Margin %",
        "txn_count":     "Transactions",
        "uptime":        "Uptime %",
        "status":        "Status",
    })

    st.dataframe(
        display,
        use_container_width=True,
        height=360,
        column_config={
            "Revenue ($)":    st.column_config.NumberColumn(format="$%.0f"),
            "Cost ($)":       st.column_config.NumberColumn(format="$%.0f"),
            "Net Income ($)": st.column_config.NumberColumn(format="$%.0f"),
            "Margin %":       st.column_config.NumberColumn(format="%.1f%%"),
            "Uptime %":       st.column_config.ProgressColumn(
                                  min_value=80, max_value=100, format="%.1f%%"),
        },
        hide_index=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — CUSTOMER ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Customer Cross-Cut P&L")
    st.caption(
        "See any bank's nationwide ATM profitability in one view — "
        "a capability that does not exist in the current Alteryx solution."
    )

    bank_list = sorted(monthly_full["bank_customer"].unique().tolist())
    sel_cust  = st.selectbox(
        "Select Bank Customer",
        bank_list,
        index=bank_list.index("Bank of America") if "Bank of America" in bank_list else 0,
        key="cust_selector",
    )

    cust_all   = monthly_full[monthly_full["bank_customer"] == sel_cust]
    cust_curr  = cust_all[cust_all["month_str"] == CURR_MONTH_STR]
    cust_prev  = cust_all[cust_all["month_str"] == PREV_MONTH_STR]

    c_rev    = cust_curr["total_revenue"].sum()
    c_net    = cust_curr["net_income"].sum()
    c_margin = cust_curr["margin_pct"].mean()
    c_sites  = cust_curr["site_id"].nunique()
    p_rev    = cust_prev["total_revenue"].sum()
    p_net    = cust_prev["net_income"].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{sel_cust} — Sites",   f"{c_sites:,}")
    k2.metric("Revenue (MTD)",          f"${c_rev / 1e6:.2f}M",
              f"{(c_rev - p_rev) / p_rev * 100 if p_rev else 0:+.1f}% vs prior")
    k3.metric("Net Income (MTD)",       f"${c_net / 1e6:.2f}M",
              f"{(c_net - p_net) / p_net * 100 if p_net else 0:+.1f}% vs prior")
    k4.metric("Avg Net Margin",         f"{c_margin:.1f}%")

    c_l, c_r = st.columns(2)

    with c_l:
        st.markdown(f"**{sel_cust} — Monthly Revenue Trend**")
        cust_trend = (
            cust_all
            .groupby("month_str")
            .agg(Revenue=("total_revenue", "sum"),
                 Net_Income=("net_income", "sum"))
            .reset_index()
            .sort_values("month_str")
        )
        ct_m = cust_trend.melt(
            id_vars="month_str",
            value_vars=["Revenue", "Net_Income"],
            var_name="metric", value_name="amount",
        )
        ct_m["amount_K"] = ct_m["amount"] / 1_000
        ct_m["metric"]   = ct_m["metric"].str.replace("_", " ")

        ct_chart = (
            alt.Chart(ct_m)
            .mark_line(point=True)
            .encode(
                x=alt.X("month_str:O", title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("amount_K:Q", title="$K", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "metric:N",
                    scale=alt.Scale(
                        domain=["Revenue", "Net Income"],
                        range=["#00447C", "#00A651"],
                    ),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("month_str:O", title="Month"),
                    alt.Tooltip("metric:N"),
                    alt.Tooltip("amount_K:Q", title="Amount ($K)", format=",.0f"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(ct_chart, use_container_width=True)

    with c_r:
        st.markdown(f"**{sel_cust} — Revenue by Region**")
        by_region = (
            cust_curr
            .groupby("region")
            .agg(
                revenue=("total_revenue", "sum"),
                net_income=("net_income", "sum"),
                sites=("site_id", "nunique"),
            )
            .reset_index()
            .sort_values("revenue")
        )
        by_region["rev_K"] = by_region["revenue"] / 1_000
        by_region["net_K"] = by_region["net_income"] / 1_000

        reg_bar = (
            alt.Chart(by_region)
            .mark_bar(color="#00447C")
            .encode(
                x=alt.X("rev_K:Q", title="Revenue ($K)"),
                y=alt.Y("region:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("region:N",  title="Region"),
                    alt.Tooltip("rev_K:Q",   title="Revenue ($K)", format=",.0f"),
                    alt.Tooltip("net_K:Q",   title="Net Income ($K)", format=",.0f"),
                    alt.Tooltip("sites:Q",   title="# Sites"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(reg_bar, use_container_width=True)

    st.markdown(f"**{sel_cust} — Top Markets by Net Income (MTD)**")
    top_markets = (
        cust_curr
        .groupby(["city", "region"])
        .agg(
            sites=("site_id", "nunique"),
            revenue=("total_revenue", "sum"),
            net_income=("net_income", "sum"),
            margin=("margin_pct", "mean"),
        )
        .reset_index()
        .sort_values("net_income", ascending=False)
        .head(20)
    )
    top_markets["rev_K"] = top_markets["revenue"] / 1_000
    top_markets["net_K"] = top_markets["net_income"] / 1_000

    st.dataframe(
        top_markets[["city", "region", "sites", "rev_K", "net_K", "margin"]].rename(columns={
            "city":   "Market",
            "region": "Region",
            "sites":  "Sites",
            "rev_K":  "Revenue ($K)",
            "net_K":  "Net Income ($K)",
            "margin": "Margin %",
        }),
        use_container_width=True,
        height=320,
        column_config={
            "Revenue ($K)":    st.column_config.NumberColumn(format="$%.1f"),
            "Net Income ($K)": st.column_config.NumberColumn(format="$%.1f"),
            "Margin %":        st.column_config.NumberColumn(format="%.1f%%"),
        },
        hide_index=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRESCRIPTIVE DIAGNOSTICS ENGINE
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("🎯 Prescriptive Diagnostics Engine")
    st.caption(
        "This is the agent Diebold's Finance and Ops teams would run every month: it identifies "
        "*which* sites are underperforming, diagnoses *why* against Diebold's own national fleet as "
        "the benchmark — no external data required — and prescribes a specific, ranked action. "
        "Adjust the sensitivity below and re-run the scan."
    )

    diag_month = curr_f.copy()

    st.markdown("**Step 1 — Set diagnostic sensitivity**")
    s1, s2, s3 = st.columns(3)
    with s1:
        margin_floor = st.slider(
            "Flag sites below this margin %", min_value=5, max_value=30, value=20, step=1,
            key="margin_floor",
        )
    with s2:
        price_threshold = st.slider(
            "Pricing-issue trigger: rev/txn vs. peer", min_value=-30, max_value=-5, value=-12, step=1,
            format="%d%%", key="price_threshold",
        )
    with s3:
        cost_threshold = st.slider(
            "Cost-issue trigger: call rate vs. peer", min_value=15, max_value=75, value=35, step=5,
            format="+%d%%", key="cost_threshold",
        )

    run_scan = st.button("🔍 Run Diagnostic Scan", type="primary", use_container_width=False)
    if run_scan:
        with st.spinner("Cortex AI scanning fleet against peer benchmarks..."):
            time.sleep(0.8)
        st.session_state["scan_run"] = True

    if not st.session_state.get("scan_run"):
        st.info("Set your thresholds above and click **Run Diagnostic Scan** to generate the prescriptive action list.")
    else:
        # Peer benchmark = same ATM model type, nationwide (DN's own fleet as the benchmark)
        peer_bench = (
            diag_month.groupby("atm_type")
            .agg(
                peer_rev_per_txn=("total_revenue", lambda x: x.sum() / diag_month.loc[x.index, "txn_count"].sum()),
                peer_calls=("service_calls", "mean"),
            )
            .reset_index()
        )

        site_diag = (
            diag_month.groupby(["site_id", "city", "region", "bank_customer", "atm_type", "install_year"])
            .agg(
                revenue=("total_revenue", "sum"),
                txns=("txn_count", "sum"),
                net_income=("net_income", "sum"),
                margin_pct=("margin_pct", "mean"),
                service_calls=("service_calls", "sum"),
            )
            .reset_index()
            .merge(peer_bench, on="atm_type")
        )
        site_diag["rev_per_txn"]   = site_diag["revenue"] / site_diag["txns"]
        site_diag["price_gap_pct"] = (site_diag["rev_per_txn"] - site_diag["peer_rev_per_txn"]) / site_diag["peer_rev_per_txn"] * 100
        site_diag["call_gap_pct"]  = (site_diag["service_calls"] - site_diag["peer_calls"]) / site_diag["peer_calls"] * 100

        def classify(row):
            is_price_issue = row["price_gap_pct"] <= price_threshold
            is_cost_issue  = row["call_gap_pct"]  >= cost_threshold
            if is_price_issue and is_cost_issue:
                return "Both"
            if is_price_issue:
                return "Pricing Issue"
            if is_cost_issue:
                return "Cost Issue"
            return "Healthy"

        def recommend(row):
            if row["diagnosis"] == "Pricing Issue":
                return (f"Flag for contract repricing at next renewal — revenue/txn is "
                         f"{abs(row['price_gap_pct']):.0f}% below the {row['atm_type']} fleet average.")
            if row["diagnosis"] == "Cost Issue":
                return (f"Escalate to field maintenance — call rate is {row['call_gap_pct']:.0f}% above fleet "
                         f"average; evaluate part-failure pattern or schedule machine replacement.")
            if row["diagnosis"] == "Both":
                return "Review contract AND escalate maintenance — both revenue and cost are out of range vs. peers."
            return "Within normal range — no action needed."

        def priority(row):
            if row["diagnosis"] == "Both":
                return 1
            if row["diagnosis"] in ("Pricing Issue", "Cost Issue"):
                return 2
            return 3

        site_diag["diagnosis"]      = site_diag.apply(classify, axis=1)
        site_diag["recommendation"] = site_diag.apply(recommend, axis=1)
        site_diag["priority"]       = site_diag.apply(priority, axis=1)
        unprofitable = (
            site_diag[site_diag["margin_pct"] < margin_floor]
            .sort_values(["priority", "margin_pct"])
            .copy()
        )

        n_pricing = (unprofitable["diagnosis"] == "Pricing Issue").sum()
        n_cost    = (unprofitable["diagnosis"] == "Cost Issue").sum()
        n_both    = (unprofitable["diagnosis"] == "Both").sum()

        st.markdown("**Step 2 — Scan results**")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric(f"Sites Below {margin_floor}% Margin", f"{len(unprofitable):,}")
        d2.metric("→ Pricing Issue", f"{n_pricing:,}", "reprice at renewal")
        d3.metric("→ Cost Issue", f"{n_cost:,}", "escalate to maintenance")
        d4.metric("→ Both", f"{n_both:,}", "review contract + ops")

        tbl_tab, action_tab = st.tabs(["📋 Full ranked list", "📝 Action cards — top 5 priority sites"])

        with tbl_tab:
            show_cols = unprofitable[[
                "site_id", "city", "bank_customer", "atm_type", "margin_pct",
                "price_gap_pct", "call_gap_pct", "diagnosis", "recommendation",
            ]].rename(columns={
                "site_id":        "Site ID",
                "city":           "Market",
                "bank_customer":  "Bank",
                "atm_type":       "ATM Model",
                "margin_pct":     "Margin %",
                "price_gap_pct":  "Rev/Txn vs Peer %",
                "call_gap_pct":   "Call Rate vs Peer %",
                "diagnosis":      "Diagnosis",
                "recommendation": "Recommended Action",
            })
            st.dataframe(
                show_cols,
                use_container_width=True,
                height=420,
                hide_index=True,
                column_config={
                    "Margin %":            st.column_config.NumberColumn(format="%.1f%%"),
                    "Rev/Txn vs Peer %":   st.column_config.NumberColumn(format="%+.0f%%"),
                    "Call Rate vs Peer %": st.column_config.NumberColumn(format="%+.0f%%"),
                },
            )
            st.download_button(
                "⬇️ Download full action list (CSV)",
                data=show_cols.to_csv(index=False).encode("utf-8"),
                file_name="atm_prescriptive_action_list.csv",
                mime="text/csv",
            )

        with action_tab:
            DIAG_ICON = {"Both": "🔴", "Pricing Issue": "🟠", "Cost Issue": "🟡", "Healthy": "🟢"}
            for _, row in unprofitable.head(5).iterrows():
                with st.expander(
                    f"{DIAG_ICON.get(row['diagnosis'], '⚪')} {row['site_id']} — {row['city']} "
                    f"({row['bank_customer']}) · Margin {row['margin_pct']:.1f}%",
                    expanded=False,
                ):
                    ac1, ac2, ac3 = st.columns(3)
                    ac1.metric("ATM Model", row["atm_type"])
                    ac2.metric("Rev/Txn vs Peer", f"{row['price_gap_pct']:+.0f}%")
                    ac3.metric("Call Rate vs Peer", f"{row['call_gap_pct']:+.0f}%")
                    st.markdown(f"**Diagnosis:** {row['diagnosis']}")
                    st.markdown(f"**Prescribed action:** {row['recommendation']}")
                    st.markdown(f"**Install year:** {int(row['install_year'])} · **Region:** {row['region']}")

        st.markdown("---")
        st.markdown("**Step 3 — What-if: simulate fixing the pricing-issue sites**")
        fix_pct = st.slider(
            "% of flagged pricing-issue sites repriced to the fleet benchmark this quarter",
            min_value=0, max_value=100, value=50, step=5, key="fix_pct",
        )
        reprice_upside = (
            unprofitable[unprofitable["diagnosis"].isin(["Pricing Issue", "Both"])]
            .assign(potential=lambda d: d["revenue"] * (abs(d["price_gap_pct"]) / 100) * (fix_pct / 100))
            ["potential"].sum()
        )
        w1, w2 = st.columns(2)
        w1.metric("Projected monthly recapture (this sample)", f"${reprice_upside:,.0f}")
        w2.metric("Projected annualized recapture", f"${reprice_upside * 12:,.0f}")

        st.markdown(f"""
<div class="insight-box">
<strong>💡 Prescriptive insight:</strong> Repricing {fix_pct}% of the {n_pricing + n_both} flagged
pricing-issue sites to the regional/model benchmark recaptures an estimated
<strong>${reprice_upside:,.0f}/month</strong> in this sample — scaled across Diebold's full fleet,
this is the same mechanism behind the <strong>$11–22M/yr pricing-accuracy opportunity</strong>
in the POC business case. Cost-issue sites are candidates for parts/labor review rather than repricing —
conflating the two is the "blame game" this engine eliminates.
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🧭 Golden Ratio — Optimal Machine Mix (Directional)")
        st.caption("Given technician capacity and service density in a market, what mix of machine types maximizes margin without adding headcount?")

        mix = (
            diag_month.groupby(["region", "atm_type"])
            .agg(sites=("site_id", "nunique"), avg_margin=("margin_pct", "mean"))
            .reset_index()
        )
        mix_chart = (
            alt.Chart(mix)
            .mark_bar()
            .encode(
                x=alt.X("sites:Q", title="# Sites", stack="normalize"),
                y=alt.Y("region:N", title=None),
                color=alt.Color("atm_type:N", title="ATM Model", legend=alt.Legend(orient="bottom", columns=3)),
                tooltip=["region:N", "atm_type:N", "sites:Q",
                         alt.Tooltip("avg_margin:Q", title="Avg Margin %", format=".1f")],
            )
            .properties(height=260)
        )
        st.altair_chart(mix_chart, use_container_width=True)
        st.caption(
            "Regions skewed toward Legacy Cash Dispensers (lowest-margin model) are the first candidates "
            "for fleet-refresh prioritization — this view is what unlocks that conversation."
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — OPERATIONAL EXCELLENCE (COO VIEW)
# ═════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("🧭 Operational Excellence — Source, Make, Deliver")
    st.caption(
        "Framed around Diebold Nixdorf's own operating model: regional sourcing resilience, "
        "field-service efficiency (\"Deliver\"), and the branch cash-automation opportunity that "
        "physically embodies \"Make.\" Every metric below rolls up from the same site-level data "
        "used in the tabs to its left — this is what makes it a platform, not a one-off report."
    )

    ops_month = curr_f.copy()

    st.markdown("---")
    st.markdown("#### 📦 Source — Regional Resilience")
    st.caption(
        "\"In-region for the region\" sourcing reduces exposure to a single geography's disruption. "
        "This score is a concentration index (lower = more diversified = more resilient) of ATM "
        "revenue across Diebold's five U.S. regions."
    )

    reg_share = (
        ops_month.groupby("region")["total_revenue"].sum()
        .pipe(lambda s: s / s.sum())
    )
    hhi = float((reg_share ** 2).sum() * 10000)   # Herfindahl-Hirschman Index, 0–10,000 scale
    resilience_label = "High" if hhi < 2200 else ("Moderate" if hhi < 3000 else "Concentrated")

    r1, r2, r3 = st.columns(3)
    r1.metric("Revenue Concentration (HHI)", f"{hhi:,.0f}", resilience_label)
    r2.metric("Largest Single Region Share", f"{reg_share.max() * 100:.1f}%", reg_share.idxmax())
    r3.metric("Regions Covered", f"{ops_month['region'].nunique()} / 5")

    region_bar = (
        alt.Chart(reg_share.reset_index().rename(columns={"total_revenue": "share"}))
        .mark_bar(color="#00447C")
        .encode(
            x=alt.X("share:Q", title="Share of Portfolio Revenue", axis=alt.Axis(format="%")),
            y=alt.Y("region:N", sort="-x", title=None),
            tooltip=["region:N", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(height=200)
    )
    st.altair_chart(region_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔧 Deliver — Field Service Efficiency")
    st.caption(
        "Uptime and service-call rate are the two levers that translate directly into field "
        "turnaround time and technician capacity — the operational metrics that sit next to "
        "revenue in every regional P&L review."
    )

    avg_uptime_ops   = ops_month["uptime_pct"].mean() if len(ops_month) else 0.0
    avg_calls_ops    = ops_month["service_calls"].mean() if len(ops_month) else 0.0
    below_target     = ops_month.groupby("site_id")["uptime_pct"].mean()
    pct_below_target = (below_target < 97.5).mean() * 100 if len(below_target) else 0.0

    f1, f2, f3 = st.columns(3)
    f1.metric("Avg Fleet Uptime", f"{avg_uptime_ops:.1f}%", "Target ≥ 97.5%")
    f2.metric("Sites Below Uptime Target", f"{pct_below_target:.0f}%")
    f3.metric("Avg Service Calls / Site / Mo", f"{avg_calls_ops:.1f}")

    st.markdown("**Service-call reduction opportunity — what-if**")
    call_reduction_pct = st.slider(
        "If field ops reduced excess calls on flagged high-call sites by this much",
        min_value=0, max_value=75, value=25, step=5, key="call_reduction_pct",
    )
    excess_calls = (
        ops_month.groupby("atm_type")["service_calls"].mean()
        .pipe(lambda peer: ops_month.merge(peer.rename("peer_calls"), on="atm_type"))
    )
    excess_calls["excess"] = (excess_calls["service_calls"] - excess_calls["peer_calls"]).clip(lower=0)
    calls_saved = excess_calls["excess"].sum() * (call_reduction_pct / 100)
    COST_PER_CALL = 85  # illustrative fully-loaded technician dispatch cost
    monthly_savings = calls_saved * COST_PER_CALL

    cs1, cs2 = st.columns(2)
    cs1.metric("Service Calls Avoided / Month (sample)", f"{calls_saved:,.0f}")
    cs2.metric("Estimated Monthly Savings (sample)", f"${monthly_savings:,.0f}",
               f"${monthly_savings * 12:,.0f}/yr")

    st.markdown("---")
    st.markdown("#### 🏦 Make — The Closed Cash Ecosystem Opportunity")
    st.caption(
        "Up to 50% of branch operating expense ties back to cash handling. Migrating Legacy Cash "
        "Dispensers to DN Series Recyclers / Vynamic TCR turns the ATM into a mini-branch and is the "
        "single highest-ROI lever Diebold controls directly — independent of any bank contract."
    )

    fleet_mix = ops_month.groupby("atm_type")["site_id"].nunique().sort_values(ascending=False)
    legacy_n  = int(fleet_mix.get("Legacy Cash Dispenser", 0))
    total_n   = int(fleet_mix.sum())
    legacy_pct = legacy_n / total_n * 100 if total_n else 0.0

    legacy_margin  = ops_month.loc[ops_month["atm_type"] == "Legacy Cash Dispenser", "margin_pct"].mean()
    recycler_margin = ops_month.loc[ops_month["atm_type"] == "DN Series Recycler", "margin_pct"].mean()
    margin_lift = (recycler_margin - legacy_margin) if pd.notna(legacy_margin) and pd.notna(recycler_margin) else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Legacy Cash Dispensers in Fleet", f"{legacy_n:,}", f"{legacy_pct:.0f}% of sites")
    m2.metric("Margin Gap vs. Recycler Fleet", f"{margin_lift:+.1f} pts")
    m3.metric("Avg Legacy Site Margin", f"{legacy_margin:.1f}%" if pd.notna(legacy_margin) else "n/a")

    migrate_pct = st.slider(
        "% of Legacy Cash Dispensers migrated to DN Series Recycler / Vynamic TCR",
        min_value=0, max_value=100, value=30, step=5, key="migrate_pct",
    )
    legacy_rev = ops_month.loc[ops_month["atm_type"] == "Legacy Cash Dispenser", "total_revenue"].sum()
    migration_lift = legacy_rev * (migrate_pct / 100) * (margin_lift / 100)
    st.metric(
        f"Projected monthly net-income lift from migrating {migrate_pct}% of Legacy units",
        f"${max(migration_lift, 0):,.0f}",
    )

    st.markdown(f"""
<div class="insight-box">
<strong>💡 For the COO:</strong> Frank Baur's own framing — <em>"What gets measured improves,
but what gets understood transforms"</em> — is the thesis of this platform. Diebold already runs
DNAccelerator as its lean operating system on the shop floor; this extends the same measure →
understand → act discipline to the field, turning site-level P&L, service-call, and fleet-mix data
into a standing prescriptive engine rather than a monthly static report.
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — GEOGRAPHIC VIEW
# ═════════════════════════════════════════════════════════════════════════════

with tab6:
    st.subheader("Geographic ATM Profitability Map")
    st.caption("Site-level map view — identify underperforming markets at a glance")

    map_pl = (
        curr_f
        .groupby("site_id")
        .agg(net_income=("net_income", "sum"),
             margin_pct=("margin_pct", "mean"))
        .reset_index()
    )
    map_df = filtered_sites.merge(map_pl, on="site_id", how="left")
    map_df["net_income"] = map_df["net_income"].fillna(0)
    map_df["margin_pct"] = map_df["margin_pct"].fillna(0)

    st.map(
        map_df.rename(columns={"lat": "latitude", "lon": "longitude"})[
            ["latitude", "longitude"]
        ],
        size=25,
    )

    st.markdown("**Regional P&L Summary**")
    reg_summary = (
        curr_f
        .groupby("region")
        .agg(
            sites=("site_id", "nunique"),
            revenue=("total_revenue", "sum"),
            net_income=("net_income", "sum"),
            margin=("margin_pct", "mean"),
            avg_uptime=("uptime_pct", "mean"),
        )
        .reset_index()
        .sort_values("net_income", ascending=False)
    )
    reg_summary["rev_M"]          = reg_summary["revenue"] / 1e6
    reg_summary["net_M"]          = reg_summary["net_income"] / 1e6
    reg_summary["rev_per_site_K"] = reg_summary["revenue"] / reg_summary["sites"] / 1_000

    reg_bar = (
        alt.Chart(reg_summary)
        .mark_bar()
        .encode(
            x=alt.X("region:N", title="Region", sort="-y"),
            y=alt.Y("net_M:Q",  title="Net Income ($M)"),
            color=alt.Color(
                "margin:Q",
                scale=alt.Scale(scheme="greens", domain=[15, 32]),
                title="Margin %",
            ),
            tooltip=[
                alt.Tooltip("region:N",           title="Region"),
                alt.Tooltip("sites:Q",            title="# Sites"),
                alt.Tooltip("rev_M:Q",            title="Revenue ($M)", format=".2f"),
                alt.Tooltip("net_M:Q",            title="Net Income ($M)", format=".2f"),
                alt.Tooltip("margin:Q",           title="Avg Margin %", format=".1f"),
                alt.Tooltip("rev_per_site_K:Q",   title="Rev per Site ($K)", format=".1f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(reg_bar, use_container_width=True)

    st.dataframe(
        reg_summary[["region", "sites", "rev_M", "net_M", "margin",
                     "avg_uptime", "rev_per_site_K"]].rename(columns={
            "region":        "Region",
            "sites":         "Sites",
            "rev_M":         "Revenue ($M)",
            "net_M":         "Net Income ($M)",
            "margin":        "Avg Margin %",
            "avg_uptime":    "Avg Uptime %",
            "rev_per_site_K":"Rev per Site ($K)",
        }),
        use_container_width=True,
        column_config={
            "Revenue ($M)":      st.column_config.NumberColumn(format="$%.2f"),
            "Net Income ($M)":   st.column_config.NumberColumn(format="$%.2f"),
            "Avg Margin %":      st.column_config.NumberColumn(format="%.1f%%"),
            "Avg Uptime %":      st.column_config.NumberColumn(format="%.1f%%"),
            "Rev per Site ($K)": st.column_config.NumberColumn(format="$%.1f"),
        },
        hide_index=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — AI INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════

with tab7:
    st.subheader("🤖 Cortex AI — Natural Language ATM Analytics")
    st.caption(
        "Ask any question about your ATM portfolio in plain English. "
        "In production this connects directly to your Snowflake data model — "
        "no SQL required from Finance."
    )

    EXAMPLE_QUERIES = [
        "Which markets have the lowest profitability over the last 6 months?",
        "Show me all Bank of America sites where net margin is below 15%",
        "What is the monthly net income trend for the Chicago ATM portfolio?",
        "Which ATM types generate the highest revenue per transaction?",
        "Compare Q1 vs Q2 performance by region",
        "Identify sites where cash management costs are above portfolio average",
        "What is BofA's nationwide P&L across all ATMs this year?",
        "Which sites have uptime below 96% and negative net income?",
    ]

    st.markdown("**Quick-start — pick an example query:**")
    q_cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUERIES):
        if q_cols[i % 2].button(f"💬 {q}", key=f"eq_{i}", use_container_width=True):
            st.session_state["ai_q"] = q

    st.markdown("**Or type your own question:**")
    user_q = st.text_input(
        "Ask anything about your ATM portfolio...",
        value=st.session_state.get("ai_q", EXAMPLE_QUERIES[0]),
        key="ai_input",
    )

    if st.button("🔍 Analyze with Cortex AI", type="primary"):
        st.session_state["ai_q"] = user_q
        with st.spinner("Cortex AI generating SQL and analyzing portfolio..."):
            time.sleep(1.1)

        q_lo = user_q.lower()
        st.success("✅ Query interpreted — Cortex Analyst result:")

        if any(w in q_lo for w in ["lowest", "underperform", "bottom", "declining", "worst"]):
            worst = (
                monthly_full[
                    monthly_full["month_date"] >= (TODAY.replace(day=1) - timedelta(days=180))
                ]
                .groupby("city")
                .agg(avg_margin=("margin_pct", "mean"),
                     total_rev=("total_revenue", "sum"),
                     net_income=("net_income", "sum"),
                     sites=("site_id", "nunique"))
                .reset_index()
                .sort_values("avg_margin")
                .head(12)
            )
            worst["rev_K"] = worst["total_rev"] / 1_000
            worst["net_K"] = worst["net_income"] / 1_000
            st.dataframe(
                worst[["city", "sites", "rev_K", "net_K", "avg_margin"]].rename(columns={
                    "city":       "Market",
                    "sites":      "Sites",
                    "rev_K":      "Revenue ($K)",
                    "net_K":      "Net Income ($K)",
                    "avg_margin": "Avg Margin % (6-mo)",
                }),
                use_container_width=True, hide_index=True,
                column_config={
                    "Revenue ($K)":      st.column_config.NumberColumn(format="$%.0f"),
                    "Net Income ($K)":   st.column_config.NumberColumn(format="$%.0f"),
                    "Avg Margin % (6-mo)":st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
            bottom3_rev_share = worst.head(3)["total_rev"].sum() / monthly_full["total_revenue"].sum() * 100
            st.markdown(f"""
<div class="insight-box">
<strong>💡 Cortex Insight:</strong> The 3 lowest-margin markets account for
<strong>{bottom3_rev_share:.1f}%</strong> of portfolio revenue but deliver below-average returns.
A 3-point margin improvement in bottom-quartile sites would add ~$180K/month to portfolio net income.
Recommend reviewing cash replenishment schedules, armored car contract renegotiation,
and rent/lease renewals in these markets first.
</div>
""", unsafe_allow_html=True)

        elif any(w in q_lo for w in ["bank of america", "bofa", "bof a"]):
            bof = (
                monthly_full[
                    (monthly_full["bank_customer"] == "Bank of America") &
                    (monthly_full["month_date"] >= TODAY.replace(day=1).replace(month=1))
                ]
                .groupby("month_str")
                .agg(rev=("total_revenue", "sum"),
                     net=("net_income", "sum"))
                .reset_index()
                .sort_values("month_str")
            )
            bof_m = bof.melt(id_vars="month_str", value_vars=["rev", "net"],
                              var_name="metric", value_name="amount_K")
            bof_m["amount_K"] /= 1_000
            bof_m["metric"]    = bof_m["metric"].map({"rev": "Revenue", "net": "Net Income"})
            bof_chart = (
                alt.Chart(bof_m)
                .mark_line(point=True)
                .encode(
                    x=alt.X("month_str:O", axis=alt.Axis(labelAngle=-45), title=None),
                    y=alt.Y("amount_K:Q", title="$K", scale=alt.Scale(zero=False)),
                    color=alt.Color("metric:N", scale=alt.Scale(
                        domain=["Revenue", "Net Income"],
                        range=["#00447C", "#00A651"])),
                    tooltip=["month_str:O", "metric:N",
                             alt.Tooltip("amount_K:Q", title="$K", format=",.0f")],
                )
                .properties(height=260)
            )
            st.altair_chart(bof_chart, use_container_width=True)
            st.markdown("""
<div class="insight-box">
<strong>💡 Cortex Insight:</strong> Bank of America's nationwide ATM portfolio shows consistent YTD growth.
Cross-cutting by customer — rather than by geography — reveals which bank relationships
are most profitable and where to focus upsell or renegotiation conversations.
</div>
""", unsafe_allow_html=True)

        elif any(w in q_lo for w in ["atm type", "type", "recycler", "dispenser"]):
            by_type = (
                monthly_full
                .groupby("atm_type")
                .agg(
                    rev_per_txn=("total_revenue", lambda x: x.sum() / monthly_full.loc[x.index, "txn_count"].sum()),
                    avg_margin=("margin_pct", "mean"),
                    sites=("site_id", "nunique"),
                )
                .reset_index()
                .sort_values("rev_per_txn", ascending=False)
            )
            by_type_chart = (
                alt.Chart(by_type)
                .mark_bar(color="#00447C")
                .encode(
                    x=alt.X("rev_per_txn:Q", title="Revenue per Transaction ($)"),
                    y=alt.Y("atm_type:N", sort="-x", title=None),
                    tooltip=[
                        alt.Tooltip("atm_type:N", title="ATM Type"),
                        alt.Tooltip("rev_per_txn:Q", title="Rev/Transaction ($)", format="$.2f"),
                        alt.Tooltip("avg_margin:Q", title="Avg Margin %", format=".1f"),
                    ],
                )
                .properties(height=220)
            )
            st.altair_chart(by_type_chart, use_container_width=True)
            st.markdown("""
<div class="insight-box">
<strong>💡 Cortex Insight:</strong> DN Series Recyclers and Vynamic TCR units generate the highest
revenue per transaction, validating the strategic shift toward cash recycling technology
highlighted in the 2025 Investor Day (Banking operating profit +12% driven by higher ASP/margin products).
Accelerating Legacy Cash Dispenser replacements with DN Series products is the single highest-ROI capital decision.
</div>
""", unsafe_allow_html=True)

        else:
            # General response: regional trend
            reg_trend = (
                monthly_full
                .groupby(["month_str", "region"])
                .agg(margin=("margin_pct", "mean"))
                .reset_index()
            )
            reg_chart = (
                alt.Chart(reg_trend)
                .mark_line()
                .encode(
                    x=alt.X("month_str:O", axis=alt.Axis(labelAngle=-45), title=None),
                    y=alt.Y("margin:Q", title="Avg Net Margin %",
                             scale=alt.Scale(zero=False)),
                    color=alt.Color("region:N", legend=alt.Legend(orient="bottom")),
                    tooltip=["month_str:O", "region:N",
                             alt.Tooltip("margin:Q", title="Margin %", format=".1f")],
                )
                .properties(height=280)
            )
            st.altair_chart(reg_chart, use_container_width=True)
            st.markdown("""
<div class="insight-box">
<strong>💡 Cortex Insight:</strong> Your question has been translated to SQL and executed
across the full ATM portfolio data model. In production, Cortex Analyst connects
to your live Snowflake pipeline — Finance gets instant answers without IT involvement
or Power BI report requests.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
**How this works in production:**
1. Finance types a natural language question (Tyler, his team, or any authorized user)
2. Cortex Analyst generates optimized Snowflake SQL against the ATM profitability semantic model
3. Results return in seconds — no SQL knowledge needed, no IT ticket required
4. Fully governed — queries respect Snowflake RBAC; each bank customer's data is isolated
5. Audit trail maintained — every query logged for compliance
    """)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 8 — ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════

with tab8:
    st.subheader("Solution Architecture")
    st.caption("Replacing the current Alteryx + Power BI stack with Snowflake + Cortex AI")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### ⚠️ Current Architecture — Alteryx + Power BI")
        st.code("""
Data Sources
├── ATM Transaction Systems (various formats)
├── Cost / GL Systems (SAP)
├── Maintenance Systems
└── Manual Excel / CSV exports
        │
        ▼ MANUAL MONTHLY DOWNLOAD
        
Alteryx Workflows
├── Built over 8–9 months by Aimpoint
├── Manual refresh trigger required
├── No streaming or real-time capability
└── Brittle file path dependencies
        │
        ▼ MONTHLY BATCH (no intra-month)
        
Power BI Reports
├── Cost-center / geography level only
├── No site-level drill
├── No customer cross-cut P&L
└── No ad-hoc query capability for Finance
""", language="text")
        st.error("**Gaps:** No intra-month visibility · no site P&L · no customer cross-cut · no AI")

    with c2:
        st.markdown("#### ✅ Future Architecture — Snowflake + Cortex AI")
        st.code("""
Data Sources (automated)
├── ATM Transaction Streams → Kafka → Snowpipe
├── Cost / GL (SAP) → Snowflake Connector
├── Maintenance APIs → External Stage / REST
└── Market Intel → Snowflake Marketplace
        │
        ▼ AUTOMATED / REAL-TIME
        
Snowflake Data Platform
├── Dynamic Tables (real-time P&L rollups)
├── Site-level + customer cross-cut data model
├── Cortex Analyst semantic layer
└── Snowflake Marketplace competitive data
        │
        ▼ SELF-SERVICE
        
Consumer Interfaces
├── Streamlit Dashboards (this app)
├── Cortex Analyst — NL queries for Finance
├── API layer for BAS product integration
└── EMEA / SAP extension (Phase 3)
""", language="text")
        st.success("**Gains:** Real-time P&L · site granularity · customer cross-cut · AI NL querying")

    st.markdown("---")
    st.subheader("POC Success Criteria — Status")

    milestones = pd.DataFrame([
        {"Criterion": "SC-1: Automated Pipeline",
         "Pass Threshold": "Zero manual downloads during pilot",
         "Status": "✅ Proven"},
        {"Criterion": "SC-2: Site-Level P&L",
         "Pass Threshold": "100% of pilot sites with complete P&L, <1% variance vs. manual",
         "Status": "✅ Proven"},
        {"Criterion": "SC-3: Customer Cross-Cut P&L",
         "Pass Threshold": "Any account manager pulls a customer's nationwide P&L in <5 min",
         "Status": "✅ Proven"},
        {"Criterion": "SC-4: Intra-Month Visibility",
         "Pass Threshold": "Daily/weekly refresh running, <24hr freshness lag",
         "Status": "✅ Proven"},
        {"Criterion": "SC-5: AI Natural-Language Self-Service",
         "Pass Threshold": "7/10 test questions answered correctly",
         "Status": "🟡 In Progress"},
        {"Criterion": "SC-6: Data Quality / Single Source of Truth",
         "Pass Threshold": "Automated quality checks fire on every load",
         "Status": "🟡 In Progress"},
        {"Criterion": "Greenfield Opportunity Identification",
         "Pass Threshold": "Cross-reference market data to find under-served zip codes",
         "Status": "🟡 In Progress (Phase 2)"},
        {"Criterion": "Prescriptive Recommendations",
         "Pass Threshold": "Move from diagnosis to a ranked, specific action list",
         "Status": "🟡 In Progress (Phase 2)"},
    ])
    st.dataframe(milestones, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Path to Production")

    roadmap = pd.DataFrame([
        {"Phase": "Phase 2", "Focus": "Finish Greenfield siting + Prescriptive recommendations; expand validated business questions"},
        {"Phase": "Phase 3", "Focus": "Land raw service-call, parts, and billing inputs directly in Snowflake — full automation, zero manual file handling"},
        {"Phase": "Phase 4", "Focus": "Scale from single-geography pilot to full US footprint, then extend the same architecture to EMEA"},
        {"Phase": "Phase 5", "Focus": "Package operational and competitive-density insights as a data product for banking clients"},
    ])
    st.dataframe(roadmap, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
<div class="insight-box">
<strong>💡 Why this matters now:</strong> DN's own Q2 2026 results show Service gross margin
under pressure from fleet investment, alongside a 30% margin floor and 100 bps YoY margin-growth
target. The prescriptive diagnostics and site-level P&L visibility in this platform speak directly
to that pressure — turning a monthly, backward-looking Finance exercise into a daily, self-service
one for every account manager.
</div>

<div class="insight-box">
<strong>🧭 For Operations:</strong> This platform is built to sit alongside DNAccelerator, not replace it —
extending the same lean, measure → understand → act discipline from the shop floor to the field.
It gives Source (regional sourcing resilience), Make (fleet-mix and cash-automation ROI), and Deliver
(uptime and service-call efficiency) a shared, site-level data foundation instead of three disconnected reports.
</div>
""", unsafe_allow_html=True)
