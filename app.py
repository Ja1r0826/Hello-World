import io
import math
import re
import time
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor,
)

# ------------------------------------------------------------
# Hispanic Heritage App 2.0 — historical fitting + function lab
# ------------------------------------------------------------
st.set_page_config(page_title="Hispanic Heritage Math Lab", page_icon="📈", layout="wide")

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
x = sp.symbols("x", real=True)

COUNTRIES = {
    "United States": "USA", "Mexico": "MEX", "Puerto Rico": "PRI",
    "Spain": "ESP", "Colombia": "COL", "Argentina": "ARG",
    "Brazil": "BRA", "Canada": "CAN", "Chile": "CHL", "Peru": "PER",
}
INDICATORS = {
    "Population, total": ("SP.POP.TOTL", "people"),
    "Life expectancy at birth, total (years)": ("SP.DYN.LE00.IN", "years"),
    "GDP per capita (current US$)": ("NY.GDP.PCAP.CD", "current US dollars per person"),
    "CO₂ emissions (metric tons per capita)": ("EN.ATM.CO2E.PC", "metric tons per person"),
    "Individuals using the Internet (% of population)": ("IT.NET.USER.ZS", "% of population"),
    "Unemployment, total (% of total labor force)": ("SL.UEM.TOTL.ZS", "% of labor force"),
}

CONTEXTS = {
    "Business": {
        "description": "A simple model of revenue or demand over time.",
        "independent": "time", "xunit": "years", "dependent": "revenue", "yunit": "dollars",
        "formula": "0.4*(x-5)**2 + 10", "visual": "Business / economic indicator",
    },
    "Engineering": {
        "description": "A model of an object's position or system output.",
        "independent": "time", "xunit": "seconds", "dependent": "position", "yunit": "meters",
        "formula": "2*x**2 + 3*x", "visual": "Moving object / projectile",
    },
    "Medical": {
        "description": "An illustrative concentration curve; not medical advice or a dosing guide.",
        "independent": "time", "xunit": "hours", "dependent": "concentration", "yunit": "arbitrary units",
        "formula": "10*x*exp(-0.5*x)", "visual": "Tank / concentration indicator",
    },
    "Sports science": {
        "description": "A simplified model of an athlete's performance metric over time.",
        "independent": "time", "xunit": "seconds", "dependent": "performance", "yunit": "meters",
        "formula": "-0.3*(x-4)**2 + 8", "visual": "Moving object / projectile",
    },
    "Biology": {
        "description": "A simplified population or colony model.",
        "independent": "time", "xunit": "days", "dependent": "population", "yunit": "organisms",
        "formula": "100*exp(0.12*x)", "visual": "Population / cell colony",
    },
    "Bioengineering": {
        "description": "A simplified model of a changing biological system.",
        "independent": "time", "xunit": "hours", "dependent": "signal", "yunit": "signal units",
        "formula": "5 + 2*sin(x)", "visual": "Tank / concentration indicator",
    },
    "Economic forecasting": {
        "description": "An illustrative index that can rise or fall over time.",
        "independent": "time", "xunit": "years", "dependent": "index", "yunit": "index points",
        "formula": "100 + 3*x - 0.08*x**2", "visual": "Business / economic indicator",
    },
    "Weather": {
        "description": "A simplified periodic temperature pattern.",
        "independent": "time", "xunit": "hours", "dependent": "temperature", "yunit": "°C",
        "formula": "15 + 8*sin(2*pi*x/24)", "visual": "Thermometer / weather indicator",
    },
    "Custom": {
        "description": "Describe your own model and assumptions.",
        "independent": "input", "xunit": "units", "dependent": "output", "yunit": "units",
        "formula": "x**2 - 4*x + 3", "visual": "Business / economic indicator",
    },
}

def world_bank_series(country_code, indicator_code, start_year, end_year):
    """Fetch annual observations from the public World Bank API."""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {"format": "json", "date": f"{start_year}:{end_year}", "per_page": 20000}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise ValueError("The data source returned no observations for this selection.")
    rows = []
    for item in payload[1]:
        if item.get("value") is not None and item.get("date"):
            rows.append({"Year": int(item["date"]), "Actual": float(item["value"])})
    df = pd.DataFrame(rows).drop_duplicates("Year").sort_values("Year")
    if df.empty:
        raise ValueError("No non-missing observations were returned.")
    return df

def fit_polynomial(df, degree, base_year):
    clean = df[["Year", "Actual"]].dropna().copy()
    clean["t"] = clean["Year"].astype(float) - float(base_year)
    if clean["Year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree + 1} distinct years; only {clean['Year'].nunique()} are available.")
    coeff = np.polyfit(clean["t"].to_numpy(), clean["Actual"].to_numpy(), degree)
    model = np.poly1d(coeff)
    fitted = model(clean["t"].to_numpy())
    residuals = clean["Actual"].to_numpy() - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((clean["Actual"].to_numpy() - clean["Actual"].mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(residuals)))
    clean["Fitted"] = fitted
    clean["Residual"] = residuals
    return model, coeff, clean, r2, mae

def fmt_num(value):
    try:
        if not np.isfinite(float(value)):
            return "undefined"
        return f"{float(value):,.6g}"
    except Exception:
        return str(value)

def polynomial_latex(coeff, base_year):
    pieces = []
    degree = len(coeff) - 1
    for i, c in enumerate(coeff):
        power = degree - i
        if abs(c) < 1e-12:
            continue
        ctext = f"{abs(c):.8g}"
        if power == 0:
            term = ctext
        elif power == 1:
            term = f"{ctext}t"
        else:
            term = f"{ctext}t^{power}"
        if not pieces:
            pieces.append(("-" if c < 0 else "") + term)
        else:
            pieces.append((" - " if c < 0 else " + ") + term)
    return "f(t) = " + "".join(pieces) + f", where t = year − {base_year}"

def safe_parse_function(text):
    local_dict = {
        "x": x, "pi": sp.pi, "E": sp.E, "e": sp.E,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "ln": sp.log,
        "sqrt": sp.sqrt, "abs": sp.Abs,
    }
    # Keep input mathematical, avoid attribute access and Python statements.
    if "__" in text or ";" in text or "[" in text or "]" in text or "=" in text:
        raise ValueError("Enter one mathematical expression in x; do not include assignments or brackets.")
    expr = parse_expr(text, local_dict=local_dict, transformations=TRANSFORMS, evaluate=True)
    if not expr.free_symbols.issubset({x}):
        raise ValueError("Use x as the independent variable.")
    return sp.simplify(expr)

def numeric_eval(expr, values):
    fn = sp.lambdify(x, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        out = np.asarray(fn(values), dtype=float)
    if out.shape == ():
        out = np.full(np.asarray(values).shape, float(out))
    return out

def calc_metrics(actual, fitted):
    resid = np.asarray(actual, dtype=float) - np.asarray(fitted, dtype=float)
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((np.asarray(actual, dtype=float) - np.mean(actual))**2))
    r2 = float(1 - ss_res/ss_tot) if ss_tot else float("nan")
    return r2, mae, resid

st.title("📈 Hispanic Heritage Math Lab")
st.caption("Historical data, curves of best fit, and interactive function analysis. Educational models are simplified and should not be treated as forecasts without validation.")

with st.sidebar:
    st.header("1. Historical data")
    country_name = st.selectbox("Country / economy", list(COUNTRIES.keys()), index=0)
    indicator_name = st.selectbox("Indicator", list(INDICATORS.keys()), index=0)
    start_year = st.number_input("Start year", min_value=1950, max_value=2025, value=1990, step=1)
    end_year = st.number_input("End year", min_value=1950, max_value=2025, value=2023, step=1)
    degree = st.selectbox("Polynomial degree", [1, 2, 3, 4, 5], index=2)
    sample_step = st.slider("Model marker / sample-table increment", 1, 10, 5)
    fetch_button = st.button("Fetch / refresh data", type="primary", use_container_width=True)
    st.divider()
    st.header("2. Course mode")
    course_mode = st.selectbox("Analysis level", ["Honors Algebra 2", "Precalculus", "Calculus"])
    st.caption("The historical-data model and the function lab are separate: use the fitted model in the function lab with the transfer button.")

data_key = f"{COUNTRIES[country_name]}|{INDICATORS[indicator_name][0]}|{start_year}|{end_year}"
if "raw_cache" not in st.session_state:
    st.session_state.raw_cache = {}
if fetch_button or data_key not in st.session_state.raw_cache:
    if start_year > end_year:
        st.error("Start year must be less than or equal to end year.")
    else:
        with st.spinner("Retrieving World Bank observations…"):
            try:
                st.session_state.raw_cache[data_key] = world_bank_series(
                    COUNTRIES[country_name], INDICATORS[indicator_name][0], int(start_year), int(end_year)
                )
                st.session_state.data_error = None
            except Exception as exc:
                st.session_state.data_error = f"{type(exc).__name__}: {exc}"
if getattr(st.session_state, "data_error", None):
    st.warning("Data could not be retrieved. Check your connection, the selected years, or the World Bank API response.")
    st.code(st.session_state.data_error)
df_original = st.session_state.raw_cache.get(data_key, pd.DataFrame(columns=["Year", "Actual"]))
unit = INDICATORS[indicator_name][1]

tab_fit, tab_func, tab_deploy = st.tabs(["Fit, calculate & interpret", "Function & calculus lab", "Deployment & verification"])

with tab_fit:
    st.subheader("Fit a centered-time polynomial")
    st.write("Edit the observations if needed. The model uses **t = year − base year** and fits the displayed edited data. Missing observations are excluded.")
    if df_original.empty:
        st.info("No data are available yet. Fetch a valid country/indicator/year range from the sidebar.")
    else:
        edited = st.data_editor(df_original, num_rows="dynamic", use_container_width=True, key=f"editor_{data_key}",
                                column_config={"Year": st.column_config.NumberColumn("Year", format="%d"),
                                               "Actual": st.column_config.NumberColumn("Actual value")})
        edited = edited.copy()
        edited["Year"] = pd.to_numeric(edited["Year"], errors="coerce")
        edited["Actual"] = pd.to_numeric(edited["Actual"], errors="coerce")
        edited = edited.dropna(subset=["Year", "Actual"])
        edited["Year"] = edited["Year"].astype(int)
        edited = edited.sort_values("Year")
        if edited["Year"].duplicated().any():
            st.error("Duplicate years detected. Keep one observation per year before fitting.")
        else:
            base_year = int(edited["Year"].min())
            try:
                model, coeff, fitdf, r2, mae = fit_polynomial(edited, int(degree), base_year)
                st.session_state.fit_payload = {
                    "key": f"{data_key}|{degree}|{hash(tuple(map(tuple, edited[['Year','Actual']].to_numpy())))}",
                    "country": country_name, "indicator": indicator_name, "unit": unit,
                    "degree": int(degree), "base_year": base_year, "model": model, "coeff": coeff,
                    "fitdf": fitdf, "r2": r2, "mae": mae, "data": edited.copy(),
                    "start": int(edited.Year.min()), "end": int(edited.Year.max()),
                }
                payload = st.session_state.fit_payload
                st.markdown("**Fitted equation**")
                st.latex(polynomial_latex(coeff, base_year).replace("t", r"(year-" + str(base_year) + ")") if False else polynomial_latex(coeff, base_year))
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Degree", degree)
                m2.metric("In-sample R²", fmt_num(r2))
                m3.metric("In-sample MAE", f"{fmt_num(mae)} {unit}")
                m4.metric("Observations", len(fitdf))
                st.caption(f"Base year: {base_year} · Fitted historical range: {payload['start']}–{payload['end']} · Mean actual value: {fmt_num(edited['Actual'].mean())} {unit}")
                st.write("**Interpretation:** R² describes how much of the variation in these same observations is represented by this fit. MAE is the average absolute difference between actual and fitted values, in the indicator's units. These are in-sample metrics, not proof that the model predicts the future accurately.")
                # Chronological holdout when there are enough points
                if len(edited) >= max(10, int(degree) + 5):
                    ordered = edited.sort_values("Year")
                    split = max(int(degree) + 1, int(np.floor(len(ordered) * 0.8)))
                    train, test = ordered.iloc[:split], ordered.iloc[split:]
                    if len(test) >= 2 and train["Year"].nunique() >= int(degree) + 1:
                        hold_base = int(train.Year.min())
                        hold_model = np.poly1d(np.polyfit(train.Year.to_numpy() - hold_base, train.Actual.to_numpy(), int(degree)))
                        hold_pred = hold_model(test.Year.to_numpy() - hold_base)
                        hold_r2, hold_mae, _ = calc_metrics(test.Actual.to_numpy(), hold_pred)
                        st.write(f"**Chronological holdout (later years held out):** R² = {fmt_num(hold_r2)}, MAE = {fmt_num(hold_mae)} {unit}, test observations = {len(test)}. This is separate from the full-data fit.")
                # Single-year and batch predictions
                st.divider()
                st.subheader("Calculate values")
                a, b = st.columns([1, 2])
                with a:
                    year_one = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=max(1950, min(2100, int(payload["end"]))), step=1, key="single_year")
                    calc_one = st.button("Calculate Value", key="calculate_one")
                with b:
                    batch_mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
                    if batch_mode == "Year range":
                        r1, r2c, rstep = st.columns(3)
                        with r1: batch_start = st.number_input("From", 1950, 2100, max(1950, min(2100, payload["start"])), 1, key="batch_start")
                        with r2c: batch_end = st.number_input("To", 1950, 2100, max(1950, min(2100, payload["end"])), 1, key="batch_end")
                        with rstep: batch_step = st.number_input("Step (1–10)", 1, 10, 1, 1, key="batch_step")
                        batch_text = ""
                    else:
                        batch_text = st.text_input("Comma-separated years", "2000, 2010, 2020, 2030", key="batch_custom")
                        batch_start, batch_end, batch_step = 1950, 2100, 1
                    calc_batch = st.button("Calculate batch", key="calculate_batch")
                if calc_one:
                    pred = float(model(int(year_one) - base_year))
                    classification = "within fitted historical range" if payload["start"] <= year_one <= payload["end"] else "outside fitted historical range (extrapolation)"
                    actual_match = edited.loc[edited.Year == int(year_one), "Actual"]
                    st.session_state.single_result = {"year": int(year_one), "estimate": pred, "classification": classification,
                                                      "observed": float(actual_match.iloc[0]) if len(actual_match) else None,
                                                      "key": payload["key"]}
                if calc_batch:
                    try:
                        if batch_mode == "Year range":
                            if batch_start > batch_end:
                                raise ValueError("'From' year must be less than or equal to 'To' year.")
                            years = list(range(int(batch_start), int(batch_end)+1, int(batch_step)))
                        else:
                            tokens = [v.strip() for v in batch_text.split(",") if v.strip()]
                            if not tokens:
                                raise ValueError("Enter at least one year.")
                            years = []
                            for token in tokens:
                                if not re.fullmatch(r"\d{4}", token):
                                    raise ValueError(f"'{token}' is not a valid four-digit year.")
                                y = int(token)
                                if y < 1950 or y > 2100:
                                    raise ValueError(f"Year {y} is outside the allowed range 1950–2100.")
                                years.append(y)
                            years = list(dict.fromkeys(years))
                        batch = pd.DataFrame({"Year": years})
                        batch["Estimate"] = [float(model(y-base_year)) for y in years]
                        batch["Classification"] = batch["Year"].apply(lambda y: "within fitted historical range" if payload["start"] <= y <= payload["end"] else "outside fitted historical range (extrapolation)")
                        actual_map = edited.set_index("Year")["Actual"].to_dict()
                        batch["Actual observation (if available)"] = batch["Year"].map(actual_map)
                        batch["Fitted value at observed year"] = batch["Year"].map(lambda y: float(model(y-base_year)) if y in actual_map else np.nan)
                        st.session_state.batch_result = {"df": batch, "key": payload["key"]}
                    except Exception as exc:
                        st.error(f"Invalid batch input: {exc}")
                current_key = payload["key"]
                single = st.session_state.get("single_result")
                if single and single.get("key") == current_key:
                    st.markdown("**Single-year result**")
                    st.write(f"Year: **{single['year']}** · Estimate: **{fmt_num(single['estimate'])} {unit}** · Classification: **{single['classification']}**")
                    if single["observed"] is not None:
                        st.write(f"Actual observation: {fmt_num(single['observed'])} {unit}. The estimate is the fitted value, not the actual value.")
                    else:
                        st.write("No actual observation is available for this year in the selected dataset.")
                    if single["classification"].startswith("outside"):
                        scale = max(abs(float(edited["Actual"].median())), 1.0)
                        if abs(single["estimate"]) > 10 * scale:
                            st.warning("This extrapolated value is far outside the scale of the historical data. Polynomial extrapolation can become implausible; the value is not clipped.")
                batch_result = st.session_state.get("batch_result")
                if batch_result and batch_result.get("key") == current_key:
                    st.markdown("**Batch results**")
                    st.dataframe(batch_result["df"], use_container_width=True)
                    st.caption("Single-year and batch estimates use the same fitted polynomial and centered-time transformation.")
                # Plot
                st.divider()
                st.subheader("Historical data and fitted curve")
                plot_min = min(int(edited.Year.min()), int(edited.Year.max()))
                plot_max = max(int(edited.Year.max()), int(edited.Year.max()))
                xs = np.linspace(plot_min, max(plot_max, plot_min+1), 350)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=edited.Year, y=edited.Actual, mode="markers", name="Observed data", marker=dict(size=7)))
                fig.add_trace(go.Scatter(x=xs, y=model(xs-base_year), mode="lines", name="Smooth fitted curve"))
                sampled = edited.iloc[::int(sample_step)]
                fig.add_trace(go.Scatter(x=sampled.Year, y=sampled.Actual, mode="markers", name=f"Sample markers (every {sample_step})", marker=dict(symbol="circle-open", size=11)))
                fig.add_vline(x=payload["start"], line_dash="dash", annotation_text="Historical boundary")
                if payload["end"] != payload["start"]:
                    fig.add_vline(x=payload["end"], line_dash="dash")
                single = st.session_state.get("single_result")
                if single and single.get("key") == current_key:
                    fig.add_trace(go.Scatter(x=[single["year"]], y=[single["estimate"]], mode="markers", name="Single-year estimate", marker=dict(size=12, symbol="star")))
                batch_result = st.session_state.get("batch_result")
                if batch_result and batch_result.get("key") == current_key:
                    bdf = batch_result["df"]
                    fig.add_trace(go.Scatter(x=bdf.Year, y=bdf.Estimate, mode="markers", name="Batch estimates", marker=dict(size=9, symbol="diamond")))
                fig.update_layout(xaxis_title="Year", yaxis_title=f"{indicator_name} ({unit})", legend=dict(orientation="h"), margin=dict(l=20,r=20,t=20,b=20))
                st.plotly_chart(fig, use_container_width=True)
                st.write("**Residuals (actual − fitted)**")
                st.dataframe(fitdf[["Year", "Actual", "Fitted", "Residual"]].round(6), use_container_width=True)
                st.write(f"**In-sample residual summary:** mean absolute residual = {fmt_num(mae)} {unit}; mean residual = {fmt_num(fitdf.Residual.mean())} {unit}. Residual patterns may indicate that the selected polynomial is a poor model.")
                # Country comparison
                st.divider()
                st.subheader("Compare another country — same indicator")
                comparison_name = st.selectbox("Comparison country", [c for c in COUNTRIES if c != country_name], key="comparison_country")
                if st.button("Load comparison", key="load_comparison"):
                    try:
                        cdf = world_bank_series(COUNTRIES[comparison_name], INDICATORS[indicator_name][0], int(start_year), int(end_year))
                        st.session_state.comparison = {"key": data_key, "name": comparison_name, "df": cdf}
                    except Exception as exc:
                        st.error(f"Comparison data unavailable: {exc}")
                comparison = st.session_state.get("comparison")
                if comparison and comparison.get("key") == data_key:
                    comp = comparison["df"]
                    fcomp = go.Figure()
                    fcomp.add_trace(go.Scatter(x=edited.Year, y=edited.Actual, mode="lines+markers", name=country_name))
                    fcomp.add_trace(go.Scatter(x=comp.Year, y=comp.Actual, mode="lines+markers", name=comparison["name"]))
                    fcomp.update_layout(xaxis_title="Year", yaxis_title=f"{indicator_name} ({unit})", legend=dict(orientation="h"))
                    st.plotly_chart(fcomp, use_container_width=True)
                    st.caption("Country coverage can differ because indicators have missing years or different reporting histories. Compare only overlapping years when making direct comparisons.")
                # Export files
                st.divider()
                st.subheader("Export current report")
                export_data = edited.copy()
                export_data["Fitted"] = model(export_data.Year.to_numpy()-base_year)
                export_data["Residual"] = export_data.Actual.to_numpy()-export_data.Fitted.to_numpy()
                export_summary = pd.DataFrame([{
                    "country": country_name, "indicator": indicator_name, "units": unit, "degree": int(degree),
                    "base_year": base_year, "equation": polynomial_latex(coeff, base_year), "in_sample_R2": r2,
                    "in_sample_MAE": mae, "observation_count": len(edited), "historical_start": int(edited.Year.min()),
                    "historical_end": int(edited.Year.max()), "mean_actual": float(edited.Actual.mean()),
                    "data_edited": not edited.equals(df_original.sort_values("Year").reset_index(drop=True)),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }])
                batch_df = st.session_state.batch_result["df"] if st.session_state.get("batch_result", {}).get("key") == current_key else pd.DataFrame()
                csv_bytes = export_data.to_csv(index=False).encode("utf-8")
                xlsx_buffer = io.BytesIO()
                with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
                    export_summary.to_excel(writer, index=False, sheet_name="Summary")
                    export_data.to_excel(writer, index=False, sheet_name="Working data")
                    batch_df.to_excel(writer, index=False, sheet_name="Calculated years")
                # PDF report using matplotlib image + reportlab
                pdf_bytes = None
                try:
                    import matplotlib.pyplot as plt
                    from reportlab.lib import colors
                    from reportlab.lib.pagesizes import letter, landscape
                    from reportlab.lib.styles import getSampleStyleSheet
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
                    pdfbuf = io.BytesIO()
                    doc = SimpleDocTemplate(pdfbuf, pagesize=landscape(letter), rightMargin=28, leftMargin=28, topMargin=24, bottomMargin=24)
                    styles = getSampleStyleSheet()
                    story = [Paragraph("Historical Curve-of-Best-Fit Report", styles["Title"]),
                             Paragraph(f"{country_name} — {indicator_name} ({unit})", styles["Heading2"]),
                             Paragraph(f"Equation: {polynomial_latex(coeff, base_year)}", styles["BodyText"]),
                             Paragraph(f"Degree {degree}; base year {base_year}; in-sample R² {fmt_num(r2)}; in-sample MAE {fmt_num(mae)} {unit}; n={len(edited)}; range {int(edited.Year.min())}–{int(edited.Year.max())}.", styles["BodyText"]),
                             Spacer(1, 8)]
                    fig2, ax = plt.subplots(figsize=(8, 3))
                    ax.scatter(edited.Year, edited.Actual, label="Observed")
                    ax.plot(xs, model(xs-base_year), label="Fitted curve")
                    ax.axvline(int(edited.Year.min()), linestyle="--", linewidth=0.8)
                    ax.axvline(int(edited.Year.max()), linestyle="--", linewidth=0.8)
                    ax.set_xlabel("Year"); ax.set_ylabel(f"{indicator_name} ({unit})"); ax.legend(); fig2.tight_layout()
                    img_buf = io.BytesIO(); fig2.savefig(img_buf, format="png", dpi=130); plt.close(fig2); img_buf.seek(0)
                    story.append(RLImage(img_buf, width=520, height=195))
                    story.append(Paragraph("Findings and limitations", styles["Heading2"]))
                    story.append(Paragraph(f"Mean observed value: {fmt_num(edited.Actual.mean())} {unit}. MAE describes typical in-sample error. Polynomial extrapolation outside {int(edited.Year.min())}–{int(edited.Year.max())} can be unreliable and is not a validated forecast or a 95% prediction interval.", styles["BodyText"]))
                    story.append(Spacer(1, 8))
                    preview = [["Year", "Actual", "Fitted", "Residual"]] + [[str(int(r.Year)), fmt_num(r.Actual), fmt_num(r.Fitted), fmt_num(r.Residual)] for r in export_data.head(12).itertuples()]
                    tbl = Table(preview, repeatRows=1)
                    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("PADDING",(0,0),(-1,-1),4)]))
                    story.append(tbl)
                    if not batch_df.empty:
                        story += [Spacer(1,8), Paragraph("Calculated-year results", styles["Heading2"])]
                        batch_preview = [list(batch_df.columns)] + [[str(v) for v in row] for row in batch_df.head(12).itertuples(index=False, name=None)]
                        bt = Table(batch_preview, repeatRows=1)
                        bt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),6),("PADDING",(0,0),(-1,-1),3)]))
                        story.append(bt)
                    doc.build(story)
                    pdf_bytes = pdfbuf.getvalue()
                except Exception as exc:
                    st.warning(f"PDF creation failed: {exc}")
                st.download_button("Download CSV", csv_bytes, file_name="historical_working_data.csv", mime="text/csv")
                st.download_button("Download Excel workbook (.xlsx)", xlsx_buffer.getvalue(), file_name="historical_model_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                if pdf_bytes:
                    st.download_button("Download PDF report", pdf_bytes, file_name="historical_model_report.pdf", mime="application/pdf")
                st.checkbox("Printer-friendly report mode", key="printer_mode", help="Use your browser's Print command; the app simplifies the report layout when selected.")
                if st.session_state.get("printer_mode"):
                    st.markdown("---")
                    st.markdown(f"### Printable summary\n**{country_name} — {indicator_name}**\n\n{polynomial_latex(coeff, base_year)}\n\nDegree {degree}; in-sample R² {fmt_num(r2)}; in-sample MAE {fmt_num(mae)} {unit}; {len(edited)} observations; range {int(edited.Year.min())}–{int(edited.Year.max())}.\n\nMAE and residuals describe the fit to observed data; they do not guarantee future accuracy.")
            except Exception as exc:
                st.error(f"Unable to fit the selected model: {exc}")

with tab_func:
    st.subheader("Context, function, and simulation")
    mode_context = st.selectbox("Context", list(CONTEXTS.keys()), key="function_context")
    defaults = CONTEXTS[mode_context]
    c1, c2 = st.columns(2)
    with c1:
        context_description = st.text_area("Context description", defaults["description"], key=f"desc_{mode_context}")
        independent_name = st.text_input("Independent variable", defaults["independent"], key=f"ind_{mode_context}")
        x_unit = st.text_input("Independent-variable units", defaults["xunit"], key=f"xunit_{mode_context}")
    with c2:
        dependent_name = st.text_input("Dependent variable", defaults["dependent"], key=f"dep_{mode_context}")
        y_unit = st.text_input("Dependent-variable units", defaults["yunit"], key=f"yunit_{mode_context}")
        visual = st.selectbox("Contextual visual", ["Walking person with prosthetic leg", "Moving object / projectile", "Tank / concentration indicator", "Population / cell colony", "Thermometer / weather indicator", "Business / economic indicator"], index=["Walking person with prosthetic leg","Moving object / projectile","Tank / concentration indicator","Population / cell colony","Thermometer / weather indicator","Business / economic indicator"].index(defaults["visual"]) if defaults["visual"] in ["Walking person with prosthetic leg","Moving object / projectile","Tank / concentration indicator","Population / cell colony","Thermometer / weather indicator","Business / economic indicator"] else 5, key=f"visual_{mode_context}")
    suggested = st.selectbox("Suggested function", ["Use context default", "Linear", "Quadratic", "Exponential", "Logarithmic", "Trigonometric", "Rational", "Use fitted historical model"], key="function_kind")
    if suggested == "Use context default":
        function_text = defaults["formula"]
    elif suggested == "Linear":
        function_text = "2*x + 1"
    elif suggested == "Quadratic":
        function_text = "x**2 - 4*x + 3"
    elif suggested == "Exponential":
        function_text = "3*exp(0.2*x)"
    elif suggested == "Logarithmic":
        function_text = "2*log(x) + 1"
    elif suggested == "Trigonometric":
        function_text = "2*sin(x) + 3"
    elif suggested == "Rational":
        function_text = "(x**2 + 1)/(x - 2)"
    else:
        function_text = "x**2"
        fp = st.session_state.get("fit_payload")
        if fp:
            # Use years relative to the fitted base year as x.
            function_text = polynomial_latex(fp["coeff"], fp["base_year"]).split(", where")[0].replace("f(t) = ", "").replace("t", "x")
            st.caption(f"Using historical fit: t = year − {fp['base_year']}. The function lab variable x represents centered time t.")
        else:
            st.info("Fit a historical model first; a context default is used until a fit exists.")
    function_text = st.text_input("Function f(x) — use x, ** for powers, exp(), log(), sin(), cos(), sqrt()", value=function_text, key="function_expression")
    colp, colq, colr = st.columns(3)
    with colp: p_a = st.number_input("Parameter a", value=2.0, key=f"param_a_{mode_context}")
    with colq: p_b = st.number_input("Parameter b", value=1.0, key=f"param_b_{mode_context}")
    with colr: p_c = st.number_input("Parameter c", value=0.0, key=f"param_c_{mode_context}")
    st.caption("Parameter controls are a quick builder for a*x² + b*x + c. The expression field is the source used for analysis.")
    if st.button("Build quadratic from parameters"):
        st.session_state.function_expression = f"{p_a}*x**2 + {p_b}*x + {p_c}"
        st.rerun()
    try:
        expr = safe_parse_function(function_text)
        d1 = sp.diff(expr, x)
        d2 = sp.diff(d1, x)
        st.markdown("**Function and derivatives**")
        eq1, eq2, eq3 = st.columns(3)
        eq1.latex("f(x) = " + sp.latex(expr))
        eq2.latex("f'(x) = " + sp.latex(d1))
        eq3.latex("f''(x) = " + sp.latex(d2))
        xmin, xmax = st.slider("Analysis interval", min_value=-20.0, max_value=20.0, value=(-5.0, 10.0), step=0.5, key="analysis_interval")
        if xmin >= xmax:
            st.error("The interval's left endpoint must be less than its right endpoint.")
        else:
            lo, hi = float(xmin), float(xmax)
            point = st.slider("Current input x", min_value=lo, max_value=hi, value=min(max(1.0,lo),hi), step=max((hi-lo)/200,0.01), key="current_x")
            aoc_cols = st.columns(2)
            with aoc_cols[0]:
                left_x = st.slider("Secant interval — left endpoint", min_value=lo, max_value=hi, value=lo, step=max((hi-lo)/100,0.01), key="secant_left")
            with aoc_cols[1]:
                right_x = st.slider("Secant interval — right endpoint", min_value=lo, max_value=hi, value=hi, step=max((hi-lo)/100,0.01), key="secant_right")
            play_cols = st.columns(4)
            with play_cols[0]:
                if st.button("▶ Play", key="play"):
                    st.session_state.playing = True
            with play_cols[1]:
                if st.button("⏸ Pause", key="pause"):
                    st.session_state.playing = False
            with play_cols[2]:
                if st.button("↺ Reset", key="reset"):
                    st.session_state.playing = False
                    st.session_state.current_x = min(max(1.0, lo), hi)
                    st.rerun()
            with play_cols[3]:
                speed = st.selectbox("Playback speed", ["Slow", "Normal", "Fast"], index=1, key="play_speed")
            if st.session_state.get("playing", False):
                # One increment per rerun; browser auto-refresh is intentionally avoided to keep the app server-safe.
                delta = {"Slow": 0.05, "Normal": 0.15, "Fast": 0.4}[speed]
                st.session_state.current_x = min(hi, float(st.session_state.get("current_x", point)) + delta)
                if st.session_state.current_x >= hi:
                    st.session_state.playing = False
                st.info("Playback is step-based. Move the slider or press Play again to advance; this avoids an uncontrolled server-side loop.")
            point = float(st.session_state.get("current_x", point))
            try:
                fval = float(sp.N(expr.subs(x, point)))
                dval = float(sp.N(d1.subs(x, point)))
                ddval = float(sp.N(d2.subs(x, point)))
                st.write(f"At x = {fmt_num(point)}: f(x) = **{fmt_num(fval)} {y_unit}**; f′(x) = **{fmt_num(dval)} {y_unit}/{x_unit}**; f″(x) = **{fmt_num(ddval)} {y_unit}/{x_unit}²**.")
                if dval > 1e-8: trend = "increasing at this point"
                elif dval < -1e-8: trend = "decreasing at this point"
                else: trend = "has approximately zero instantaneous rate at this point"
                if ddval > 1e-8: concavity = "concave up here (the rate is increasing)"
                elif ddval < -1e-8: concavity = "concave down here (the rate is decreasing)"
                else: concavity = "has approximately zero second derivative here"
                st.write(f"**Context interpretation:** {context_description} At this input, {dependent_name} is {fmt_num(fval)} {y_unit}; it is {trend}. Its rate of change is {concavity}.")
            except Exception:
                st.warning("The function or one of its derivatives is undefined at the selected point (for example, at a logarithm restriction or denominator zero).")
            # Plot avoid connecting across invalid values
            grid = np.linspace(lo, hi, 900)
            yv = numeric_eval(expr, grid)
            yv[~np.isfinite(yv)] = np.nan
            show_f = st.checkbox("Show f(x)", value=True, key="show_f")
            show_d1 = st.checkbox("Show f′(x)", value=(course_mode == "Calculus"), key="show_d1")
            show_d2 = st.checkbox("Show f″(x)", value=(course_mode == "Calculus"), key="show_d2")
            fig = go.Figure()
            if show_f: fig.add_trace(go.Scatter(x=grid, y=yv, mode="lines", name="f(x)"))
            if show_d1:
                dy = numeric_eval(d1, grid); dy[~np.isfinite(dy)] = np.nan
                fig.add_trace(go.Scatter(x=grid, y=dy, mode="lines", name="f′(x)", line=dict(dash="dash")))
            if show_d2:
                ddy = numeric_eval(d2, grid); ddy[~np.isfinite(ddy)] = np.nan
                fig.add_trace(go.Scatter(x=grid, y=ddy, mode="lines", name="f″(x)", line=dict(dash="dot")))
            if np.isfinite(numeric_eval(expr, np.array([point]))[0]):
                fig.add_trace(go.Scatter(x=[point], y=[float(sp.N(expr.subs(x,point)))], mode="markers", name="Current point", marker=dict(size=11, symbol="circle")))
                slope = float(sp.N(d1.subs(x, point)))
                intercept = float(sp.N(expr.subs(x, point))) - slope*point
                tangent_y = slope*grid + intercept
                tangent_y[(grid < max(lo, point-(hi-lo)*0.18)) | (grid > min(hi, point+(hi-lo)*0.18))] = np.nan
                fig.add_trace(go.Scatter(x=grid, y=tangent_y, mode="lines", name="Tangent line"))
            if left_x != right_x:
                try:
                    yl, yr = float(sp.N(expr.subs(x,left_x))), float(sp.N(expr.subs(x,right_x)))
                    if np.isfinite(yl) and np.isfinite(yr):
                        slope_sec = (yr-yl)/(right_x-left_x)
                        sec_y = slope_sec*(grid-left_x)+yl
                        sec_y[(grid < left_x) | (grid > right_x)] = np.nan
                        fig.add_trace(go.Scatter(x=grid, y=sec_y, mode="lines", name="Secant line", line=dict(dash="dashdot")))
                        st.write(f"Average rate of change on [{fmt_num(left_x)}, {fmt_num(right_x)}]: **{fmt_num(slope_sec)} {y_unit}/{x_unit}**.")
                    else:
                        st.warning("The secant endpoints include an undefined function value.")
                except Exception:
                    st.warning("Unable to calculate the secant line for these endpoints.")
            fig.update_layout(xaxis_title=f"{independent_name} ({x_unit})", yaxis_title=f"Value / rate (units differ by curve)", legend=dict(orientation="h"), margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
            if course_mode in ["Honors Algebra 2", "Precalculus", "Calculus"]:
                st.markdown("**Domain and behavior**")
                st.write(f"Analysis window: [{lo}, {hi}]. For polynomial functions, the domain is all real numbers. For rational, logarithmic, or even-root expressions, domain restrictions may apply; undefined graph points are left as gaps.")
                st.write(f"End behavior is determined by the dominant term for polynomials; for other families, use the displayed graph and domain restrictions. The function's range over this finite window is approximated from valid plotted samples.")
                valid_y = yv[np.isfinite(yv)]
                if len(valid_y):
                    st.write(f"Approximate plotted range on this window: [{fmt_num(np.min(valid_y))}, {fmt_num(np.max(valid_y))}] {y_unit}.")
                # Average rates at multiple intervals
                if course_mode in ["Precalculus", "Calculus"]:
                    st.markdown("**Changing average rates (turning and concavity clues)**")
                    if len(valid_y) > 2:
                        slopes = np.diff(yv) / np.diff(grid)
                        finite_slopes = slopes[np.isfinite(slopes)]
                        if len(finite_slopes):
                            st.write(f"Across the plotted window, sampled slopes range from {fmt_num(np.min(finite_slopes))} to {fmt_num(np.max(finite_slopes))} {y_unit}/{x_unit}. A change in slope can indicate changing concavity.")
                if course_mode == "Calculus":
                    st.markdown("**Numerical critical points, extrema, and inflection candidates**")
                    critical = []
                    try:
                        roots = sp.solve(sp.Eq(d1,0), x)
                        for root in roots:
                            if root.is_real is True:
                                rv = float(sp.N(root))
                                if lo <= rv <= hi:
                                    eps = min(1e-3, max((hi-lo)/10000,1e-5))
                                    left_s = float(sp.N(d1.subs(x, max(lo,rv-eps))))
                                    right_s = float(sp.N(d1.subs(x, min(hi,rv+eps))))
                                    kind = "local minimum (derivative changes − to +)" if left_s < 0 < right_s else ("local maximum (derivative changes + to −)" if left_s > 0 > right_s else "stationary point; not classified as an extremum")
                                    critical.append((rv, float(sp.N(expr.subs(x,rv))), kind))
                    except Exception:
                        pass
                    if critical:
                        st.dataframe(pd.DataFrame(critical, columns=["x (approx.)","f(x) (approx.)","classification"]), use_container_width=True)
                    else:
                        st.write("No real derivative-zero critical points were found in the selected interval, or the symbolic solver could not classify them.")
                    # numerical inflection candidates; verify sign change in second derivative
                    inflections = []
                    try:
                        roots2 = sp.solve(sp.Eq(d2,0), x)
                        for root in roots2:
                            if root.is_real is True:
                                rv = float(sp.N(root))
                                if lo < rv < hi:
                                    eps = min(1e-3, max((hi-lo)/10000,1e-5))
                                    lval = float(sp.N(d2.subs(x,rv-eps)))
                                    rval = float(sp.N(d2.subs(x,rv+eps)))
                                    if np.isfinite(lval) and np.isfinite(rval) and lval*rval < 0:
                                        inflections.append((rv,float(sp.N(expr.subs(x,rv))),"concavity changes"))
                    except Exception:
                        pass
                    st.write("**Verified inflection points (numerical sign-change check):**")
                    if inflections:
                        st.dataframe(pd.DataFrame(inflections, columns=["x (approx.)","f(x) (approx.)","verification"]), use_container_width=True)
                    else:
                        st.write("No verified inflection point was found in the selected interval. A zero of f″ alone is not enough.")
                    # endpoint-inclusive sampled extrema
                    if len(valid_y):
                        valid_idx = np.where(np.isfinite(yv))[0]
                        imin, imax = valid_idx[np.argmin(yv[valid_idx])], valid_idx[np.argmax(yv[valid_idx])]
                        st.write(f"Approximate absolute extrema on the plotted interval: minimum ≈ ({fmt_num(grid[imin])}, {fmt_num(yv[imin])}); maximum ≈ ({fmt_num(grid[imax])}, {fmt_num(yv[imax])}). These are numerical grid estimates and may miss discontinuities or sharp features.")
            st.markdown("**Visual simulation (conceptual)**")
            normalized = 0.5
            if len(valid_y) and np.max(valid_y) != np.min(valid_y):
                normalized = float((np.clip(fval if 'fval' in locals() else np.nanmean(valid_y), np.min(valid_y), np.max(valid_y))-np.min(valid_y))/(np.max(valid_y)-np.min(valid_y)))
            st.progress(float(np.clip(normalized,0,1)), text=f"Scaled indicator for: {visual} (0–100% of the plotted output range; not a physical scale)")
            st.caption("This indicator maps the current output to the plotted range. It is illustrative, not a literal prosthetic, medical, weather, or engineering simulation.")
    except Exception as exc:
        st.error(f"Function input error: {exc}")
        st.write("Examples: `x**2 - 4*x + 3`, `3*exp(0.2*x)`, `2*log(x)`, `sin(x)`, `(x**2+1)/(x-2)`. Use `**` for powers.")

with tab_deploy:
    st.subheader("Run locally")
    st.code("pip install -r requirements.txt\npython -m streamlit run app.py", language="bash")
    st.subheader("Deploy with GitHub + Streamlit Community Cloud")
    st.markdown("""
1. Create a GitHub repository and add `app.py` and `requirements.txt` at the repository root.
2. Commit and push both files.
3. Open Streamlit Community Cloud, connect GitHub, choose the repository and branch, and set the app file path to `app.py`.
4. Deploy. Check the app logs if dependencies fail or the API is unavailable.
""")
    st.subheader("Deploy on Render")
    st.markdown("Create a **Python Web Service** connected to the repository. Set the root directory to the project root.")
    st.write("Build command")
    st.code("pip install -r requirements.txt", language="bash")
    st.write("Start command")
    st.code("python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT", language="bash")
    st.subheader("Verification checklist")
    checklist = [
        "Source accuracy: country code, indicator code, units, and date range match the World Bank API response.",
        "Edit/reset: edit a value in the data editor, fit again, and confirm exports reflect edited data; refresh the page to restore source data.",
        "Insufficient/duplicate years: degree d requires at least d+1 distinct years; duplicate years should be removed before fitting.",
        "Equation and metrics: compare centered-time equation, in-sample R², MAE, residuals, observation count, range, and mean.",
        "Single/batch agreement: request the same year in both places and confirm estimates match.",
        "Invalid inputs: test reversed year ranges, invalid comma-separated years, and years outside 1950–2100.",
        "API failure: disconnect or use an invalid range; verify that the app displays an understandable error and logs contain the cause.",
        "Function families: test polynomial, rational denominator zero, log(x) for x≤0, sqrt(x) for x<0, exponential, and trigonometric functions.",
        "Controls: change course mode, context, units, graph toggles, point slider, secant endpoints, and playback controls.",
        "Exports: open CSV, Excel workbook, and PDF; confirm current selection, model equation, metrics, chart, data preview, and calculated-year results.",
        "Deployment: test the public app URL, initial data fetch, reruns, and download buttons after deployment.",
    ]
    for item in checklist:
        st.checkbox(item, key="check_" + str(abs(hash(item))))
    st.subheader("How the app works")
    st.markdown("""
- **Data flow:** country + indicator + years → World Bank API → editable observations → centered-time polynomial fit → estimates, diagnostics, plots, and exports.
- **Centered time:** `t = year − base year` keeps the polynomial input closer to zero and can improve numerical conditioning. Predictions reuse the same base year and fitted polynomial.
- **Fit metrics:** R² and MAE are calculated on the observations used for fitting. The chronological holdout score is separately calculated on later observations not used in that holdout fit.
- **Function lab:** a mathematical expression is parsed with SymPy; its first and second derivatives are calculated symbolically, then evaluated numerically for graphs and local interpretations.
- **Assumptions and limits:** a polynomial is a descriptive model, not proof of cause. High-degree curves can oscillate and extrapolate badly. MAE is a typical in-sample error, not a validated prediction interval. Missing data and differing country coverage limit comparisons. Context visuals are normalized teaching aids, not clinical or engineering tools.
- **Troubleshooting:** `ModuleNotFoundError` usually means a dependency is missing; rerun `pip install -r requirements.txt`. API timeout/HTTP errors may be temporary or network-related. Check Streamlit's deployment logs for the first traceback line and the failing package or request.
""")
