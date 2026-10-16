# Hispanic Heritage Historical Data Explorer

A Streamlit learning app for retrieving historical indicators from the official World Bank API, editing and validating observations, fitting centered-time polynomial models, comparing countries, exploring contextual functions and derivatives, and exporting CSV, Excel, and PDF reports.

## Run locally

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt` (and optionally `README.md` and `sources.md`).
3. In Streamlit Community Cloud, create an app connected to the repository.
4. Set the main file path to `app.py` and deploy. No API key is required for the World Bank API.
5. If deployment fails, open the app logs and check dependency installation, outbound network access, and API request errors.

## Deploy to Render

Create a **Python Web Service** and connect the GitHub repository.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

## Verification checklist

- Confirm the country, indicator code, source URL, retrieval date, and requested 70-year period.
- Check actual coverage and missing years against the World Bank API; no missing values are filled with fabricated data.
- Edit a value, confirm the user-modified scenario status, then use **Reset to source data**.
- Try duplicate years, non-integer years, blank values, and invalid indicator ranges; the model should explain why it cannot fit.
- Check that degree `d` requires at least `d+1` distinct years.
- Compare the equation and fitted values with a separate polynomial calculation; inspect in-sample R², MAE, and residuals.
- Calculate one year in the single-year panel and the same year in batch mode; estimates should agree.
- Try invalid batch inputs and years outside 1950–2100.
- Change country, indicator, degree, sample increment, forecast horizon, comparison countries, course mode, function, and graph toggles; verify each changes the relevant output.
- Download each format and open it to verify the actual file format and current report content.
- Test with the network unavailable and inspect the app error/logs; no random replacement data should appear.
- On deployment, check app logs and confirm the World Bank API is reachable from the hosting environment.

## Mathematical assumptions and limitations

The historical model uses `t = year - base_year` and a polynomial least-squares fit. Centering time improves numerical conditioning. R² and MAE are **in-sample** measures; an optional chronological holdout MAE is separately labeled. Neither metric is a prediction interval. Polynomial extrapolations can become implausible outside the observed range. User edits create a scenario and do not change the original source data. The contextual-function lab uses symbolic derivatives and numerical sampling; candidate critical points and inflection points are approximate and must be checked against domain restrictions and sign changes. This app is educational, not causal analysis, professional forecasting, or medical advice.
