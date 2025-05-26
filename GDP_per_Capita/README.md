# Global GDP per Capita Dashboard

A professional dashboard for exploring, comparing, and visualizing GDP per capita across all countries and years. Designed for development economists, policymakers, and researchers.

## Features
- Interactive world map of GDP per capita by country and year
- Country/year selection sidebar for focused analysis
- Comparison table for selected countries and years
- Trend chart for GDP per capita over time
- Clean, modern UI/UX (no newsletter or unrelated content)

## How to Run Locally
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the dashboard:
   ```sh
   streamlit run dashboard/app.py
   ```

## How to Deploy on Streamlit Cloud
1. Ensure `requirements.txt` is in the root of your repo and contains all dependencies.
2. Deploy via https://share.streamlit.io/ and set the main file to `dashboard/app.py`.

## Data
- Data source: `data/gdp_per_capita.csv`
- Country codes: ISO 3166-1 alpha-3

---
For questions or improvements, open an issue or pull request.
