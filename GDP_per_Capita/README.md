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

## 🚀 Senior Data Engineering Features
- **LLM Data Copilot:** Natural language Q&A, SQL generation, and insights using OpenAI GPT-4 (tab in dashboard).
- **Web3 Data:** View and interact with economic data on Ethereum testnet (tab in dashboard, web3.py).
- **Automated Data Pipeline:** Airflow DAG for scheduled fetch, validation, and cloud upload (see `dags/fetch_gdp_data.py`).
- **Cloud Storage Ready:** Integrate with AWS S3, GCP, or Azure for data storage (add credentials and logic in Airflow DAG).

## ☁️ Cloud Deployment

### Dashboard (Streamlit Cloud)
1. Push to GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud), link your repo, set main file to `dashboard/app.py`.
3. App is live globally (requirements auto-installed).

### Dashboard (Docker/VM/Cloud)
1. Build and run with Docker Compose:
   ```powershell
   docker compose up --build
   ```
2. Deploy to any cloud VM (AWS EC2, GCP Compute, Azure VM) or container service.

### Data Pipeline (Airflow Cloud)
- Deploy `dags/fetch_gdp_data.py` to Astronomer Cloud, MWAA (AWS), or Cloud Composer (GCP).
- Or run Airflow locally:
   ```powershell
   pip install apache-airflow
   airflow db init
   airflow users create --username admin --password admin --role Admin --email admin@example.com --firstname Admin --lastname User
   airflow webserver -p 8080
   airflow scheduler
   # Place DAG in airflow/dags/
   ```

## 🦾 LLM & Web3 Usage
- **LLM Copilot:** Enter your OpenAI API key in the dashboard tab to ask questions or generate SQL.
- **Web3 Data:** Enter your Infura/Alchemy testnet URL to connect and view on-chain data.

## 📦 Requirements
- Python 3.9+
- See `requirements.txt` for all dependencies (includes streamlit, pandas, plotly, openai, web3, airflow, etc.)

## 📝 Notes
- For production, add secrets management for API keys and cloud credentials.
- Extend Airflow DAG for real S3 upload, Great Expectations validation, and notifications.
- Web3 tab is a demo; add real smart contract logic as needed.

---
For questions or improvements, open an issue or pull request.
