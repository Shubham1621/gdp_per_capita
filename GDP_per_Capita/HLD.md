# High-Level Design (HLD): Global Economic & Markets AI Dashboard

## 1. System Architecture

```
+-------------------+      +-------------------+      +-------------------+      +-------------------+
|  Data Fetching    | ---> |  Data Enhancement | ---> |   Dashboard/API   | ---> |  3D Globe/Markets |
| (fetch_data.py)   |      | (enhance_analytics|      |  (Streamlit, SQL) |      |  (CesiumJS, JS)   |
+-------------------+      +-------------------+      +-------------------+      +-------------------+
        |                        |                          |                          |
        v                        v                          v                          v
   [Raw Data CSVs]         [Analytics CSVs]           [User Interface]         [3D Globe/Market View]
```

- **Data Fetching:** Pulls data from APIs (World Bank, IMF, UN, Yahoo Finance, NewsAPI, etc.) and Wikipedia.
- **Data Enhancement:** Runs ML/AI analytics, forecasting, clustering, SHAP, DCF, macro models, sentiment analysis.
- **Dashboard/API:** Streamlit app for interactive exploration, SQL queries, AI insights, and login.
- **3D Globe/Markets:** Embedded CesiumJS/Google Earth for global visualization; market/news/valuation modules.

## 2. Data Flow Diagram

1. **Data Ingestion:**
   - `data/fetch_data.py` → `data/economic_data_all_countries.csv`, `data/gdp_per_capita.csv`, market/news data
2. **Analytics Enhancement:**
   - `dashboard/enhance_analytics.py` → updates analytics columns/outputs (ML, DCF, macro, sentiment)
3. **Dashboard:**
   - Loads processed data, provides UI, runs ML/AI/valuation on demand, exposes SQL interface, 3D globe, news, and market modules

## 3. Component Descriptions

- **fetch_data.py:** Fetches and updates economic, market, news, and demographic data from multiple sources.
- **enhance_analytics.py:** Runs scheduled analytics (ML, clustering, forecasting, SHAP, DCF, macro models, sentiment) and updates data files.
- **app.py:** Main Streamlit dashboard, handles navigation, visualization, AI/ML, SQL, login, and user interaction.
- **3D Globe Module:** CesiumJS/Google Earth API for interactive global visualization (country/market overlays, news, analytics).
- **Market/News Modules:** Integrate Yahoo Finance, NewsAPI, and sentiment analysis for global market/news coverage.
- **Valuation/Macro Modules:** DCF, scenario analysis, macroeconomic models for companies and countries.
- **custom_style.css:** Bloomberg Terminal-inspired premium UI/UX.
- **setup_scheduled_jobs.ps1:** PowerShell script for automating jobs on Windows.
- **.github/workflows/automate_data_and_analytics.yml:** GitHub Actions workflow for cloud automation.

## 4. Technology Stack
- **Python:** Data processing, ML, Streamlit dashboard
- **Pandas, NumPy:** Data wrangling
- **scikit-learn, statsmodels, SHAP:** ML/AI analytics
- **DuckDB:** In-memory SQL
- **OpenAI API:** LLM-powered insights
- **CesiumJS/Google Earth API:** 3D globe visualization
- **Yahoo Finance, NewsAPI:** Free market/news data
- **Docker:** Containerization
- **GitHub Actions:** CI/CD and automation

## 5. Security Considerations
- **Secrets:** API keys (OpenAI, Google, NewsAPI) are never stored or hardcoded; entered securely by user.
- **Data Privacy:** No user data is stored; all analysis is in-memory or local.
- **Login:** Simple username/password (extensible to OAuth).

## 6. Extensibility Points
- Add new data sources or indicators in `fetch_data.py`.
- Extend analytics in `enhance_analytics.py` (add DCF, macro, sentiment, etc.).
- Add new dashboard tabs, 3D globe overlays, or visualizations in `app.py`.
- Integrate user authentication, notifications, or external APIs.
- Add new market/news/valuation modules as plugins.

## 7. Research & Methodology
- **Singaporean Economic Frameworks:** Implement best practices and analytics as used by Singapore’s administration since the 1960s (e.g., long-term planning, scenario analysis, macroprudential policy, sovereign wealth management).
- **Citations:** All analytics and models cite relevant research papers and frameworks in the UI and documentation.
- **Continuous Improvement:** LLM/AI modules learn from user feedback and new research.

## 8. Use Cases
- **Business:** Market entry, macro risk, peer benchmarking, valuation, scenario planning.
- **Think Tanks:** Policy analysis, global trends, research, forecasting, news/sentiment.
- **Government:** Economic planning, crisis response, long-term macro analysis, DCF for public assets.

## 9. Future Enhancements
- User accounts and saved dashboards
- Custom alerts/notifications
- REST API for external access
- More data sources and real-time feeds
- Mobile and accessibility improvements
- Payment/premium features (optional)

## 10. 3D Globe & Market/News Modules
- **3D Globe:**
  - CesiumJS/Google Earth API integration for interactive global visualization
  - Country overlays, market data, news, and analytics on the globe
  - User can select countries, view time series, and compare indicators on the globe
- **Market Data Module:**
  - Integrates free APIs (Yahoo Finance, Alpha Vantage, etc.) for global stocks, indices, commodities, FX, crypto
  - Market heatmaps, sector performance, and historical trends
- **News & Sentiment Module:**
  - Integrates NewsAPI and RSS for global economic/financial news
  - AI-powered summarization and sentiment analysis
  - News impact scoring and trending topics
- **UI/UX Overhaul:**
  - Bloomberg Terminal-inspired modular panels
  - Customizable layouts, dark mode, keyboard shortcuts
  - Responsive design for desktop, tablet, and mobile

## 11. Extensibility & Research
- Modular plugin system for new data, analytics, and visualizations
- Research-backed analytics (Singaporean frameworks, global best practices)
- All models and analytics cite research in UI and documentation
- Continuous improvement via user feedback and LLM learning
