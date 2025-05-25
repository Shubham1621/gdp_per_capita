# Global Economic & Markets AI Dashboard

A premium, research-backed dashboard for exploring, analyzing, and visualizing global economic, market, and news data—powered by AI, ML, and a 3D interactive globe. Inspired by Bloomberg Terminal, designed for business, think tanks, and government.

## Features (100+)
- **3D Globe Visualization:** Interactive CesiumJS/Google Earth globe with country overlays, market data, and news.
- **Market Data:** Stocks, indices, commodities, FX, crypto (free/delayed, global coverage).
- **Economic Indicators:** GDP, inflation, unemployment, sectoral composition, budget, labor, demographics, and more.
- **News & Sentiment:** Global economic/financial news, AI-powered summarization, and sentiment analysis.
- **Valuation & Macro Models:** DCF, scenario analysis, macroeconomic stress testing for companies and countries.
- **AI/LLM Insights:** ChatGPT-powered commentary, continuous learning, research-backed analytics.
- **Advanced Analytics:** Rolling stats, volatility, outlier detection, ARIMA/Random Forest forecasting, SHAP, KMeans clustering.
- **SQL Data Exploration:** DuckDB in-memory SQL queries.
- **Bloomberg Terminal UI/UX:** Modular panels, dark mode, keyboard shortcuts, responsive design.
- **User Login:** Simple authentication (username/password, extensible).
- **Automation:** Scheduled jobs for data fetching and analytics enhancement (Windows, Linux, GitHub Actions).
- **Security:** All API keys entered securely, never stored.
- **Cloud Ready:** Dockerized, free to deploy on any cloud or locally.
- **Research-Backed:** Implements Singaporean economic frameworks and best practices, with citations.
- **Extensible:** Add new data, analytics, visualizations, or modules easily.

## Feature List (100+)

### Visualization & UI
- 3D interactive globe (CesiumJS/Google Earth API integration)
- Country overlays and selection on globe
- Bloomberg Terminal-inspired modular UI
- Dark mode and light mode
- Responsive design (desktop, tablet, mobile)
- Customizable dashboard panels
- Keyboard shortcuts for navigation
- Data cards and summary widgets
- Dynamic comparison tables
- Time series, bar, and scatter plots
- Choropleth and heatmap overlays
- Downloadable charts and tables
- Custom color themes
- User profile and login
- Saved dashboard layouts (future)

### Economic & Financial Analytics
- GDP (nominal, PPP)
- Inflation, unemployment, labor force
- Sectoral composition (primary, secondary, tertiary)
- Budget deficit/surplus
- Demographics (population, fertility, life expectancy)
- Rolling stats and moving averages
- Volatility and outlier detection
- Peer benchmarking
- Country, region, and world aggregates
- Macro risk and scenario analysis
- Long-term macroeconomic forecasting
- Country and sector clustering (KMeans)
- Economic cycle detection
- Sovereign risk analysis
- Policy effectiveness scoring
- Data quality and anomaly flags

### Markets & Valuation
- Global stock indices (S&P 500, FTSE, Nikkei, etc.)
- Major stocks (US, EU, Asia, emerging markets)
- Commodities (oil, gold, agriculture, etc.)
- FX rates (USD, EUR, JPY, CNY, etc.)
- Crypto markets (BTC, ETH, etc.)
- Market heatmaps and sector performance
- DCF valuation for companies
- DCF for countries/public assets
- Peer and sector valuation multiples
- Scenario-based valuation
- Stress testing and sensitivity analysis
- Historical and projected returns
- Market correlation and beta analysis
- Sovereign wealth fund analytics

### News & Sentiment
- Global economic and financial news feed
- Country/indicator-specific news
- AI-powered news summarization
- Sentiment analysis (positive/negative/neutral)
- News impact scoring
- Trending topics and events
- News timeline and archive
- RSS and API news integration
- Alerts for major news events (future)

### AI, ML & Research
- LLM-powered insights and commentary
- Continuous learning from user feedback
- ARIMA and Random Forest forecasting
- SHAP feature importance
- Automated report generation
- Research paper citations in UI
- Singaporean economic frameworks and best practices
- Macroprudential and long-term planning models
- Policy recommendation engine
- Explainable AI for analytics
- Custom AI prompt builder

### SQL & Data Exploration
- DuckDB in-memory SQL queries
- Custom query builder
- Download query results
- Data dictionary and metadata explorer
- Data lineage and source tracking

### Security & Deployment
- Secure API key entry (OpenAI, Google, NewsAPI, etc.)
- No user data stored or shared
- Dockerized for local/cloud deployment
- GitHub Actions for CI/CD
- Windows/Linux scheduled jobs
- Extensible authentication (OAuth ready)

### Extensibility & Automation
- Modular plugin system for new data/analytics
- Easy addition of new data sources/APIs
- Custom analytics and visualization modules
- REST API for external access (future)
- Alerts/notifications (future)
- Mobile app integration (future)

### Documentation & Support
- In-app help and tooltips
- Research and methodology documentation
- Use case guides for business, think tanks, government
- Changelog and release notes
- Community and support links

## Use Cases (Detailed)

### Business
- Market entry analysis: Identify attractive countries/sectors for expansion
- Macro risk assessment: Monitor global shocks, volatility, and policy changes
- Peer benchmarking: Compare company/country performance to global/regional peers
- Valuation: DCF and scenario-based valuation for companies and assets
- News intelligence: Track market-moving news and sentiment
- Strategic planning: Use macro models and forecasts for long-term decisions

### Think Tanks & Academia
- Policy analysis: Evaluate effectiveness of economic policies and reforms
- Global trends: Track and forecast economic, demographic, and market shifts
- Research: Access and analyze data for publications and whitepapers
- Scenario modeling: Simulate policy, market, and macroeconomic scenarios
- News and event analysis: Study impact of news and events on markets/economies
- Data-driven recommendations: Generate actionable insights for stakeholders

### Government & Public Sector
- Economic planning: Long-term macroeconomic and sectoral planning
- Crisis response: Monitor and respond to global/regional economic shocks
- Asset valuation: DCF for public assets, sovereign wealth, and infrastructure
- Policy benchmarking: Compare national performance to global best practices
- News and sentiment: Track public and market sentiment on policy and events
- Transparency: Provide open data and analytics to the public

## Research & Methodology (Singaporean & Global Best Practices)
- Implements frameworks and analytics as used by Singapore’s economic administration since the 1960s:
  - Long-term planning (e.g., Economic Development Board, GIC, Temasek)
  - Scenario analysis and macroprudential policy
  - Sovereign wealth management and public asset valuation
  - Data-driven policy and continuous improvement
- Cites and implements global research papers and best practices:
  - Macro forecasting (ARIMA, Random Forest, scenario models)
  - DCF and valuation (Damodaran, IMF, World Bank)
  - Sentiment analysis and news impact (NLP, LLMs)
  - Peer benchmarking and clustering (KMeans, PCA)
- All analytics and models cite relevant research in the UI and documentation
- LLM/AI modules learn from user feedback and new research

## Getting Started
1. Ensure you have Docker and Python 3.8+ installed.
2. Build and run the dashboard using Docker:
   ```sh
   docker compose up --build
   ```
3. For 3D globe and market/news modules, obtain free API keys (Google, NewsAPI, Yahoo Finance, etc.) and enter them in the dashboard when prompted.

## Project Structure
- `data/` - Scripts and files for data fetching and cleaning
- `dashboard/` - Dashboard app code (Streamlit, 3D globe, analytics, UI)
- `.github/` - Copilot instructions

## Extensibility
- Add new data sources or indicators in `fetch_data.py`.
- Extend analytics in `enhance_analytics.py` (add DCF, macro, sentiment, etc.).
- Add new dashboard tabs, 3D globe overlays, or visualizations in `app.py`.
- Integrate user authentication, notifications, or external APIs.
- Add new market/news/valuation modules as plugins.

## Contributing
Pull requests and suggestions welcome! See [HLD.md](HLD.md) for architecture and extension points.

## License
MIT

## Model Explainability (SHAP)

The dashboard provides model explainability using SHAP (SHapley Additive exPlanations) for supported ML models (Random Forest, XGBoost, Linear Regression):

- **SHAP Summary Plot:** Shows the most important features globally for the model's predictions.
- **SHAP Force Plot:** Visualizes how each feature contributed to a single prediction (the most recent one).

**How to interpret:**
- The summary plot (bar chart) ranks features by their average impact on model output.
- The force plot shows, for a specific prediction, which features pushed the prediction higher or lower compared to the model's baseline.

If SHAP is not installed, the dashboard will prompt you to install it for these features.
