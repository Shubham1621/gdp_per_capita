# --- Imports ---
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from iso3166 import countries_by_alpha3
import requests

# --- Helper Functions ---
def get_country_name(code):
    try:
        return countries_by_alpha3[code].name
    except Exception:
        return code

def format_value(val):
    if pd.isnull(val):
        return ''
    val = float(val)
    return f"{val:,.2f}"

# --- Load Data ---
try:
    gdp_data = pd.read_csv('data/economic_data_all_countries.csv')
    required_cols = {'country','indicator_name','year','value'}
    if not required_cols.issubset(gdp_data.columns):
        st.error(f"Data file missing required columns: {required_cols - set(gdp_data.columns)}")
        st.stop()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# --- Sidebar: Country, Indicator, Year Selection ---
countries = gdp_data['country'].unique()
country_options = [f"{get_country_name(c)} ({c})" for c in sorted(countries)]
selected_countries = st.sidebar.multiselect('Select Countries', country_options, default=country_options[:1])
country_codes = [c.split('(')[-1].replace(')','').strip() for c in selected_countries]
indicators = gdp_data['indicator_name'].unique()
selected_indicators = st.sidebar.multiselect('Select Indicators', sorted(indicators), default=list(indicators)[:2])
years = sorted(gdp_data['year'].unique())
year_range = st.sidebar.slider('Year Range', min_value=int(min(years)), max_value=int(max(years)), value=(int(min(years)), int(max(years))))

# --- Filter Data ---
filtered = gdp_data[(gdp_data['country'].isin(country_codes)) & (gdp_data['indicator_name'].isin(selected_indicators)) & (gdp_data['year'] >= year_range[0]) & (gdp_data['year'] <= year_range[1])]

# --- Main UI ---
st.markdown('''<style>
.topnav {
  background: linear-gradient(90deg,#0071e3 60%,#00c6fb 100%);
  color: #fff;
  padding: 1.1em 2em 1.1em 2em;
  border-radius: 0 0 18px 18px;
  margin-bottom: 2em;
  display: flex;
  align-items: center;
  justify-content: flex-start;
  box-shadow: 0 4px 24px rgba(0,0,0,0.08);
}
.topnav .left {
  font-size: 1.5em;
  font-weight: 700;
  letter-spacing: -1px;
  display: flex;
  align-items: center;
}
</style>
<div class="topnav">
  <div class="left">Global Economic Dashboard</div>
</div>''', unsafe_allow_html=True)

tab_names = [
    "Summary", "ML/AI Forecasting", "Correlation Explorer", "Macro Events Timeline", "Custom Insights & AI Chat", "DCF Valuation", "SQL Data Exploration", "2D Globe", "LLM Data Copilot", "Web3 Data"
]
tabs = st.tabs(tab_names)

# --- Summary Tab ---
with tabs[0]:
    st.title('🌍 Live Economic Dashboard for Job Search')
    st.markdown('''<div style="background:linear-gradient(90deg,#f5f6fa,#e3e9f7 80%);border-radius:20px;padding:1.5em 2em 1em 2em;margin-bottom:2em;box-shadow:0 4px 24px rgba(0,0,0,0.06);">
    <h2 style="margin-bottom:0.2em;font-size:2em;font-weight:700;letter-spacing:-1px;color:#0071e3;">Key Economic Indicators <span title='GDP, labor, sectoral, and more. Hover on cards for details.' style='cursor:help;'>ℹ️</span></h2>
    <div style="font-size:1.1em;color:#555;">Compare and track the most relevant economic data for your job search, interviews, and live market awareness.</div>
    </div>''', unsafe_allow_html=True)

    # --- Economic Map Section ---
    st.subheader('World Map: Economic Indicators')
    map_indicator = st.selectbox('Select indicator for map', sorted(indicators), index=0, help='Choose an economic parameter to visualize on the world map.')
    map_year = st.slider('Select year for map', min_value=int(min(years)), max_value=int(max(years)), value=int(max(years)), help='Choose the year to display on the map.')

    map_data = gdp_data[(gdp_data['indicator_name'] == map_indicator) & (gdp_data['year'] == map_year)]
    if not map_data.empty:
        fig_map = px.choropleth(
            map_data,
            locations='country',
            color='value',
            hover_name='country',
            color_continuous_scale='Blues',
            labels={'value': map_indicator.replace('_',' ').title()},
            title=f'GDP per Capita ({map_year})',
            projection='natural earth',
        )
        fig_map.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            geo=dict(showcoastlines=True, showland=True, landcolor="#f5f6fa", fitbounds="locations"),
            coloraxis_colorbar=dict(title=map_indicator.replace('_',' ').title())
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info('No data available for this indicator/year.')

    # --- Data Cards for Each Country/Indicator ---
    # (Removed per user request: replaced with bubble chart below)

    # --- Bubble Chart for All Countries/Indicators ---
    st.subheader('Bubble Chart: Latest Values by Country & Indicator')
    bubble_df = pd.DataFrame()
    for code in country_codes:
        for ind in selected_indicators:
            data = filtered[(filtered['country'] == code) & (filtered['indicator_name'] == ind)]
            if not data.empty:
                latest = data.sort_values('year').iloc[-1]
                bubble_df = pd.concat([
                    bubble_df,
                    pd.DataFrame({
                        'Country': [get_country_name(code)],
                        'Indicator': [ind.replace('_',' ').title()],
                        'Value': [latest['value']]
                    })
                ], ignore_index=True)
    if not bubble_df.empty:
        fig_bubble = px.scatter(
            bubble_df,
            x='Country',
            y='Indicator',
            size='Value',
            color='Value',
            hover_name='Country',
            size_max=60,
            color_continuous_scale='Blues',
            title='Latest Value by Country & Indicator (Bubble Size = Value)'
        )
        fig_bubble.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='#f5f6fa',
            paper_bgcolor='#f5f6fa',
            font=dict(size=14)
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.info('No data available for bubble chart.')

    # --- Comparison Table ---
    st.subheader('Country Comparison Table')
    st.markdown('<span style="color:#888;font-size:1em;">Shows last 3 years for each indicator.</span>', unsafe_allow_html=True)
    comparison_rows = []
    for code in country_codes:
        row = {'Country': get_country_name(code)}
        for y in [year_range[1]-2, year_range[1]-1, year_range[1]]:
            for ind in selected_indicators:
                val = gdp_data.query('country == @code and indicator_name == @ind and year == @y')['value']
                row[f"{ind} {y}"] = format_value(val.values[0]) if not val.empty else 'N/A'
        comparison_rows.append(row)
    st.table(pd.DataFrame(comparison_rows))

    # --- Trend Chart ---
    st.subheader('Trends Over Time')
    st.markdown('<span style="color:#888;font-size:1em;">Line chart for selected indicators and countries.</span>', unsafe_allow_html=True)
    chart_df = pd.DataFrame()
    for code in country_codes:
        for ind in selected_indicators:
            data = gdp_data[(gdp_data['country'] == code) & (gdp_data['indicator_name'] == ind) & (gdp_data['year'] >= year_range[0]) & (gdp_data['year'] <= year_range[1])]
            if not data.empty:
                temp = data[['year','value']].copy()
                temp['Country'] = get_country_name(code)
                temp['Indicator'] = ind.replace('_',' ').title()
                chart_df = pd.concat([chart_df, temp])
    if not chart_df.empty:
        fig = px.line(chart_df, x='year', y='value', color='Country', line_dash='Indicator', markers=True, title='Selected Indicators: Countries')
        fig.update_layout(font=dict(size=16), plot_bgcolor='#f5f6fa', paper_bgcolor='#f5f6fa')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div style="color:#555;font-size:1em;">Use this dashboard to quickly compare countries and indicators for interviews, applications, and live market awareness. Focus on recent data and trends for the most relevant insights.</div>', unsafe_allow_html=True)

    # --- Data Quality & Anomaly Detection ---
    st.subheader('Data Quality & Anomaly Detection')
    if not filtered.empty:
        # Outlier detection using IQR
        q1 = filtered.groupby('indicator_name')['value'].quantile(0.25)
        q3 = filtered.groupby('indicator_name')['value'].quantile(0.75)
        iqr = q3 - q1
        def is_outlier(row):
            q1_val = q1[row['indicator_name']]
            q3_val = q3[row['indicator_name']]
            iqr_val = iqr[row['indicator_name']]
            return row['value'] < (q1_val - 1.5 * iqr_val) or row['value'] > (q3_val + 1.5 * iqr_val)
        filtered['outlier'] = filtered.apply(is_outlier, axis=1)
        st.write('Outliers flagged in the table below:')
        def highlight_outlier_row(row):
            return ["background: #ffcccc" if row['outlier'] and col == 'outlier' else "" for col in row.index]
        st.dataframe(filtered[['country','indicator_name','year','value','outlier']].style.apply(highlight_outlier_row, axis=1))
    else:
        st.info('No data for anomaly detection.')

    # --- Download & API Integration ---
    st.subheader('Download & API Access')
    if not filtered.empty:
        csv = filtered.to_csv(index=False)
        st.download_button('Download Filtered Data as CSV', csv, file_name='filtered_gdp_data.csv')
        st.markdown('''<span style="color:#888;font-size:1em;">For programmatic access, use the REST API endpoint below (example only):</span>''', unsafe_allow_html=True)
        st.code('GET /api/gdp?country=USA&indicator=GDP_per_capita&year=2022', language='http')
    else:
        st.info('No data to download.')

    # --- Data Lineage & Metadata ---
    st.subheader('Data Lineage & Metadata')
    if not filtered.empty:
        st.write('**Data Source:** data/gdp_per_capita.csv')
        st.write('**Country Codes:** ISO 3166-1 alpha-3')
        st.write('**Indicators:**', ', '.join(sorted(filtered["indicator_name"].unique())))
        st.write('**Years:**', f"{filtered['year'].min()} - {filtered['year'].max()}")
    else:
        st.info('No metadata available for current selection.')

# --- ML/AI Forecasting Tab ---
with tabs[1]:
    st.header('ML/AI Forecasting')
    st.info('Forecast economic indicators using ARIMA, Random Forest, or XGBoost. (Demo: ARIMA only)')
    import statsmodels.api as sm
    forecast_country = st.selectbox('Country', sorted(countries))
    forecast_indicator = st.selectbox('Indicator', sorted(indicators))
    forecast_data = gdp_data[(gdp_data['country'] == forecast_country) & (gdp_data['indicator_name'] == forecast_indicator)].sort_values('year')
    if len(forecast_data) > 5:
        y = forecast_data['value'].values
        model = sm.tsa.ARIMA(y, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=5)
        st.line_chart(list(y) + list(forecast))
        st.write('Forecast (next 5 years):', forecast)
    else:
        st.warning('Not enough data for forecasting.')

# --- Correlation Explorer Tab ---
with tabs[2]:
    st.header('Correlation Explorer')
    st.info('Explore correlations between selected economic indicators and countries. Heatmap and pairplot for deeper insights.')
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        corr_data = filtered.pivot_table(index='year', columns=['country','indicator_name'], values='value')
        if corr_data.shape[1] > 1:
            corr = corr_data.corr()
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)
            st.pyplot(fig)
            st.write('**Tip:** Select multiple countries and indicators in the sidebar for richer correlation analysis.')
            # Pairplot for selected indicators
            if len(selected_indicators) > 1:
                pair_data = filtered.pivot(index='year', columns='indicator_name', values='value').dropna()
                if pair_data.shape[1] > 1:
                    pairplot_fig = sns.pairplot(pair_data)
                    st.pyplot(pairplot_fig.fig)
        else:
            st.info('Select at least two indicators/countries for correlation analysis.')
    except ImportError:
        st.warning('Seaborn is not installed. Please run: pip install seaborn')

# --- Macro Events Timeline Tab ---
with tabs[3]:
    st.header('Macro Events Timeline')
    st.info('Timeline of major global economic events. Useful for context and data interpretation.')
    import plotly.graph_objects as go
    events = [
        {"year": 2008, "event": "Global Financial Crisis"},
        {"year": 2010, "event": "Eurozone Debt Crisis"},
        {"year": 2014, "event": "Oil Price Crash"},
        {"year": 2020, "event": "COVID-19 Pandemic"},
        {"year": 2022, "event": "Global Inflation Surge"},
    ]
    timeline_years = [e['year'] for e in events]
    timeline_events = [e['event'] for e in events]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timeline_years, y=[1]*len(events), mode='markers+text', text=timeline_events, textposition='top center', marker=dict(size=16, color='#0071e3')))
    fig.update_layout(title='Major Global Economic Events (2008-2025)', yaxis=dict(visible=False), xaxis=dict(title='Year', tickmode='linear'), showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('''<div style="color:#555;font-size:1em;">These events often explain sudden changes in economic indicators. Use this timeline to interpret data spikes or drops.</div>''', unsafe_allow_html=True)

# --- Custom Insights & AI Chat Tab ---
with tabs[4]:
    st.header('Custom Insights & AI Chat')
    st.info('Get instant insights and answers about the selected data. (Static demo: AI chat integration coming soon)')
    # Example: Show top 3 insights for selected data
    if not filtered.empty:
        st.subheader('Top Insights')
        # Insight 1: Highest GDP per capita
        if any(filtered['indicator_name'].str.contains('gdp', case=False)):
            top_gdp = filtered[filtered['indicator_name'].str.contains('gdp', case=False)].sort_values('value', ascending=False).iloc[0]
            st.write(f"{get_country_name(top_gdp['country'])} had the highest GDP per capita ({format_value(top_gdp['value'])}) in {top_gdp['year']}.")
        # Insight 2: Fastest growth (CAGR)
        growth_list = []
        for (country, indicator), group in filtered.groupby(['country','indicator_name']):
            cagr_val = None
            if len(group['value']) > 1:
                start, end = group['value'].iloc[0], group['value'].iloc[-1]
                n = len(group['value'])-1
                if start > 0 and end > 0 and n > 0:
                    cagr_val = (end/start)**(1/n)-1
            if cagr_val is not None:
                growth_list.append({'country': country, 'indicator_name': indicator, 'CAGR': cagr_val})
        if growth_list:
            growth_df = pd.DataFrame(growth_list)
            fastest = growth_df.sort_values('CAGR', ascending=False).iloc[0]
            st.write(f"{get_country_name(fastest['country'])} had the fastest CAGR ({fastest['indicator_name']} at {fastest['CAGR']:.2%}).")
        # Insight 3: Most recent data year
        recent = filtered.sort_values('year', ascending=False).iloc[0]
        st.write(f"Most recent data: {get_country_name(recent['country'])}, {recent['indicator_name']} in {recent['year']}.")
    else:
        st.info('Select data in the sidebar to see insights.')
    st.markdown('''---\n**AI Chat (coming soon):** Ask questions about the data, get summaries, and more. For now, use the SQL tab for custom queries.''')

# --- DCF Valuation Tab ---
with tabs[5]:
    st.header('DCF Valuation')
    st.info('Discounted Cash Flow (DCF) valuation for countries/companies.')
    dcf_country = st.selectbox('Country for DCF', sorted(countries), key='dcf_country')
    dcf_indicator = st.selectbox('Indicator for DCF (e.g., GDP)', sorted(indicators), key='dcf_indicator')
    dcf_data = gdp_data[(gdp_data['country'] == dcf_country) & (gdp_data['indicator_name'] == dcf_indicator)].sort_values('year')
    if len(dcf_data) > 5:
        # Simple DCF: forecast next 5 years using CAGR, discount at 10%
        vals = dcf_data['value'].values
        years_hist = dcf_data['year'].values
        cagr = None
        if len(vals) > 1:
            start, end = vals[0], vals[-1]
            n = len(vals)-1
            if start > 0 and end > 0 and n > 0:
                cagr = (end/start)**(1/n)-1
        if cagr is not None:
            last_val = vals[-1]
            forecast = [last_val * (1 + cagr) ** i for i in range(1, 6)]
            discount_rate = 0.1
            discounted = [f / ((1 + discount_rate) ** i) for i, f in enumerate(forecast, 1)]
            dcf_value = sum(discounted)
            st.write(f"**DCF Value (next 5 years, discounted at 10%):** {format_value(dcf_value)}")
            # Ensure years_hist[-1] is int for range()
            last_year = int(years_hist[-1]) if len(years_hist) > 0 else 0
            start_year = last_year + 1
            end_year = last_year + 6
            st.line_chart(pd.DataFrame({'Year': list(range(start_year, end_year)), 'Forecast': forecast, 'Discounted': discounted}).set_index('Year'))
        else:
            st.info('Not enough data for DCF calculation.')
    else:
        st.info('Not enough data for DCF calculation.')

# --- SQL Data Exploration Tab ---
with tabs[6]:
    st.header('SQL Data Exploration')
    st.info('Run SQL queries on the dataset (DuckDB in-memory).')
    import duckdb
    query = st.text_area('Enter SQL query:', 'SELECT * FROM gdp_data LIMIT 5')
    try:
        con = duckdb.connect()
        con.register('gdp_data', gdp_data)
        # Ensure DuckDB can handle numpy types in query
        result = con.execute(str(query)).df()
        st.dataframe(result)
    except Exception as e:
        st.error(f'Query failed: {e}')

# --- 2D Globe Visualization Tab ---
with tabs[7]:
    st.header('2D Globe Visualization')
    st.info('Interactive world map of economic indicators.')
    globe_indicator = st.selectbox('Select indicator for globe', sorted(indicators), index=0, help='Choose an economic parameter to visualize on the 2D globe.', key='globe_indicator')
    globe_year = st.slider('Select year for globe', min_value=int(min(years)), max_value=int(max(years)), value=int(max(years)), help='Choose the year to display on the globe.', key='globe_year')
    globe_data = gdp_data[(gdp_data['indicator_name'] == globe_indicator) & (gdp_data['year'] == globe_year)]
    if not globe_data.empty:
        # Ensure globe_year is int for key and range
        globe_year_int = int(globe_year)
        fig_globe = px.choropleth(
            globe_data,
            locations='country',
            color='value',
            hover_name='country',
            color_continuous_scale='Blues',
            labels={'value': globe_indicator.replace('_',' ').title()},
            title=f'{globe_indicator.replace("_"," ").title()} ({globe_year_int})',
            projection='natural earth',
        )
        fig_globe.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            geo=dict(showcoastlines=True, showland=True, landcolor="#f5f6fa", fitbounds="locations"),
            coloraxis_colorbar=dict(title=globe_indicator.replace('_',' ').title())
        )
        st.plotly_chart(fig_globe, use_container_width=True, key=f'globe_plot_{globe_indicator}_{globe_year_int}')
    else:
        st.info('No data available for this indicator/year.')

# --- LLM Data Copilot Tab ---
with tabs[8]:
    st.header('LLM Data Copilot')
    st.info('Ask questions about the data, generate SQL, and get instant insights using your own LLM (Ollama, LocalAI, etc.).')
    ollama_url = st.text_input('Ollama/LocalAI API URL', value='http://localhost:11434', help='URL of your self-hosted LLM server')
    user_query = st.text_area('Ask a question or request a SQL query:', 'Show top 5 countries by GDP per capita in 2022')
    if st.button('Ask LLM') and ollama_url and user_query:
        try:
            prompt = f"You are a senior data engineer. Given the following data columns: {list(gdp_data.columns)}, generate a SQL query for DuckDB to answer: '{user_query}'. Only output the SQL query."
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": "llama3", "prompt": prompt, "stream": False}
            )
            sql = response.json().get('response', '').strip()
            st.code(sql, language='sql')
            import duckdb
            con = duckdb.connect()
            con.register('gdp_data', gdp_data)
            result = con.execute(sql).df()
            st.dataframe(result)
        except Exception as e:
            st.error(f'LLM or SQL error: {e}')
    st.markdown('''<span style="color:#888;font-size:1em;">LLM Copilot uses your own local LLM (Ollama, LocalAI, etc.). No data leaves your machine.</span>''', unsafe_allow_html=True)

# --- Web3 Data Tab ---
with tabs[9]:
    st.header('Web3 Data')
    st.info('View and interact with economic data on Ethereum testnet (demo).')
    try:
        from web3 import Web3
        infura_url = st.text_input('Infura/Alchemy Testnet URL', value='https://sepolia.infura.io/v3/YOUR_KEY', help='Get a free endpoint at infura.io or alchemy.com')
        if infura_url:
            w3 = Web3(Web3.HTTPProvider(infura_url))
            st.write('Connected:', w3.is_connected())
            # Demo: Show latest block and a sample on-chain GDP value (mock)
            block = w3.eth.block_number
            st.write('Latest Block:', block)
            # Simulate on-chain GDP data (replace with real contract call as needed)
            st.write('Sample On-Chain GDP (mock):', 12345.67)
            st.markdown('<span style="color:#888;font-size:1em;">Web3 integration lets you push/pull economic data to/from blockchain for transparency and auditability.</span>', unsafe_allow_html=True)
    except ImportError:
        st.warning('web3.py is not installed. Please run: pip install web3')