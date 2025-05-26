# --- Imports ---
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from iso3166 import countries_by_alpha3

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
gdp_data = pd.read_csv('data/gdp_per_capita.csv')

# --- Sidebar: Country & Year Selection ---
countries = gdp_data['country'].unique()
years = sorted(gdp_data['year'].unique())
country_options = [f"{get_country_name(c)} ({c})" for c in sorted(countries)]
selected_countries = st.sidebar.multiselect('Select Countries', country_options, default=country_options[:3])
country_codes = [c.split('(')[-1].replace(')','').strip() for c in selected_countries]
year_range = st.sidebar.slider('Year Range', min_value=int(min(years)), max_value=int(max(years)), value=(int(min(years)), int(max(years))))

# --- Filter Data ---
filtered = gdp_data[(gdp_data['country'].isin(country_codes)) & (gdp_data['year'] >= year_range[0]) & (gdp_data['year'] <= year_range[1])]

# --- Main UI ---
st.title('🌍 Global GDP per Capita Dashboard')
st.markdown('''<div style="background:#f5f6fa;border-radius:16px;padding:1.2em 2em 1em 2em;margin-bottom:2em;box-shadow:0 2px 12px rgba(0,0,0,0.07);">
<h2 style="margin-bottom:0.2em;font-size:2em;font-weight:700;letter-spacing:-1px;color:#0071e3;">GDP per Capita Analysis <span title='Nominal GDP per capita by country. Hover on map or table for details.' style='cursor:help;'>ℹ️</span></h2>
<div style="font-size:1.1em;color:#555;">Explore, compare, and visualize GDP per capita across countries and years. Use the sidebar to select countries and time periods for focused economic analysis.</div>
</div>''', unsafe_allow_html=True)

# --- World Map ---
st.subheader('World Map: GDP per Capita')
map_year = st.slider('Select year for map', min_value=int(min(years)), max_value=int(max(years)), value=int(max(years)), help='Year to display on the map.')
map_data = gdp_data[gdp_data['year'] == map_year]
if not map_data.empty:
    fig_map = px.choropleth(
        map_data,
        locations='country',
        color='value',
        hover_name='country',
        color_continuous_scale='Blues',
        labels={'value': 'GDP per Capita'},
        title=f'GDP per Capita ({map_year})',
        projection='natural earth',
    )
    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, geo=dict(showcoastlines=True, showland=True, landcolor="#f5f6fa", fitbounds="locations"), coloraxis_colorbar=dict(title='GDP per Capita'))
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info('No data available for this year.')

# --- Data Table ---
st.subheader('Country Comparison Table')
st.markdown('<span style="color:#888;font-size:1em;">GDP per capita for selected countries and years.</span>', unsafe_allow_html=True)
if not filtered.empty:
    table = filtered.pivot(index='year', columns='country', values='value')
    st.dataframe(table.style.format('{:,.2f}'))
else:
    st.info('No data for selected countries/years.')

# --- Trend Chart ---
st.subheader('Trends Over Time')
st.markdown('<span style="color:#888;font-size:1em;">Line chart for selected countries.</span>', unsafe_allow_html=True)
if not filtered.empty:
    fig = px.line(filtered, x='year', y='value', color='country', markers=True, title='GDP per Capita Over Time')
    fig.update_layout(font=dict(size=16), plot_bgcolor='#f5f6fa', paper_bgcolor='#f5f6fa')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('No data for selected countries/years.')