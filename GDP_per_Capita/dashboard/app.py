import pandas as pd
import plotly.express as px
import streamlit as st

# Load data
df = pd.read_csv('data/gdp_per_capita.csv')

st.title('Global GDP per Capita Dashboard')
st.markdown('Explore and compare GDP per capita (nominal and PPP) for all countries.')

# Scatter plot: Nominal vs PPP
df_clean = df.dropna(subset=['value_nominal', 'value_ppp'])
fig = px.scatter(
    df_clean,
    x='value_nominal',
    y='value_ppp',
    hover_name='country',
    title='GDP per Capita: Nominal vs PPP',
    labels={'value_nominal': 'Nominal (USD)', 'value_ppp': 'PPP (Int. $)'}
)
st.plotly_chart(fig)

# Highlight countries with high nominal but low PPP and vice versa
st.header('Countries with Unusual GDP per Capita Patterns')

# Define thresholds (can be adjusted)
high_nominal = df_clean['value_nominal'].quantile(0.85)
low_ppp = df_clean['value_ppp'].quantile(0.15)
high_ppp = df_clean['value_ppp'].quantile(0.85)
low_nominal = df_clean['value_nominal'].quantile(0.15)

high_nominal_low_ppp = df_clean[(df_clean['value_nominal'] > high_nominal) & (df_clean['value_ppp'] < low_ppp)]
high_ppp_low_nominal = df_clean[(df_clean['value_ppp'] > high_ppp) & (df_clean['value_nominal'] < low_nominal)]

st.subheader('High Nominal, Low PPP')
st.dataframe(high_nominal_low_ppp[['country', 'value_nominal', 'value_ppp']])

st.subheader('High PPP, Low Nominal')
st.dataframe(high_ppp_low_nominal[['country', 'value_nominal', 'value_ppp']])

st.markdown('---')
st.markdown('This dashboard is a starting point. More economic indicators and deeper analysis can be added.')
