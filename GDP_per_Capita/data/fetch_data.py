import requests
import pandas as pd
import time

# World Bank API indicators for GDP per capita (nominal and PPP) and more
INDICATORS = {
    'gdp_per_capita_nominal': 'NY.GDP.PCAP.CD',
    'gdp_per_capita_ppp': 'NY.GDP.PCAP.PP.CD',
    'gdp_nominal': 'NY.GDP.MKTP.CD',
    'gdp_ppp': 'NY.GDP.MKTP.PP.CD',
    'inflation': 'FP.CPI.TOTL.ZG',
    'unemployment': 'SL.UEM.TOTL.ZS',
    'labor_force': 'SL.TLF.TOTL.IN',
    'exports': 'NE.EXP.GNFS.CD',
    'imports': 'NE.IMP.GNFS.CD',
    'fdi': 'BX.KLT.DINV.CD.WD',
    'reserves': 'FI.RES.TOTL.CD',
    'budget_deficit': 'GC.BAL.CASH.GD.ZS',
    'population': 'SP.POP.TOTL',
    'life_expectancy': 'SP.DYN.LE00.IN',
    'govt_debt_gdp': 'GC.DOD.TOTL.GD.ZS',
    'current_account_gdp': 'BN.CAB.XOKA.GD.ZS',
    'poverty_headcount': 'SI.POV.DDAY',
    'education_gdp': 'SE.XPD.TOTL.GD.ZS',
    'health_gdp': 'SH.XPD.CHEX.GD.ZS',
    # Add more as needed
}

INDICATOR_DESCRIPTIONS = {
    'gdp_per_capita_nominal': 'GDP per capita (current US$)',
    'gdp_per_capita_ppp': 'GDP per capita, PPP (current international $)',
    'gdp_nominal': 'GDP (current US$)',
    'gdp_ppp': 'GDP, PPP (current international $)',
    'inflation': 'Inflation, consumer prices (annual %)',
    'unemployment': 'Unemployment, total (% of total labor force)',
    'labor_force': 'Labor force, total',
    'exports': 'Exports of goods and services (current US$)',
    'imports': 'Imports of goods and services (current US$)',
    'fdi': 'Foreign direct investment, net inflows (BoP, current US$)',
    'reserves': 'Total reserves (includes gold, current US$)',
    'budget_deficit': 'Cash surplus/deficit (% of GDP)',
    'population': 'Population, total',
    'life_expectancy': 'Life expectancy at birth, total (years)',
    'govt_debt_gdp': 'Central government debt, total (% of GDP)',
    'current_account_gdp': 'Current account balance (% of GDP)',
    'poverty_headcount': 'Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)',
    'education_gdp': 'Government expenditure on education (% of GDP)',
    'health_gdp': 'Current health expenditure (% of GDP)',
}

COUNTRY_URL = 'http://api.worldbank.org/v2/country?format=json&per_page=400'
API_URL = 'http://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=100&date=2000:2023'


def fetch_countries():
    resp = requests.get(COUNTRY_URL)
    countries = resp.json()[1]
    return [
        {
            'id': c['id'],
            'name': c['name'],
            'region': c['region']['value'],
            'income_level': c['incomeLevel']['value']
        }
        for c in countries if c['region']['id'] != 'NA'
    ]


def fetch_indicator_for_all_countries(indicator):
    countries = fetch_countries()
    data = []
    for i, country in enumerate(countries):
        url = API_URL.format(country=country['id'], indicator=indicator)
        resp = requests.get(url)
        try:
            for entry in resp.json()[1]:
                data.append({
                    'country': country['id'],
                    'country_name': country['name'],
                    'region': country['region'],
                    'income_level': country['income_level'],
                    'year': int(entry['date']),
                    'indicator': indicator,
                    'value': entry['value']
                })
        except Exception:
            continue
        if i % 20 == 0:
            print(f"Fetched {i+1}/{len(countries)} countries for {indicator}")
        time.sleep(0.1)  # Be nice to the API
    return pd.DataFrame(data)


def fetch_all_indicators():
    dfs = []
    for key, indicator in INDICATORS.items():
        print(f"Fetching {key} ({indicator})...")
        df = fetch_indicator_for_all_countries(indicator)
        df['indicator_name'] = key
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)
    return all_data


def main():
    df = fetch_all_indicators()
    df = df[['country', 'country_name', 'region', 'income_level', 'year', 'indicator', 'indicator_name', 'value']]
    df = df.sort_values(['country', 'indicator_name', 'year'])
    df.to_csv('data/economic_data_all_countries.csv', index=False)
    print('Saved all economic data to data/economic_data_all_countries.csv')
    # Save indicator metadata
    pd.DataFrame([
        {'indicator_name': k, 'indicator_code': v, 'description': INDICATOR_DESCRIPTIONS.get(k, '')}
        for k, v in INDICATORS.items()
    ]).to_csv('data/indicator_metadata.csv', index=False)
    print('Saved indicator metadata to data/indicator_metadata.csv')


if __name__ == '__main__':
    main()
