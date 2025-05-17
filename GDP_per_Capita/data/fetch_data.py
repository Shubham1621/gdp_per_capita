import requests
import pandas as pd

# World Bank API indicators for GDP per capita (nominal and PPP)
INDICATORS = {
    'gdp_per_capita_nominal': 'NY.GDP.PCAP.CD',
    'gdp_per_capita_ppp': 'NY.GDP.PCAP.PP.CD',
    # Add more indicators as needed
}

COUNTRY_URL = 'http://api.worldbank.org/v2/country?format=json&per_page=400'
API_URL = 'http://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=100&date=2022'


def fetch_countries():
    resp = requests.get(COUNTRY_URL)
    countries = resp.json()[1]
    return [c['id'] for c in countries if c['region']['id'] != 'NA']


def fetch_indicator_for_all_countries(indicator):
    countries = fetch_countries()
    data = []
    for country in countries:
        url = API_URL.format(country=country, indicator=indicator)
        resp = requests.get(url)
        try:
            value = resp.json()[1][0]['value']
        except Exception:
            value = None
        data.append({'country': country, 'value': value})
    return pd.DataFrame(data)


def fetch_gdp_per_capita():
    dfs = {}
    for key, indicator in INDICATORS.items():
        dfs[key] = fetch_indicator_for_all_countries(indicator)
    df = dfs['gdp_per_capita_nominal'].merge(
        dfs['gdp_per_capita_ppp'], on='country', suffixes=('_nominal', '_ppp')
    )
    return df


def main():
    df = fetch_gdp_per_capita()
    df.to_csv('data/gdp_per_capita.csv', index=False)
    print('Saved GDP per capita data to data/gdp_per_capita.csv')


if __name__ == '__main__':
    main()
