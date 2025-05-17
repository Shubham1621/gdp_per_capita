import requests
import pandas as pd

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
    # Add more as needed
}

COUNTRY_URL = 'http://api.worldbank.org/v2/country?format=json&per_page=400'
API_URL = 'http://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=100&date=2000:2023'


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
            for entry in resp.json()[1]:
                data.append({
                    'country': country,
                    'year': entry['date'],
                    'indicator': indicator,
                    'value': entry['value']
                })
        except Exception:
            continue
    return pd.DataFrame(data)


def fetch_all_indicators():
    dfs = []
    for key, indicator in INDICATORS.items():
        df = fetch_indicator_for_all_countries(indicator)
        df['indicator_name'] = key
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)
    return all_data


def main():
    df = fetch_all_indicators()
    df.to_csv('data/economic_data_all_countries.csv', index=False)
    print('Saved all economic data to data/economic_data_all_countries.csv')


if __name__ == '__main__':
    main()
