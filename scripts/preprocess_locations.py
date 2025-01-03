import pandas as pd
import us

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
    
soy_df = pd.read_csv("data/Soybeans_Loc_ID.csv")
soy_df['FIPS_state'] = soy_df['State'].apply(lambda x: us.states.lookup(x).fips)
soy_df['FIPS_state'] = soy_df['FIPS_state'].astype(str)

counties_df = pd.DataFrame(counties['features'])
counties_fips = pd.DataFrame()
counties_fips['County'] = counties_df['properties'].apply(lambda x: x['NAME'].lower())
counties_fips['FIPS_state'] = counties_df['properties'].apply(lambda x: x['STATE'])
counties_fips['FIPS_state'] = counties_fips['FIPS_state'].astype(str)
counties_fips['FIPS_cty'] = counties_df['properties'].apply(lambda x: x['STATE'] + x['COUNTY'])
counties_fips['FIPS_cty'] = counties_fips['FIPS_cty'].astype(str)

merged = soy_df.merge(counties_fips, on=['FIPS_state', 'County'], how='left')

cty_dic = {}
for cty in merged[merged['FIPS_cty'].isna()]['County'].unique():
    if 'st ' in cty:
        cty_dic[cty] = cty.replace('st ', 'st. ')
    elif 'ste ' in cty:
        cty_dic[cty] = cty.replace('ste ', 'ste. ')
    elif 'o brien' in cty:
        cty_dic[cty] = cty.replace('o brien', "o'brien")
    else:
        cty_dic[cty] = cty.replace(' ', '')
        
soy_df['County'] = soy_df['County'].apply(lambda x: cty_dic[x] if x in cty_dic else x)
soy_df = soy_df.merge(counties_fips, on=['FIPS_state', 'County'], how='left')

assert soy_df.isna().sum().sum() == 1
print("County not find:")
print(soy_df[soy_df['FIPS_cty'].isna()])

soy_df = soy_df.dropna(subset=['FIPS_cty'])
soy_df.to_csv("data/Soybeans_Loc_ID_FIPS.csv", index=False)


