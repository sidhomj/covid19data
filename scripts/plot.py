import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns

sigma = 3
state_sel = 'NY'
df = pd.read_csv('../covidactnow/states.timeseries.csv')
df_sel = df[df['state']==state_sel]
df_sel['metrics.vaccinationsCompletedRatio'] = df_sel['metrics.vaccinationsCompletedRatio'].interpolate()
df_sel['metrics.vaccinationsCompletedRatio'].fillna(0,inplace=True)
df_sel['vax_7day'] = df_sel['metrics.vaccinationsCompletedRatio'].rolling(window=7).mean()
df_sel['actuals.newCases'] = df_sel['actuals.newCases'].rolling(window=7).mean()
df_sel['actuals.newDeaths'] = df_sel['actuals.newDeaths'].rolling(window=7).mean()
df_sel.dropna(subset=['actuals.newCases','actuals.newDeaths'],inplace=True)
df_sel['deaths/cases'] = df_sel['actuals.newDeaths']/df_sel['actuals.newCases']
df_sel.dropna(subset=['deaths/cases'],inplace=True)

fig,ax = plt.subplots(4,1,figsize=(4,8))
ax[0].plot(df_sel['date'],gaussian_filter1d(df_sel['actuals.newCases'],sigma))
ax[0].bar(df_sel['date'],df_sel['actuals.newCases'])
ax[0].set_xticks([])
ax[0].set_title('Cases')
ax[1].plot(df_sel['date'],gaussian_filter1d(df_sel['actuals.newDeaths'],sigma))
ax[1].bar(df_sel['date'],df_sel['actuals.newDeaths'])
ax[1].set_xticks([])
ax[1].set_title('Deaths')
ax[2].plot(df_sel['date'],gaussian_filter1d(df_sel['deaths/cases'],sigma))
ax[2].set_xticks([])
ax[2].set_title('Deaths/Cases')
ax[3].plot(df_sel['date'],gaussian_filter1d(df_sel['metrics.vaccinationsCompletedRatio'],sigma))
ax[3].set_xticks([])
ax[3].set_title('Vax_Ratio')
fig.suptitle(state_sel)
plt.tight_layout()
#
# df['deaths/cases'] = df['actuals.newDeaths']/df['actuals.newCases']
# df_agg = df.groupby(['state']).agg({'deaths/cases':'mean','metrics.vaccinationsCompletedRatio':'last'})
# sns.scatterplot(data=df_agg,x='metrics.vaccinationsCompletedRatio',y='deaths/cases')