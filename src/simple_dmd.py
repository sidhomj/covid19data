#%%
import pydmd
from pydmd import DMD
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
def gen_data(focus_state=None):
    if focus_state is None:
        pass
    df = pd.read_csv('../data/states.timeseries.csv')
    state_data = df[df['state'] == focus_state]

    return state_data


nyc_data = gen_data(focus_state='NY')

data = nyc_data['actuals.newCases'].to_numpy()
data2 = nyc_data['actuals.newDeaths'].to_numpy()
data3 = nyc_data['metrics.vaccinationsCompletedRatio'].to_numpy()

full_stack = np.array([data, data2, data3])

fig, ax= plt.subplots()

ax.plot(full_stack[0,:])
ax2 = ax.twinx()
ax2.plot(full_stack[1,:],color='red')
plt.show()

full_stack[np.isnan(full_stack)] = 0

#%%
#simple DMD run
model = DMD(svd_rank=3)
model.fit(full_stack)


for mode in model.modes.T:
    plt.plot(mode.real)
    plt.title('Modes')
plt.show()

t = np.linspace(0,1,full_stack.shape[-1])
for dynamic in model.dynamics:
    plt.plot(t, dynamic.real)
    plt.title('Dynamics')
plt.show()