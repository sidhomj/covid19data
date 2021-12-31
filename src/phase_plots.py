#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gen_data(focus_state=None):

    cases = pd.read_csv(
        "../data/data_table_for_daily_case_trends__the_united_states.csv",
        sep=",",
    )

    deaths = pd.read_csv(
        "../data/data_table_for_daily_death_trends__the_united_states.csv",
        sep=",",
    )

    return (cases, deaths)


def add_arrow(line, position=None, direction="right", size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == "right":
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate(
        "",
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size,
    )


#%%
us_data = gen_data(focus_state=None)

deaths = np.array(us_data[0]["New Cases"])[::-1]
cases = np.array(us_data[1]["New Deaths"])[::-1]

#%%
import scipy.signal as sig
import scipy.stats as stats

lpf_sos = sig.butter(10, 0.05, btype="low", output="sos")

lp_cases = sig.sosfilt(lpf_sos, cases)
lp_deaths = sig.sosfilt(lpf_sos, deaths)

# lp_cases = stats.zscore(lp_cases)
# lp_deaths = stats.zscore(lp_deaths)

#%%
fig, ax = plt.subplots(figsize=(15, 10))
lines = plt.plot(lp_cases, lp_deaths)
arrow_skip = 15
color_map = np.arange(lp_cases.shape[0])[::arrow_skip]
# add_arrow(lines)
# plt.scatter(
#    lp_cases[::arrow_skip],
#    lp_deaths[::arrow_skip],
#    marker=">",
#    c=color_map,
#    cmap="cool",
#    s=100,
# )
diff_lp_cases = np.pad(np.diff(lp_cases), (0, 1), "constant")
diff_lp_deaths = np.pad(np.diff(lp_deaths), (0, 1), "constant")

normed_arrow_dir =

for plot_arrow in range(0, len(lp_cases), arrow_skip):
    plt.arrow(
        lp_cases[plot_arrow],
        lp_deaths[plot_arrow],
        diff_lp_cases[plot_arrow],
        diff_lp_deaths[plot_arrow],
        head_width=20,
        head_length=20,
        width=0,
        shape="full",
    )

# plt.colorbar()
plt.xlabel("Cases per day")
plt.ylabel("Deaths per day")
plt.title("Phase plot of US-COVID")
ax.set_aspect("auto")
plt.show()


#%%
plt.figure()
plt.plot(lp_cases)
plt.show()
