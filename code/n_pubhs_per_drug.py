# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import csv
from sklearn import metrics

# %% Some functions and predetermined parameters for plots
def exp(x, a, b):
    return a * np.exp(b * x)

def fit(dru, dat):
    x_axis = np.arange(len(dat[dru]))
    # print(np.array(dat[dru]))
    out, out_cov = curve_fit(exp, x_axis, np.array(dat[dru]), p0=(4, 0.1))
    # print(out[0], out[1])
    pred_exp_d = exp(x_axis, out[0], out[1])
    # print(out)
    return [pred_exp_d, out[0], out[1]]

def r2_score(dru, dat):
    pred_exp_d, out, out_cov = fit(dru, dat)
    # print(pred_exp_d, out, out_cov)
    r2_value = metrics.r2_score(dat[dru], pred_exp_d)
    return round(r2_value, 2)

mc = 'black'
plt.rcParams.update({'font.size': 41, 
                        'axes.labelcolor' : "{}".format(mc), 
                        'xtick.color': "{}".format(mc),
                        'ytick.color': "{}".format(mc),
                        'font.family': 'sans-serif'})

# %% COLOURS
turkish_sea = '#255498'
blue_moon = '#3388A6'
pink_dnm = '#EC96BA'
ultra_violet = '#5F4B8B'
v_yellow = '#FFDC01'
pale_gold = '#B78B5F'
sunset_gold = '#FAC668'
copper = '#AD6B4E'
dark_orange = '#ff8c00'
blue_dnm = '#89F0FF'
green_dnm = '#5ABE8A'
red_dnm = '#E96C6C'

coloh = {'Alcohol': ['#89F0FF', 1],
        'Cannabis': ['#5ABE8A', 1],
        'MDMA': ['#EC96BA', 1],
        'Benzodiazepines': ['#3388A6' , 1],
        'Ketamine': ['#000000', 1],
        'LSD': ['#ff8c00', 1],
        'Heroin': ['#5F4B8B', 1],
        'Psilocybin': ['#ff8c00', 0.5],
        'Khat': ['#3388A6', 1/3],
        'Cocaine': ['#E96C6C', 2/3],
        'Amphetamines': ['#E96C6C', 1/3],
        'Methamphetamines': ['#E96C6C', 1],
        'Methadone': ['#5F4B8B', 2/3],
        'Codeine': ['#5F4B8B', 1/3],
        'GHB': ['#3388A6', 2/3] }

# %% Import data
file = './../data/ppy_drugs.csv'
raw = pd.read_csv(file)

# %%
drugs = raw.iloc[:,1:17]
years = raw.iloc[:, 0]

# Drugs by schedules (US, 2019)
sched_I = ['MDMA', 'LSD', 'Heroin']
sched_I_low = ['Psilocybin', 'Khat']
sched_I_can = ['Cannabis']
sched_II = ['Cocaine', 'Amphetamines', 'Methamphetamines', 'Methadone']
sched_III = ['Ketamine']
sched_III_ghb = ['GHB']
sched_IV = ['Benzodiazepines']
sched_V = ['Codeine']
legal = ['Alcohol']
wos = ['WoS']

# Drugs by drug category
depressants = ['Benzodiazepines', 'GHB', 'Khat']
depressant_legal = ['Alcohol']
cannabinoids = ['Cannabis']
empathogens = ['MDMA']
dissociatives = ['Ketamine']
psychedelics = ['LSD', 'Psilocybin']
opioids = ['Heroin', 'Methadone', 'Codeine']
stimulants = ['Methamphetamines', 'Cocaine', 'Amphetamines']


# %% FIGURE 2 individual drugs
sch_cols = {'sched_I': blue_moon, 'sched_II': green_dnm, 'sched_III': dark_orange, 
            'sched_IV': v_yellow, 'sched_V': pink_dnm, 'legal': ultra_violet, 'wos': 'k'}

def fig_2_indv(data, sch_color, max_y, steps, dict_params):
    fig, ax = plt.subplots(figsize = (18, 15))

    lwDf = 6
    lwD = 15
    widthtick = 15
    lenD = 20
    s_bub = 150

    pred_t, a, b = fit(data, raw)
    r_square = r2_score(data, raw)
    print(data, r_square, a, b)

    ax.plot(years, raw[data], color = sch_cols[sch_color], label=data, lw = lwD)
    ax.plot(years, pred_t, color='black', alpha=0.8, linestyle='--', lw = lwDf)

    dict_params[data] = {'r_square': r_square, 'a_init': a, 'g_rate': b}

    ax.set_ylabel('Number of publications', labelpad=20)
    ax.set_xlabel('Time (years)', labelpad=20)
    print(np.arange(0, (max_y + 0.1), steps))

    ax.set_yticks(np.arange(0, (max_y + 0.1), steps))
    ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])

    ax.set_xlim([1960, 2018])
    ax.set_ylim([0, max_y])

    ax.spines['left'].set_linewidth(lwD)
    ax.spines['bottom'].set_linewidth(lwD)
    ax.spines['right'].set_linewidth(lwD)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
    ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)

    # leg = ax.legend(loc='upper left')
    # leg.get_frame().set_linewidth(0.0)
    # leg.get_frame().set_facecolor('none')

    ax.set_title(data)

    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.tight_layout()

    plt.savefig(f'./../figures/dev_write_up/sched_2_{data.lower()}.png'.format(), transparent = True, bbox_inches='tight')

    return dict_params

params_sche = {}

# %% Alcohol
params_sche = fig_2_indv('Alcohol', 'legal', 40000, 5000, params_sche)

# %% wos
params_sche = fig_2_indv('WoS', 'wos', 3500000, 500000, params_sche)

# %% 'Psilocybin'
params_sche = fig_2_indv('Psilocybin', 'sched_I', 150, 25, params_sche)

# %% 'Khat'
params_sche = fig_2_indv('Khat', 'sched_I', 125, 25, params_sche)

# %% 'Cannabis'
params_sche = fig_2_indv('Cannabis', 'sched_I', 5000, 1000, params_sche)

# %% 'MDMA'
params_sche = fig_2_indv('MDMA', 'sched_I', 500, 100, params_sche)

# %% 'LSD'
params_sche = fig_2_indv('LSD', 'sched_I', 500, 100, params_sche)

# %% 'Heroin'
params_sche = fig_2_indv('Heroin', 'sched_I', 1200, 200, params_sche)

# %% 'Cocaine'
params_sche = fig_2_indv('Cocaine', 'sched_II', 2500, 500, params_sche)

# %% 'Amphetamines'
params_sche = fig_2_indv('Amphetamines', 'sched_II', 1200, 200, params_sche)

# %% 'Methamphetamines'
params_sche = fig_2_indv('Methamphetamines', 'sched_II', 1000, 200, params_sche)

# %% 'Methadone'
params_sche = fig_2_indv('Methadone', 'sched_II', 1000, 200, params_sche)

# %% 'Ketamine'
params_sche = fig_2_indv('Ketamine', 'sched_III', 1500, 250, params_sche)

# %% 'GHB'
params_sche = fig_2_indv('GHB', 'sched_III', 150, 25, params_sche)

# %% 'Benzodiazepines'
params_sche = fig_2_indv('Benzodiazepines', 'sched_IV', 1500, 250, params_sche)

# %% 'Codeine'
params_sche = fig_2_indv('Codeine', 'sched_V', 300, 50, params_sche)

# %% LEGEND SCHEDULES
fig, ax = plt.subplots(figsize = (18, 15))

ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)

sI = mlines.Line2D([], [], color='#3388A6', label = 'Schedule I')
sII = mlines.Line2D([], [], color='#5ABE8A', label = 'Schedule II')
sIII = mlines.Line2D([], [], color='#ff8c00', label = 'Schedule III')
sIV = mlines.Line2D([], [], color='#FFDC01', label = 'Schedule IV')
sV = mlines.Line2D([], [], color='#e96c6c', label = 'Schedule V')
leg = mlines.Line2D([], [], color='#5F4B8B', label = 'Legal')

leg = ax.legend(handles=[sI, sII, sIII, sIV, sV, leg], bbox_to_anchor=(1.05, 0.75), ncol=2)

leg.get_frame().set_facecolor('none')
leg.get_frame().set_linewidth(0.0)

for lo in leg.legendHandles:
    lo.set_linewidth(20)

ax.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

plt.savefig(f'./../figures/dev_write_up/solo_legend.png', transparent = True, bbox_inches='tight')

# %% Figure 2 fitting by schedule
def fig_2_sche(data, color, lineswag, name_file, max_y, steps, dict_params):
    fig, ax = plt.subplots(figsize = (18, 15))

    lwDf = 6
    lwD = 15
    widthtick = 15
    lenD = 20
    s_bub = 150

    for i, z in enumerate(data):

        pred_t, a, b = fit(z, raw)
        r_square = r2_score(z, raw)
        print(z, r_square, a, b)

        ax.plot(years, raw[z], color = color, label=z, linestyle=lineswag[i], lw = lwD)
        ax.plot(years, pred_t, color='black', alpha=0.6, lw = lwDf)

        dict_params[z] = {'r_square': r_square, 'a_init': a, 'g_rate': b}

    ax.set_ylabel('Number of publications', labelpad=20)
    ax.set_xlabel('Time (years)', labelpad=20)
    print(np.arange(0, (max_y + 0.1), steps))

    ax.set_yticks(np.arange(0, (max_y + 0.1), steps))
    ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])

    ax.set_xlim([1960, 2018])
    ax.set_ylim([0, max_y])

    ax.spines['left'].set_linewidth(lwD)
    ax.spines['bottom'].set_linewidth(lwD)
    ax.spines['right'].set_linewidth(lwD)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
    ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)

    leg = ax.legend(loc='upper left')
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor('none')

    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.tight_layout()

    plt.savefig(f'./../figures/dev_write_up/{name_file}.png'.format(), transparent = True, bbox_inches='tight')

    return dict_params

# params_sche = {}

# %% Schedule I

ls_sched_I = ['-', '-', '-']

# fig_2_sche(sched_I, blue_moon,ls_sched_I, 'sche_I', 1200, 200, params_sche)

# %% plots fitted

# %% Figure 2 fitting by drug cateogry
# Function to plot drugs by drug category
def fig_2_drug_cat(data, color, title, name_file, max_y, steps, dict_params):
    fig, ax = plt.subplots(figsize = (18, 15))

    lwD = 10
    widthtick = 15
    lenD = 20
    s_bub = 150

    for i, z in enumerate(data):
        # print((1 - (1/len(data))* i), z)

        pred_t, a, b = fit(z, raw)
        r_square = r2_score(z, raw)
        print(z, r_square, a, b)

        ax.plot(years, raw[z], color = color, label = z, alpha = (1 - (1/len(data))* i), lw = lwD*1.2)
        ax.plot(years, pred_t, '--', color='black', alpha=(1 - (1/len(data))* i), lw = lwD)

        dict_params[z] = {'r_square': r_square, 'a_init': a, 'g_rate': b}

    # ax.text(1970, mid_y, 'R-square = {}'.format(r_square))
    # start, end = ax.get_ylim()
    # mid_y = end/2

    # ax.set_title(title)
    ax.set_ylabel('Number of publications', labelpad=20)
    ax.set_xlabel('Time (years)', labelpad=20)
    print(np.arange(0, (max_y + 0.1), steps))

    ax.set_yticks(np.arange(0, (max_y + 0.1), steps))
    ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])

    ax.set_xlim([1960, 2018])
    ax.set_ylim([0, max_y])

    ax.spines['left'].set_linewidth(lwD)
    ax.spines['bottom'].set_linewidth(lwD)
    ax.spines['right'].set_linewidth(lwD)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
    ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)

    leg = ax.legend(loc='upper left')
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor('none')
    # plt.legend(frameon=False)

    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.tight_layout()

    plt.savefig('./../figures/dev_write_up/{}.png'.format(name_file), transparent = True, bbox_inches='tight')

    return dict_params

# %% Depressants
dict_params = {}

dict_params = fig_2_drug_cat(depressants, blue_moon, 'Number of publications about depressant drugs.\n Excluding alcohol', 'depressants_excl_alc', 1600, 200, dict_params)

# %% WoS
dict_params = fig_2_drug_cat(wos, 'black', 'Number of publications about psychedelic drugs.', 'wos_pubs', 4000000, 1000000, dict_params)


# %% Dissociatives
ma_black = '#000000'
dict_params = fig_2_drug_cat(dissociatives, ma_black, 'Number of publications about dissociatives', 'disso', 1600, 200, dict_params)

# %% Psychedelics
dict_params = fig_2_drug_cat(psychedelics, dark_orange, 'Number of publications about psychedelic drugs.', 'psychedelics', 600, 100, dict_params)

# %% empathogens
dict_params = fig_2_drug_cat(empathogens, pink_dnm, 'Number of publications about empathogens', 'mdma', 600, 100, dict_params)


# %% opioids
dict_params = fig_2_drug_cat(opioids, ultra_violet, 'Number of publications about opioids', 'opioids',1200, 200, dict_params)

# %% Alcohol
dict_params = fig_2_drug_cat(depressant_legal, blue_dnm, 'Number of publications about alcohol', 'alcohol',50000, 10000, dict_params)

# %% Cannabis
dict_params = fig_2_drug_cat(cannabinoids, green_dnm, 'Number of publications about cannabis', 'cannabis', 5000, 1000, dict_params)


# %% stimulants

dict_params = fig_2_drug_cat(stimulants, red_dnm, 'Number of publications about stimulants', 'stimulants',3000, 1000, dict_params)


#### Figure 3 Rs, Grs and table
# %% Save R^2, a_init, g_rate as csv for table
file_table = 'table_params_exp'
table_exp_params = open('./../figures/dev_write_up/{}.csv'.format(file_table), 'a')

data_writer = csv.writer(table_exp_params)
header_pure = dict_params['Alcohol'].keys()
header = list(header_pure)
header.insert(0, 'Drugs')

data_writer.writerow(header)

drugs_dict = list(dict_params.keys())

for i in np.arange(len(drugs_dict)):
    temp_drug = drugs_dict[i]
    rowToWrite = [temp_drug]
    for j in header_pure:
        datapoint = dict_params[temp_drug][j]
        rowToWrite.append(datapoint)

    data_writer.writerow(rowToWrite)

table_exp_params.close()



# %%

drugs_r = []
drugs_r2 = []
drugs_init = []
rs_ds = {}
r_squares = np.zeros(len(dict_params.keys()))
ases = np.zeros(len(dict_params.keys()))
bses = np.zeros(len(dict_params.keys()))

t = np.arange(0, 16, 1)

for i, k in enumerate(dict_params.keys()):
    rs_ds[k] = [dict_params[k]['r_square'], dict_params[k]['g_rate'], dict_params[k]['a_init']]

# %%
order_sched = ['Heroin', 'Cannabis', 'Khat', 'Psilocybin', 'LSD',  'MDMA',

                'Methamphetamines', 'Methadone', 'Cocaine',  'Amphetamines',

                'Ketamine', 'GHB',
                'Benzodiazepines',
                'Codeine',
                'Alcohol',
                'WoS']

r_squares_sched = []
for ore in order_sched:
    print(ore)
    r_squares_sched.append(rs_ds[ore][0])

# %%

fig, ax = plt.subplots(figsize = (18, 15))

lwD = 10
widthtick = 15
lenD = 20
s_bub = 150

colobars_sched = [ blue_moon, blue_moon, blue_moon, blue_moon, blue_moon, blue_moon,
            green_dnm, green_dnm, green_dnm, green_dnm, dark_orange, dark_orange,
            v_yellow, pink_dnm, ultra_violet, 'k']

ys = np.linspace(0, 56, len(r_squares_sched))
ax.barh(ys, width = r_squares_sched[::-1], color=colobars_sched[::-1], height=2)

# ax.set_axis_off()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.axes.xaxis.set_visible(False)
ax.set_yticks(ys)

labels = [item.get_text() for item in ax.get_yticklabels()]

for i, l in enumerate(labels):
    labels[i] = order_sched[::-1][i]

ax.set_yticklabels(labels)

# cb.ax.get_yaxis().labelpad = 45
ax.set_xlabel('$R^2$', labelpad=20)
ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)
ax.tick_params(axis='y', which='major', pad=30)
ax.tick_params(axis='x', which='major', pad=10)
ax.spines['left'].set_linewidth(lwD)
ax.spines['bottom'].set_linewidth(lwD)

ax.set_xlim([0, 1])
# ax.set_ylim([1, 58])

# # name_file = 'heatmap_r_values'
plt.tight_layout(pad=1.6)
name_file = 'hbars_r_values_ORD_sched'
plt.savefig('./../figures/dev_write_up/{}.png'.format(name_file), transparent = True, bbox_inches='tight')

# %%
order_sched = ['Psilocybin', 'Cannabis', 'Khat', 'LSD', 'Heroin', 'MDMA',

                'Methamphetamines', 'Methadone', 'Cocaine',  'Amphetamines',

                'Ketamine', 'GHB',
                'Benzodiazepines',
                'Codeine',
                'Alcohol',
                'WoS']

bses_squares_sched = []
for ore in order_sched:
    bses_squares_sched.append(rs_ds[ore][1])

# %%

fig, ax = plt.subplots(figsize = (18, 15))

ys = np.linspace(0, 56, len(bses_squares_sched))
ax.barh(ys, width = bses_squares_sched[::-1], color=colobars_sched[::-1], height=2)

ax.yaxis.set_ticks(ys)
ax.xaxis.set_ticks(np.arange(0, 0.201, 0.05))

labels = [item.get_text() for item in ax.get_yticklabels()]

for i, l in enumerate(labels):
    labels[i] = order_sched[::-1][i]

ax.set_yticklabels(labels)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)

ax.spines['left'].set_linewidth(lwD)
ax.spines['bottom'].set_linewidth(lwD)

ax.tick_params(axis='y', which='major', pad=30)
ax.tick_params(axis='x', which='major', pad=10)

ax.set_xlabel('Growth rate (r)', labelpad=20)

plt.tight_layout(pad=1.6)
name_file = 'exp_hbars_sched_ORD'
plt.savefig('./../figures/dev_write_up/{}.png'.format(name_file), transparent = True, bbox_inches='tight')

# %% R-square values coloured by schedule ORDERED MAX TO LOW
sort_orders_r2 = sorted(rs_ds.items(), key=lambda x: x[1][0], reverse=True)
sort_orders_r = sorted(rs_ds.items(), key=lambda x: x[1][1], reverse=True)
sort_orders_init = sorted(rs_ds.items(), key=lambda x: x[1][2], reverse=True)

for i, k in enumerate(sort_orders_r2):
    r_squares[i] = k[1][0]
    drugs_r2.append(k[0])

for i, k in enumerate(sort_orders_r):
    bses[i] = k[1][1]
    drugs_r.append(k[0])

for i, k in enumerate(sort_orders_init):
    ases[i] = k[1][2]
    drugs_init.append(k[0])

# %%
fig, ax = plt.subplots(figsize = (18, 15))

lwD = 10
widthtick = 15
lenD = 20
s_bub = 150

colobars = [v_yellow, dark_orange, green_dnm, blue_moon, blue_moon, green_dnm, blue_moon, pink_dnm, blue_moon,
            green_dnm, 'k', green_dnm, blue_moon, blue_moon, dark_orange, ultra_violet]

ys = np.linspace(0, 56, len(r_squares))
ax.barh(ys, width = r_squares[::-1], color=colobars, height=2)

# ax.set_axis_off()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.axes.xaxis.set_visible(False)
ax.set_yticks(ys)

labels = [item.get_text() for item in ax.get_yticklabels()]

for i, l in enumerate(labels):
    labels[i] = drugs_r2[::-1][i]

ax.set_yticklabels(labels)

# cb.ax.get_yaxis().labelpad = 45
ax.set_xlabel('$R^2$', labelpad=20)
ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)
ax.tick_params(axis='y', which='major', pad=30)
ax.tick_params(axis='x', which='major', pad=10)
ax.spines['left'].set_linewidth(lwD)
ax.spines['bottom'].set_linewidth(lwD)

ax.set_xlim([0, 1])
# ax.set_ylim([1, 58])

# # name_file = 'heatmap_r_values'
plt.tight_layout(pad=1.6)
name_file = 'hbars_r_values_sched'
plt.savefig('./../figures/dev_write_up/{}.png'.format(name_file), transparent = True, bbox_inches='tight')

# %% PLOT R squares
zs = np.reshape(r_squares, (16, 1))

# %% R-square values ALL BLACK
fig, ax = plt.subplots(figsize = (18, 15))

lwD = 10
widthtick = 15
lenD = 20
s_bub = 150

ys = np.linspace(0, 56, len(r_squares))
ax.barh(ys, width = r_squares[::-1], color='k', height=2)

# ax.set_axis_off()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.axes.xaxis.set_visible(False)
ax.set_yticks(ys)

labels = [item.get_text() for item in ax.get_yticklabels()]

for i, l in enumerate(labels):
    labels[i] = drugs_r2[::-1][i] 

ax.set_yticklabels(labels)

# cb.ax.get_yaxis().labelpad = 45
ax.set_xlabel('$R^2$', labelpad=20)
ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)
ax.tick_params(axis='y', which='major', pad=30)
ax.tick_params(axis='x', which='major', pad=10)
ax.spines['left'].set_linewidth(lwD)
ax.spines['bottom'].set_linewidth(lwD)

ax.set_xlim([0, 1])
ax.set_ylim([1, 58])

# # name_file = 'heatmap_r_values'
plt.tight_layout(pad=1.6)
name_file = 'hbars_r_values'
plt.savefig('./../figures/dev_write_up/{}.png'.format(name_file), transparent = True, bbox_inches='tight')

# %% GROWTH RATE by SCHEDULE

sched_I = ['MDMA', 'LSD', 'Heroin']
sched_I_low = ['Psilocybin', 'Khat']
sched_I_can = ['Cannabis'] #blue_moon
sched_II = ['Cocaine', 'Amphetamines', 'Methamphetamines', 'Methadone'] #green_dnm
sched_III = ['Ketamine']
sched_III_ghb = ['GHB'] #dark_orange
sched_IV = ['Benzodiazepines'] #v_yellow
sched_V = ['Codeine'] #pink_dnm
legal = ['Alcohol'] #ultra_violet
wos = ['WoS'] #'k'


colobars_g = [blue_moon, blue_moon, blue_moon, green_dnm, dark_orange, ultra_violet, blue_moon, green_dnm,
                blue_moon, blue_moon, dark_orange, green_dnm, pink_dnm, green_dnm, 'k', v_yellow]

fig, ax = plt.subplots(figsize = (18, 15))

ys = np.linspace(0, 56, len(bses))
ax.barh(ys, width = bses[::-1], color=colobars_g[::-1], height=2)

ax.yaxis.set_ticks(ys)
ax.xaxis.set_ticks(np.arange(0, 0.201, 0.05))

labels = [item.get_text() for item in ax.get_yticklabels()]

for i, l in enumerate(labels):
    labels[i] = drugs_r[::-1][i]
    # print(names_barh[i]) 

ax.set_yticklabels(labels)
# ax.set_ylim([1, 58])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)

ax.spines['left'].set_linewidth(lwD)
ax.spines['bottom'].set_linewidth(lwD)

ax.tick_params(axis='y', which='major', pad=30)
ax.tick_params(axis='x', which='major', pad=10)

ax.set_xlabel('Growth rate (r)', labelpad=20)

plt.tight_layout(pad=1.6)
name_file = 'exp_hbars_sched'
plt.savefig('./../figures/dev_write_up/{}.png'.format(name_file), transparent = True, bbox_inches='tight')


# %%

fig, ax = plt.subplots(figsize = (18, 15))

ys = np.linspace(0, 56, len(bses))
ax.barh(ys, width = bses[::-1], color='k', height=2)

ax.yaxis.set_ticks(ys)
ax.xaxis.set_ticks(np.arange(0, 0.201, 0.05))

labels = [item.get_text() for item in ax.get_yticklabels()]

for i, l in enumerate(labels):
    labels[i] = drugs_r[::-1][i]
    # print(names_barh[i]) 

ax.set_yticklabels(labels)
ax.set_ylim([1, 58])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)

ax.spines['left'].set_linewidth(lwD)
ax.spines['bottom'].set_linewidth(lwD)

ax.tick_params(axis='y', which='major', pad=30)
ax.tick_params(axis='x', which='major', pad=10)

ax.set_xlabel('Growth rate (r)', labelpad=20)

plt.tight_layout(pad=1.6)
name_file = 'exp_hbars'
plt.savefig('./../figures/dev_write_up/{}.png'.format(name_file), transparent = True, bbox_inches='tight')


# %% Initial value
for i in ases:
    print(i)

for i in drugs_init:
    print(i)

# %% Figure 3 RGR
# Calculation of relative growth rate (RGR) i.e. Increase in number of publications per unit of time
# RGR = [log e W 2 - log e W 1 ]/(T2 - T1) 
# log e W1: log of initial number of articles
# log e W2: log of final number of articles after a specific period of interval
# T2-T1: the unit difference between the initial time and the final time.

import math

def rgr(drs, ys, data):
    rgr_drug = np.zeros((len(ys) - 1))

    for i in np.arange(len(ys)):
        try:
            present = data.loc[data['Year'] == ys[i]]
            future = data.loc[data['Year'] == ys[i + 1]]
            print(drs)
            log_present = 0
            log_future = 0

            if present[drs].values[0] != 0:
                log_present = math.log(present[drs].values[0])
            
            if future[drs].values[0] != 0:
                log_future = math.log(future[drs].values[0])

            # print(log_future - log_present)
            # print(ys[i+1] - ys[i])
            rgr_step = (log_future - log_present) / float((ys[i+1] - ys[i]))
            # print(rgr_step)
            rgr_drug[i] = rgr_step
        except Exception as e: 
            print(e)

    return rgr_drug
# %%
years = [1960, 1970, 1980, 1990, 2000, 2010, 2018]

rgr(['LSD'], years, raw)

# raw['Ketamine']

# %%

rgr_all = {}

for i in drugs:
    # print(i)
    d_rgr = rgr([i], years, raw)
    rgr_all[i] = d_rgr

# RGR by schedule
sched_Irgr = ['MDMA', 'LSD', 'Heroin', 'Psilocybin', 'Khat', 'Cannabis'] #blue_moon
sched_IIrgr = ['Cocaine', 'Amphetamines', 'Methamphetamines', 'Methadone'] #green_dnm
sched_IIIrgr = ['Ketamine', 'GHB'] #dark_orange
sched_IVrgr = ['Benzodiazepines'] #v_yellow
sched_Vrgr = ['Codeine'] #pink_dnm
legalrgr = ['Alcohol'] #ultra_violet
wosrgr = ['WoS'] #'k'

# %% Separate RGRs

def fig_3_sched(data, color, name_file):
    fig, ax = plt.subplots(figsize = (18, 15))

    lwD = 10
    widthtick = 15
    lenD = 20
    s_bub = 30

    ls = ['-', ':', '-.']
    ma = [".", "^", "s"]
    phalpha = [1, 0.5, 0.1]

    for i, z in enumerate(data):
        print(1 - (1/len(data))* i*1.4)
        ax.plot(years[1:], rgr_all[z], color = color[i], label = z, alpha = 1,lw = lwD) #linestyle=ls[i])

    ax.hlines(0, 1970, 2018, lw = 4, linestyle='--', color ='k')
    # ax.set_title(title)
    ax.set_ylabel('Relative Growth Rate', labelpad=20)
    ax.set_xlabel('Time (decades)', labelpad=20)

    ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])

    ax.set_xlim([1970, 2018])
    ax.set_ylim([-0.4, 0.4])
    ax.set_yticks(np.arange(-0.4, 0.5, 0.2))

    ax.spines['left'].set_linewidth(lwD)
    ax.spines['bottom'].set_linewidth(lwD)
    ax.spines['right'].set_linewidth(lwD)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
    ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)

    leg = ax.legend(loc='upper right', bbox_to_anchor=(1, 0.4))
    leg.get_frame().set_facecolor('none')
    leg.get_frame().set_linewidth(0.0)

    # plt.tight_layout()
    plt.savefig('./../figures/dev_write_up/{}.png'.format(name_file), transparent = True, bbox_inches='tight')

# %% Schedule I
fig_3_sched(sched_Irgr[:3], ['#235f74', '#3388a6', '#99c3d2'], 'sched_I_lsd_md_hero')

# %% Schedule I
fig_3_sched(sched_Irgr[3:], ['#235f74', '#3388a6', '#99c3d2'], 'sched_I_psilo_khat_cann')

# %% Schedule II
fig_3_sched(sched_IIrgr, ['#1b3929', '#3e8560', '#5abe8a', '#9cd8b8'], 'sched_II')

# %% Schedule III
fig_3_sched(sched_IIIrgr, ['#cc7000', '#ffae4c'], 'sched_III')

# %% Schedule IV
fig_3_sched(sched_IVrgr, [v_yellow], 'sched_IV')

# %% Schedule V
fig_3_sched(sched_Vrgr, [pink_dnm], 'sched_V')

# %% Legal
fig_3_sched(legalrgr, [ultra_violet], 'legal')

# %% WoS
fig_3_sched(wosrgr, ['k'], 'wos_rgr')

# %%


# %%
# One graph per category, one colour and then do opacity gradient
# keep colour code throughout paper
# FIX Y AXIS

def fig_3_rgr(data, color, title, name_file):
    fig, ax = plt.subplots(figsize = (18, 15))
    
    lwD = 15
    widthtick = 15
    lenD = 20
    s_bub = 15

    for i, z in enumerate(data):
        print(1 - (1/len(data))* i)
        ax.plot(years[1:], rgr_all[z], color = color, label = z, alpha = (1 - (1/len(data))* i*1.2), lw = lwD*1.8)
    
    ax.hlines(0, 1970, 2018, lw = 4, linestyle='--', color = 'k')
    # ax.set_title(title)
    ax.set_ylabel('Relative Growth Rate', labelpad=20)
    ax.set_xlabel('Time (years)', labelpad=20)

    ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])

    ax.set_xlim([1970, 2018])
    ax.set_ylim([-0.4, 0.4])
    ax.set_yticks(np.arange(-0.4, 0.5, 0.2))

    ax.spines['left'].set_linewidth(lwD)
    ax.spines['bottom'].set_linewidth(lwD)
    ax.spines['right'].set_linewidth(lwD)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
    ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)

    leg = ax.legend(loc='upper right', bbox_to_anchor=(1, 0.3))
    leg.get_frame().set_facecolor('none')
    leg.get_frame().set_linewidth(0.0)

    # plt.tight_layout()
    plt.savefig('./../figures/dev_write_up/{}.png'.format(name_file), transparent = True, bbox_inches='tight')

# %% Depressants
fig_3_rgr(depressants, blue_moon, 'Relative Growth Rate.\n Excluding alcohol', 'rgr_depressants_excl_alc')

# %% Psychedelics
fig_3_rgr(psychedelics, dark_orange, 'Relative Growth Rate for publications on psychedelics.', 'rgr_psychedelics')

# %% Alcohol
fig_3_rgr(depressant_legal, blue_dnm, 'Relative Growth Rate for publications on Alcohol.', 'rgr_alcohol')

# %% Cannabis
fig_3_rgr(cannabinoids, green_dnm, 'Number of publications about Cannabis', 'rgr_cannabis')

# %% empathogens
fig_3_rgr(empathogens, pink_dnm, 'Number of publications about Empathogens', 'rgr_mdma')

# %% dissociatives
fig_3_rgr(dissociatives, ma_black, 'Number of publications about Dissociatives', 'rgr_disso')

# %% stimulants
fig_3_rgr(stimulants, red_dnm, 'Number of publications about Stimulants', 'rgr_stimulants')

# %% opioids
fig_3_rgr(opioids, ultra_violet, 'Number of publications about Opioids', 'rgr_opioids')

############ Figure 4 Pie chart in time
# %% Set-up
perc_y = pd.DataFrame(np.zeros((np.asarray(raw.shape) - [0,2])), columns=raw.columns[1:-1], index=raw['Year'])

for i, y in enumerate(perc_y.index):
    perc_y.loc[y] = np.round(100*np.asarray(raw.iloc[i, 1:-1]/sum(raw.iloc[i, 1:-1])), 2)

# %% PLOT


scheds_drugs_indvs = {'MDMA': '#0a1b21', 
    'LSD': '#1e5163', 
    'Heroin': '#2d7a95',
    'Psilocybin': '#4793ae',
    'Khat': '#84b7c9', 
    'Cannabis': '#eaf3f6',
    'Cocaine': '#1b3929',
    'Amphetamines': '#3e8560',
    'Methamphetamines': '#7acba1',
    'Methadone': '#bde5d0',
    'Ketamine': '#cc7000', 
    'GHB': '#ffae4c',
    'Benzodiazepines': '#FFDC01',
    'Codeine': '#e96c6c',
    'Alcohol': '#5F4B8B'}

scheds_drugs_cols = {'MDMA': '#3388A6', 
    'LSD': '#3388A6', 
    'Heroin': '#3388A6',
    'Psilocybin': '#3388A6',
    'Khat': '#3388A6', 
    'Cannabis': '#3388A6',
    'Cocaine': '#5ABE8A',
    'Amphetamines': '#5ABE8A',
    'Methamphetamines': '#5ABE8A',
    'Methadone': '#5ABE8A',
    'Ketamine': '#ff8c00',
    'GHB': '#ff8c00',
    'Benzodiazepines': '#FFDC01',
    'Codeine': '#e96c6c',
    'Alcohol': '#5F4B8B'}

cats_drugs_cols = {
    'MDMA': '#e9c7df', 

    'LSD': '#87abda', 
    'Psilocybin': '#87abda',

    'Cannabis': '#857fe3',

    'Khat': '#ee7f96',
    'Cocaine': '#ee7f96',
    'Amphetamines': '#ee7f96',
    'Methamphetamines': '#ee7f96',

    'Methadone': '#b7d433',
    'Heroin': '#b7d433',
    'Codeine': '#b7d433',

    'Ketamine': '#c1addb',

    'GHB': '#ffa512',
    'Benzodiazepines': '#ffa512',
    'Alcohol': '#ffa512'}

# %%

fig, ax = plt.subplots(figsize=(28,15))  
lwD = 10
widthtick = 15
lenD = 20
s_bub = 150

margin_bottom = np.zeros(len(raw.iloc[:, 0]))

for i, drug in enumerate(list(scheds_drugs_cols.keys())[::-1]):
    # print(i)
    perc_dru_t = perc_y[drug]
    # print(perc_dru_t)

    ax.bar(perc_y.index.values, perc_dru_t,
                bottom = margin_bottom, label=drug, color=scheds_drugs_indvs[drug], alpha=coloh[drug][1])
    margin_bottom += perc_dru_t

ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)

ax.set_ylabel('%', labelpad=20)
ax.set_xlabel('Time (years)', labelpad=20)
ax.set_ylim(0, 100)
ax.set_xlim([1959, 2019])

ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])


ax.spines['left'].set_linewidth(lwD)
ax.spines['bottom'].set_linewidth(lwD)
ax.spines['right'].set_linewidth(lwD)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 0.75), ncol=2)
leg.get_frame().set_facecolor('none')
leg.get_frame().set_linewidth(0.0)

name_file = 'time_pie_chart_sched_indv'
plt.savefig('./../figures/dev_write_up/{}.png'.format(name_file), transparent = True, bbox_inches='tight')
name_file = 'figure5_time_pie_chart_sched_indv'
plt.savefig('./../figures/final_paper/{}.pdf'.format(name_file), transparent = True, bbox_inches='tight')

# %%

fig, ax = plt.subplots(figsize=(28,15))  
lwD = 10
widthtick = 15
lenD = 20
s_bub = 150

margin_bottom = np.zeros(len(raw.iloc[:, 0]))

for i, drug in enumerate(list(scheds_drugs_cols.keys())[::-1]):
    # print(i)
    perc_dru_t = perc_y[drug]
    # print(perc_dru_t)

    ax.bar(perc_y.index.values, perc_dru_t,
                bottom = margin_bottom, label=drug, color=scheds_drugs_indvs[drug], alpha=coloh[drug][1])
    margin_bottom += perc_dru_t

ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)

ax.set_ylabel('%', labelpad=20)
ax.set_xlabel('Time (years)', labelpad=20)
ax.set_ylim(50, 100)
ax.set_xlim([1959, 2019])

ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])


ax.spines['left'].set_linewidth(lwD)
ax.spines['bottom'].set_linewidth(lwD)
ax.spines['right'].set_linewidth(lwD)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 0.75), ncol=2)
leg.get_frame().set_facecolor('none')
leg.get_frame().set_linewidth(0.0)

# plt.tight_layout()

name_file = 'time_pie_chart_inset'
plt.savefig('./../figures/dev_write_up/{}.png'.format(name_file), transparent = True, bbox_inches='tight')
plt.savefig('./../figures/final_paper/{}.pdf'.format(name_file), transparent = True, bbox_inches='tight')


# %%

fig, ax = plt.subplots(figsize=(28,15))  
lwD = 10
widthtick = 15
lenD = 20
s_bub = 150

margin_bottom = np.zeros(len(raw.iloc[:, 0]))

for i, drug in enumerate(list(scheds_drugs_cols.keys())[::-1]):
    # print(i)
    perc_dru_t = perc_y[drug]
    print(drug)

    ax.bar(perc_y.index.values, perc_dru_t,
                bottom = margin_bottom, label=drug, color=scheds_drugs_cols[drug], alpha=1)
    margin_bottom += perc_dru_t

ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)

ax.set_ylabel('%', labelpad=20)
ax.set_xlabel('Time (years)', labelpad=20)
ax.set_ylim(0, 100)
ax.set_xlim([1959, 2019])

ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])


ax.spines['left'].set_linewidth(lwD)
ax.spines['bottom'].set_linewidth(lwD)
ax.spines['right'].set_linewidth(lwD)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

sI = mlines.Line2D([], [], color='#3388A6', label = 'Schedule I')
sII = mlines.Line2D([], [], color='#5ABE8A', label = 'Schedule II')
sIII = mlines.Line2D([], [], color='#ff8c00', label = 'Schedule III')
sIV = mlines.Line2D([], [], color='#FFDC01', label = 'Schedule IV')
sV = mlines.Line2D([], [], color='#e96c6c', label = 'Schedule V')
leg = mlines.Line2D([], [], color='#5F4B8B', label = 'Legal')

leg = ax.legend(handles=[sI, sII, sIII, sIV, sV, leg], bbox_to_anchor=(1.05, 0.75), ncol=2)

leg.get_frame().set_facecolor('none')
leg.get_frame().set_linewidth(0.0)

for lo in leg.legendHandles:
    lo.set_linewidth(20)

# plt.tight_layout()

name_file = 'time_pie_chart_by_schedule'
plt.savefig('./../figures/dev_write_up/{}.png'.format(name_file), transparent = True, bbox_inches='tight')
name_file = 'figure5_time_pie_chart_by_schedule'
plt.savefig('./../figures/final_paper/{}.pdf'.format(name_file), transparent = True, bbox_inches='tight')

# %%

fig, ax = plt.subplots(figsize=(28,15))  
lwD = 10
widthtick = 15
lenD = 20
s_bub = 150

margin_bottom = np.zeros(len(raw.iloc[:, 0]))

for i, drug in enumerate(list(cats_drugs_cols.keys())[::-1]):
    # print(i)
    perc_dru_t = perc_y[drug]
    print(drug)

    ax.bar(perc_y.index.values, perc_dru_t,
                bottom = margin_bottom, label=drug, color=cats_drugs_cols[drug], alpha=1)
    margin_bottom += perc_dru_t

ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)

ax.set_ylabel('%', labelpad=20)
ax.set_xlabel('Time (years)', labelpad=20)
ax.set_ylim(0, 100)
ax.set_xlim([1959, 2019])

ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])


ax.spines['left'].set_linewidth(lwD)
ax.spines['bottom'].set_linewidth(lwD)
ax.spines['right'].set_linewidth(lwD)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

dep = mlines.Line2D([], [], color='#ffa512', label = 'Depressants')
dis = mlines.Line2D([], [], color='#c1addb', label = 'Dissociatives')
opi = mlines.Line2D([], [], color='#b7d433', label = 'Opioids')
sti = mlines.Line2D([], [], color='#ee7f96', label = 'Stimulants')
can = mlines.Line2D([], [], color='#857fe3', label = 'Cannabinoids')
psi = mlines.Line2D([], [], color='#87abda', label = 'Psychedelics')
emp = mlines.Line2D([], [], color='#e9c7df', label = 'Empathogens')

leg = ax.legend(handles=[dep, dis, opi, sti, can, psi, emp][::-1], bbox_to_anchor=(1.05, 0.75), ncol=2)

leg.get_frame().set_facecolor('none')
leg.get_frame().set_linewidth(0.0)

for lo in leg.legendHandles:
    lo.set_linewidth(20)

# plt.tight_layout()

name_file = 'time_pie_chart_by_cat'
plt.savefig('./../figures/dev_write_up/{}.png'.format(name_file), transparent = True, bbox_inches='tight')
name_file = 'figure5_time_pie_chart_by_cat'
plt.savefig('./../figures/final_paper/{}.pdf'.format(name_file), transparent = True, bbox_inches='tight')


# %% Figure 5 Per countries
# Top 5 countries
# Main text: alcohol, cannabis, ketamine, LSD
# benzos, heroin, methamphetamines, psilocybin

def sort_top(dat):
    sum = dat.sum(axis = 0)
    sorted_sum = sum[1:].sort_values(ascending=False)

    top_countries = sorted_sum[0:5]

    return top_countries.index.values

def extract_country_dat(drug):
    file = './../data/ppy_{}.csv'.format(drug)
    raw_d = pd.read_csv(file)
    # print(raw_d)

    top_countries = sort_top(raw_d)
    top_countries_dat = raw_d[top_countries]
    years = raw_d['Years']
    
    return top_countries_dat, years


# %% Figure 5 Alcohol

alc_countries, years = extract_country_dat('alcohol')

# %%
def capiletters(words):
    capited_words = []
    for w in words:
        temp_w = w.lower()
        if temp_w == 'usa':
            temp_w = 'USA'
        else:
            temp_w = temp_w.capitalize()

        capited_words.append(temp_w)

    return capited_words

def fig_5_countries(dat, years, title, color, y_max, steps, name_file):
    mc = 'black'
    plt.rcParams.update({'font.size': 40, 
                            'axes.labelcolor' : "{}".format(mc), 
                            'xtick.color': "{}".format(mc),
                            'ytick.color': "{}".format(mc),
                            'font.family': 'sans-serif'})

    fig, ax = plt.subplots(figsize=(20,15))  

    capited_words = capiletters(dat)
    lwD = 10
    widthtick = 15
    lenD = 20
    s_bub = 150

    for i, z in enumerate(dat):
        print(color[i])
        ax.plot(years, dat[z], color=color[i], label = capited_words[i], alpha = 1, lw = lwD*1.4)
        # print((1 - (1/len(dat))* i))

    ax.set_title(title)
    ax.set_ylabel('Number of publications', labelpad=20)
    ax.set_xlabel('Time (years)', labelpad=20)

    ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])

    ax.set_xlim([1960, 2018])
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, (y_max + 0.1), steps))

    ax.spines['left'].set_linewidth(lwD)
    ax.spines['bottom'].set_linewidth(lwD)
    ax.spines['right'].set_linewidth(lwD)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_tick_params(width = lwD, length = lenD, pad=15)
    ax.xaxis.set_tick_params(width = lwD, length = lenD, pad=15)

    leg = ax.legend(loc='upper left')
    leg.get_frame().set_facecolor('none')
    leg.get_frame().set_linewidth(0.0)
    plt.tight_layout()

    plt.savefig('./../figures/dev_write_up/{}.png'.format(name_file), transparent = True, bbox_inches='tight')


# %% Figure 5 Alcohol
legal_countries = ['#000000', '#55437d', '#7e6ea2', '#afa5c5', '#dfdbe7']
sheI_countries = ['#000000', '#286c84', '#70abc0', '#adcfdb', '#eaf3f6']
sheII_countries = ['#000000', '#3e8560', '#6ac495', '#9cd8b8', '#def2e7']
sheIII_countries = ['#000000', '#995400', '#ff8c00', '#ffae4c', '#ffdcb2']
sheIV_countries = ['#000000', '#b29a00', '#ffdc01', '#ffea66', '#fff8cc']
sheV_countries = ['#000000', '#a34b4b', '#e96c6c', '#f1a6a6', '#fae1e1']

# %% Figure 5 Alcohol
fig_5_countries(alc_countries, years, 'Alcohol', legal_countries, 12000, 2000, 'alcohol_countries')


# %% Figure 5 Cannabis
can_countries, years = extract_country_dat('cannabis')
fig_5_countries(can_countries, years, 'Cannabis', sheI_countries, 2500, 500, 'cannabis_countries')

# %% Figure 5  ketamine
ket_countries, years = extract_country_dat('ketamine')
fig_5_countries(ket_countries, years, 'Ketamine', sheIII_countries, 800, 200, 'ketamine_countries')

# %% Figure 5  LSD
lsd_countries, years = extract_country_dat('lsd')
fig_5_countries(lsd_countries, years, 'LSD', sheI_countries, 120, 20, 'lsd_countries')

# %% Figure 5 MDMA
mdma_countries, years = extract_country_dat('mdma')
fig_5_countries(mdma_countries, years, 'MDMA', sheI_countries, 160, 20, 'mdma_countries')

# %% SUPPLEMENTARY Figure 1 benzos
benzos_countries, years = extract_country_dat('benzos')
fig_5_countries(benzos_countries, years, 'Benzodiazepines', sheIV_countries, 800, 200, 'benzos_countries')

# %% SUPPLEMENTARY Figure 1 heroin
heroin_countries, years = extract_country_dat('heroin')
fig_5_countries(heroin_countries, years, 'Heroin', sheI_countries, 800, 200, 'heroin_countries')

# %% SUPPLEMENTARY Figure 1 methamphetamines
metham_countries, years = extract_country_dat('methamphetamine')
fig_5_countries(metham_countries, years, 'Methamphetamine', sheII_countries, 800, 200, 'methamphetamine_countries')

# %% SUPPLEMENTARY Figure 1 psilocybin
psilo_countries, years = extract_country_dat('psilocybin')
fig_5_countries(psilo_countries, years, 'Psilocybin', sheI_countries, 120, 20, 'psilo_countries')

# %% SUPPLEMENTARY Figure 1 Khat
khat_countries, years = extract_country_dat('khat')
fig_5_countries(khat_countries, years, 'Khat', sheI_countries, 60, 10, 'khat_countries')

# %% SUPPLEMENTARY Figure 1 Methadone
methadone_countries, years = extract_country_dat('Methadone')
fig_5_countries(methadone_countries, years, 'Methadone', sheII_countries, 500, 100, 'methadone_countries')

# %% SUPPLEMENTARY Figure 1 GHB
ghb_countries, years = extract_country_dat('ghb')
fig_5_countries(ghb_countries, years, 'GHB', sheIII_countries, 100, 20, 'ghb_countries')

# %% SUPPLEMENTARY Figure 1 Cocaine
cocaine_countries, years = extract_country_dat('cocaine')
fig_5_countries(cocaine_countries, years, 'Cocaine', sheII_countries, 1600, 200, 'cocaine_countries')

# %% SUPPLEMENTARY Figure 1 Amphetamine
amphetamine_countries, years = extract_country_dat('amphetamine')
fig_5_countries(amphetamine_countries, years, 'Amphetamines', sheII_countries, 800, 200, 'amphetamine_countries')

# %% SUPPLEMENTARY Figure 1 Codeine
codeine_countries, years = extract_country_dat('codeine')
fig_5_countries(codeine_countries, years, 'Codeine', sheV_countries, 100, 20, 'codeine_countries')


# %% 

r_cua_all = {}

for i in drugs.columns:
    # print(i)
    r_cua_all[i] = r2_score(i, raw)

# %%
sched_I_rgr = [r_cua_all[i] for i in sched_I]
sched_II_rgr = [r_cua_all[i] for i in sched_II]
sched_III_rgr = [r_cua_all[i] for i in sched_III]
sched_IV_rgr = [r_cua_all[i] for i in sched_IV]
legal_rgr = [r_cua_all[i] for i in legal]

shes = [sched_I_rgr, sched_II_rgr, sched_III_rgr, sched_IV_rgr, legal_rgr]

means_scheds = []
for i in shes:
    means_sched = np.nanmean(i)
    means_scheds.append(means_sched)

# %% plot r squares

for i, s in enumerate(shes):
    plt.scatter(np.repeat(i + 1, len(s)), s)


# fig_fitted('Alcohol', raw, 'Fitted exp curve Alcohol', years, coloh, '{}_fitted'.format(i.lower()))


# %%

for i in drugs.columns:
    fig_fitted(i, raw, 'Fitted exp curve {}'.format(i), years, coloh, '{}_fitted'.format(i.lower()))

####################################################################################
############################ TRASH FIGURE pie chart ########################################
####################################################################################
#%% Figure

fig, ax = plt.subplots(figsize=(20,7))  

mc = 'black'

plt.rcParams.update({'font.size': 30, 
                     'axes.labelcolor' : "{}".format(mc), 
                     'xtick.color': "{}".format(mc),
                     'ytick.color': "{}".format(mc),
                     'font.family' : 'sans-serif'})

margin_bottom = np.zeros(len(raw.iloc[:, 0]))

for i, drug in enumerate(perc_y.columns):
    # print(i)
    perc_dru_t = perc_y[drug]

    perc_y[drug].plot.bar(x='Year',y='Percentage', ax=ax, stacked=True, 
                                    bottom = margin_bottom, label=drug, color=np.random.rand(3,))
    margin_bottom += perc_dru_t
    # print(margin_bottom)

lwD = 5
lenD = 10

ax.spines['left'].set_linewidth(lwD)
ax.spines['bottom'].set_linewidth(lwD)
ax.spines['right'].set_linewidth(lwD)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# years = np.arange(1960, 2018.1, 6)

start, end = ax.get_xlim()
# ax.xaxis.set_ticks(np.arange(start, end, 1))

# labelsx = [item.get_text() for item in ax.get_xticklabels()]

# for j in enumerate(seconds):
#     labelsx[j[0]] = int(j[1])

# axis.set_xticklabels(labelsx)

ax.yaxis.set_tick_params(width = lwD, length = lenD)
ax.xaxis.set_tick_params(width = lwD, length = lenD)

ax.set_ylabel('Percentage', labelpad=20)
ax.set_xlabel('Year', labelpad=20)
ax.set_ylim(0, 100)


####################################################################################
############################ TRASH FIGURE Fitting ########################################
####################################################################################

# %% Fitting

def linear(x, a, b):
    return a*x + b

x_axis = np.arange(len(raw['Alcohol']))
out, out_cov = curve_fit(exp, x_axis, np.array(raw['Alcohol']), p0=(4, 0.1))
# outL, out_covL = curve_fit(linear, x_axis, np.array(raw['Alcohol']))

# %%
exp_line = exp(x_axis, out[0], out[1])
# lin_line = linear(x_axis, outL[0], outL[1])

fig, ax = plt.subplots()

ax.plot(x_axis, raw['Alcohol'], '.')
ax.plot(x_axis, exp_line)
# ax.plot(x_axis, lin_line)

# %%
residuals_lin = np.array(raw['Alcohol']) - exp(x_axis, out[0], out[1])
ss_res = np.sum(residuals**2)

ss_tot = np.sum((np.array(raw['Alcohol'])-np.mean(np.array(raw['Alcohol'])))**2)

r_squared = 1 - (ss_res / ss_tot)

# %%
residuals_lin = np.array(raw['Alcohol']) - linear(x_axis, outL[0], outL[1])
ss_res_lin = np.sum(residuals**2)

ss_tot_lin = np.sum((np.array(raw['Alcohol'])-np.mean(np.array(raw['LSD'])))**2)

r_squared_lin = 1 - (ss_res / ss_tot)

# %%

from scipy.optimize import curve_fit 
  
from matplotlib import pyplot as plt 
  
# numpy.linspace with the given arguments 
# produce an array of 40 numbers between 0 
# and 10, both inclusive 
x = np.linspace(0, 10, num = 40) 
  
  
# y is another array which stores 3.45 times 
# the sine of (values in x) * 1.334.  
# The random.normal() draws random sample  
# from normal (Gaussian) distribution to make 
# them scatter across the base line 
y = 3.45 * np.sin(1.334 * x) + np.random.normal(size = 40) 
  
# Test function with coefficients as parameters 
def test(x, a, b): 
    return a * np.sin(b * x) 
  
# curve_fit() function takes the test-function 
# x-data and y-data as argument and returns  
# the coefficients a and b in param and 
# the estimated covariance of param in param_cov 
param, param_cov = curve_fit(test, x, y) 
  
  
print("Sine funcion coefficients:") 
print(param) 
print("Covariance of coefficients:") 
print(param_cov) 
  
# ans stores the new y-data according to  
# the coefficients given by curve-fit() function 
ans = (param[0]*(np.sin(param[1]*x))) 
  
'''Below 4 lines can be un-commented for plotting results  
using matplotlib as shown in the first example. '''
  
plt.plot(x, y, 'o', color ='red', label ="data") 
plt.plot(x, ans, '--', color ='blue', label ="optimized data") 
plt.legend() 
plt.show() 

####################################################################################
############################ TRASH FIGURE 2 ########################################
####################################################################################
# %% Figure 2  all
drugs = raw.iloc[:,1:16]
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize = (20, 10))
lwD = 3
colors = ['b', 'rosybrown', 'r', 'sienna', 'pink', 'magenta', 'blueviolet', 'black', 'g', 'lime', 'darkorange', 'olivedrab', 'deepskyblue', 'slategrey', 'lawngreen', 'darkviolet']

for z, c in zip(drugs, colors):
    # print(z)
    plt.plot(years, raw[z], color = c, linewidth=lwD, label = z)

ax.set_title('Number of publications for 15 psychoactive drugs.\n Including alcohol')
ax.set_ylabel('Number of publications')
ax.set_xlabel('Years')

ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])

ax.set_xlim([1960, 2018])

ax.spines['left'].set_linewidth(lwD)
ax.spines['bottom'].set_linewidth(lwD)
ax.spines['right'].set_linewidth(lwD)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

lenD = 8

ax.yaxis.set_tick_params(width = lwD, length = lenD)
ax.xaxis.set_tick_params(width = lwD, length = lenD)


leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
leg.get_frame().set_linewidth(0.0)

ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

name = 'all'
# plt.savefig('./../figures/dev_write_up/{}.png'.format(name), transparent = True, bbox_inches='tight')

# %% ############# Figure 2 all no alcohol
drugs = raw.iloc[:,2:16]
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize = (20, 10))

colors = ['b', 'rosybrown', 'r', 'sienna', 'pink', 'magenta', 'blueviolet', 'black', 'g', 'lime', 'darkorange', 'olivedrab', 'deepskyblue', 'slategrey', 'lawngreen', 'darkviolet']

for z, c in zip(drugs, colors):
    # print(z)
    plt.plot(years, raw[z], color = c, linewidth=3.0, label = z)

ax.set_title('Number of publications for 14 psychoactive drugs.\n Excluding alcohol')
ax.set_ylabel('Number of publications')
ax.set_xlabel('Years')

ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])

ax.set_xlim([1960, 2018])
ax.set_ylim([0, 5000.01])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.savefig('./../figures/dev_write_up/all_no_alc.png', transparent = True, bbox_inches='tight')

# %% ############ Figure 2 number pubs selected_two_axes
drugs_two = ['Cannabis', 'Ketamine', 'MDMA', 'Psilocybin']
alc_two = ['Alcohol']
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize = (20, 10))

colors = ['green', 'grey', 'pink', 'sienna']
lns = []
for z, c in zip(drugs_two, colors):
    l = plt.plot(years, raw[z], color = c, linewidth=3.0, label = z)
    lns.append(l)

ax.set_title('Number of publications for common recreational drugs.\n')
ax.set_ylabel('Number of publications')
ax.set_xlabel('Years')

ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])

ax.set_xlim([1960, 2018])
# ax.set_ylim([0, 5000.01])

ax2 = ax.twinx()
lns_alc = ax2.plot(years, raw['Alcohol'], color='b', label = 'Alcohol')

# added these three lines
# lns.append(lns_alc)


# for l in lns:
#     print(l[0].get_label())
# labs = [l[0].get_label() for l in lns]
# lns = [lns[0], lns[1], lns[2], lns[3], lns[4]]

alc = mlines.Line2D([], [], color='blue', label = 'Alcohol')
can = mlines.Line2D([], [], color='green', label = 'Cannabis')
md = mlines.Line2D([], [], color='pink', label = 'MDMA')
ket = mlines.Line2D([], [], color='grey', label = 'Ketamine')
psi = mlines.Line2D([], [], color='sienna', label = 'Psilocybin')

ax.legend(handles=[alc, can, md, ket, psi])

ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.savefig('./../figures/dev_write_up/selected_two_axes.png', transparent = True, bbox_inches='tight')


# %% ############ Figure 2 WoS
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize = (20, 10))

colors = ['k'] #, 'r', 'sienna', 'pink']
names_iso = ['WoS']  #'Cannabis', 'LSD', 'Psilocybin']

for z, c in zip(names_iso, colors):
    # print(z)
    plt.plot(years, raw[z], color = c, linewidth=3.0, label = 'Web of Science')

ax.legend(frameon=False)
ax.set_title('Total number of research publications')
ax.set_ylabel('Number of publications')
ax.set_xlabel('Years')


ax.set_xlim([1960, 2018])
ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.savefig('./../figures/dev_write_up/WoS.png', transparent = True, bbox_inches='tight')


# %% ############ Figure 2 LSD, Psilocybin, alcohol
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize = (20, 10))

colors = ['lawngreen', 'darkorange'] #, 'r', 'sienna', 'pink']
names_iso = ['LSD', 'Psilocybin']  #'Cannabis', 'LSD', 'Psilocybin']


for z, c in zip(names_iso, colors):
    # print(z)
    plt.plot(years, raw[z], color = c, linewidth=3.0, label = z)

# ax.legend(frameon=False)
ax.set_title('Total number of research publications')
ax.set_ylabel('Number of publications: LSD & Psilocybin')
ax.set_xlabel('Years')

ax.set_xlim([1960, 2018])
ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])

ax2 = ax.twinx()
lns_alc = ax2.plot(years, raw['Alcohol'], color='b', label = 'Alcohol')

# added these three lines
# lns.append(lns_alc)


# for l in lns:
#     print(l[0].get_label())
# labs = [l[0].get_label() for l in lns]
# lns = [lns[0], lns[1], lns[2], lns[3], lns[4]]

alc = mlines.Line2D([], [], color='b', label = 'Alcohol')
psi = mlines.Line2D([], [], color='darkorange', label = 'Psilocybin')
lsd = mlines.Line2D([], [], color='lawngreen', label = 'LSD')

ax.legend(handles=[alc, psi, lsd])

ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax2.set_ylabel('Number of publications: Alcohol')
ax2.yaxis.label.set_color('b')
ax2.spines['right'].set_color('b')

plt.savefig('./../figures/dev_write_up/psyche_alc.png', transparent = True, bbox_inches='tight')

# %%############ Figure 2 Depressants
fig, ax = plt.subplots(figsize = (20, 10))
lwD = 3

turkish_sea = '#255498'
blue_moon = '#3388A6' 

for i, z in enumerate(depressants):
    # print(z)
    plt.plot(years, raw[z], color = blue_moon, linewidth=lwD, label = z, alpha = (1 - (1/len(depressants))* i))

ax.set_title('Number of publications for depressant drugs.\n Excluding alcohol')
ax.set_ylabel('Number of publications')
ax.set_xlabel('Years')

ax.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2018])

ax.set_xlim([1960, 2018])
ax.set_ylim([0, 1600])

ax.spines['left'].set_linewidth(lwD)
ax.spines['bottom'].set_linewidth(lwD)
ax.spines['right'].set_linewidth(lwD)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

lenD = 8

ax.yaxis.set_tick_params(width = lwD, length = lenD)
ax.xaxis.set_tick_params(width = lwD, length = lenD)

leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
leg.get_frame().set_linewidth(0.0)

ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

name = 'depressants'
plt.savefig('./../figures/dev_write_up/{}.png'.format(name), transparent = True, bbox_inches='tight')