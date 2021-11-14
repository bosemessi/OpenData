import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import json_normalize
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
import os
from matplotlib import rcParams
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Slabo 27px'
mpl.rcParams['xtick.color'] = '#edece9'
mpl.rcParams['ytick.color'] = '#edece9'
mpl.rcParams['text.color'] = '#edece9' 

data_dir = 'C:\\Users\\bosem\\OneDrive\\Desktop\\opendata\\'
files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]

df = pd.DataFrame()
for file in files:
    df = df.append(pd.read_csv(file, encoding='utf-8-sig'))
df.drop(columns='Unnamed: 0',inplace=True)
df = df[df.player_role!='Goalkeeper']
duration = 'in_poss_median_duration'
other_cols = [cols for cols in df.columns if cols not in [ 'in_poss_median_duration','player_name','player_role'] ]
aggdict = {duration:'min'}
for cols in other_cols:
    aggdict[cols] = 'max'
df = df.groupby('player_name').agg(aggdict).reset_index()

df.head()

plot_columns = ['top_speed', 'top_acc', 'median_sprint_speed','median_sprint_maintainance',
 'median_10_20m_sprint_speed','median_20_30m_sprint_speed', 'median_30_40m_sprint_speed',
'median_40mplus_sprint_speed', 'in_poss_top_speed', 'in_poss_top_acc',
'in_poss_median_sprint_speed', 'in_poss_median_duration',
'def_top_speed', 'def_top_acc', 'def_median_sprint_speed']

plot_titles = ["Top speed ", "Top accl ", " Median sprint speed", "Median speed while maintaining sprint",
"Median speed during 10 to 20m sprints","Median speed during 20 to 30m sprints",
"Median speed during 30 to 40m sprints","Median speed during 40m or more sprints",
"Top speed with ball", "Top accl with ball", "Median sprint speed with ball", "Median ball hold duration",
"Top speed in defensive phase","Top accl in defensive phase", "Median sprint speed in defensive phase"]

reverse = [True,True,True,True,True,True,True,True,True,True,True,False,True,True,True]

def bar_plotter(arr,names,title,reversed):
    barWidth = 0.85
    r = [i for i in range(10)]
    fig,ax = plt.subplots(figsize=(10,5))
    fig1 = ax.barh(r, arr, color='lightcoral', edgecolor='white')
    if reversed:
        for i in range(10):
            ax.text(arr[i]+0.1,r[i]+0.2,str(round(arr[i],1)),fontsize=15)
    else:
        for i in range(10):
            ax.text(arr[i]+0.1,r[i]-0.2,str(round(arr[i],1)),fontsize=15)
    plt.yticks(r, names, fontsize=15)
    plt.xticks([])
    for s in ['top','left','right','bottom']:
        ax.spines[s].set_visible(False)
    if reversed:
        ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_facecolor('#082630')
    fig.set_facecolor('#082630')
    ax.set_title(title,fontsize=30,loc='left')
    plt.tight_layout()
    fig.savefig(title+'.png', dpi=500, bbox_inches='tight',
        facecolor = fig.get_facecolor())

for i, cols in enumerate(plot_columns):
    temp_df = df.sort_values(by=cols, ascending=False).reset_index(drop=True)
    bar_plotter(temp_df[cols][:10],temp_df['player_name'][:10],plot_titles[i],reverse[i])

