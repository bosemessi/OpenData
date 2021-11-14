import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kloppy import datasets
from kloppy.domain import Transformer, Provider
from kloppy.helpers import to_pandas,\
                   load_skillcorner_tracking_data      

from pandas import json_normalize
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
import scipy.signal as signal

"""#Choose a data file and read it; then some basic pre-processing"""

data_dir = 'C:\\Users\\bosem\\OneDrive\\Desktop\\opendata\\data\\matches\\'
game_num = input("Enter game number : ")
matchdata_file = data_dir + game_num + '\\match_data.json'
tracking_file = data_dir + game_num + '\\structured_data.json'

dataset = load_skillcorner_tracking_data(
    metadata_filename=matchdata_file,
    raw_data_filename=tracking_file,
    options={"coordinate_system": Provider.SKILLCORNER})

metadata = dataset.metadata
home_team, away_team = metadata.teams

df = to_pandas(dataset)

with open(tracking_file,'r') as f:
    matchdict = json.load(f)

with open(matchdata_file,'r') as f:
    mdict = json.load(f)

hometeamid = mdict['home_team']['id']
awayteamid = mdict['away_team']['id']
teamdict = {hometeamid:'home', awayteamid:'away'}

playerids = {}
playerdict = {}
positiondict = {}

for players in mdict['players']:
    if players['team_id'] == hometeamid:
        playerids[players['trackable_object']]='home_'+str(players['number'])
        playerdict['home_'+str(players['number'])]=players['first_name']+' '+players['last_name']
        positiondict['home_'+str(players['number'])]=players['player_role']['name']
    else:
        playerids[players['trackable_object']]='away_'+str(players['number'])
        playerdict['away_'+str(players['number'])]=players['first_name']+' '+players['last_name']
        positiondict['away_'+str(players['number'])]=players['player_role']['name']


sdf = json_normalize(matchdict,sep="_")
sdf = sdf[sdf['time'].notna()].reset_index(drop=True)
sdf[['min','sec']] = sdf['time'].str.split(':', expand=True)
sdf['min'] = sdf['min'].astype('float64')
sdf['sec'] = sdf['sec'].astype('float64')
sdf['timestamp'] = sdf['min']*60 + sdf['sec']
df = df.merge(sdf[['timestamp','possession_trackable_object']],
              on='timestamp',how='inner')
df['player_in_possession'] = df['possession_trackable_object'].map(playerids)
df['team_in_possession'] = df['ball_owning_team_id'].map(teamdict)

non_anon = [col for col in df.columns
            if 'anon' not in col]
df = df[non_anon]
xcols = [col for col in non_anon if '_x' in col]
ycols = [col for col in non_anon if '_y' in col]
df[xcols] = df[xcols] + 52.5
df[ycols] = df[ycols] + 34
df['dT'] = round(df.timestamp.shift(-1) - df.timestamp.shift(),1)
df['dt'] = round(df.timestamp.diff(),1)
df['same_poss'] = np.where(df['possession_trackable_object'].diff()==0,
                           1,0)
df['poss_number'] = (df['possession_trackable_object'].diff()!=0).astype('int').cumsum()*df['same_poss']

home_x = [cols for cols in xcols if 'home' in cols]
home_y = [cols for cols in ycols if 'home' in cols]
away_x = [cols for cols in xcols if 'away' in cols]
away_y = [cols for cols in ycols if 'away' in cols]

def calc_player_velocities(df, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12,
                           maxacc = 10):
    # remove any velocity data already in the dataframe
    df = remove_player_velocities(df)
    
    # Get the player ids
    player_ids = np.unique( [ c[:-2] for c in df.columns if c[:4] in ['home','away'] ] )

    # index of first frame in second half
    second_half_idx = df.period_id.idxmax(2)
    
    #condition for calculating velocity - only if dT = 0.1
    delta_cond = df.dT == 0.2

    # estimate velocities for players in team
    for player in player_ids: # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = np.where(delta_cond,
                      (df[player+"_x"].shift(-1)-df[player+"_x"].shift())/df.dT,
                      np.nan)
         
        vy = np.where(delta_cond,
                      (df[player+"_y"].shift(-1)-df[player+"_y"].shift())/df.dT,
                      np.nan)

        ax = np.where(delta_cond,
                      (df[player+"_x"].shift(-1)+df[player+"_x"].shift()-2.0*df[player+"_x"])/(df.dT/2.0)**2,
                      np.nan)
        
        ay = np.where(delta_cond,
                      (df[player+"_y"].shift(-1)+df[player+"_y"].shift()-2.0*df[player+"_y"])/(df.dT/2.0)**2,
                      np.nan)

        if maxspeed>0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt( vx**2 + vy**2 )
            vx[ raw_speed>maxspeed ] = np.nan
            vy[ raw_speed>maxspeed ] = np.nan

        if maxacc>0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_acc = np.sqrt( ax**2 + ay**2 )
            ax[ raw_acc>maxacc ] = np.nan
            ay[ raw_acc>maxacc ] = np.nan
            
        if smoothing:
            if filter_=='Savitzky-Golay':
                # calculate first half velocity
                vx[:second_half_idx] = signal.savgol_filter(vx[:second_half_idx],window_length=window,polyorder=polyorder)
                vy[:second_half_idx] = signal.savgol_filter(vy[:second_half_idx],window_length=window,polyorder=polyorder)
                ax[:second_half_idx] = signal.savgol_filter(ax[:second_half_idx],window_length=window,polyorder=polyorder)
                ay[:second_half_idx] = signal.savgol_filter(ay[:second_half_idx],window_length=window,polyorder=polyorder)        
                # calculate second half velocity
                vx[second_half_idx:] = signal.savgol_filter(vx[second_half_idx:],window_length=window,polyorder=polyorder)
                vy[second_half_idx:] = signal.savgol_filter(vy[second_half_idx:],window_length=window,polyorder=polyorder)
                ax[second_half_idx:] = signal.savgol_filter(ax[second_half_idx:],window_length=window,polyorder=polyorder)
                ay[second_half_idx:] = signal.savgol_filter(ay[second_half_idx:],window_length=window,polyorder=polyorder)
            elif filter_=='moving average':
                ma_window = np.ones( window ) / window 
                # calculate first half velocity
                vx[:second_half_idx] = np.convolve( vx[:second_half_idx] , ma_window, mode='same' ) 
                vy[:second_half_idx] = np.convolve( vy[:second_half_idx] , ma_window, mode='same' )
                ax[:second_half_idx] = np.convolve( ax[:second_half_idx] , ma_window, mode='same' ) 
                ay[:second_half_idx] = np.convolve( ay[:second_half_idx] , ma_window, mode='same' )      
                # calculate second half velocity
                vx[second_half_idx:] = np.convolve( vx[second_half_idx:] , ma_window, mode='same' ) 
                vy[second_half_idx:] = np.convolve( vy[second_half_idx:] , ma_window, mode='same' ) 
                ax[second_half_idx:] = np.convolve( ax[second_half_idx:] , ma_window, mode='same' ) 
                ay[second_half_idx:] = np.convolve( ay[second_half_idx:] , ma_window, mode='same' ) 
                
        
        # put player speed in x,y direction, and total speed back in the data frame
        df[player + "_vx"] = vx
        df[player + "_vy"] = vy
        df[player + "_speed"] = np.sqrt( vx**2 + vy**2 )
        df[player + "_ax"] = ax
        df[player + "_ay"] = ay
        df[player + "_acceleration"] = np.sqrt( ax**2 + ay**2 )
        df[player + "_dx"] = df[player + "_x"].diff()
        df[player + "_dy"] = df[player + "_y"].diff()
        df[player + "_dist"] = np.sqrt(df[player+"_dx"]**2 + df[player+"_dy"]**2)

    return df

def remove_player_velocities(df):
    # remove player velocoties and acceleeration measures that are already in the 'team' dataframe
    columns = [c for c in df.columns if c.split('_')[-1] in ['vx','vy','ax','ay','speed','acceleration']] # Get the player ids
    df = df.drop(columns=columns)
    return df

df = calc_player_velocities(df, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12)


players = np.unique( [ c.split('_')[0]+'_'+c.split('_')[1] 
                      for c in df.columns if c[:4] in ['home','away'] ] )
player_summary = pd.DataFrame(index=players)
max_speed = []
max_acc = []
walking = []
jogging = []
running = []
sprinting = []
sprint_maint = []
sprint_v1 = []
sprint_v2 = []
sprint_v3 = []
sprint_v4 = []
in_poss_max_speed = []
in_poss_max_acc = []
in_poss_sprinting = []
in_poss_duration = []
def_max_speed = []
def_max_acc = []
def_sprinting = []

sprint_threshold = 7 # minimum speed to be defined as a sprint (m/s)
sprint_window = 1*10

for player in players:
    col1 = player+'_speed'
    col2 = player+'_acceleration'
    col3 = player+'_dist'
    max_speed.append(df[col1].max())
    max_acc.append(df[col2].max())
    playerpossdf = df[df.player_in_possession==player]
    if len(playerpossdf)>0:
        in_poss_max_speed.append(playerpossdf[col1].max())
        in_poss_max_acc.append(playerpossdf[col2].max())
        player_speed = playerpossdf.loc[(playerpossdf[col1] >= 7), col1].median()
        in_poss_sprinting.append(player_speed)
        gdf = playerpossdf.groupby(['poss_number']).agg({'dt':'sum'}).reset_index()
        in_poss_duration.append(gdf.dt.median())
    else:
    	in_poss_max_speed.append(np.nan)
    	in_poss_max_acc.append(np.nan)
    	in_poss_sprinting.append(np.nan)
    	in_poss_duration.append(np.nan)

    player_speed = df.loc[df[col1] < 2, col1].median()
    walking.append( player_speed )
    # jogging (between 2 and 4 m/s)
    player_speed = df.loc[(df[col1] >= 2) & (df[col1] < 4), col1].median()
    jogging.append( player_speed )
    # running (between 4 and 7 m/s)
    player_speed = df.loc[(df[col1] >= 4) & (df[col1] < 7), col1].median()
    running.append( player_speed )
    # sprinting (greater than 7 m/s)
    player_speed = df.loc[(df[col1] >= 7), col1].median()
    sprinting.append( player_speed )
    player_sprints = np.diff( 1*( np.convolve( 1*(df[col1]>=sprint_threshold),
                        np.ones(sprint_window), mode='same' ) >= sprint_window ) )

    player_sprints_start = np.where( player_sprints == 1 )[0] - int(sprint_window/2) + 1 
    player_sprints_end = np.where( player_sprints == -1 )[0] + int(sprint_window/2) + 1
    sprintm = []
    sprintdist1 = []
    sprintdist2 = []
    sprintdist3 = []
    sprintdist4 = []
    for p in range(len(player_sprints_start)):
        sprintm.append( ( df[col1][player_sprints_start[p]:player_sprints_end[p]+1] ).median() )
        dist =  df[col3][player_sprints_start[p]:player_sprints_end[p]+1].sum()
        T = df['dT'][player_sprints_start[p]:player_sprints_end[p]+1].sum()/2.0
        if ((dist>=10)&(dist<20)):
            sprintdist1.append(dist/T)
        elif ((dist>=20)&(dist<30)):
            sprintdist2.append(dist/T)
        elif ((dist>=30)&(dist<40)):
            sprintdist3.append(dist/T)
        elif ((dist>=40)):
            sprintdist4.append(dist/T)
    if len(sprintm)>0:
        sprint_maint.append(np.median(sprintm))
    else:
        sprint_maint.append(np.nan)

    if len(sprintdist1)>0:
        sprint_v1.append(np.median(sprintdist1))
    else:
        sprint_v1.append(np.nan)

    if len(sprintdist2)>0:
        sprint_v2.append(np.median(sprintdist2))
    else:
        sprint_v2.append(np.nan)

    if len(sprintdist3)>0:
        sprint_v3.append(np.median(sprintdist3))
    else:
        sprint_v3.append(np.nan)

    if len(sprintdist4)>0:
        sprint_v4.append(np.median(sprintdist4))
    else:
        sprint_v4.append(np.nan)    

    if 'away' in player:
        deftransition = df[df.team_in_possession=='home']
    else:
        deftransition = df[df.team_in_possession=='home']
    def_max_speed.append(deftransition[col1].max())
    def_max_acc.append(deftransition[col2].max())
    player_speed = deftransition.loc[(deftransition[col1] >= 7), col1].median()
    def_sprinting.append(player_speed)


player_summary['top_speed'] = max_speed
player_summary['top_acc'] = max_acc
player_summary['median_walk_speed'] = walking
player_summary['median_jog_speed'] = jogging
player_summary['median_run_speed'] = running
player_summary['median_sprint_speed'] = sprinting
player_summary['median_sprint_maintainance'] = sprint_maint
player_summary['median_10_20m_sprint_speed'] = sprint_v1
player_summary['median_20_30m_sprint_speed'] = sprint_v2
player_summary['median_30_40m_sprint_speed'] = sprint_v3
player_summary['median_40mplus_sprint_speed'] = sprint_v4
player_summary['in_poss_top_speed'] = in_poss_max_speed
player_summary['in_poss_top_acc'] = in_poss_max_acc
player_summary['in_poss_median_sprint_speed'] = in_poss_sprinting
player_summary['in_poss_median_duration'] = in_poss_duration
player_summary['def_top_speed'] = def_max_speed
player_summary['def_top_acc'] = def_max_acc
player_summary['def_median_sprint_speed'] = def_sprinting
player_summary['player_name'] = player_summary.index.map(playerdict)
player_summary['player_role'] = player_summary.index.map(positiondict)

player_summary.to_csv('summary_'+game_num+'.csv', encoding='utf-8-sig')

print('Code run done !')