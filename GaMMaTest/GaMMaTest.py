import pandas as pd
import os
import numpy as np
import json
from datetime import datetime, timezone
from gamma.utils import association, from_seconds
from tqdm import tqdm
from contexttimer import Timer
from pyproj import Proj


def run_gamma(picks, config, stations):
    pbar = tqdm(sorted(list(set(picks["time_idx"])))) #I also added this
    picks["timestamp"] = picks["timestamp"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))

    event_idx0 = 1  ## current earthquake index
    assignments = []
    if (len(picks) > 0) and (len(picks) < 5000):
        catalogs, assignments = association(picks, stations, config, event_idx0, config["method"], pbar=pbar)
        event_idx0 += len(catalogs)
    else:
        catalogs = []
        picks["time_idx"] = picks["timestamp"].apply(lambda x: x.strftime("%Y-%m-%dT%H"))  ## process by hours
        for hour in sorted(list(set(picks["time_idx"]))):
            picks_ = picks[picks["time_idx"] == hour]
            if len(picks_) == 0:
                continue

            catalog, assign = association(picks, stations, config, event_idx0, config["method"], pbar=pbar)
            event_idx0 += len(catalog)
            catalogs.extend(catalog)
            assignments.extend(assign)

    catalogs = pd.DataFrame(
        catalogs,
        columns=["time(s)"]
        + config["dims"]
        + ["magnitude", "sigma_time", "sigma_amp", "cov_time_amp", "covariance","event_idx", "prob_gamma",],
    )

    catalogs["time"] = catalogs["time(s)"].apply(lambda x: from_seconds(x))
    catalogs["longitude"] = catalogs["x(km)"].apply(lambda x: x / config["degree2km"] + config["center"][0])
    catalogs["latitude"] = catalogs["y(km)"].apply(lambda x: x / config["degree2km"] + config["center"][1])
    catalogs["depth(m)"] = catalogs["z(km)"].apply(lambda x: x * 1e3)

    if config["use_amplitude"]:
        catalogs["covariance"] = catalogs["covariance"].apply(lambda x: f"{x[0][0]:.3f},{x[1][1]:.3f},{x[0][1]:.3f}")
    else: 
        catalogs["covariance"] = catalogs["covariance"].apply(lambda x: f"{x[0][0]:.3f}")

    catalogs = catalogs[['time', 'magnitude', 'longitude', 'latitude', 'depth(m)', 'covariance', "event_idx","prob_gamma"]]   

    assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gmma"])
    picks_gamma = picks.join(assignments.set_index("pick_idx")).fillna(-1).astype({'event_idx': int})
    picks_gamma["timestamp"] = picks_gamma["timestamp"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3])

    picks_gamma["timestamp_actual"] = picks_gamma["timestamp"].apply(lambda x: (datetime.strptime(x,"%Y-%m-%dT%H:%M:%S.%f")).replace(tzinfo=timezone.utc).timestamp())

    if "time_idx" in picks_gamma:
        picks_gamma.drop(columns=["time_idx"], inplace=True)
    
    return catalogs,picks_gamma


#CODE STARTS HERE
with Timer() as run_gamma_seconds:
    config = {'center': (-117.504, 35.705), 
        'xlim_degree': [-118.004, -117.004], 
        'ylim_degree': [35.205, 36.205], 
        'degree2km': 111.19492474777779, 
        'starttime': datetime(2019, 7, 4, 17, 0), 
        'endtime': datetime(2019, 7, 5, 0, 0)}

    config["x(km)"] = (np.array(config["xlim_degree"]) - np.array(config["center"][0])) * config["degree2km"] #times cos(lat)
    config["y(km)"] = (np.array(config["ylim_degree"]) - np.array(config["center"][1])) * config["degree2km"]
    config["z(km)"] = (0, 20) #from (0,60) 


    
    PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__)))
    STATION_CSV = os.path.join(PROJECT_ROOT, "tests/SCSN_station_response.csv") #This is for our new set of stations
    PICK_CSV='picks.csv'
    # PICK_CSV = os.path.join(PROJECT_ROOT, "tests/2022_06_10_Amplitude_MLpicks.csv")
    # PICK_CSV = os.path.join(PROJECT_ROOT, "tests/picks_modded_dictionaryform_unsorted.csv")
    # STATION_CSV = os.path.join(PROJECT_ROOT, "tests/stations_modded.csv") #This is for our new set of stations

    stations = pd.read_csv(STATION_CSV)
    stations = stations.rename(columns={"station": "id"}) 

    proj = Proj(f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")
    stations[["x(km)", "y(km)"]] = stations.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
    stations["z(km)"] = stations["elevation(m)"].apply(lambda x: -x / 1e3)

    #FIRST STEP: set up configuration parameters (in the Ridgecrest example we're using, we use the below)
    ### setting GMMA configs
    config["dims"] = ['x(km)', 'y(km)', 'z(km)']
    config["use_amplitude"] = True
    config["use_dbscan"] = True
    config["vel"] = {"p": 6.0, "s": 6.0 / 1.75} #prev s 6.0/1.73
    config["method"] = "BGMM"
    if config["method"] == "BGMM":
        config["oversample_factor"] = 8 #prev 4 
    if config["method"] == "GMM":
        config["oversample_factor"] = 1

    config["bfgs_bounds"] = (
        (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
        (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
        (0, config["z(km)"][1] + 1),  # x
        (None, None),
    )  # t
    config["dbscan_eps"] = min(
        10,
        np.sqrt(
            (stations["x(km)"].max() - stations["x(km)"].min()) ** 2
            + (stations["y(km)"].max() - stations["y(km)"].min()) ** 2
        )
        / (6.0 / 1.75),
    )  
    config["dbscan_min_samples"] = min(3, len(stations))

    # Filtering
    config["min_picks_per_eq"] = min(5, len(stations) // 2)  
    config["max_sigma11"] = 2.0  # s 
    config["max_sigma22"] = 1.0  # m/s 
    config["max_sigma12"] = 1.0  # covariance

    for k, v in config.items():
        print(f"{k}: {v}")

    picks = pd.read_csv(PICK_CSV)
    picks = picks.rename(columns={"station_id": "id","phase_time": "datetime","phase_score": "prob","phase_amp": "amp","phase_type": "type"}) 
    picks[['sta', 'net','nothing','inst']] = picks['id'].str.split(".", expand = True) 
    picks=picks.drop(['nothing'], axis=1)
    picks['loc']=['--']*len(picks) 

    picks["timestamp"] = picks["datetime"].apply(lambda x: datetime.timestamp(datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f')))
    picks['pickwidth']=[0]*len(picks) 
    now = datetime.now()
    ts = datetime.timestamp(now)
    picks['curr_timestamp']=[ts]*len(picks)


    print("Picks")
    print(picks)


    picks = picks.rename(columns={"confidence": "prob"}) 
    picks=picks.sort_values(by = ['id', 'type'], ascending = [True, True])
    picks['timestamp_older']=picks['timestamp'] 

    picks["time_idx"] = picks["datetime"].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f').strftime("%Y-%m-%dT%H"))
    picks["timestamp"] = picks["datetime"].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f').strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]) 
    picks=picks.drop(columns=['sta', 'net','inst','loc','datetime','pickwidth'])

    ## if use amplitude
    if config["use_amplitude"]:
        picks = picks[picks["amp"] != 0]

    picks["timestamp"] = picks["timestamp"].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))

    unique_picks=list(set(picks['id'].tolist()))

    stations = stations[stations['id'].isin(unique_picks)] 
    stations=stations.sort_values(by = ['id'], ascending = [True]) 
    # stations.to_csv('ascending_stations.csv') 
    print("# of stations")
    print(len(stations))
    # picks.to_csv('modded_picks.csv')

    pbar = tqdm(sorted(list(set(picks["time_idx"])))) 

    event_idx0 = 1  
    assignments = []
    if (len(picks) > 0) and (len(picks) < 5000):
        catalogs, assignments = association(picks, stations, config, event_idx0, method=config["method"], pbar=pbar,)
        event_idx0 += len(catalogs)
    else:
        catalogs = []
        for i, segment in enumerate(pbar):
            picks_ = picks[picks["time_idx"] == segment]
            if len(picks_) == 0:
                continue
            catalog, assign = association(picks_, stations, config, event_idx0, method=config["method"], pbar=pbar,)
            event_idx0 += len(catalog)
            catalogs.extend(catalog)
            assignments.extend(assign)

    catalogs = pd.DataFrame(
        catalogs,
        columns=["time(s)"]
        + config["dims"]
        + ["magnitude", "sigma_time", "sigma_amp", "cov_time_amp", "event_idx", "prob_gamma",],
    )
    catalogs["time"] = catalogs["time(s)"].apply(lambda x: from_seconds(x))
    catalogs["longitude"] = catalogs["x(km)"].apply(lambda x: x / config["degree2km"] + config["center"][0])
    catalogs["latitude"] = catalogs["y(km)"].apply(lambda x: x / config["degree2km"] + config["center"][1])
    catalogs["depth(m)"] = catalogs["z(km)"].apply(lambda x: x * 1e3)
    catalogs.sort_values(by=["time"], inplace=True)
    catalog_csv='catalog_SCSN_found_PROJ.csv'
    with open(catalog_csv, 'w') as fp:
        catalogs.to_csv(
            fp,
            sep="\t",
            index=False,
            float_format="%.3f",
            date_format='%Y-%m-%dT%H:%M:%S.%f',
            columns=[
                "time",
                "magnitude",
                "longitude",
                "latitude",
                "depth(m)",
                "sigma_time",
                "sigma_amp",
                "cov_time_amp",
                "prob_gamma",
                # "x(km)",
                # "y(km)",
                # "z(km)",
                "event_idx",
            ],
        )

    print("Num events")
    print(catalogs)
    print(len(catalogs))

    #Also save the associated picks here
    assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gmma"])
    picks_gamma = picks.join(assignments.set_index("pick_idx")).fillna(-1).astype({'event_idx': int})
    picks['timestamp_older']=picks['timestamp'] #introduced this so we can readily get the timestamp to iterate across 

    picks_gamma["timestamp"] = picks_gamma["timestamp"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3])
    picks_gamma["timestamp_actual"] = picks_gamma["timestamp"].apply(lambda x: (datetime.strptime(x,"%Y-%m-%dT%H:%M:%S.%f")).replace(tzinfo=timezone.utc).timestamp())

    if "time_idx" in picks_gamma:
        picks_gamma.drop(columns=["time_idx"], inplace=True)
    
    print("GaMMA association:")
    picks_gamma=picks_gamma[picks_gamma.event_idx != -1] #reduce information in kinesis stream, only send relevant associated picks 
    picks_gamma=picks_gamma.sort_values(by='event_idx', ascending=False)

    unique_event_ids_currpicks=list(set(picks_gamma['event_idx'].tolist()))

    for id_used in unique_event_ids_currpicks:
        curr_assoc_picks_by_event=picks_gamma[picks_gamma.event_idx == id_used] 
        highprob_picks=curr_assoc_picks_by_event[curr_assoc_picks_by_event.prob > 0.6] 

    picks_gamma.to_csv('associated_picks_SCSN_found.csv')

    catalog_dictionary = catalogs.to_dict() 
    df_catalog_str = json.dumps(catalog_dictionary) 
    catalog_redict = json.loads(df_catalog_str) 

    if 'magnitude' in catalog_redict:
        for key in catalog_redict['magnitude']: 
            origin_timestamp=catalog_redict['time(s)'][key]
            magnitude=round(catalog_redict['magnitude'][key],2)
            event_id=catalog_redict['event_idx'][key]
            prob_gamma_event=round(catalog_redict['prob_gamma'][key],2)
            origin_datetime=catalog_redict['time'][key]
            longitude=catalog_redict['longitude'][key]
            latitude=catalog_redict['latitude'][key]
            depth=catalog_redict['depth(m)'][key] 

    picks_dictionary = picks_gamma.to_dict() 
    df_picks_str = json.dumps(picks_dictionary) 
    picks_redict = json.loads(df_picks_str) 

    if 'event_idx' in picks_redict:
        event_ids=picks_redict['event_idx']

        uniqueEventIDs = list(set(event_ids.values()))

        catalog_event_nums=catalog_redict['event_idx']
        for eventID in uniqueEventIDs:
            corresponding_key=list(catalog_event_nums.keys())[list(catalog_event_nums.values()).index(eventID)]
            corr_mag=catalog_redict['magnitude'][corresponding_key]
            prob_gamma=catalog_redict['prob_gamma'][corresponding_key]
            break

run_gamma_time_elapsed=run_gamma_seconds.elapsed
print('Run gamma time elapsed (in seconds)')
print(run_gamma_time_elapsed)




