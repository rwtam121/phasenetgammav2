import os
import numpy as np
from scipy.interpolate import interp1d
from postprocess import extract_amplitude, extract_picks
from datetime import datetime, timedelta
import pandas as pd
import obspy.core as oc
import json
import numpy as np
from scipy.interpolate import interp1d
import logging
from model import UNet
import tensorflow as tf
from tensorflow.python.util import deprecation
from contexttimer import Timer

deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class LoadModel:
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        # PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
        self.model = UNet(mode="pred")
        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.path  = f"model/190703-214543" #{PROJECT_ROOT}/
        self.sess  = tf.compat.v1.Session(config=sess_config)
        self.restored_checkpoint = self.load()

    def load(self):
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        latest_check_point = tf.train.latest_checkpoint(self.path)
        print(f"restoring model {latest_check_point}") 
        saver.restore(self.sess, latest_check_point)

    def get_prediction(self,data,return_preds=False):
        vec = np.array(data['vec'])
        vec, vec_raw = preprocess(vec)

        feed = {load_model.model.X: vec, load_model.model.drop_rate: 0, load_model.model.is_training: False}
        preds = load_model.sess.run(load_model.model.preds, feed_dict=feed)

        picks = extract_picks(preds, fnames=data['id'], station_ids=data['id'], t0=data['timestamp'])
        amps = extract_amplitude(vec_raw, picks)

        picks = format_picks(picks, data['dt'], amps)

        if return_preds:
                return picks, preds

        return picks



def normalize_batch(data, window=3000):
       shift = window // 2
       nsta, nt, nch = data.shape

       data_pad = np.pad(data, ((0, 0), (window // 2, window // 2), (0, 0)), mode="reflect")
       t = np.arange(0, nt, shift, dtype="int")
       std = np.zeros([nsta, len(t) + 1, nch])
       mean = np.zeros([nsta, len(t) + 1, nch])
       for i in range(1, len(t)):
              std[:, i, :] = np.std(data_pad[:, i * shift : i * shift + window, :], axis=1)
              mean[:, i, :] = np.mean(data_pad[:, i * shift : i * shift + window, :], axis=1)

       t = np.append(t, nt)
       std[:, -1, :], mean[:, -1, :] = std[:, -2, :], mean[:, -2, :]
       std[:, 0, :], mean[:, 0, :] = std[:, 1, :], mean[:, 1, :]
       std[std == 0] = 1

       t_interp = np.arange(nt, dtype="int")
       std_interp = interp1d(t, std, axis=1, kind="slinear")(t_interp)
       mean_interp = interp1d(t, mean, axis=1, kind="slinear")(t_interp)
       data = (data - mean_interp) / std_interp

       return data

def preprocess(data):
       raw = data.copy()
       data = normalize_batch(data)
       if len(data.shape) == 3:
              data = data[:, :, np.newaxis, :]
              raw = raw[:, :, np.newaxis, :]
       return data, raw

def calc_timestamp(timestamp, sec):
       timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f") + timedelta(seconds=sec)
       return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

def format_picks(picks, dt, amplitudes):
       picks_ = []
       for pick, amplitude in zip(picks, amplitudes):
              for idxs, probs, amps in zip(pick.p_idx, pick.p_prob, amplitude.p_amp):
                     for idx, prob, amp in zip(idxs, probs, amps):
                            picks_.append(
                            {
                                   "id": pick.fname,
                                   "timestamp": calc_timestamp(pick.t0, float(idx) * dt),
                                   "prob": prob,
                                   "amp": amp,
                                   "type": "p",
                            }
                            )
              for idxs, probs, amps in zip(pick.s_idx, pick.s_prob, amplitude.s_amp):
                     for idx, prob, amp in zip(idxs, probs, amps):
                            picks_.append(
                            {
                                   "id": pick.fname,
                                   "timestamp": calc_timestamp(pick.t0, float(idx) * dt),
                                   "prob": prob,
                                   "amp": amp,
                                   "type": "s",
                            }
                            )
       return picks_




def convert_mseed(mseed, station_locs):
    sampling_rate = 100
    n_channel = 3
    dtype = "float32"
    # amplitude = True
    remove_resp = True

    try:
        mseed = mseed.detrend("spline", order=2, dspline=5 * mseed[0].stats.sampling_rate)
    except:
        logging.error(f"Error: spline detrend failed at file")
        mseed = mseed.detrend("demean")
    mseed = mseed.merge(fill_value=0)
    starttime = min([st.stats.starttime for st in mseed])
    endtime = max([st.stats.endtime for st in mseed])
    mseed = mseed.trim(starttime, endtime, pad=True, fill_value=0)

    for i in range(len(mseed)):
        if mseed[i].stats.sampling_rate != sampling_rate:
            logging.warning(
                f"Resampling {mseed[i].id} from {mseed[i].stats.sampling_rate} to {sampling_rate} Hz"
            )
            mseed[i] = mseed[i].interpolate(sampling_rate, method="linear")

    order = ['3', '2', '1', 'E', 'N', 'Z']
    order = {key: i for i, key in enumerate(order)}
    comp2idx = {"3": 0, "2": 1, "1": 2, "E": 0, "N": 1, "Z": 2}

    nsta = len(station_locs)
    nt = max(len(mseed[i].data) for i in range(len(mseed)))
    data = []
    station_id = []
    t0 = []
    for i in range(nsta):
        trace_data = np.zeros([nt, n_channel], dtype=dtype)
        empty_station = True
        sta = station_locs.iloc[i]["id"]
        comp = station_locs.iloc[i]["component"].split(",")
        if remove_resp:
            resp = station_locs.iloc[i]["response"].split(",")

        for j, c in enumerate(sorted(comp, key=lambda x: order[x[-1]])):
            resp_j = float(resp[j])
            if len(comp) != 3:  ## less than 3 component
                j = comp2idx[c]

            if len(mseed.select(id=sta + c)) == 0:
                continue
            else:
                empty_station = False

            tmp = mseed.select(id=sta + c)[0].data.astype(dtype)
            trace_data[: len(tmp), j] = tmp[:nt]

            if station_locs.iloc[i]["unit"] == "m/s**2":
                tmp = mseed.select(id=sta + c)[0]
                tmp = tmp.integrate()
                tmp = tmp.filter("highpass", freq=1.0)
                tmp = tmp.data.astype(dtype)
                trace_data[: len(tmp), j] = tmp[:nt]
            elif station_locs.iloc[i]["unit"] == "m/s":
                tmp = mseed.select(id=sta + c)[0].data.astype(dtype)
                trace_data[: len(tmp), j] = tmp[:nt]
            else:
                print(
                    f"Error in {station_locs.iloc[i]['station']}\n{station_locs.iloc[i]['unit']} should be m/s**2 or m/s!"
                )
            
            if remove_resp:
                trace_data[:, j] /= resp_j
                
        if not empty_station:
            data.append(trace_data)
            station_id.append(sta)
            t0.append(starttime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3])

    data = np.stack(data)

    meta = {"data": data, "t0": t0, "station_id": station_id, "fname": station_id}
    
    return meta



#CODE STARTS HERE
with Timer() as run_phasenet_seconds:
    load_model=LoadModel() #this will also load the model

    #sample_set='sample_chunked_6.csv'
    #sample_set='sample_chunked_fromRedis.csv'
    #sample_set='sample_chunked_RedisRidgecrest.csv'
    sample_set='sample_chunked_0.csv'
    #sample_set='sample_chunked_10.csv'

    samprate=100 

    with Timer() as read_aggregate_data:
        query_df=pd.read_csv(sample_set)  
        query_df.drop(['timestamp', 'num_stations'], axis=1, inplace=True)
        df_candidates_two = query_df[['station','network','channel']].drop_duplicates()

        query_df['channel_two'] = query_df.channel.str[:-1]
        df_candidates_three = query_df[['station','network','channel_two']].drop_duplicates()

        stream = oc.Stream()
        trace_channel_list=[]

        data_station=df_candidates_two['station'].tolist()
        data_network=df_candidates_two['network'].tolist()
        data_channel=df_candidates_two['channel'].tolist()

        for i in range(len(data_station)):
            df_specificchannels=query_df[(query_df['station']==data_station[i]) & (query_df['network']==data_network[i]) & (query_df['channel']==data_channel[i])]
            starttime=min(df_specificchannels['startt'].tolist())
            endtime=max(df_specificchannels['endt'].tolist())

            stacked_data=[]
            for _, row in df_specificchannels.iterrows():
                    data_used=json.loads(row['data']) 
                    stacked_data.extend(data_used) 

            data_channels=np.array(stacked_data)

            header = {
                    'sampling_rate': samprate,
                    'delta': 1.0 / samprate,
                    'starttime': oc.utcdatetime.UTCDateTime(starttime),
                    'endtime': oc.utcdatetime.UTCDateTime(endtime),
                    'station': df_specificchannels.iloc[0]['station'],
                    'network': df_specificchannels.iloc[0]['network'],
                    'channel': df_specificchannels.iloc[0]['channel'] 
            }
            trace = oc.Trace(data=data_channels, header=header)
            trace_channel_list.append(trace.stats.channel)

            stream.append(trace)


    with Timer() as convert_mseed_secs:
        station_response_found='SCSN_station_response.csv'
        station_locs=pd.read_csv(station_response_found)  
        meta = convert_mseed(stream, station_locs)

    stations_separated_by_meta=meta['station_id']
    num_HN_inmeta=[i for i in meta['station_id'] if 'HN' in i]
    nonHN_meta=len(stations_separated_by_meta)-len(num_HN_inmeta)

    with Timer() as batch_pred_secs:
        batch = 4
        phasenet_picks = []

        for j in range(0, len(meta["station_id"]), batch):
            req = {"id": meta['station_id'][j:j+batch],
                    "timestamp": meta["t0"][j:j+batch],
                    "vec": meta["data"][j:j+batch].tolist(),
                    "dt":0.01}

            picks=load_model.get_prediction(req) 
            phasenet_picks.extend(picks)

    print("All picks")
    print(phasenet_picks)
    print(len(phasenet_picks))

    #iterate list of dictionary picks
    station_id_pick_list=[]
    HN_station_list=[]
    nonHN_station_list=[]
    for pick in phasenet_picks:
        station_id_pick_list.append(pick['id'])

        if 'HN' in pick['id']:
            HN_station_list.append(pick['id'])
        else:
            nonHN_station_list.append(pick['id'])


    num_unique_stations_in_picks=len(list(set(station_id_pick_list)))
    num_unique_HNstations_in_picks=len(list(set(HN_station_list)))
    num_unique_nonHNstations_in_picks=len(list(set(nonHN_station_list)))


run_phasenet_time_elapsed=run_phasenet_seconds.elapsed
print('Run phasenet time elapsed (in seconds)')
print(run_phasenet_time_elapsed)

read_aggregate_data_elapsed=read_aggregate_data.elapsed
print('Read aggregate data elapsed (in seconds)')
print(read_aggregate_data_elapsed)

convert_mseed_secs_elapsed=convert_mseed_secs.elapsed
print('Convert mseed (in seconds)')
print(convert_mseed_secs_elapsed)

batch_pred_secs_elapsed=batch_pred_secs.elapsed
print('Batch pred elapsed (in seconds)')
print(batch_pred_secs_elapsed)
