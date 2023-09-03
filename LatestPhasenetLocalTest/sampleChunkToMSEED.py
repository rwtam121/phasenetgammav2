import pandas as pd
import obspy.core as oc
import json
import numpy as np
from obspy import read
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os
from contexttimer import Timer

samprate=100
sample_set='sample_chunked_6.csv'
#sample_set='sample_chunked_10.csv'
#sample_set='sample_chunked.csv'

mseed_desired_name='sample_chunked.mseed'
path_name="MSEEDFromCSV"

tmp_path = Path(path_name)
query_df=pd.read_csv(sample_set)  

with Timer() as comprehensive_time:

    #STEP 1: After reading csv, get the comprehensive MSEED file
    df_candidates_two = query_df[['station','network','channel']].drop_duplicates()

    stream = oc.Stream()

    for _, row in df_candidates_two.iterrows():
        df_specificchannels=query_df[(query_df['station']==row['station']) & (query_df['network']==row['network']) & (query_df['channel']==row['channel'])]
        starttime=min(df_specificchannels['startt'].tolist())
        endtime=max(df_specificchannels['endt'].tolist())

        stacked_data=[]
        for _, row in df_specificchannels.iterrows():
            data_used=json.loads(row['data']) 
            stacked_data.extend(data_used) 

        data_channels=np.array(stacked_data).astype(np.int32) 

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
        stream.append(trace)


    mseed_desired_name='total_used.mseed'
    if not tmp_path.exists():
        tmp_path.mkdir()

    output_file=tmp_path / mseed_desired_name 
    stream=stream.sort() 
    stcopy_saved = stream.copy()
    stcopy_saved.write(output_file, format="MSEED") 


    #STEP 2: Split MSEED file into 30-second stations in a separate subfolder.
    output_used=path_name+'/'+mseed_desired_name
    st_opened = read(output_used) 

    output_path = Path("waveformsFromCSV")
    if not output_path.exists():
        output_path.mkdir(parents=True)

    for trace in tqdm(st_opened):
        tmp_path = output_path / f"{trace.stats.starttime.datetime.isoformat(timespec='seconds')}"
        tmp_file = tmp_path/f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}.mseed"
        if not tmp_path.exists():
            tmp_path.mkdir(parents=True)
        trace.write(tmp_file, format="MSEED")

    #STEP 3: Create a csv file to read everything within  subfolder
    mseed_list = sorted(list(output_path.rglob("*.mseed")))
    mseeds = []
    for f in mseed_list:
        mseeds.append(str(f).split(".mseed")[0][:-1]+"*.mseed")

    #remove triplicates
    mseeds=list(set(mseeds))

    file_name='test_data/mseed_from_chunk.csv'
    with open(file_name, "w") as fp:
        fp.write("fname\n")
        fp.write("\n".join(mseeds))


    with Timer() as predict_time:
        #Pass in our file into the predict engine
        command_used="python phasenet/predict.py --model=model/190703-214543 --chunked_data="+sample_set+" --data_list=test_data/mseed_from_chunk.csv --data_dir=./ --format=mseed --amplitude --batch_size=1"
        os.system(command_used)

    pred_time = predict_time.elapsed

    print("Predict time elapsed")
    print(pred_time)

infer_time = comprehensive_time.elapsed

print("Total time elapsed")
print(infer_time)
