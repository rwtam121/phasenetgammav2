This repository provides local testing scripts used for 1) `Earlier Phasenet`, 2) `Latest Phasenet`, and 3) `GaMMa` . A summary of the scripts, and how to run them, are provided below.

### Latest Phasenet (currently used in Quakes2AWS): 

**Summary**: I trust this Phasenet’s results since it matches Weiqiang’s results (in terms of the pick arrival time and finding all of the relevant picks). But this code also runs slower. It utilizes a MSEED conversion of the Quakes2AWS realtime data, and its coding style is rooted in research-type testing. Elements of this code are derived from Weiqiang Zhu’s code found here: https://github.com/AI4EPS/PhaseNet/blob/ac2b736f514735bd058860033f4cc89c99593f4b/phasenet/predict.py 

**To run:**

* Access the `LatestPhasenetLocalTest` subfolder. The entryway is `sampleChunkToMSEED.py`. Focus on line 13 of the code and change the sample_set variable to reference any of the example real time data I am vetting (these files consist of 30-seconds, 100Hz of raw waveforms from 50+ stations. We have three example tests: sample_chunked, sample_chunked_6 and sample_chunked_10.csv). 
* Run `python sampleChunkToMSEED.py` 
* The resulting `pick.csv` will be stored in the /picks subfolder, where pick results can be analyzed. 
* The resulting amplitudes are wrong, but this is the latest Phasenet I believe matches Weiqiang’s results (in terms of the arrival time/finding all picks). For vetting, those picks could be plot on top of their waveforms, or digging into specific elements of the code (specifically in data_reader.py or predict.py) could prove helpful. 



### Earlier Phasenet (previously utilized): 

**Summary**: Unreliable results, but good speed and written with API calls, which lends itself to realtime implementation for processing many stations. This is motivation to ideally get this code to work over the later Phasenet. Elements of this code are derived from Weiqiang Zhu’s code found [here](https://github.com/AI4EPS/PhaseNet/blob/ac2b736f514735bd058860033f4cc89c99593f4b/phasenet/app.py)

**To run:**

* Access the `EarlierPhasenetLocalTest` subfolder. The entryway is `app_RT_ampCalc.py`. Focus on line 215 of the code and change the sample_set variable to reference any of the example real time data I am vetting (we have four example tests; sample_chunked, sample_chunked_6, sample_chunked_fromRedis.csv and sample_chunked_RedisRidgecrest.csv). 
* Run `python app_RT_ampCalc.py` 
* Picks will be printed onto your console screen, after “All picks”, IE: 
_{'id': 'CE.13070..HN', 'timestamp': '2019-07-04T01:11:17.090', 'prob': 0.59686100482940674, 'amp': 3.0161194445099682e-05, 'type': 'p'}_
* These picks are known to be unreliable. For vetting, those picks could be plot on top of their waveforms, or digging into specific elements of the code (specifically in data_reader.py or predict.py) could prove helpful. Specifically, the convert_mseed function in app_RT_ampCalc.py is the preprocessing step I use for the real time data for the prediction. This might affect the format of some of the results, and is ripe for making changes (I might lack some of the seismological background as I manipulate the data here). 


### Running the GaMMa Associator: 

**Summary** Needs tuning, since in realtime we finding more events than there are actual events (overly aggressive). Elements of this code are derived from Weiqiang’s code found [here](https://github.com/AI4EPS/GaMMA/blob/master/gamma/app.py)
 
**To run:**

* Access the `GaMMaTest` subfolder. The entryway is `GaMMaTest.py`. Focus on line ~82 of the code, which contains the PICK_CSV variable. This references our Phasenet-found picks in csv format (it references a file called picks.csv, which consist of picks found from a 20-minute Ridgecrest replay segment (back in July 6, 2019).  
* Run `python GaMMaTest.py `
* The outputs are two files: the associated events and the picks associated with each of those events. `catalog_SCSN_found_PROJ.csv` display the events, and `associated_picks_SCSN_found.csv` display the associated picks. 
* The events found in catalog_SCSN_found_PROJ.csv can be compared to known Ridgecrest events in the catalog, and as GaMMa is tuned, comparing GaMMa’s event times with those event times can be a useful barometer. For tuning, the configuration parameters are in lines 96-126 of GaMMaTest.py. 
















