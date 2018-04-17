# setup and paths

import numpy as np
import pandas as pd
import pickle
import time
import warnings
import gc
import psutil

from statsmodels.tsa.stattools import kpss

import progressbar as pb

data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"

TOTAL_MEM_GB = psutil.virtual_memory().total/1024**3


# set up hdf5 storage

ts_store = pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5")

# Define functions to find stationary transformations
# ---------------------------------------------------
# Use KPSS test in waterfall way to determine order of differencing, order of seasonal differencing, and transformations. Only consider if d+D<=2 and D<=1. Do S=7 for daily, but S=0 for weekly since the seasonality would be long if did 52.
#
# Perform KPSS test in this order until stationary test result (null hypothesis is not rejected).
#
# 1. d=0, D=0, untransformed
# 2. d=1, D=0, untransformed
# 3. d=0, D=1, untransformed (daily only)
# 4. d=1, D=1, untransformed (daily only)
# 5. d=2, D=0, untransformed
# 6. repeat above with log transformation
# 7. if still no stationary transformation at this point, flag it

# define a function to find a stationary combination of differencing (d) and seasonal differencing (D, and S)
def find_stationary_ts(ts, S=0, pval_thres=0.05):
    # set return defaults
    found_stationary = False
    d_stationary = np.nan
    D_stationary = np.nan
    S_stationary = np.nan
    ts_stationary = None

    # straight level
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, kpss_pval, _, _ = kpss(ts, regression="c")

    if kpss_pval > pval_thres:
        found_stationary = True
        d_stationary = 0
        D_stationary = 0
        S_stationary = S
        ts_stationary = ts

    # first difference
    if not found_stationary:
        ts_d1D0 = ts - ts.shift(1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, kpss_pval, _, _ = kpss(ts_d1D0[1:], regression="c")

        if kpss_pval > pval_thres:
            found_stationary = True
            d_stationary = 1
            D_stationary = 0
            S_stationary = S
            ts_stationary = ts_d1D0

    # seasonal difference
    if not found_stationary and S>1:
        ts_d0D1 = ts - ts.shift(S)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, kpss_pval, _, _ = kpss(ts_d0D1[S:], regression="c")

        if kpss_pval > pval_thres:
            found_stationary = True
            d_stationary = 0
            D_stationary = 1
            S_stationary = S
            ts_stationary = ts_d0D1

    # seasonal first difference
    if not found_stationary and S>1:
        ts_d1D1 = ts_d1D0 -ts_d1D0.shift(S)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, kpss_pval, _, _ = kpss(ts_d1D1[(S+1):], regression="c")

        if kpss_pval > pval_thres:
            found_stationary = True
            d_stationary = 1
            D_stationary = 1
            S_stationary = S
            ts_stationary = ts_d1D1

    # second difference
    if not found_stationary:
        ts_d2D0 = ts_d1D0 - ts_d1D0.shift(1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, kpss_pval, _, _ = kpss(ts_d2D0[2:], regression="c")

        if kpss_pval > pval_thres:
            found_stationary = True
            d_stationary = 2
            D_stationary = 0
            S_stationary = S
            ts_stationary = ts_d2D0

    # if nothing is found
    if ts_stationary is None:
        ts_stationary = pd.Series(np.repeat(np.nan, len(ts)), index=ts.index, dtype=ts.dtype)
        ts_stationary.name = ts.name

        S_stationary = S

    # return
    ts_stationary.name = ts_stationary.name + "_stn"

    return found_stationary, d_stationary, D_stationary, S_stationary, ts_stationary

# define a function to find a stationary combination that also considers functional transformations as well
def find_stationary_ts_considerFunctional(ts, S=0, consider_log=True, pval_thres=0.05):
    transf_stationary = np.nan

    found_stationary, d_stationary, D_stationary, S_stationary, ts_stationary = find_stationary_ts(ts, S)

    if found_stationary:
        transf_stationary = "asis"
    elif consider_log:
        ts_0to1_log = ts.copy()
        ts_0to1_log[ts_0to1_log==0] = 1
        ts_0to1_log = np.log(ts_0to1_log)

        found_stationary, d_stationary, D_stationary, S_stationary, ts_stationary = find_stationary_ts(ts_0to1_log, S)
        del ts_0to1_log

        if found_stationary:
            transf_stationary = "log"

    # return
    return found_stationary, transf_stationary, d_stationary, D_stationary, S_stationary, ts_stationary


# Find Stationary Transformation for Topline Time Series
# ------------------------------------------------------
# set up what is common to both daily and weekly

## pull out data
agg_daily_long = ts_store["agg_daily_long"]
agg_weekly_long = ts_store["agg_weekly_long"]

## get unique keys for time series
ts_keys = [x[:3] for x in agg_daily_long.index.values]
ts_keys = list(dict.fromkeys(ts_keys))

## create dataframe to hold params that get a stationary transformation
agg_stn_params1 = pd.DataFrame(ts_keys, columns=["project","access","agent"])
agg_stn_params1["variable"] = np.nan
agg_stn_params1["found_stn"] = np.nan
agg_stn_params1["func"] = np.nan
agg_stn_params1["d"] = np.nan
agg_stn_params1["D"] = np.nan
agg_stn_params1["S"] = np.nan
agg_stn_params1.set_index(["project","access","agent"], inplace=True)

agg_stn_params2 = agg_stn_params1.copy()
agg_stn_params3 = agg_stn_params1.copy()
agg_stn_params4 = agg_stn_params1.copy()

agg_stn_params1["variable"] = "daily_level_shrtAdj"
agg_stn_params2["variable"] = "daily_wowGr_shrtAdj"
agg_stn_params3["variable"] = "weekly_level_shrtAdj"
agg_stn_params4["variable"] = "weekly_wowGr_shrtAdj"

agg_stn_params = pd.concat([agg_stn_params1, agg_stn_params2, agg_stn_params3, agg_stn_params4])

agg_stn_params.set_index("variable", append=True, inplace=True)

del agg_stn_params1, agg_stn_params2, agg_stn_params3, agg_stn_params4

## sort
agg_stn_params.sort_index(inplace=True)

# set seasonality
S_daily = 7
S_weekly = 1

# loop through each time series and find stationary transformation for daily level and daily growth, and same for weekly
stnlist_daily_level = []
stnlist_daily_wowGr = []

stnlist_weekly_level = []
stnlist_weekly_wowGr = []

#for ts_key in [ts_keys[10]]:
for ts_key in ts_keys:
    ### pull out untransformed variables for this time series
    ts_daily_level = agg_daily_long.xs(ts_key, level=["project","access","agent"])["daily_level_shrtAdj"]

    ts_daily_wowGr = agg_daily_long.xs(ts_key, level=["project","access","agent"])["daily_wowGr_shrtAdj"]

    ts_weekly_level = agg_weekly_long.xs(ts_key, level=["project","access","agent"])["weekly_level_shrtAdj"]

    ts_weekly_wowGr = agg_weekly_long.xs(ts_key, level=["project","access","agent"])["weekly_wowGr_shrtAdj"]

    ### get stationary transformations
    retobj_daily_level = find_stationary_ts_considerFunctional(ts_daily_level, S_daily, consider_log=True)

    retobj_daily_wowGr = find_stationary_ts_considerFunctional(ts_daily_wowGr[7:], S_daily, consider_log=False)

    retobj_weekly_level = find_stationary_ts_considerFunctional(ts_weekly_level, S_weekly, consider_log=True)

    retobj_weekly_wowGr = find_stationary_ts_considerFunctional(ts_weekly_wowGr[1:], S_weekly, consider_log=False)

    ### store the params and time series
    #### define keys
    k_daily_level = list(ts_key)
    k_daily_level.append("daily_level_shrtAdj")
    k_daily_level = tuple(k_daily_level)

    k_daily_wowGr = list(ts_key)
    k_daily_wowGr.append("daily_wowGr_shrtAdj")
    k_daily_wowGr = tuple(k_daily_wowGr)

    k_weekly_level = list(ts_key)
    k_weekly_level.append("weekly_level_shrtAdj")
    k_weekly_level = tuple(k_weekly_level)

    k_weekly_wowGr = list(ts_key)
    k_weekly_wowGr.append("weekly_wowGr_shrtAdj")
    k_weekly_wowGr = tuple(k_weekly_wowGr)

    #### store params
    agg_stn_params.loc[k_daily_level, ["found_stn","func","d","D","S"]] = retobj_daily_level[:5]

    agg_stn_params.loc[k_daily_wowGr, ["found_stn","func","d","D","S"]] = retobj_daily_wowGr[:5]

    agg_stn_params.loc[k_weekly_level, ["found_stn","func","d","D","S"]] = retobj_weekly_level[:5]

    agg_stn_params.loc[k_weekly_wowGr, ["found_stn","func","d","D","S"]] = retobj_weekly_wowGr[:5]

    ### store the stationary time series in a list
    #### daily level
    ts_stn_daily_level = retobj_daily_level[5]

    ts_stn_daily_level = ts_stn_daily_level.to_frame()
    ts_stn_daily_level["project"] = k_daily_level[0]
    ts_stn_daily_level["access"] = k_daily_level[1]
    ts_stn_daily_level["agent"] = k_daily_level[2]
    ts_stn_daily_level.set_index(["project","access","agent"], append=True, inplace=True)
    ts_stn_daily_level = ts_stn_daily_level.reorder_levels(["project","access","agent","time_d"])

    stnlist_daily_level.append(ts_stn_daily_level)

    #### daily growth
    ts_stn_daily_wowGr = pd.concat([pd.Series(np.repeat(np.nan, 7), index=ts_daily_wowGr.index[:7]), retobj_daily_wowGr[5]])
    ts_stn_daily_wowGr.name = retobj_daily_wowGr[5].name

    ts_stn_daily_wowGr = ts_stn_daily_wowGr.to_frame()
    ts_stn_daily_wowGr["project"] = k_daily_wowGr[0]
    ts_stn_daily_wowGr["access"] = k_daily_wowGr[1]
    ts_stn_daily_wowGr["agent"] = k_daily_wowGr[2]
    ts_stn_daily_wowGr.set_index(["project","access","agent"], append=True, inplace=True)
    ts_stn_daily_wowGr = ts_stn_daily_wowGr.reorder_levels(["project","access","agent","time_d"])

    stnlist_daily_wowGr.append(ts_stn_daily_wowGr)

    #### weekly level
    ts_stn_weekly_level = retobj_weekly_level[5]

    ts_stn_weekly_level = ts_stn_weekly_level.to_frame()
    ts_stn_weekly_level["project"] = k_weekly_level[0]
    ts_stn_weekly_level["access"] = k_weekly_level[1]
    ts_stn_weekly_level["agent"] = k_weekly_level[2]
    ts_stn_weekly_level.set_index(["project","access","agent"], append=True, inplace=True)
    ts_stn_weekly_level = ts_stn_weekly_level.reorder_levels(["project","access","agent","time_w"])

    stnlist_weekly_level.append(ts_stn_weekly_level)


    #### weekly growth
    ts_stn_weekly_wowGr = pd.concat([pd.Series(np.repeat(np.nan, 1), index=ts_weekly_wowGr.index[:1]), retobj_weekly_wowGr[5]])
    ts_stn_weekly_wowGr.name = retobj_weekly_wowGr[5].name

    ts_stn_weekly_wowGr = ts_stn_weekly_wowGr.to_frame()
    ts_stn_weekly_wowGr["project"] = k_weekly_wowGr[0]
    ts_stn_weekly_wowGr["access"] = k_weekly_wowGr[1]
    ts_stn_weekly_wowGr["agent"] = k_weekly_wowGr[2]
    ts_stn_weekly_wowGr.set_index(["project","access","agent"], append=True, inplace=True)
    ts_stn_weekly_wowGr = ts_stn_weekly_wowGr.reorder_levels(["project","access","agent","time_w"])

    stnlist_weekly_wowGr.append(ts_stn_weekly_wowGr)


# turn list of stationary time series into dataframe
agg_daily_stn_long = pd.concat(
    [pd.concat(stnlist_daily_level, axis=0), pd.concat(stnlist_daily_wowGr, axis=0)],
    axis=1)

agg_weekly_stn_long = pd.concat(
    [pd.concat(stnlist_weekly_level, axis=0), pd.concat(stnlist_weekly_wowGr, axis=0)],
    axis=1)

del stnlist_daily_level, stnlist_daily_wowGr, stnlist_weekly_level, stnlist_weekly_wowGr

assert(len(agg_daily_stn_long) == len(agg_daily_long))
assert(len(agg_weekly_stn_long) == len(agg_weekly_long))

# store
## params
agg_stn_params["found_stn"] = agg_stn_params["found_stn"].astype("bool")

if np.sum(np.isnan(agg_stn_params["d"]))==0:
    agg_stn_params["d"] = agg_stn_params["d"].astype("int")

if np.sum(np.isnan(agg_stn_params["D"]))==0:
    agg_stn_params["D"] = agg_stn_params["D"].astype("int")

if np.sum(np.isnan(agg_stn_params["S"]))==0:
    agg_stn_params["S"] = agg_stn_params["S"].astype("int")

ts_store.put(key="agg_stn_params", value=agg_stn_params, format="table")

## time series
ts_store.put(key="agg_daily_stn_long", value=agg_daily_stn_long, format="table")
ts_store.put(key="agg_weekly_stn_long", value=agg_weekly_stn_long, format="table")

# cleanup for this section
del agg_daily_long, agg_weekly_long
del agg_daily_stn_long, agg_weekly_stn_long
del agg_stn_params



# Find Stationary Time Series for Weekly Individual Series
# --------------------------------------------------------
# set up for individual weekly time series

## pull out data
ts_weekly_long = ts_store["ts_weekly_long"]

## get unique keys for time series
ts_keys = [x[0] for x in ts_weekly_long.index.values]
ts_keys = list(dict.fromkeys(ts_keys))

## create dataframe to hold params that get a stationary transformation
ts_weekly_stn_params1 = pd.DataFrame(ts_keys, columns=["ts_id"])
ts_weekly_stn_params1["variable"] = np.nan
ts_weekly_stn_params1["found_stn"] = np.nan
ts_weekly_stn_params1["func"] = np.nan
ts_weekly_stn_params1["d"] = np.nan
ts_weekly_stn_params1["D"] = np.nan
ts_weekly_stn_params1["S"] = np.nan
ts_weekly_stn_params1.set_index(["ts_id"], inplace=True)

ts_weekly_stn_params2 = ts_weekly_stn_params1.copy()

ts_weekly_stn_params1["variable"] = "weekly_level"
ts_weekly_stn_params2["variable"] = "weekly_wowGr"

ts_weekly_stn_params = pd.concat([ts_weekly_stn_params1, ts_weekly_stn_params2])

ts_weekly_stn_params.set_index("variable", append=True, inplace=True)

del ts_weekly_stn_params1, ts_weekly_stn_params2

## sort
ts_weekly_stn_params.sort_index(inplace=True)


# init progress bar
widgets = ["Progress for {} time series: ".format(len(ts_keys)), pb.Percentage(), ' ',
        pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
pbtimer = pb.ProgressBar(widgets=widgets, maxval=len(ts_keys)).start()

i = 0

# set seasonality
S_weekly = 1

# loop through each time series and find stationary transformation for daily level and daily growth, and same for weekly
t0 = time.time()

stnlist_weekly_level = []
stnlist_weekly_wowGr = []

for ts_key in ts_keys:
    ### pull out untransformed variables for this time series
    ts_weekly_level = ts_weekly_long.xs(ts_key, level="ts_id")["weekly_level"]

    ts_weekly_wowGr = ts_weekly_long.xs(ts_key, level="ts_id")["weekly_wowGr"]

    ### get stationary transformations
    retobj_weekly_level = find_stationary_ts_considerFunctional(ts_weekly_level, S_weekly, consider_log=True)

    retobj_weekly_wowGr = find_stationary_ts_considerFunctional(ts_weekly_wowGr[1:], S_weekly, consider_log=False)

    ### store the params and time series
    #### define keys
    k_weekly_level = (ts_key, "weekly_level")

    k_weekly_wowGr = (ts_key, "weekly_wowGr")

    #### store params
    ts_weekly_stn_params.loc[k_weekly_level, ["found_stn","func","d","D","S"]] = retobj_weekly_level[:5]

    ts_weekly_stn_params.loc[k_weekly_wowGr, ["found_stn","func","d","D","S"]] = retobj_weekly_wowGr[:5]

    ### store the stationary time series in a list
    #### weekly level
    ts_stn_weekly_level = retobj_weekly_level[5]

    ts_stn_weekly_level = ts_stn_weekly_level.to_frame()
    ts_stn_weekly_level["ts_id"] = k_weekly_level[0]
    ts_stn_weekly_level.set_index(["ts_id"], append=True, inplace=True)
    ts_stn_weekly_level = ts_stn_weekly_level.reorder_levels(["ts_id","time_w"])

    stnlist_weekly_level.append(ts_stn_weekly_level)


    #### weekly growth
    ts_stn_weekly_wowGr = pd.concat([pd.Series(np.repeat(np.nan, 1), index=ts_weekly_wowGr.index[:1]), retobj_weekly_wowGr[5]])
    ts_stn_weekly_wowGr.name = retobj_weekly_wowGr[5].name

    ts_stn_weekly_wowGr = ts_stn_weekly_wowGr.to_frame()
    ts_stn_weekly_wowGr["ts_id"] = k_weekly_wowGr[0]
    ts_stn_weekly_wowGr.set_index(["ts_id"], append=True, inplace=True)
    ts_stn_weekly_wowGr = ts_stn_weekly_wowGr.reorder_levels(["ts_id","time_w"])

    stnlist_weekly_wowGr.append(ts_stn_weekly_wowGr)

    ### update progress bar
    i += 1

    if (i % 100 == 0) or (i==len(ts_keys)):
        pbtimer.update(i)


# turn list of stationary time series into dataframe
ts_weekly_stn_long = pd.concat(
    [pd.concat(stnlist_weekly_level, axis=0), pd.concat(stnlist_weekly_wowGr, axis=0)],
    axis=1)

del stnlist_weekly_level, stnlist_weekly_wowGr

assert(len(ts_weekly_stn_long) == len(ts_weekly_long))

# store
## params
ts_weekly_stn_params["found_stn"] = ts_weekly_stn_params["found_stn"].astype("bool")

if np.sum(np.isnan(ts_weekly_stn_params["d"]))==0:
    ts_weekly_stn_params["d"] = ts_weekly_stn_params["d"].astype("int")

if np.sum(np.isnan(ts_weekly_stn_params["D"]))==0:
    ts_weekly_stn_params["D"] = ts_weekly_stn_params["D"].astype("int")

if np.sum(np.isnan(ts_weekly_stn_params["S"]))==0:
    ts_weekly_stn_params["S"] = ts_weekly_stn_params["S"].astype("int")

ts_store.put(key="ts_weekly_stn_params", value=ts_weekly_stn_params, format="table")

## time series
ts_store.put(key="ts_weekly_stn_long", value=ts_weekly_stn_long, format="table", complevel=9)

# print timing info
tdelta = time.time() - t0
print("Weekly stationary time series found at rate of " + str(round(tdelta/len(ts_keys), 4)) + " seconds per time series.")

# cleanup for this section
del ts_weekly_long
del ts_weekly_stn_long
del ts_weekly_stn_params

gc.collect()


# Find Stationary Time Series for Daily Individual Series
# -------------------------------------------------------
# set up constant data

## seasonality
S_daily = 7

## remove potentially stale data
try:
    del ts_store["ts_daily_stn_params"]
except:
    pass

try:
    del ts_store["ts_daily_stn_long"]
except:
    pass

## get counts
num_t = len(ts_store.select(key="ts_daily_long", where="ts_id==%i" % ts_store["page_to_id_map"].index.values[0]))
assert(ts_store.get_storer("ts_daily_long").nrows % num_t == 0)

num_n = ts_store.get_storer("ts_daily_long").nrows

num_timeseries = num_n/num_t
num_timeseries = num_timeseries.astype(int)

## set iteration params
NUM_TS_PER_CHUNK = int(round(TOTAL_MEM_GB * 100))
chunksize = NUM_TS_PER_CHUNK * num_t

# set up timing and progress bar
t0 = time.time()

widgets = ["Progress for {} time series: ".format(num_timeseries), pb.Percentage(), ' ',
        pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
pbtimer = pb.ProgressBar(widgets=widgets, maxval=num_timeseries).start()

i = 0

# iterate by chunk
iter = ts_store.select("ts_daily_long", columns=["daily_level","daily_wowGr"], iterator=True, chunksize=chunksize)
for ts_daily_long_chunk in iter:
    ## represent component data into lists that can be fed into a worker function as a list comprehension
    ts_daily_id_vec = []
    ts_daily_level_vec = []
    ts_daily_wowGr_vec = []

    def helper1(df):
        ts_daily_id_vec.append(df.index.get_level_values("ts_id")[0])
        ts_daily_level_vec.append(df["daily_level"])
        ts_daily_wowGr_vec.append(df["daily_wowGr"])
        return(None)

    dumret = ts_daily_long_chunk.groupby(level="ts_id", sort=False).apply(helper1)

    del dumret
    del helper1
    del ts_daily_long_chunk

    ## find stationary
    stnlist_daily_level = [find_stationary_ts_considerFunctional(ts, S_daily, consider_log=True) for ts in ts_daily_level_vec]
    stnlist_daily_wowGr = [find_stationary_ts_considerFunctional(ts[7:], S_daily, consider_log=False) for ts in ts_daily_wowGr_vec]

    del ts_daily_level_vec, ts_daily_wowGr_vec

    ## organize dataframe of stationary time series
    ts_daily_stn_long_chunk1 = pd.concat([x[5] for x in stnlist_daily_level], axis=0)
    ts_daily_stn_long_chunk2 = pd.concat([x[5] for x in stnlist_daily_wowGr], axis=0)

    ts_daily_stn_long_chunk = pd.concat([ts_daily_stn_long_chunk1, ts_daily_stn_long_chunk2], axis=1)

    del ts_daily_stn_long_chunk1, ts_daily_stn_long_chunk2

    ## organize dataframe of stationary params
    ts_daily_stn_params1 = pd.DataFrame([x[:5] for x in stnlist_daily_level], columns=["found_stn","func","d","D","S"])
    ts_daily_stn_params1["ts_id"] = ts_daily_id_vec
    ts_daily_stn_params1["variable"] = "daily_level"

    ts_daily_stn_params2 = pd.DataFrame([x[:5] for x in stnlist_daily_wowGr], columns=["found_stn","func","d","D","S"])
    ts_daily_stn_params2["ts_id"] = ts_daily_id_vec
    ts_daily_stn_params2["variable"] = "daily_wowGr"

    ts_daily_stn_params_chunk = pd.concat([ts_daily_stn_params1, ts_daily_stn_params2], axis=0)
    ts_daily_stn_params_chunk.set_index(["ts_id","variable"], inplace=True)
    ts_daily_stn_params_chunk.sort_index(inplace=True)

    del stnlist_daily_level, stnlist_daily_wowGr
    del ts_daily_stn_params1, ts_daily_stn_params2
    del ts_daily_id_vec

    ## store this chunk result into hdf5
    #todo
    ts_store.append("ts_daily_stn_long",
            ts_daily_stn_long_chunk,
            format="table",
            index=False,
            expectedrows=num_n,
            complevel=9)

    ts_store.append("ts_daily_stn_params",
            ts_daily_stn_params_chunk,
            format="table",
            index=False,
            expectedrows=num_timeseries*2,
            complevel=9)

    ## update progress bar
    i += NUM_TS_PER_CHUNK
    i = min(i, num_timeseries)

    pbtimer.update(i)

# create hdf5 indices
ts_store.create_table_index("ts_daily_stn_long")
ts_store.create_table_index("ts_daily_stn_params")

# print timing info
tdelta = time.time() - t0
print("Daily stationary time series found at rate of " + str(round(tdelta/num_timeseries, 4)) + " seconds per time series.")

# size expectation checks
assert(ts_store.get_storer("ts_daily_stn_long").nrows == num_n)
assert(ts_store.get_storer("ts_daily_stn_params").nrows == num_timeseries*2)



# Post-processing after both weekly and daily stationary series are found
# -----------------------------------------------------------------------

# append together the time series-level params dataframes for daily and weekly
ts_daily_stn_params = ts_store["ts_daily_stn_params"]
ts_weekly_stn_params = ts_store["ts_weekly_stn_params"]

ts_stn_params = pd.concat([ts_daily_stn_params, ts_weekly_stn_params], axis=0)
ts_stn_params.sort_index(inplace=True)

ts_store.put(key="ts_stn_params", value=ts_stn_params, format="table")

del ts_store["ts_daily_stn_params"]
del ts_store["ts_weekly_stn_params"]


# clean up
# --------

ts_store.close()
