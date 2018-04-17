
# For each time series, compute the following summary statistics both overall for each day of the week:
# * sum of all visits
# * mean
# * mean last 90, 60, 30
# * median
# * median last 90, 60, 30
#
# And calculate
# * average historical proportions

# setup and paths
## prebuilt libraries
import numpy as np
import pandas as pd
import pickle
from collections import namedtuple, OrderedDict
import time
import psutil
import itertools
import more_itertools
import warnings
import gc
from importlib import reload


## custom and project specific libraries
import projutils_backtrans

## paths
data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"

TOTAL_MEM_GB = psutil.virtual_memory().total/1024**3


# set up hdf5 storage
ts_store = pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r")
summary_stats_store = pd.HDFStore(data_intermed_nb_fldrpath + "/summary_stats_store.h5")


# Calculate Summary Stats excl Historical Proportions

## Define a function that given a time series will output summary statistics excluding the historical proportions,
## which need all the time series level calculated first before the historical proportions can be calculated.
def ts_summ(y):
    # pull out time series
    v = "daily_level"
    ts = y.loc[:,v]
    ts.index = ts.index.get_level_values("time_d")

    # subsets
    ## subset to last x days
    ts90 = ts[ts.last_valid_index()-pd.DateOffset(90-1, 'D'):]
    ts60 = ts[ts.last_valid_index()-pd.DateOffset(60-1, 'D'):]
    ts30 = ts[ts.last_valid_index()-pd.DateOffset(30-1, 'D'):]

    ## subset to last 120th to 60th day
    ts_trunc60_last60 = ts[ts.last_valid_index()-pd.DateOffset(120-1, 'D'):ts.last_valid_index()-pd.DateOffset(60, 'D')]

    ## subset from beginning to last 60th day
    ts_trunc60 = ts[:ts.last_valid_index()-pd.DateOffset(60, 'D')]

    # calculate summary statistics
    ## overall
    ts_sum = np.nansum(ts)
    ts_mean = np.nanmean(ts)
    ts_median = np.nanmedian(ts)

    ## last x days
    ts_sum90 = np.nansum(ts90)
    ts_mean90 = np.nanmean(ts90)
    ts_median90 = np.nanmedian(ts90)

    ts_sum60 = np.nansum(ts60)
    ts_mean60 = np.nanmean(ts60)
    ts_median60 = np.nanmedian(ts60)

    ts_sum30 = np.nansum(ts30)
    ts_mean30 = np.nanmean(ts30)
    ts_median30 = np.nanmedian(ts30)

    ## truncate last y days, summary stats on last x days
    ts_trunc60_sum60 = np.nansum(ts_trunc60_last60)
    ts_trunc60_mean60 = np.nanmean(ts_trunc60_last60)
    ts_trunc60_median60 = np.nanmedian(ts_trunc60_last60)

    ## truncate last y days
    ts_trunc60_sum = np.nansum(ts_trunc60)
    ts_trunc60_mean = np.nanmean(ts_trunc60)
    ts_trunc60_median = np.nanmedian(ts_trunc60)

    # return
    df = {"sum":[ts_sum], "mean":[ts_mean], "median":[ts_median],
         "sum90":[ts_sum90], "mean90":[ts_mean90], "median90":[ts_median90],
         "sum60":[ts_sum60], "mean60":[ts_mean60], "median60":[ts_median60],
         "sum30":[ts_sum30], "mean30":[ts_mean30], "median30":[ts_median30],
         "trunc60_sum":[ts_trunc60_sum], "trunc60_mean":[ts_trunc60_mean], "trunc60_median":[ts_trunc60_median],
         "trunc60_sum60":[ts_trunc60_sum60], "trunc60_mean60":[ts_trunc60_mean60], "trunc60_median60":[ts_trunc60_median60]}
    df = pd.DataFrame(df)
    return(df)


## Calculate summary statistics by chunk.
## get counts
num_t = len(ts_store.select(key="ts_daily_long", where="ts_id==%i" % ts_store["page_to_id_map"].index.values[0]))
assert(ts_store.get_storer("ts_daily_long").nrows % num_t == 0)

num_n = ts_store.get_storer("ts_daily_long").nrows

num_timeseries = num_n/num_t
num_timeseries = num_timeseries.astype(int)

## set iteration params
NUM_TS_PER_CHUNK = int(round(TOTAL_MEM_GB * 100))
chunksize = NUM_TS_PER_CHUNK * num_t

ts_stat_overall = []
ts_stat_dayofweek = []

iter = ts_store.select("ts_daily_long", columns=["day_of_week","daily_level"], iterator=True, chunksize=chunksize)
for chunk in iter:
    chunk_stat_overall = chunk.groupby(by="ts_id").apply(ts_summ)
    chunk_stat_overall.reset_index(level=-1, drop=True, inplace=True)

    chunk_stat_dayofweek = chunk.groupby(by=["ts_id","day_of_week"]).apply(ts_summ)
    chunk_stat_dayofweek.reset_index(level=-1, drop=True, inplace=True)

    ts_stat_overall.append(chunk_stat_overall)
    ts_stat_dayofweek.append(chunk_stat_dayofweek)

ts_stat_overall = pd.concat(ts_stat_overall, axis=0)
ts_stat_dayofweek = pd.concat(ts_stat_dayofweek, axis=0)

assert(len(ts_stat_overall) == num_timeseries)
assert(len(ts_stat_dayofweek) == num_timeseries * 7)

summary_stats_store.put(key="ts_stat_overall", value=ts_stat_overall, format="table", complevel=9)
summary_stats_store.put(key="ts_stat_dayofweek", value=ts_stat_dayofweek, format="table", complevel=9)


# Calculate Historical Proportions for allocating aggregate predictions to time series

# merge in aggregation categories
page_to_id_map = ts_store["page_to_id_map"]

## overall data
ts_hist_prop = ts_stat_overall[["sum","sum30","sum60","sum90","trunc60_sum","trunc60_sum60"]].merge(
    right=page_to_id_map[["project","access","agent"]],
    how="left",
    left_index=True, right_index=True,
    sort=False)

assert(len(ts_hist_prop)==len(ts_stat_overall))

## day of week data
ts_hist_prop_dayofweek = ts_stat_dayofweek[["sum","sum30","sum60","sum90","trunc60_sum","trunc60_sum60"]].merge(
    right=page_to_id_map[["project","access","agent"]],
    how="left",
    left_index=True, right_index=True,
    sort=False)

assert(len(ts_hist_prop_dayofweek)==len(ts_stat_dayofweek))


del page_to_id_map

# calculate aggregated sums
## overall data
agg_stat_overall = ts_hist_prop\
    .groupby(["project","access","agent"])\
    .sum()

## day of week data
agg_stat_dayofweek = ts_hist_prop_dayofweek\
    .reset_index(level="day_of_week")\
    .groupby(["project","access","agent","day_of_week"])\
    .sum()

## save
summary_stats_store.put(key="agg_stat_overall", value=agg_stat_overall, format="table", complevel=9)
summary_stats_store.put(key="agg_stat_dayofweek", value=agg_stat_dayofweek, format="table", complevel=9)

# merge back to ts for overall

ts_hist_prop = ts_hist_prop.reset_index().merge(
    right = agg_stat_overall.reset_index(),
    how = "left",
    on = ["project","access","agent"],
    sort=False,
    suffixes=("_ts","_agg")
)
ts_hist_prop.set_index("ts_id", inplace=True)

assert(len(ts_hist_prop)==len(ts_stat_overall))

# calculate historical proportions for allocation of aggregate predictions to time series level
ts_hist_prop["hist_prop"] = ts_hist_prop.sum_ts / ts_hist_prop.sum_agg
ts_hist_prop["hist_prop_last90days"] = ts_hist_prop.sum90_ts / ts_hist_prop.sum90_agg
ts_hist_prop["hist_prop_last60days"] = ts_hist_prop.sum60_ts / ts_hist_prop.sum60_agg
ts_hist_prop["hist_prop_last30days"] = ts_hist_prop.sum30_ts / ts_hist_prop.sum30_agg
ts_hist_prop["hist_prop_trunc60"] = ts_hist_prop.trunc60_sum_ts / ts_hist_prop.trunc60_sum_agg
ts_hist_prop["hist_prop_trunc60_last60days"] = ts_hist_prop.trunc60_sum60_ts / ts_hist_prop.trunc60_sum60_agg

ts_hist_prop = ts_hist_prop[["project","access","agent",
                             "hist_prop","hist_prop_last90days","hist_prop_last60days","hist_prop_last30days",
                             "hist_prop_trunc60","hist_prop_trunc60_last60days"]]

# save
summary_stats_store.put(key="ts_hist_prop", value=ts_hist_prop, format="table", complevel=9)

# checks
TOL = 1e-6
check1 = ts_hist_prop.groupby(["project","access","agent"]).sum()
assert(all((check1 > 1-TOL) & (check1 < 1+TOL)))

# clean up
ts_store.close()
summary_stats_store.close()
