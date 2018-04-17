# setup and paths

import numpy as np
import pandas as pd
import pickle
import time

data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"

# set up hdf5 storage

ts_store = pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5")

# Create Topline Data Aggregations
# ----------------------------------------------------------------
# Aggregate time series by summing up to the level of project by access by agent.
#
# Some of the component time series are short. To account for the downward bias:
#
# 1. Take the global median of all time series.
# 2. At each date, scale the summed value of the available series up by multiplying by the ratio of the sum of all global medians to sum of available medians.

## Topline Daily Aggregations

# define a function that given a chunk of time series in the same aggregation pool, return
# 1. sum of the daily level
# 2. sum of the medians of the time series that contribute to the sum at each date
# 3. '' that do not contribute ''
# 4. sum of the daily level that is untouched
def ts_aggregate_shrtAdj_chunk(chunk):
    ts_contribs = chunk.groupby(level="ts_id")["daily_level"].median(skipna=True).to_frame(name="contrib")

    chunk = chunk.join(ts_contribs, how="left", sort=False)

    chunk["contrib_ok"] = chunk["contrib"]
    chunk.rename(columns={"contrib":"contrib_short"}, inplace=True)
    chunk.loc[chunk.missing_type=="missing_short", "contrib_ok"]  = 0
    chunk.loc[chunk.missing_type!="missing_short", "contrib_short"]  = 0

    chunk_agg = chunk.groupby(level="time_d")["daily_level","contrib_short","contrib_ok","daily_untouched"].sum(skipna=True)
    chunk_agg.rename(columns={"daily_level":"daily_level_noShrtAdj", "daily_untouched":"daily_untouched_noShrtAdj"}, inplace=True)

    return(chunk_agg)

# iterate through chunks
## define maps for the aggregation pools
agg_id_map = ts_store["page_to_id_map"].reset_index().set_index(["project","access","agent"])
agg_id_map.sort_index(inplace=True)

agg_id_unique = agg_id_map.index.unique().tolist()

## init list to store daily aggregation for each aggregation pool
chunk_shrtAdj_list = []

## loop through by each aggregation pool
for agg_id in agg_id_unique:
    ### get the time series ids associated with this aggregation pool
    agg_ts_id_list = agg_id_map.loc[agg_id, "ts_id"].tolist()

    ### pull those time series from storage
    keepcols = ["daily_level","daily_untouched","missing_type"]
    chunk = [ts_store.select("ts_daily_long_tmp3", where="ts_id==%i" % ts_id, columns=keepcols) for ts_id in agg_ts_id_list]
    chunk = pd.concat(chunk)
    del keepcols

    ### perform aggregation
    chunk_shrtAdj = ts_aggregate_shrtAdj_chunk(chunk)

    ### store in list
    chunk_shrtAdj["project"] = agg_id[0]
    chunk_shrtAdj["access"] = agg_id[1]
    chunk_shrtAdj["agent"] = agg_id[2]
    chunk_shrtAdj.set_index(["project","access","agent"], append=True, inplace=True)
    chunk_shrtAdj = chunk_shrtAdj.reorder_levels(["project","access","agent","time_d"])

    chunk_shrtAdj_list.append(chunk_shrtAdj)

## concat aggregations for each aggregation pool into one dataframe
chunk_shrtAdj = pd.concat(chunk_shrtAdj_list)

agg_daily_long = chunk_shrtAdj
del chunk_shrtAdj


# calculate daily level with the adjustment for short time series
agg_daily_long["shrtAdj_factor"] = (agg_daily_long.contrib_short + agg_daily_long.contrib_ok)/agg_daily_long.contrib_ok
agg_daily_long.drop(["contrib_short","contrib_ok"], axis=1, inplace=True)

agg_daily_long["daily_level_shrtAdj"] = agg_daily_long.daily_level_noShrtAdj * agg_daily_long.shrtAdj_factor


## calculate daily week-over-week growth, treating 0s as 1s
agg_daily_long["daily_level_shrtAdj_0to1"] = agg_daily_long["daily_level_shrtAdj"]
agg_daily_long.loc[agg_daily_long.daily_level_shrtAdj==0, "daily_level_shrtAdj_0to1"] = 1

agg_daily_long["daily_wowGr_shrtAdj"] = agg_daily_long\
    .groupby(level=["project","access","agent"], sort=False)["daily_level_shrtAdj_0to1"]\
    .pct_change(periods=7)

agg_daily_long.drop("daily_level_shrtAdj_0to1", axis=1, inplace=True)

## add column for day of the week
agg_daily_long["day_of_week"] = agg_daily_long.index.get_level_values("time_d").dayofweek.tolist()

## reorder columns
col_order = [
    'day_of_week',
    'daily_level_shrtAdj', 'daily_wowGr_shrtAdj',
    'shrtAdj_factor',
    'daily_level_noShrtAdj', 'daily_untouched_noShrtAdj'
]
agg_daily_long = agg_daily_long[col_order]
del col_order

# store
ts_store.put(key="agg_daily_long", value=agg_daily_long, format="table")


## Topline Weekley Aggregations

# define a function that given a time series will produce weekly aggregation and transformations
def wk_agg_helper2(ts):
    ## keep only time index since the other indices should be the same
    ts = ts.reset_index()
    ts = ts.set_index("time_d")

    ## only consider complete weeks in the data for aggregation
    offset_top = (7-ts.iloc[0]["day_of_week"]).astype(int)
    offset_btm = ts.iloc[-1]["day_of_week"].astype(int) + 1

    if not offset_top==7:
        ts = ts.iloc[offset_top:,]

    if not offset_btm==7:
        ts = ts.iloc[:-offset_btm,]

    del offset_top, offset_btm

    ## aggregate daily level to weekly level
    ts_subset = ts["daily_level"]

    ts_weekly = ts_subset.resample("W").sum()
    ts_weekly = ts_weekly.to_frame(name="weekly_level")

    ## calculate weekly week-over-week growth, treating 0s as 1s
    ts_weekly["weekly_level_0to1"] = ts_weekly["weekly_level"]
    ts_weekly.loc[ts_weekly.weekly_level==0, "weekly_level_0to1"] = 1

    ts_weekly["weekly_wowGr"] = ts_weekly["weekly_level_0to1"].pct_change(periods=1)

    ts_weekly.drop("weekly_level_0to1", axis=1, inplace=True)

    ## change index name to signify that this is a weekly series
    ts_weekly.index.names = ["time_w"]

    return(ts_weekly)

# create the weekly topline aggregations
## convert daily level adjusted for short series to weekly level adjusted for short series, and
## base the weekly growth adjusted for short series off that
agg_weekly_long1 = agg_daily_long\
    .rename(columns={"daily_level_shrtAdj":"daily_level"})[["day_of_week","daily_level"]]\
    .groupby(level=["project","access","agent"], sort=False)\
    .apply(wk_agg_helper2)\
    .rename(columns={"weekly_level":"weekly_level_shrtAdj", "weekly_wowGr":"weekly_wowGr_shrtAdj"})

## convert daily level without adjustment to weekly level without adjustment
agg_weekly_long2 = agg_daily_long\
    .rename(columns={"daily_level_noShrtAdj":"daily_level"})[["day_of_week","daily_level"]]\
    .groupby(level=["project","access","agent"], sort=False)\
    .apply(wk_agg_helper2)\
    .rename(columns={"weekly_level":"weekly_level_noShrtAdj"})\
    .drop("weekly_wowGr", axis=1)

## convert daily untouched to weekly untouched
agg_weekly_long3 = agg_daily_long\
    .rename(columns={"daily_untouched_noShrtAdj":"daily_level"})[["day_of_week","daily_level"]]\
    .groupby(level=["project","access","agent"], sort=False)\
    .apply(wk_agg_helper2)\
    .rename(columns={"weekly_level":"weekly_untouched_noShrtAdj"})\
    .drop("weekly_wowGr", axis=1)

## put these together into one dataframe
agg_weekly_long = agg_weekly_long1.merge(agg_weekly_long2, left_index=True, right_index=True, sort=False)
agg_weekly_long = agg_weekly_long.merge(agg_weekly_long3, left_index=True, right_index=True, sort=False)

assert(len(agg_weekly_long)==len(agg_weekly_long1))
assert(len(agg_weekly_long)==len(agg_weekly_long2))

del agg_weekly_long1, agg_weekly_long2, agg_weekly_long3

# store
ts_store.put(key="agg_weekly_long", value=agg_weekly_long, format="table")

# clean up
del ts_store["ts_daily_long_tmp1"]
del ts_store["ts_daily_long_tmp2"]

ts_store.get_node("ts_daily_long_tmp3")._f_rename("ts_daily_long")
ts_store.get_node("ts_weekly_long_tmp1")._f_rename("ts_weekly_long")

ts_store.close()
