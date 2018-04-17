# setup and paths

import numpy as np
import pandas as pd
import pickle
import time
import psutil

import progressbar as pb

data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"

TOTAL_MEM_GB = psutil.virtual_memory().total/1024**3

# set up hdf5 storage

ts_store = pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5")

# Daily Data Transformations
# --------------------------
# (Calculate the week-over-week growth rate. When the denominator would be zero, assume there was one view for the denominator to avoid infinities.
# Also add the day of the week.)


try:
    del ts_store["ts_daily_long_tmp3"]
except:
    pass

# iterate by chunk and perform daily data transformations
num_t = len(ts_store.select(key="ts_daily_long_tmp2", where="ts_id==%i" % ts_store["page_to_id_map"].index.values[0]))
assert(ts_store.get_storer("ts_daily_long_tmp2").nrows % num_t == 0)

num_n = ts_store.get_storer("ts_daily_long_tmp2").nrows

NUM_TS_PER_CHUNK = int(round(TOTAL_MEM_GB * 100))
chunksize = NUM_TS_PER_CHUNK * num_t

iter = ts_store.select("ts_daily_long_tmp2", iterator=True, chunksize=chunksize)
for chunk in iter:
    chunk_transf = chunk

    ## calculate daily week-over-week growth, treating 0s as 1s
    chunk_transf["daily_level_0to1"] = chunk_transf["daily_level"]
    chunk_transf.loc[chunk_transf.daily_level==0, "daily_level_0to1"] = 1

    chunk_transf["daily_wowGr"] = chunk_transf.groupby(level="ts_id", sort=False)["daily_level_0to1"].pct_change(periods=7)

    chunk_transf.drop("daily_level_0to1", axis=1, inplace=True)

    ## add column for day of the week
    chunk_transf["day_of_week"] = chunk_transf.index.get_level_values("time_d").dayofweek.tolist()

    ## reorder columns
    col_order = [
        'day_of_week',
        'daily_level', 'daily_wowGr', 'holiday',
        'daily_cleaned', 'outlier',
        'daily_untouched', 'missing_type'
    ]
    chunk_transf = chunk_transf[col_order]
    del col_order

    ## store this chunk
    ts_store.append("ts_daily_long_tmp3",
            chunk_transf,
            format="table",
            index=False,
            expectedrows=num_n,
            complevel=9)

del chunk_transf

# create index on stored table
ts_store.create_table_index("ts_daily_long_tmp3")

assert(num_n == ts_store.get_storer("ts_daily_long_tmp3").nrows)


# Weekly Data Aggregations and Transformations
# --------------------------------------------
# Aggregate daily time series to weekly by sum to create weekly level and weekly growth variables.

# define a function that given a time series will produce weekly aggregation and transformations
def wk_agg_helper(ts):
    ts = ts.copy()

    ## make sure that this is only one time series
    my_ts_id = ts.index.get_level_values("ts_id").unique().tolist()
    assert(len(my_ts_id)==1)
    my_ts_id = my_ts_id[0]

    ## only consider complete weeks in the data for aggregation
    offset_top = (7-ts.iloc[0]["day_of_week"]).astype(int)
    offset_btm = ts.iloc[-1]["day_of_week"].astype(int) + 1

    if not offset_top==7:
        ts = ts.iloc[offset_top:,]

    if not offset_btm==7:
        ts = ts.iloc[:-offset_btm,]

    del offset_top, offset_btm

    ## aggregate daily level to weekly level
    ts_subset = ts.xs(my_ts_id, level="ts_id", drop_level=True)["daily_level"]

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


# calculate various rows and counts
## daily series counts
num_t_daily = len(ts_store.select(key="ts_daily_long_tmp3", where="ts_id==%i" % ts_store["page_to_id_map"].index.values[0]))
num_rows_daily_long = ts_store.get_storer("ts_daily_long_tmp3").nrows

## number of time series
num_timeseries = int(num_rows_daily_long/num_t_daily)

assert(num_rows_daily_long % num_t_daily == 0)

## weekly counts
num_t_weekly = len(
    wk_agg_helper(ts_store.select(key="ts_daily_long_tmp3", where="ts_id==%i" % ts_store["page_to_id_map"].index.values[0]))
)
num_rows_weekly_long = num_t_weekly * num_timeseries

# create weekly data

## clear from the hdf5 storage if already exists
try:
    del ts_store["ts_weekly_long_tmp1"]
except:
    pass

## init progress bar
widgets = ["Progress for {} time series: ".format(num_timeseries), pb.Percentage(), ' ',
        pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
pbtimer = pb.ProgressBar(widgets=widgets, maxval=num_timeseries).start()

## loop through time series by chunks
### set up chunk parameters
NUM_TS_PER_CHUNK = int(round(TOTAL_MEM_GB * 100))
chunksize = NUM_TS_PER_CHUNK * num_t_daily

t0 = time.time()

i = 0

### iterate by chunks
iter = ts_store.select("ts_daily_long_tmp3", iterator=True, chunksize=chunksize)
for chunk_daily_i in iter:
    #### get weekly aggregations for this chunk
    chunk_weekly_i = chunk_daily_i.groupby(level="ts_id", sort=False)[["day_of_week","daily_level"]].apply(wk_agg_helper)

    #### store the weekly aggregation
    ts_store.append("ts_weekly_long_tmp1",
            chunk_weekly_i,
            format="table",
            index=False,
            expectedrows=num_rows_weekly_long,
            complevel=9)

    #### update the progress bar
    i += NUM_TS_PER_CHUNK
    i = min(i, num_timeseries)

    pbtimer.update(i)

### create an index for the hdf5 table where the weekly aggregations were stored
ts_store.create_table_index("ts_weekly_long_tmp1")

## print timing info
tdelta = time.time() - t0
print("Weekly data created at an average rate of " + str(round(tdelta/num_timeseries, 4)) + " seconds per time series.")

# clean up
ts_store.close()
