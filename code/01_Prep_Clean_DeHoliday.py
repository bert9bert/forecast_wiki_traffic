

# setup and paths

import numpy as np
import pandas as pd
import pickle
import time
import psutil

import progressbar as pb

data_raw_fldrpath = "/media/disk1/forecast_wiki_traffic/data_raw"
data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"

modeling_data_csv = data_raw_fldrpath + "/train_2.csv"
key_data_csv = data_raw_fldrpath + "/key_2.csv"

TOTAL_MEM_GB = psutil.virtual_memory().total/1024**3

# set up hdf5 storage

ts_store = pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5")


# load and prep modeling data
## load
modeling_data = pd.read_csv(modeling_data_csv)

## format data types
idx = [tuple(x.rsplit("_", 3)) for x in modeling_data["Page"]]
idx = pd.DataFrame(idx, columns=["name","project","access","agent"])

modeling_data = pd.concat([idx, modeling_data], axis=1)
del idx
modeling_data.drop("Page", axis=1, inplace=True)

modeling_data.set_index(["name","project","access","agent"], inplace=True)

modeling_data.sort_index(inplace=True)

modeling_data.columns = pd.to_datetime(modeling_data.columns, format="%Y-%m-%d")


## store untouched daily data as dictionary
ts_daily = dict()

datevec = modeling_data.columns

for row in modeling_data.itertuples():
    k = row[0]

    v = pd.DataFrame(data=list(row[1:]), dtype=np.float64, columns=["daily_untouched"], index=datevec)
    v.index.set_names("time_d", inplace=True)
    v.sort_index(inplace=True)

    ts_daily[k] = v

del modeling_data


# clean daily time series
# 1. impute missing values
# 2. account for outliers
# nb. short series are left in place

## define function that will clean one time series

def daily_clean(df):
    # identify the type of missing
    df = df.assign(missing_type = pd.Categorical(
        values=np.repeat("no_missing", len(df)),
        categories=["no_missing","missing_short","missing_hole"]))

    rows_missing = np.isnan(df["daily_untouched"])
    rows_missing_short = np.cumsum(rows_missing) == (np.array(range(len(df)))+1)

    df.loc[rows_missing, "missing_type"] = "missing_hole"
    df.loc[rows_missing_short, "missing_type"] = "missing_short"

    # identify outliers as obs more than 2 pop std dev from the 30-day center rolling mean
    df = df.assign(outlier = pd.Categorical(
        values=np.repeat("ok", len(df)),
        categories=["ok","outlier"]))

    rolling_mean_t = df.daily_untouched.rolling(window=30, min_periods=1, center=True).mean()
    full_std = np.std(df.daily_untouched)

    df.loc[df.daily_untouched < rolling_mean_t - 2*full_std, "outlier"] = "outlier"
    df.loc[df.daily_untouched > rolling_mean_t + 2*full_std, "outlier"] = "outlier"


    # for holes, impute first using the average of the prior day of week and following day of week
    df["daily_cleaned"] = df.daily_untouched
    df.loc[df.outlier=="outlier", "daily_cleaned"] = np.nan

    impute_values = np.mean(
        np.array([np.append(np.repeat(np.nan, 7), df.daily_cleaned[:-7]),
                  np.append(df.daily_cleaned[7:], np.repeat(np.nan, 7))]),
        axis=0)

    df.loc[df.missing_type=="missing_hole", "daily_cleaned"] = impute_values[df.missing_type=="missing_hole"]
    df.loc[df.outlier=="outlier", "daily_cleaned"] = impute_values[df.outlier=="outlier"]

    # for holes, second impute by using the rolling average
    df.loc[np.isnan(df.daily_cleaned) & (df.missing_type=="missing_hole"), "daily_cleaned"] = \
        rolling_mean_t[np.isnan(df.daily_cleaned) & (df.missing_type=="missing_hole")]
    df.loc[np.isnan(df.daily_cleaned) & (df.outlier=="outlier"), "daily_cleaned"] = \
        rolling_mean_t[np.isnan(df.daily_cleaned) & (df.outlier=="outlier")]

    # for holes, impute third by forward filling

    df[["daily_cleaned"]] = df[["daily_cleaned"]].fillna(method="ffill")

    # there should be no negative view counts, make negative view counts zero
    df.loc[df.daily_cleaned < 0, "daily_cleaned"] = 0

    # make sure there are no missing values
    isprob = np.isnan(df.daily_cleaned[df.missing_type!="missing_short"])
    if np.sum(isprob) > 0:
        print(isprob)
        print(df.loc["2015-09"])
        print(df.loc["2015-10"])
        raise Exception("There are missing values in the clean daily series.")

    return(df)


## apply the cleaning function to all the time series, and store in hdf5

### create map from page to id
page_to_id_map = pd.DataFrame(list(ts_daily.keys()), columns=["name","project","access","agent"])
page_to_id_map.index.set_names(["ts_id"], inplace=True)

try:
    del ts_store["page_to_id_map"]
except:
    pass

ts_store.put("page_to_id_map", page_to_id_map)


### loop through pages to clean and store
#### delete hdf5 node if it already exists
try:
    del ts_store["ts_daily_long_tmp1"]
except:
    pass

#### get length of the component time series
num_t = len(ts_daily[list(ts_daily.keys())[0]])

#### init progress bar
widgets = ["Cleaning Data -- Progress for {} time series: ".format(len(page_to_id_map)), pb.Percentage(), ' ',
        pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
pbtimer = pb.ProgressBar(widgets=widgets, maxval=len(page_to_id_map)).start()

t0 = time.time()

#### loop through and store by chunks
CHUNKSIZE = int(round(TOTAL_MEM_GB * 100))
chunk_list = []

i = 0
idx = page_to_id_map.index
for ts_id in idx:
    #### clean this chunk and add to holding list
    k = tuple(page_to_id_map.loc[ts_id])

    ts_daily_cleaned_this = daily_clean(df=ts_daily[k])
    ts_daily_cleaned_this["ts_id"] = ts_id
    ts_daily_cleaned_this = ts_daily_cleaned_this.reset_index().set_index(["ts_id","time_d"])

    chunk_list.append(ts_daily_cleaned_this)

    i+=1

    #### when the desired max size for this chunk has been reached,
    #### store it and reset the holding list
    if i % CHUNKSIZE == 0:
        ts_store.append("ts_daily_long_tmp1",
                pd.concat(chunk_list),
                format="table",
                index=False,
                expectedrows=num_t*len(idx),
                complevel=9)

        chunk_list = []

        pbtimer.update(i)

#### in case the last chunk does not reach the max size, make sure that
#### this last chunk is still saved
if i % CHUNKSIZE > 0:
    ts_store.append("ts_daily_long_tmp1",
            pd.concat(chunk_list),
            format="table",
            index=False,
            expectedrows=num_t*len(idx),
            complevel=9)

    pbtimer.update(i)

del chunk_list
del ts_daily_cleaned_this

### create an index for this hdf5 table
ts_store.create_table_index("ts_daily_long_tmp1")

### output timing info
t1 = time.time()
tdelta = t1 - t0
print("Data cleaned at an average rate of " + str(round(tdelta/len(idx),2)) + " seconds per time series.")

del ts_daily


# data adjustments for holidays


def apply_daily_holiday_adjustments(df, holidays, ts_id, ts_project):
    df = df.copy()

    ### pull out some values for easy access
    #### dates
    d = df.index.get_level_values("time_d")

    #### holiday names
    holiday_names = [x["holiday"] for x in holidays]

    ### initialize holidays variable
    cats = ["regular_day"]
    cats.extend([x + "_adj" for x in holiday_names])
    cats.extend([x + "_noadj" for x in holiday_names])
    df["holiday"] = pd.Categorical(
            values=np.repeat("regular_day", len(df)),
            categories=cats)
    del cats

    ### the daily level is the cleaned time series
    df["daily_level"] = df["daily_cleaned"]

    ### loop through holidays
    holiday_adjustment_factors = []

    for h in holidays:
        holiday_dateonly_i = h["dateonly"]
        holiday_name_i = h["holiday"]
        holiday_projects_i = h["projects"]

        ### if holiday applies to this page's project...
        if (holiday_projects_i is None) or (ts_project in holiday_projects_i):
            holiday_idx_i = (d.month==holiday_dateonly_i[0]) & (d.day==holiday_dateonly_i[1])

            if all(df.loc[holiday_idx_i, "outlier"] == "outlier"):
                df.loc[holiday_idx_i, "holiday"] = holiday_name_i + "_adj"
                adjustment_factor = np.mean(df.loc[holiday_idx_i, "daily_untouched"]/df.loc[holiday_idx_i, "daily_level"])
                holiday_adjustment_factors.extend(
                    [{"ts_id": ts_id, "holiday": holiday_name_i, "dateonly": holiday_dateonly_i, "adjustment_factor": adjustment_factor}]
                )
            else:
                df.loc[holiday_idx_i, "holiday"] = holiday_name_i + "_noadj"

    return df, holiday_adjustment_factors

# for holidays where all holidays in the time series are outliers, save the
# ratio between the cleaned figure to the untouched figure as the
# adjustment factor to apply for projection

try:
    del ts_store["ts_daily_long_tmp2"]
except:
    pass

## define fixed-date holidays
holidays = [
    {"holiday": "chinese_teachers_day", "dateonly": ( 9,10), "projects": ["zh.wikipedia.org"]},
    {"holiday": "halloween", "dateonly": (10,31), "projects": ["en.wikipedia.org"]},
    {"holiday": "toussaint", "dateonly": (11, 1), "projects": ["fr.wikipedia.org"]},
    {"holiday": "christmas", "dateonly": (12,25), "projects": None},
    {"holiday": "new_years", "dateonly": ( 1, 1), "projects": None}
]

## initialize list to hold the holiday adjustment factors
holiday_adjustment_factors = []

## loop through each page and identify holidays that need adjustment and store adjustment factors
page_to_id_map = ts_store["page_to_id_map"]

# page_to_id_map_tup = page_to_id_map.reset_index().itertuples()
# ts_info_vec = [{"ts_id": getattr(x, "ts_id"), "ts_project": getattr(x, "ts_id")} for x in page_to_id_map_tup]
# del page_to_id_map_tup

### get time series length (num_t) and number of obs (num_n)
num_t = len(ts_store.select(key="ts_daily_long_tmp1", where="ts_id==%i" % page_to_id_map.index.values[0]))
assert(ts_store.get_storer("ts_daily_long_tmp1").nrows % num_t == 0)

num_n = ts_store.get_storer("ts_daily_long_tmp1").nrows

### init progress bar
widgets = ["Data Adjustments -- Progress for {} time series: ".format(len(page_to_id_map)), pb.Percentage(), ' ',
        pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
pbtimer = pb.ProgressBar(widgets=widgets, maxval=len(page_to_id_map)).start()

### loop through time series by chunks
NUM_TS_PER_CHUNK = int(round(TOTAL_MEM_GB * 10))
chunksize = NUM_TS_PER_CHUNK * num_t

t0 = time.time()

i = 0

holiday_adjustment_factors = []

iter = ts_store.select("ts_daily_long_tmp1", iterator=True, chunksize=chunksize)
for chunk in iter:
    ts_id_chunkvec = chunk.index.get_level_values("ts_id").unique().tolist()

    chunk_adj = []

    for ts_id in ts_id_chunkvec:
        ts_project = page_to_id_map.loc[ts_id, "project"]
        df = chunk.xs(ts_id, level="ts_id", drop_level=False)

        assert(len(df)==num_t)

        ### apply
        df_this, holiday_adjustment_factors_this = apply_daily_holiday_adjustments(df, holidays, ts_id, ts_project)

        holiday_adjustment_factors.extend(holiday_adjustment_factors_this)
        chunk_adj.extend([df_this])

    chunk_adj = pd.concat(chunk_adj)
    assert(len(chunk) == len(chunk_adj))

    #### store this chunk adjusted for holidays
    ts_store.append("ts_daily_long_tmp2",
                chunk_adj,
                format="table",
                index=False,
                expectedrows=num_n,
                complevel=9)

    i += NUM_TS_PER_CHUNK
    i = min(i, len(page_to_id_map))

    pbtimer.update(i)


ts_store.create_table_index("ts_daily_long_tmp2")

assert(num_n == ts_store.get_storer("ts_daily_long_tmp2").nrows)

t1 = time.time()
tdelta = t1 - t0
print("Data adjusted at an average rate of " + str(round(tdelta/i, 4)) + " seconds per time series.")


### store the holiday adjustment factors
holiday_adjustment_factors = pd.DataFrame(holiday_adjustment_factors)
holiday_adjustment_factors["month"] = [x[0] for x in holiday_adjustment_factors.loc[:,"dateonly"]]
holiday_adjustment_factors["day"] = [x[1] for x in holiday_adjustment_factors.loc[:,"dateonly"]]
holiday_adjustment_factors.drop("dateonly", axis=1, inplace=True)

holiday_adjustment_factors = holiday_adjustment_factors[["holiday","month","day","ts_id","adjustment_factor"]]

ts_store.put(key="holiday_adjustment_factors", value=holiday_adjustment_factors, format="table")

# clean up
assert(ts_store.get_storer("ts_daily_long_tmp1").nrows == ts_store.get_storer("ts_daily_long_tmp2").nrows)

ts_store.close()
