# setup and paths
## prebuilt libraries
import numpy as np
import pandas as pd
import pickle
from collections import namedtuple, OrderedDict
import time
import itertools
import more_itertools
import warnings
import gc
from importlib import reload


## custom and project specific libraries
import projutils_ts
import helperfns_sarima_mapreduce

## paths
data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"

# set up hdf5 storage
ts_store = pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5")
sarima_agg_store = pd.HDFStore(data_intermed_nb_fldrpath + "/sarima_agg_store.h5")


# Fit Topline Models

## Fit Models
agg_stn_params = ts_store.select("agg_stn_params")

# aggregated daily level
sarima_fits_agg_daily_level = dict()

x = helperfns_sarima_mapreduce.mapperfn_reducerfn(
    store = ts_store,
    tbl_series = "agg_daily_long",
    stn_params_df = agg_stn_params,
    var = "daily_level_shrtAdj",
    freq = "D"
)

x[0]["include_intercept"] = x[0]["include_intercept"].astype(np.float)
x[0]["var"] = "daily_level_shrtAdj"
x[0].set_index("var", append=True, inplace=True)
sarima_fits_agg_daily_level["sarima_model_spec_df"] = x[0]

sarima_fits_agg_daily_level["fitted_df"] = x[1]
sarima_fits_agg_daily_level["forecast_df"] = x[2]
sarima_fits_agg_daily_level["valicast_df"] = x[3]
del x

# aggregated daily w-o-w growth
sarima_fits_agg_daily_wowGr = dict()

x = helperfns_sarima_mapreduce.mapperfn_reducerfn(
    store = ts_store,
    tbl_series = "agg_daily_long",
    stn_params_df = agg_stn_params,
    var = "daily_wowGr_shrtAdj",
    freq = "D"
)

x[0]["include_intercept"] = x[0]["include_intercept"].astype(np.float)
x[0]["var"] = "daily_wowGr_shrtAdj"
x[0].set_index("var", append=True, inplace=True)
sarima_fits_agg_daily_wowGr["sarima_model_spec_df"] = x[0]

sarima_fits_agg_daily_wowGr["fitted_df"] = x[1]
sarima_fits_agg_daily_wowGr["forecast_df"] = x[2]
sarima_fits_agg_daily_wowGr["valicast_df"] = x[3]
del x

# aggregated weekly level
sarima_fits_agg_weekly_level = dict()

x = helperfns_sarima_mapreduce.mapperfn_reducerfn(
    store = ts_store,
    tbl_series = "agg_weekly_long",
    stn_params_df = agg_stn_params,
    var = "weekly_level_shrtAdj",
    freq = "W"
)

x[0]["include_intercept"] = x[0]["include_intercept"].astype(np.float)
x[0]["var"] = "weekly_level_shrtAdj"
x[0].set_index("var", append=True, inplace=True)
sarima_fits_agg_weekly_level["sarima_model_spec_df"] = x[0]

sarima_fits_agg_weekly_level["fitted_df"] = x[1]
sarima_fits_agg_weekly_level["forecast_df"] = x[2]
sarima_fits_agg_weekly_level["valicast_df"] = x[3]
del x

# aggregated weekly w-o-w growth
sarima_fits_agg_weekly_wowGr = dict()

x = helperfns_sarima_mapreduce.mapperfn_reducerfn(
    store = ts_store,
    tbl_series = "agg_weekly_long",
    stn_params_df = agg_stn_params,
    var = "weekly_wowGr_shrtAdj",
    freq = "W"
)

x[0]["include_intercept"] = x[0]["include_intercept"].astype(np.float)
x[0]["var"] = "weekly_wowGr_shrtAdj"
x[0].set_index("var", append=True, inplace=True)
sarima_fits_agg_weekly_wowGr["sarima_model_spec_df"] = x[0]

sarima_fits_agg_weekly_wowGr["fitted_df"] = x[1]
sarima_fits_agg_weekly_wowGr["forecast_df"] = x[2]
sarima_fits_agg_weekly_wowGr["valicast_df"] = x[3]
del x

## Store Fitted Models

# consolidate data
## model specs
sarima_spec_agg = pd.concat([
    sarima_fits_agg_daily_level["sarima_model_spec_df"],
    sarima_fits_agg_daily_wowGr["sarima_model_spec_df"],
    sarima_fits_agg_weekly_level["sarima_model_spec_df"],
    sarima_fits_agg_weekly_wowGr["sarima_model_spec_df"]
], axis=0)

## fitted values
sarima_fitted_agg_daily = pd.concat([
    sarima_fits_agg_daily_level["fitted_df"],
    sarima_fits_agg_daily_wowGr["fitted_df"]
], axis=1)

sarima_fitted_agg_weekly = pd.concat([
    sarima_fits_agg_weekly_level["fitted_df"],
    sarima_fits_agg_weekly_wowGr["fitted_df"]
], axis=1)

## forecasted values
sarima_forecast_agg_daily = pd.concat([
    sarima_fits_agg_daily_level["forecast_df"],
    sarima_fits_agg_daily_wowGr["forecast_df"]
], axis=1)

sarima_forecast_agg_weekly = pd.concat([
    sarima_fits_agg_weekly_level["forecast_df"],
    sarima_fits_agg_weekly_wowGr["forecast_df"]
], axis=1)

## predicted values on validation sample
sarima_valicast_agg_daily = pd.concat([
    sarima_fits_agg_daily_level["valicast_df"],
    sarima_fits_agg_daily_wowGr["valicast_df"]
], axis=1)

sarima_valicast_agg_weekly = pd.concat([
    sarima_fits_agg_weekly_level["valicast_df"],
    sarima_fits_agg_weekly_wowGr["valicast_df"]
], axis=1)


# close open stores
ts_store.close()
sarima_agg_store.close()

# save

sarima_spec_agg.to_hdf(sarima_agg_store, key="sarima_spec_agg",
                       format="table", complevel=9)

sarima_fitted_agg_daily.to_hdf(sarima_agg_store, key="sarima_fitted_agg_daily",
                               format="table", complevel=9)
sarima_fitted_agg_weekly.to_hdf(sarima_agg_store, key="sarima_fitted_agg_weekly",
                                format="table", complevel=9)

sarima_forecast_agg_daily.to_hdf(sarima_agg_store, key="sarima_forecast_agg_daily",
                                 format="table", complevel=9)
sarima_forecast_agg_weekly.to_hdf(sarima_agg_store, key="sarima_forecast_agg_weekly",
                                  format="table", complevel=9)

sarima_valicast_agg_daily.to_hdf(sarima_agg_store, key="sarima_valicast_agg_daily",
                                 format="table", complevel=9)
sarima_valicast_agg_weekly.to_hdf(sarima_agg_store, key="sarima_valicast_agg_weekly",
                                  format="table", complevel=9)
