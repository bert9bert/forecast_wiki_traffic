# Convert Topline SARIMA Models to Daily Level #
# ============================================ #

# The SARIMA models fit to the topline aggregates need to have their forecasts converted back to the daily level. Weekly series will then need to be re-seasonalized.
#
# | Modeling Transformation | Back Transformations |
# | :---------------------- | :------------------- |
# | Daily Level             | None                 |
# | Daily W-o-W Growth      | Convert growth to level |
# | Weekly Level            | Re-seasonalize to daily level |
# | Weekly W-o-W Growth     | Convert weekly growth to weekly level, then <br> re-seasonalize to daily level
#
# Do this for the fitted values, the projection forecast, and the validation set forecast.

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
import projutils_backtrans
import helperfns_backtrans_mapreduce

## paths
data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"

# set up hdf5 storage
ts_store = pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r")
sarima_agg_store = pd.HDFStore(data_intermed_nb_fldrpath + "/sarima_agg_store.h5", mode="r")
summary_stats_store = pd.HDFStore(data_intermed_nb_fldrpath + "/summary_stats_store.h5", mode="r")

sarima_agg_backtrans_store = pd.HDFStore(data_intermed_nb_fldrpath + "/sarima_agg_backtrans_store.h5")

# Back-transform Topline Modeled to Topline at Daily Level
# | When mapping weekly back to daily, use the historical seasonal proportions calculated from the historical seasonal proportions without the validation data.
# | Re-introducing adjustments such as holiday adjusts will be done once all models have been converted to the time series daily level.

# inputs for data chunk
SEASONAL_VAR = "trunc60_sum60"

## historical data
chunk_y_hist = ts_store.select("agg_daily_long", columns=["daily_level_shrtAdj"])
chunk_y_hist_weekly = ts_store.select("agg_weekly_long", columns=["weekly_level_shrtAdj"])

## intraweek seasonal distribution
chunk_intraweek_seasonal_df = summary_stats_store.select("agg_stat_dayofweek", columns=[SEASONAL_VAR])

## inputs for forecast

chunk_y_daily_level_forecast = sarima_agg_store.select("sarima_forecast_agg_daily", columns=["daily_level_shrtAdj_pred"])
chunk_y_daily_wowGr_forecast = sarima_agg_store.select("sarima_forecast_agg_daily", columns=["daily_wowGr_shrtAdj_pred"])
chunk_y_weekly_level_forecast = sarima_agg_store.select("sarima_forecast_agg_weekly", columns=["weekly_level_shrtAdj_pred"])
chunk_y_weekly_wowGr_forecast = sarima_agg_store.select("sarima_forecast_agg_weekly", columns=["weekly_wowGr_shrtAdj_pred"])

## inputs for valicast

chunk_y_daily_level_valicast = sarima_agg_store.select("sarima_valicast_agg_daily", columns=["daily_level_shrtAdj_pred"])
chunk_y_daily_wowGr_valicast = sarima_agg_store.select("sarima_valicast_agg_daily", columns=["daily_wowGr_shrtAdj_pred"])
chunk_y_weekly_level_valicast = sarima_agg_store.select("sarima_valicast_agg_weekly", columns=["weekly_level_shrtAdj_pred"])
chunk_y_weekly_wowGr_valicast = sarima_agg_store.select("sarima_valicast_agg_weekly", columns=["weekly_wowGr_shrtAdj_pred"])

## inputs for fitted

chunk_y_daily_level_fitted = sarima_agg_store.select("sarima_fitted_agg_daily", columns=["daily_level_shrtAdj_pred"])
chunk_y_daily_wowGr_fitted = sarima_agg_store.select("sarima_fitted_agg_daily", columns=["daily_wowGr_shrtAdj_pred"])
chunk_y_weekly_level_fitted = sarima_agg_store.select("sarima_fitted_agg_weekly", columns=["weekly_level_shrtAdj_pred"])
chunk_y_weekly_wowGr_fitted = sarima_agg_store.select("sarima_fitted_agg_weekly", columns=["weekly_wowGr_shrtAdj_pred"])

# apply backtransformations
mapper_inputs = helperfns_backtrans_mapreduce.create_mapper_input(chunk_y_hist, chunk_y_hist_weekly,
    chunk_intraweek_seasonal_df,
    chunk_y_daily_level_forecast, chunk_y_daily_wowGr_forecast, chunk_y_weekly_level_forecast, chunk_y_weekly_wowGr_forecast,
    chunk_y_daily_level_valicast, chunk_y_daily_wowGr_valicast, chunk_y_weekly_level_valicast, chunk_y_weekly_wowGr_valicast,
    chunk_y_daily_level_fitted, chunk_y_daily_wowGr_fitted, chunk_y_weekly_level_fitted, chunk_y_weekly_wowGr_fitted)

mapper_output = [helperfns_backtrans_mapreduce.mapperfn(v) for v in mapper_inputs]

reducer_output = helperfns_backtrans_mapreduce.reducerfn(mapper_output)

# store the processed data
reducer_output["z_forecast"].to_hdf(
    sarima_agg_backtrans_store,
    key = "sarima_agg_backtrans_to_dailyAgg/forecast",
    format = "table",
    append = False,
    complevel = 9
)

reducer_output["z_valicast"].to_hdf(
    sarima_agg_backtrans_store,
    key = "sarima_agg_backtrans_to_dailyAgg/valicast",
    format = "table",
    append = False,
    complevel = 9
)

reducer_output["z_fitted"].to_hdf(
    sarima_agg_backtrans_store,
    key = "sarima_agg_backtrans_to_dailyAgg/fitted",
    format = "table",
    append = False,
    complevel = 9
)

# clean up
ts_store.close()
sarima_agg_store.close()
summary_stats_store.close()

sarima_agg_backtrans_store.close()
