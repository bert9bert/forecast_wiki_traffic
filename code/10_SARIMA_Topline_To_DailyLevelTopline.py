# setup and paths
## prebuilt libraries
import pandas as pd
import dask
import dask.dataframe as dd


## paths
data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"

# define function that distributes topline views to time series
def distribute_topline_to_ts(hist_prop_store_path, hist_prop_key,
    hist_prop_var,
    topline_dailyAgg_store_path, topline_dailyAgg_key,
    topline_dailyTS_store_path, topline_dailyTS_key,
    hist_prop_chunksize = 1000000, topline_dailyAgg_chunksize = 1000000):

    ## load historical proportions of each time series in the aggregation
    ts_hist_prop = dd.read_hdf(
        hist_prop_store_path,
        key = hist_prop_key,
        chunksize = hist_prop_chunksize
    )
    ts_hist_prop = ts_hist_prop[["project","access","agent",hist_prop_var]]
    ts_hist_prop = ts_hist_prop.reset_index()

    ## get topline forecasts
    backtrans_to_dailyAgg_forecast = dd.read_hdf(
        topline_dailyAgg_store_path,
        key = topline_dailyAgg_key,
        chunksize = topline_dailyAgg_chunksize
    )
    backtrans_to_dailyAgg_forecast = backtrans_to_dailyAgg_forecast.reset_index()

    ## create cartesian product of historical proportions and daily topline forecasts
    backtrans_to_dailyTS_forecast = dd.merge(ts_hist_prop, backtrans_to_dailyAgg_forecast, on=["project","access","agent"])

    ## use historical proportions to distribute page views from topline to time series
    backtrans_to_dailyTS_forecast["daily_level_shrtAdj_predbt"] = backtrans_to_dailyTS_forecast[hist_prop_var] \
        * backtrans_to_dailyTS_forecast["daily_level_shrtAdj_predbt"]
    backtrans_to_dailyTS_forecast["daily_wowGr_shrtAdj_predbt"] = backtrans_to_dailyTS_forecast[hist_prop_var] \
        * backtrans_to_dailyTS_forecast["daily_wowGr_shrtAdj_predbt"]
    backtrans_to_dailyTS_forecast["weekly_level_shrtAdj_predbt"] = backtrans_to_dailyTS_forecast[hist_prop_var] \
        * backtrans_to_dailyTS_forecast["weekly_level_shrtAdj_predbt"]
    backtrans_to_dailyTS_forecast["weekly_wowGr_shrtAdj_predbt"] = backtrans_to_dailyTS_forecast[hist_prop_var] \
        * backtrans_to_dailyTS_forecast["weekly_wowGr_shrtAdj_predbt"]

    backtrans_to_dailyTS_forecast = backtrans_to_dailyTS_forecast[["ts_id","time_d",
        "daily_level_shrtAdj_predbt","daily_wowGr_shrtAdj_predbt",
        "weekly_level_shrtAdj_predbt","weekly_wowGr_shrtAdj_predbt"]]

    ## save to hdf5
    backtrans_to_dailyTS_forecast.to_hdf(
        topline_dailyTS_store_path,
        key = topline_dailyTS_key,
        compute = True,
        format = "table",
        data_columns = ["ts_id","time_d"]
    )

    ## check counts
    TOL = 1e-8

    sums_actual = backtrans_to_dailyAgg_forecast[["daily_level_shrtAdj_predbt","daily_wowGr_shrtAdj_predbt",
        "weekly_level_shrtAdj_predbt","weekly_wowGr_shrtAdj_predbt"]]\
        .sum()\
        .compute()

    sums_calculated = dd.read_hdf(topline_dailyTS_store_path,
        key = topline_dailyTS_key)[["daily_level_shrtAdj_predbt","daily_wowGr_shrtAdj_predbt",
        "weekly_level_shrtAdj_predbt","weekly_wowGr_shrtAdj_predbt"]]\
        .sum()\
        .compute()

    sums_reldiff = abs(sums_actual-sums_calculated)/sums_actual

    assert(all(sums_reldiff < TOL))

    del sums_actual, sums_calculated, sums_reldiff, TOL


    ### number of rows stored
    with pd.HDFStore(topline_dailyTS_store_path, mode="r") as s:
        nrows_df = s.get_storer(topline_dailyTS_key).nrows

    with pd.HDFStore(hist_prop_store_path, mode="r") as s:
        n_timeseries = s.get_storer(hist_prop_key).nrows

    with pd.HDFStore(topline_dailyAgg_store_path, mode="r") as s:
        key = topline_dailyAgg_key
        idx1 = s.select(key, stop=1).index.values[0][:3]
        n_t = s.select(key,
            where=["project=='{}'".format(idx1[0]), "access=='{}'".format(idx1[1]), "agent=='{}'".format(idx1[2])]).shape[0]
        del key, idx1

    assert(nrows_df == n_timeseries * n_t)
    del nrows_df, n_timeseries, n_t


# apply to forecasts
distribute_topline_to_ts(
    hist_prop_store_path = data_intermed_nb_fldrpath + "/summary_stats_store.h5",
    hist_prop_key = "/ts_hist_prop",
    hist_prop_var = "hist_prop_last60days",
    topline_dailyAgg_store_path = data_intermed_nb_fldrpath + "/sarima_agg_backtrans_store.h5",
    topline_dailyAgg_key = "/sarima_agg_backtrans_to_dailyAgg/forecast",
    topline_dailyTS_store_path = data_intermed_nb_fldrpath + "/sarima_agg_backtrans_store.h5",
    topline_dailyTS_key = "/sarima_agg_backtrans_to_dailyTS/forecast"
    )

# apply to validation
distribute_topline_to_ts(
    hist_prop_store_path = data_intermed_nb_fldrpath + "/summary_stats_store.h5",
    hist_prop_key = "/ts_hist_prop",
    hist_prop_var = "hist_prop_trunc60_last60days",
    topline_dailyAgg_store_path = data_intermed_nb_fldrpath + "/sarima_agg_backtrans_store.h5",
    topline_dailyAgg_key = "/sarima_agg_backtrans_to_dailyAgg/valicast",
    topline_dailyTS_store_path = data_intermed_nb_fldrpath + "/sarima_agg_backtrans_store.h5",
    topline_dailyTS_key = "/sarima_agg_backtrans_to_dailyTS/valicast"
    )

# apply to fitted
distribute_topline_to_ts(
    hist_prop_store_path = data_intermed_nb_fldrpath + "/summary_stats_store.h5",
    hist_prop_key = "/ts_hist_prop",
    hist_prop_var = "hist_prop_last60days",
    topline_dailyAgg_store_path = data_intermed_nb_fldrpath + "/sarima_agg_backtrans_store.h5",
    topline_dailyAgg_key = "/sarima_agg_backtrans_to_dailyAgg/fitted",
    topline_dailyTS_store_path = data_intermed_nb_fldrpath + "/sarima_agg_backtrans_store.h5",
    topline_dailyTS_key = "/sarima_agg_backtrans_to_dailyTS/fitted",
    hist_prop_chunksize = 100, topline_dailyAgg_chunksize = 100*803
    )
