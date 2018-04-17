
# setup and paths
## prebuilt libraries
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd

## paths
data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"

sarima_ts_backtrans_store_path = data_intermed_nb_fldrpath + "/sarima_ts_backtrans_store.h5"
sarima_agg_backtrans_store_path = data_intermed_nb_fldrpath + "/sarima_agg_backtrans_store.h5"
summary_stats_store_path = data_intermed_nb_fldrpath + "/summary_stats_store.h5"
ts_store_path = data_intermed_nb_fldrpath + "/ts_store.h5"
component_models_store_path = data_intermed_nb_fldrpath + "/component_models_store.h5"

dask_tmpdir = "/media/disk1/forecast_wiki_traffic/data_intermed/tmp"

# compute amount to trim SARIMA estimates at
# (trim at max observed times a multiplier)
TRIM_MULTIPLIER = 10

## read in time series
ts_daily_long = dd.read_hdf(ts_store_path, "/ts_daily_long", chunksize=803*2048)

## get max per time series
trimmax = ts_daily_long.groupby("ts_id")["daily_level"].max().compute()
trimmax = trimmax.to_frame("trimmax")

## get max per time series with last 60 truncated
with pd.HDFStore(ts_store_path) as s:
    n = s.get_storer("/ts_daily_long").nrows
    t_max = s.select("/ts_daily_long", start=n-1, stop=n)
    t_max = t_max.index.values[0][1]
    del n

ts_daily_long_trunc60 = ts_daily_long.reset_index()
ts_daily_long_trunc60 = ts_daily_long_trunc60[ts_daily_long_trunc60.time_d < (t_max + pd.Timedelta("-60 days"))]

del t_max

trimmax_trunc60 = ts_daily_long_trunc60.groupby("ts_id")["daily_level"].max().compute()
trimmax_trunc60 = trimmax_trunc60.to_frame("trimmax_trunc60")

## merge together the overall max and the truncated max
trimmax_df = pd.merge(trimmax, trimmax_trunc60, left_index=True, right_index=True)
assert(len(trimmax_df) == len(trimmax))
assert(len(trimmax_df) == len(trimmax_trunc60))
del trimmax, trimmax_trunc60

assert(all(np.sum(np.isnan(trimmax_df))==0))

## multiply trims by multiplier
trimmax_df = trimmax_df * TRIM_MULTIPLIER


# define a function to prepare final unadjusted component models
def prep_final_unadj_component_models(sarima_ts_backtrans_key,
    sarima_agg_backtrans_key,
    component_models_intermed_key,
    trimmax_var,
    seas_stat_mod_vars,
    chunksize):
    ## combine SARIMA estimates for both aggregates and time series levels
    ### prep time series level model intermediate estimates
    sarima_ts_backtrans = dd.read_hdf(sarima_ts_backtrans_store_path, sarima_ts_backtrans_key, chunksize=chunksize)
    sarima_ts_backtrans = sarima_ts_backtrans.reset_index()
    col_ren_dict = {"daily_level_predbt":"mod_ts_daily_level_Bt",
        "daily_wowGr_predbt":"mod_ts_daily_wowGr_Bt",
        "weekly_level_predbt":"mod_ts_weekly_level_Bt",
        "weekly_wowGr_predbt":"mod_ts_weekly_wowGr_Bt"}
    sarima_ts_backtrans = sarima_ts_backtrans.rename(columns=col_ren_dict)
    del col_ren_dict

    ### prep aggregate level model intermediate estimates
    sarima_agg_backtrans = dd.read_hdf(sarima_agg_backtrans_store_path, sarima_agg_backtrans_key, chunksize=chunksize)
    col_ren_dict = {"daily_level_shrtAdj_predbt":"mod_agg_daily_level_Bt",
        "daily_wowGr_shrtAdj_predbt":"mod_agg_daily_wowGr_Bt",
        "weekly_level_shrtAdj_predbt":"mod_agg_weekly_level_Bt",
        "weekly_wowGr_shrtAdj_predbt":"mod_agg_weekly_wowGr_Bt"}
    sarima_agg_backtrans = sarima_agg_backtrans.rename(columns=col_ren_dict)
    del col_ren_dict

    ### put together intermediate SARIMA estimates
    component_models_intermed = dd.merge(
        sarima_ts_backtrans,
        sarima_agg_backtrans,
        on=["ts_id","time_d"]
        )


    ## trim outliers from SARIMA estimates
    component_models_intermed = dd.merge(
        component_models_intermed,
        trimmax_df[[trimmax_var]].rename(columns={trimmax_var:"trimmax_this"}),
        on=["ts_id"],
        left_index=False, right_index=True
        )

    cols = ["mod_ts_daily_level_Bt","mod_ts_daily_wowGr_Bt","mod_ts_weekly_level_Bt","mod_ts_weekly_wowGr_Bt",
        "mod_agg_daily_level_Bt","mod_agg_daily_wowGr_Bt","mod_agg_weekly_level_Bt","mod_agg_weekly_wowGr_Bt"]
    for v in cols:
        component_models_intermed[v+"Trim"] = component_models_intermed[[v,"trimmax_this"]].min(axis=1, skipna=False)
    del cols


    ## combine summary stat estimates
    with pd.HDFStore(summary_stats_store_path, mode="r") as s:
        sstat = s.select("/ts_stat_dayofweek", columns=list(seas_stat_mod_vars.keys()))
        sstat.rename(columns=seas_stat_mod_vars, inplace=True)
        sstat.reset_index(inplace=True)

    component_models_intermed["day_of_week"] = component_models_intermed.time_d.dt.dayofweek

    component_models_intermed = dd.merge(
        component_models_intermed,
        sstat,
        on = ["ts_id","day_of_week"]
    )

    del component_models_intermed["day_of_week"]


    # save results to h5
    component_models_intermed.to_hdf(
        component_models_store_path,
        key = component_models_intermed_key,
        compute = True,
        format = "table",
        data_columns = ["ts_id","time_d"]
    )

    # check counts
    with pd.HDFStore(component_models_store_path, mode="r") as s:
        n0 = s.get_storer(component_models_intermed_key).nrows

    with pd.HDFStore(sarima_ts_backtrans_store_path, mode="r") as s:
        n1 = s.get_storer(sarima_ts_backtrans_key).nrows

    with pd.HDFStore(sarima_agg_backtrans_store_path, mode="r") as s:
        n2 = s.get_storer(sarima_agg_backtrans_key).nrows

    assert(n0==n1)
    assert(n1==n2)



# prepare final unadjusted component models
## forecast
seas_stat_mod_vars_forecast = {
    "mean": "mod_sstat_mean_overall",
    "mean60": "mod_sstat_mean_last60",
    "median": "mod_sstat_median_overall",
    "median60": "mod_sstat_median_last60"
}

prep_final_unadj_component_models(
    sarima_ts_backtrans_key = "/sarima_ts_backtrans_to_dailyts/forecast",
    sarima_agg_backtrans_key = "/sarima_agg_backtrans_to_dailyTS/forecast",
    component_models_intermed_key = "/component_models_intermed/forecast",
    trimmax_var = "trimmax",
    seas_stat_mod_vars = seas_stat_mod_vars_forecast,
    chunksize = 60 * 2**15
    )

del seas_stat_mod_vars_forecast

## validation
seas_stat_mod_vars_valicast = {
    "trunc60_mean": "mod_sstat_mean_overall",
    "trunc60_mean60": "mod_sstat_mean_last60",
    "trunc60_median": "mod_sstat_median_overall",
    "trunc60_median60": "mod_sstat_median_last60"
}

prep_final_unadj_component_models(
    sarima_ts_backtrans_key = "/sarima_ts_backtrans_to_dailyts/valicast",
    sarima_agg_backtrans_key = "/sarima_agg_backtrans_to_dailyTS/valicast",
    component_models_intermed_key = "/component_models_intermed/valicast",
    trimmax_var = "trimmax_trunc60",
    seas_stat_mod_vars = seas_stat_mod_vars_valicast,
    chunksize = 60 * 2**15
    )

del seas_stat_mod_vars_valicast

## fitted
seas_stat_mod_vars_fitted = {
    "mean": "mod_sstat_mean_overall",
    "mean60": "mod_sstat_mean_last60",
    "median": "mod_sstat_median_overall",
    "median60": "mod_sstat_median_last60"
}

with dask.set_options(temporary_directory=dask_tmpdir):
    prep_final_unadj_component_models(
        sarima_ts_backtrans_key = "/sarima_ts_backtrans_to_dailyts/fitted",
        sarima_agg_backtrans_key = "/sarima_agg_backtrans_to_dailyTS/fitted",
        component_models_intermed_key = "/component_models_intermed/fitted",
        trimmax_var = "trimmax",
        seas_stat_mod_vars = seas_stat_mod_vars_fitted,
        chunksize = 803 * 2**8
        )

del seas_stat_mod_vars_fitted
