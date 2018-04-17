
# setup and paths
## prebuilt libraries
import pandas as pd
from multiprocessing import Pool, cpu_count
import pickle

from projutils import smape

NUM_PROCESSES = cpu_count()
P_CHK = 256

## paths
data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"

component_models_store_path = data_intermed_nb_fldrpath + "/component_models_store.h5"
ts_store_path = data_intermed_nb_fldrpath + "/ts_store.h5"

all_models_store_path = data_intermed_nb_fldrpath + "/all_models_store.h5"

# define constants
component_model_varnames = [
       'mod_ts_daily_level_BtTrim', 'mod_ts_daily_wowGr_BtTrim',
       'mod_ts_weekly_level_BtTrim', 'mod_ts_weekly_wowGr_BtTrim',
       'mod_agg_daily_level_BtTrim', 'mod_agg_daily_wowGr_BtTrim',
       'mod_agg_weekly_level_BtTrim', 'mod_agg_weekly_wowGr_BtTrim',
       'mod_sstat_mean_overall', 'mod_sstat_median_overall',
       'mod_sstat_mean_last60', 'mod_sstat_median_last60']
num_mods = len(component_model_varnames)


# define function that creates calibration data for ensemble
def create_ensemble_calibdata(component_models_store_path, component_models_key, ts_store_path=None):
    # merge component model estimates with prepared daily level for validation data
    ## load component model estimates
    keepvars1 = ['ts_id', 'time_d',
           'mod_ts_daily_level_BtTrim', 'mod_ts_daily_wowGr_BtTrim',
           'mod_ts_weekly_level_BtTrim', 'mod_ts_weekly_wowGr_BtTrim',
           'mod_agg_daily_level_BtTrim', 'mod_agg_daily_wowGr_BtTrim',
           'mod_agg_weekly_level_BtTrim', 'mod_agg_weekly_wowGr_BtTrim',
           'mod_sstat_mean_overall', 'mod_sstat_median_overall',
           'mod_sstat_mean_last60', 'mod_sstat_median_last60']
    with pd.HDFStore(component_models_store_path, mode="r") as s:
        component_models_valicast = s.select(key=component_models_key, columns=keepvars1)
    del keepvars1

    component_models_valicast.set_index(["ts_id","time_d"], inplace=True)
    component_models_valicast.sort_index(inplace=True)

    time_d_min = component_models_valicast.iloc[0].name[1].strftime('%Y-%m-%d')
    time_d_max = component_models_valicast.iloc[-1].name[1].strftime('%Y-%m-%d')

    ## load prepared daily level if will merge it
    if ts_store_path is not None:
        keepvars2 = ["daily_level","daily_untouched"]
        with pd.HDFStore(ts_store_path, mode="r") as s:
            ts_daily = s.select(key="ts_daily_long", columns=keepvars2,
                where=["time_d >= '{}'".format(time_d_min),"time_d <= '{}'".format(time_d_max)])
        del keepvars2

        del time_d_min, time_d_max

    ## merge
    if ts_store_path is not None:
        n1 = len(component_models_valicast)
        n2 = len(ts_daily)
        assert(n1==n2)

        ensemble_calibdata = pd.merge(
            component_models_valicast,
            ts_daily,
            how = "inner",
            left_index = True, right_index = True
        )

        assert(n1 == len(ensemble_calibdata))
        del n1, n2
        del ts_daily
    else:
        ensemble_calibdata = component_models_valicast

    assert(ensemble_calibdata.index.names==["ts_id","time_d"])

    del component_models_valicast

    return(ensemble_calibdata)



# define function to create candidate ensemble models
def create_candidate_ensemble_models(ensemble_ts):
    # set up function environ variables
    component_predictions = ensemble_ts
    num_mods = component_predictions.shape[1]

    # create ensemble models
    ## point-wise mean
    y_pointwise_mean = component_predictions.mean(axis=1, skipna=True).to_frame("ensmod_pointwise_mean")

    ## point-wise median
    y_pointwise_median = component_predictions.median(axis=1, skipna=True).to_frame("ensmod_pointwise_median")

    ## function median by choosing the series where the sum is the median, and
    ## break tie by closest to mean if even number of models
    component_prediction_rowsums = component_predictions.sum(axis=0, skipna=True)
    component_prediction_rowsums = component_prediction_rowsums.sort_values()

    if num_mods % 2 == 1:
        medrow = num_mods/2 + 0.5
        medvar = component_prediction_rowsums.index[medrow]
        del medrow
    else:
        tmp1 = (component_prediction_rowsums.iloc[int(num_mods/2):int(num_mods/2 + 2)] \
            - component_prediction_rowsums.mean()).abs()
        medvar = tmp1.idxmin()
        del tmp1

    y_funcwise_median = component_predictions.loc[:,medvar].to_frame("ensmod_funcwise_median")

    del medvar

    # combine candidate ensemble models into one dataframe
    candidate_ensemble_models = pd.concat([y_pointwise_mean, y_pointwise_median, y_funcwise_median], axis=1)
    return(candidate_ensemble_models)


# create candidate ensemble models on the validation set
def create_candidate_ensemble_models_batch(component_models_store_path, component_models_key, ts_store_path = None):
    ## create calibration data for ensemble
    ensemble_calibdata = create_ensemble_calibdata(
        component_models_store_path = component_models_store_path,
        component_models_key = component_models_key,
        ts_store_path = ts_store_path
        )

    ## create the candidate ensemble models
    grp = ensemble_calibdata[component_model_varnames].groupby("ts_id")

    all_candidate_ensemble_models = grp\
        .apply(create_candidate_ensemble_models)

    assert(len(all_candidate_ensemble_models)==len(ensemble_calibdata))

    ## combine models into one datafrae
    all_models = pd.concat(
        [all_candidate_ensemble_models, ensemble_calibdata[component_model_varnames]],
        axis=1)

    if ts_store_path is not None:
        all_models = pd.concat(
            [all_models, ensemble_calibdata[["daily_level", "daily_untouched"]]],
            axis=1)

    assert(len(all_candidate_ensemble_models)==len(all_models))

    ## return
    return(all_models)


if __name__ == "__main__":
    # create candidate ensemble models on the validation set
    all_models_valicast = create_candidate_ensemble_models_batch(
        component_models_store_path = component_models_store_path,
        component_models_key = "/component_models_intermed/valicast",
        ts_store_path = ts_store_path
        )

    all_models_valicast.to_hdf(all_models_store_path,
        key = "/all_models_NotReadjusted/valicast",
        format = "table",
        append = False,
        complevel = 9)

    del all_models_valicast

    # create candidate ensemble models on the forecast set
    all_models_forecast = create_candidate_ensemble_models_batch(
        component_models_store_path = component_models_store_path,
        component_models_key = "/component_models_intermed/forecast",
        ts_store_path = None
        )

    all_models_forecast.to_hdf(all_models_store_path,
        key = "/all_models_NotReadjusted/forecast",
        format = "table",
        append = False,
        complevel = 9)

    del all_models_forecast
