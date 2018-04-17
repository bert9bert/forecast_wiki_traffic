
# setup and paths
## prebuilt libraries
import pandas as pd
import numpy as np
import dask.dataframe as dd

## paths
data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"

ts_store_path = data_intermed_nb_fldrpath + "/ts_store.h5"
all_models_store_path = data_intermed_nb_fldrpath + "/all_models_store.h5"


# get holiday adjustments
## get map of page x date to holiday
ts_daily_long_holidays = dd.read_hdf(ts_store_path, key="/ts_daily_long", columns=["holiday"])
ts_daily_long_holidays = ts_daily_long_holidays.loc[ts_daily_long_holidays.holiday != "regular_day"]
ts_daily_long_holidays = ts_daily_long_holidays.loc[ts_daily_long_holidays.holiday.str[-4:] == "_adj"]
ts_daily_long_holidays = ts_daily_long_holidays.reset_index()
ts_daily_long_holidays = ts_daily_long_holidays.compute()

## get map of holiday to adjustment
with pd.HDFStore(ts_store_path, mode="r") as s:
    holiday_adjustment_factors = s.select("/holiday_adjustment_factors")
    holiday_adjustment_factors = holiday_adjustment_factors.loc[:,["ts_id","holiday","adjustment_factor"]]
    holiday_adjustment_factors["holiday"] = holiday_adjustment_factors["holiday"].astype(str) + "_adj"

## construct map of page x date to adjustment for holidays
ts_holiday_premerge = pd.merge(
    ts_daily_long_holidays,
    holiday_adjustment_factors,
    on = ["ts_id", "holiday"]
)
ts_holiday_premerge.set_index(["ts_id","time_d"], inplace=True)
ts_holiday_premerge = ts_holiday_premerge[["adjustment_factor"]]

assert(len(ts_holiday_premerge) == len(ts_daily_long_holidays))

del ts_daily_long_holidays, holiday_adjustment_factors


# re-adjust all candidate models for holidays (re-holiday)

def reholiday_helper(all_models_in_store_path, store_key_in, pageXdate_to_adjustment):
    ## merge in the adjustment factors
    with pd.HDFStore(all_models_in_store_path, mode="r") as s:
        models_in = s.select(store_key_in)

    models_out = pd.merge(
        models_in,
        pageXdate_to_adjustment,
        how = "left",
        left_index = True, right_index = True
    )

    assert(len(models_out) == len(models_in))

    ## re-holiday (re-adjust) the candidate model outputs
    mod_names = [v for v in models_out.columns if v[:4]=="mod_" or v[:7]=="ensmod_"]

    rows_to_adj = ~np.isnan(models_out.adjustment_factor)

    for v in mod_names:
        models_out.loc[rows_to_adj, v] = models_out.loc[rows_to_adj, v] * models_out.loc[rows_to_adj, "adjustment_factor"]

    models_out.drop("adjustment_factor", axis=1, inplace=True)

    ## return
    return(models_out)


all_models_readj_valicast = reholiday_helper(
    all_models_in_store_path = all_models_store_path,
    store_key_in = "/all_models_NotReadjusted/valicast",
    pageXdate_to_adjustment = ts_holiday_premerge
)


all_models_readj_forecast = reholiday_helper(
    all_models_in_store_path = all_models_store_path,
    store_key_in = "/all_models_NotReadjusted/forecast",
    pageXdate_to_adjustment = ts_holiday_premerge
)


# save
all_models_readj_valicast.to_hdf(all_models_store_path,
    key = "/all_models_Rehol_NotPlugged/valicast",
    format = "table",
    append = False,
    complevel = 9)

all_models_readj_forecast.to_hdf(all_models_store_path,
    key = "/all_models_Rehol_NotPlugged/forecast",
    format = "table",
    append = False,
    complevel = 9)
