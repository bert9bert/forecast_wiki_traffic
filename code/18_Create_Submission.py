
# setup and paths
## prebuilt libraries
import pandas as pd
import numpy as np
import pickle
import dask.dataframe as dd

## project libraries
from projutils import smape

## paths
data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"
outputs_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/output"
raw_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_raw"

all_models_store_path = data_intermed_nb_fldrpath + "/all_models_store.h5"


## runtime settings
### create Kaggle submission files for these models
create_output_csv_list = ["mod_sstat_median_last60", "mod_ts_daily_level_BtTrim",
    "postValiMod_ens_ranked1", "postValiMod_ens_ranked5"]
create_output_csv_flag = False


if __name__ == "__main__":
    # load the key file
    submission_key = pd.read_csv(raw_nb_fldrpath + "/key_2.csv")

    num_n_t = len(submission_key)

    ## format key table so that it can be used as a merging table
    ## by page and date
    idx = [tuple(x.rsplit("_", 4)) for x in submission_key["Page"]]
    idx = pd.DataFrame(idx, columns=["name","project","access","agent","date"])

    submission_key = pd.concat([idx, submission_key], axis=1)
    del idx
    submission_key.drop("Page", axis=1, inplace=True)

    submission_key["date"] = pd.to_datetime(submission_key["date"], format="%Y-%m-%d")

    ## merge in the modeling time series IDs
    with pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r") as s:
        page_to_id_map = s.select("/page_to_id_map")
        page_to_id_map.reset_index(inplace=True)

    submission_key = pd.merge(
        submission_key,
        page_to_id_map,
        on = ["name","project","access","agent"]
    )

    assert(len(submission_key) == num_n_t)

    submission_key = submission_key[["ts_id","date","Id"]]
    submission_key.rename(columns={"date":"time_d"}, inplace=True)
    submission_key.set_index(["ts_id","time_d"], inplace=True)

    # create dataframe linking forecasts to submission Id
    ## put forecasts together
    with pd.HDFStore(all_models_store_path, mode="r") as s:
        submission_forecast_1 = s.select("/all_models_Rehol_Plugged/forecast")
        submission_forecast_2 = s.select("/post_vali_models/forecast")

    submission_forecast_all = pd.concat([submission_forecast_1, submission_forecast_2], axis=1)

    assert(len(submission_forecast_1) == len(submission_forecast_2))
    assert(len(submission_forecast_all) == len(submission_forecast_1))
    assert(submission_forecast_all.shape[1] == submission_forecast_1.shape[1] + submission_forecast_2.shape[1])

    del submission_forecast_1, submission_forecast_2

    ## merge with submission key
    submission_forecast_all = pd.merge(
        submission_forecast_all,
        submission_key,
        how = "right",
        left_index = True, right_index = True
    )
    submission_forecast_all.sort_index(inplace=True)
    submission_forecast_all.set_index("Id", append=True, inplace=True)

    assert(len(submission_forecast_all) == num_n_t)

    assert(not submission_forecast_all.isnull().values.any())

    ## save
    submission_forecast_all.to_hdf(
        all_models_store_path,
        key="/submission_forecast_all",
        format="table", append=False, complevel=9
        )

    del submission_forecast_all

    # create submission files
    if create_output_csv_flag:
        for m in create_output_csv_list:
            with pd.HDFStore(all_models_store_path, mode="r") as s:
                submission_output = s.select("/submission_forecast_all", columns=["Id",m])

            submission_output.rename(columns={m:"Visits"}, inplace=True)
            submission_output = submission_output.reset_index(drop=False)[["Id","Visits"]]

            assert(len(submission_output) == num_n_t)

            submission_output.to_csv(
                data_intermed_nb_fldrpath + "/submission_" + m + ".csv.gz",
                index=False,
                compression="gzip")

            del submission_output
