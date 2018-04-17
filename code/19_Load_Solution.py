
# setup and paths
## prebuilt libraries
import pandas as pd
import numpy as np

## paths
data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"
raw_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_raw"

all_models_store_path = data_intermed_nb_fldrpath + "/all_models_store.h5"


# load and store solution
## load
solution = pd.read_csv(raw_nb_fldrpath + "/solution_11_15.csv")
solution.set_index("Id", inplace=True)

## merge in page and time data
with pd.HDFStore(all_models_store_path, mode="r") as s:
    idxdf = s.select(key="/submission_forecast_all", columns=[])

assert(len(idxdf) == len(solution))

idxdf.reset_index(["ts_id","time_d"], drop=False, inplace=True)

solution = pd.merge(
    solution,
    idxdf,
    left_index=True, right_index=True
)

solution.reset_index(drop=False, inplace=True)
solution.set_index(["ts_id","time_d","Id"], inplace=True)

solution.sort_index(level=["ts_id","time_d"], inplace=True)

## check counts
with pd.HDFStore(all_models_store_path, mode="r") as s:
    num_n_t = s.get_storer("submission_forecast_all").nrows

assert(len(solution) == num_n_t)

## store
solution.to_hdf(
    all_models_store_path,
    key="/solution",
    format="table", append=False, complevel=9
    )
