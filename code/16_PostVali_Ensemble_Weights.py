# setup and paths
## prebuilt libraries
import pickle
import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask.array as da

## project libraries
from projutils import smape

## paths
data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"

# get SMAPE on validation sample for the component models
## load the calculated SMAPE and reshape long
_, losses_byPage, _ = pickle.load(open(data_intermed_nb_fldrpath + "/ensemble_choice.p", "rb"))

NUM_COMPONENT_MODELS = losses_byPage.columns.str.startswith("mod_").sum()

losses_byPage_modelRanked = losses_byPage.stack(dropna=False).to_frame().reset_index()
losses_byPage_modelRanked.columns = ["ts_id","model","smape"]

## keep only component models
losses_byPage_modelRanked = losses_byPage_modelRanked.loc[losses_byPage_modelRanked["model"].str.startswith("mod_")]

## add the model rank by SMAPE
losses_byPage_modelRanked.sort_values(by=["ts_id","smape"], inplace=True)
losses_byPage_modelRanked["model_rank"] = losses_byPage_modelRanked.groupby("ts_id").cumcount()+1

losses_byPage_modelRanked.set_index("ts_id", inplace=True)


# create ensemble weights from the inverse validation SMAPE from the top ranked models

## define helper function to produce ensemble weights according to top ranked models

def get_ranked_ensemble_weights(num_top_ranked):
    ranked_ensemble_weights = losses_byPage_modelRanked.copy()

    ## top ranked should be weighted by the inverse validation smape and the rest receive zero weight
    ranked_ensemble_weights["ranked_ens_wt"] = 0

    rows = ranked_ensemble_weights["model_rank"] <= num_top_ranked
    ranked_ensemble_weights.loc[rows, "ranked_ens_wt"] = 1/(ranked_ensemble_weights.loc[rows, "smape"] + 0.001)
    del rows

    ## NaN smape values should receive zero weight
    ranked_ensemble_weights.loc[pd.isnull(ranked_ensemble_weights["smape"]), "ranked_ens_wt"] = 0

    ## normalize weights
    ranked_ensemble_weights["ranked_ens_wt"] = ranked_ensemble_weights["ranked_ens_wt"].groupby("ts_id").apply(lambda df: df/df.sum())

    ## if all models have nan smape, then weight all models equally
    ts_id_smape_all_nan = ranked_ensemble_weights.groupby("ts_id").apply(lambda df: np.sum(pd.isnull(df["smape"])))
    ts_id_smape_all_nan = ts_id_smape_all_nan[ts_id_smape_all_nan==NUM_COMPONENT_MODELS]
    ts_id_smape_all_nan = ts_id_smape_all_nan.index.values

    ranked_ensemble_weights.loc[ts_id_smape_all_nan, "ranked_ens_wt"] = 1/NUM_COMPONENT_MODELS

    ## reshape to wide
    ranked_ensemble_weights_wide = ranked_ensemble_weights[["model","ranked_ens_wt"]]\
        .pivot(columns="model", values="ranked_ens_wt")
    ranked_ensemble_weights_wide.columns = ["reWt_" + v for v in ranked_ensemble_weights_wide.columns]

    ## return
    return(ranked_ensemble_weights_wide)

## create ensemble weights
### use the top 1 model on the validation set
ranked1_ensemble_weights = get_ranked_ensemble_weights(num_top_ranked=1)

### use the top 5 models weighted by inverse smape
ranked5_ensemble_weights = get_ranked_ensemble_weights(num_top_ranked=5)


# save
pickle.dump(losses_byPage_modelRanked, open(data_intermed_nb_fldrpath + "/losses_byPage_modelRanked.p", "wb"))

d = {"ranked1_ensemble_weights": ranked1_ensemble_weights, "ranked5_ensemble_weights": ranked5_ensemble_weights}
pickle.dump(d, open(data_intermed_nb_fldrpath + "/postvali_ensemble_weights.p", "wb"))
