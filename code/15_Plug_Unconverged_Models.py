
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

all_models_store_path = data_intermed_nb_fldrpath + "/all_models_store.h5"


# plug missing model projections with the last 60-day seasonal median
## get model names
with pd.HDFStore(all_models_store_path, mode="r") as s:
    df = s.select("/all_models_Rehol_NotPlugged/valicast", stop=1)
    mod_names = [v for v in df.columns if v[:4]=="mod_" or v[:7]=="ensmod_"]
del df

## set variable to plug with
plug_var = "mod_sstat_median_last60"

toplug_vars = [x for x in mod_names if x!=plug_var]

for datasub in ["valicast","forecast"]:
    with pd.HDFStore(all_models_store_path, mode="r") as s:
        all_models = s.select("/all_models_Rehol_NotPlugged/"+datasub)

    n = len(all_models)

    assert(np.sum(np.isnan(all_models[plug_var])) == 0)

    for v in toplug_vars:
        all_models.loc[np.isnan(all_models[v]), v] = all_models.loc[np.isnan(all_models[v]), plug_var]

        assert(np.sum(np.isnan(all_models[v])) == 0)

    assert(len(all_models) == n)

    all_models.to_hdf(
        all_models_store_path,
        key="/all_models_Rehol_Plugged/"+datasub,
        format="table", append=False, complevel=9
        )

    del all_models, n


# calculate losses on the plugged validation set
def helperfn1(df, xvars=mod_names, yvar="daily_untouched"):
    df1 = [smape(F=df[x], A=df[yvar]) for x in xvars]
    df1 = pd.Series(df1, index=xvars)
    return(df1)

## at the overall level
with pd.HDFStore(all_models_store_path, mode="r") as s:
    all_models_valicast = s.select("/all_models_Rehol_Plugged/valicast")

losses_overall = helperfn1(all_models_valicast)

best_model_overall_EnsembAndComp = losses_overall.idxmin()
best_model_overall_EnsembOnly = losses_overall[[x for x in losses_overall.index.values if "ensmod_" in x]].idxmin()

print("Best model after plugging is {} with SMAPE of {:.2f}.".format(
    best_model_overall_EnsembAndComp, losses_overall[best_model_overall_EnsembAndComp]))
print("Best ensemble model after plugging is {} with SMAPE of {:.2f}.".format(
    best_model_overall_EnsembOnly, losses_overall[best_model_overall_EnsembOnly]))

## at the page level
losses_byPage = all_models_valicast.groupby("ts_id").apply(helperfn1)

### distributions of results at the page level (where no model has a missing projection)
losses_byPage_distrib = losses_byPage.dropna(how="any").describe()

print(losses_byPage_distrib)

## save
losses_overall.to_csv(outputs_nb_fldrpath + "/losses_Plugged_overall.csv")
losses_byPage_distrib.to_csv(outputs_nb_fldrpath + "/losses_Plugged_byPage_distrib.csv")

savelist = [
    losses_overall,
    losses_byPage,
    losses_byPage_distrib
]

pickle.dump(savelist, open(data_intermed_nb_fldrpath + "/ensemble_choice_plugged.p", "wb"))
