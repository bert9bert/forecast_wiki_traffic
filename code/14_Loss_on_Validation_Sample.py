
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


# load validation data and results
with pd.HDFStore(all_models_store_path, mode="r") as s:
    all_models = s.select("/all_models_Rehol_NotPlugged/valicast")

mod_names = [v for v in all_models.columns if v[:4]=="mod_" or v[:7]=="ensmod_"]


# compute ensemble loss and individual component losses
def helperfn1(df, xvars=mod_names, yvar="daily_untouched"):
    df1 = [smape(F=df[x], A=df[yvar]) for x in xvars]
    df1 = pd.Series(df1, index=xvars)
    return(df1)

## at the overall level (where no model has a missing projection)
all_models_NoNan = all_models.dropna(how="any")

losses_overall = helperfn1(all_models_NoNan)

best_model_overall_EnsembAndComp = losses_overall.idxmin()
best_model_overall_EnsembOnly = losses_overall[[x for x in losses_overall.index.values if "ensmod_" in x]].idxmin()

print("Best model is {} with SMAPE of {:.2f}.".format(
    best_model_overall_EnsembAndComp, losses_overall[best_model_overall_EnsembAndComp]))
print("Best ensemble model is {} with SMAPE of {:.2f}.".format(
    best_model_overall_EnsembOnly, losses_overall[best_model_overall_EnsembOnly]))

## at the page level
losses_byPage = all_models.groupby("ts_id").apply(helperfn1)

### distributions of results at the page level (where no model has a missing projection)
losses_byPage_distrib = losses_byPage.dropna(how="any").describe()

print(losses_byPage_distrib)


# count not missing projection values
nonmissing_model_projections = 1 - all_models.isnull().sum()/len(all_models)
nonmissing_model_projections = nonmissing_model_projections[mod_names]
nonmissing_model_projections = nonmissing_model_projections.to_frame("nonmissing_model_projections")


# save
nonmissing_model_projections.to_csv(outputs_nb_fldrpath + "/nonmissing_model_projections.csv")

losses_overall.to_csv(outputs_nb_fldrpath + "/losses_NotPlugged_overall.csv")
losses_byPage_distrib.to_csv(outputs_nb_fldrpath + "/losses_NotPlugged_byPage_distrib.csv")

savelist = [
    losses_overall,
    losses_byPage,
    losses_byPage_distrib
]

pickle.dump(savelist, open(data_intermed_nb_fldrpath + "/ensemble_choice.p", "wb"))
