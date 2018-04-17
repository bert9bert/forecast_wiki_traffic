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
outputs_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/output"

all_models_store_path = data_intermed_nb_fldrpath + "/all_models_store.h5"
post_vali_models_store = all_models_store_path

## runtime settings
dd_chunksize_valicast = 60*1024
dd_chunksize_forecast = 70*1024


# define function to create ensemble models based off results on the validation sample (post-validation sample models)
def add_post_vali_mods(df_dd, ensemble_weights, ensemble_name):
    ### merge model predictions with ensemble weights
    all_models_with_postValiMods = dd.merge(
        df_dd,
        ensemble_weights,
        left_index=True,
        right_index=True)

    ### create post-validation set ensembles
    all_models_with_postValiMods[ensemble_name] = 0

    for v in list(df_dd.columns[df_dd.columns.str.startswith("mod_")]):
        m = all_models_with_postValiMods["reWt_"+v] == 0
        m1 = da.isfinite(all_models_with_postValiMods[v])

        all_models_with_postValiMods[ensemble_name] = all_models_with_postValiMods[ensemble_name]\
            .where(m,
                   all_models_with_postValiMods[ensemble_name] +
                       all_models_with_postValiMods[v].where(m1, 0) * all_models_with_postValiMods["reWt_"+v])

        del m

    ### drop the weights columns since they aren't needed anymore
    all_models_with_postValiMods = all_models_with_postValiMods.drop(labels=ensemble_weights.columns, axis=1)

    ### return dask graph
    return(all_models_with_postValiMods)


# load ensemble weights
d = pickle.load(open(data_intermed_nb_fldrpath + "/postvali_ensemble_weights.p", "rb"))

ranked1_ensemble_weights = d["ranked1_ensemble_weights"]
ranked5_ensemble_weights = d["ranked5_ensemble_weights"]

del d


# create post-validation sample models on the validation sample
## load component models
all_models_dd = dd.read_hdf(
    all_models_store_path,
    key="/all_models_Rehol_NotPlugged/valicast",
    chunksize=dd_chunksize_valicast)

## add ranked 1 and ranked 5 predictions
all_models_with_postValiMods_dd = add_post_vali_mods(
    df_dd = all_models_dd,
    ensemble_weights = ranked1_ensemble_weights,
    ensemble_name = "postValiMod_ens_ranked1")

all_models_with_postValiMods_dd = add_post_vali_mods(
    df_dd = all_models_with_postValiMods_dd,
    ensemble_weights = ranked5_ensemble_weights,
    ensemble_name = "postValiMod_ens_ranked5")

all_models_with_postValiMods_dd = all_models_with_postValiMods_dd[
    ["daily_level","daily_untouched","postValiMod_ens_ranked1","postValiMod_ens_ranked5"]]

## compute dask graph and save
all_models_with_postValiMods_dd.to_hdf(
    path_or_buf = post_vali_models_store,
    key = "/post_vali_models/valicast",
    compute = True,
    complevel = 9)

## checks
### make sure that ensembles don't have missing values
with pd.HDFStore(post_vali_models_store, mode="r") as s:
    dfcheck = s.select("/post_vali_models/valicast", columns=["postValiMod_ens_ranked1","postValiMod_ens_ranked5"])

assert(all(pd.isnull(dfcheck).sum()==0))

del dfcheck

### check that number of rows is as expected
with pd.HDFStore(post_vali_models_store, mode="r") as s:
    n1 = s.get_storer("/post_vali_models/valicast").nrows

n2 = len(all_models_dd)

assert(n1==n2)
del n1, n2

### print first few rows
with pd.HDFStore(post_vali_models_store, mode="r") as s:
    dfcheck = s.select("/post_vali_models/valicast", stop=10)
print(dfcheck.to_string())
del dfcheck



# create post-validation sample models on the forecast/test sample
## load component models
all_models_dd = dd.read_hdf(
    all_models_store_path,
    key="/all_models_Rehol_NotPlugged/forecast",
    chunksize=dd_chunksize_forecast)

## add ranked 1 and ranked 5 predictions
all_models_with_postValiMods_dd = add_post_vali_mods(
    df_dd = all_models_dd,
    ensemble_weights = ranked1_ensemble_weights,
    ensemble_name = "postValiMod_ens_ranked1")

all_models_with_postValiMods_dd = add_post_vali_mods(
    df_dd = all_models_with_postValiMods_dd,
    ensemble_weights = ranked5_ensemble_weights,
    ensemble_name = "postValiMod_ens_ranked5")

all_models_with_postValiMods_dd = all_models_with_postValiMods_dd[
    ["postValiMod_ens_ranked1","postValiMod_ens_ranked5"]]

## compute dask graph and save
all_models_with_postValiMods_dd.to_hdf(
    path_or_buf = post_vali_models_store,
    key = "/post_vali_models/forecast",
    compute = True,
    complevel = 9)

## checks
### make sure that ensembles don't have missing values
with pd.HDFStore(post_vali_models_store, mode="r") as s:
    dfcheck = s.select("/post_vali_models/forecast", columns=["postValiMod_ens_ranked1","postValiMod_ens_ranked5"])

assert(all(pd.isnull(dfcheck).sum()==0))

del dfcheck

### check that number of rows is as expected
with pd.HDFStore(post_vali_models_store, mode="r") as s:
    n1 = s.get_storer("/post_vali_models/forecast").nrows

n2 = len(all_models_dd)

assert(n1==n2)
del n1, n2

### print first few rows
with pd.HDFStore(post_vali_models_store, mode="r") as s:
    dfcheck = s.select("/post_vali_models/forecast", stop=10)
print(dfcheck.to_string())
del dfcheck



# calculate losses for the post-val ensemble models on the validation set
def helperfn1(df, xvars=["postValiMod_ens_ranked1","postValiMod_ens_ranked5"], yvar="daily_untouched"):
    df1 = [smape(F=df[x], A=df[yvar]) for x in xvars]
    df1 = pd.Series(df1, index=xvars)
    return(df1)

## at the overall level
with pd.HDFStore(all_models_store_path, mode="r") as s:
    post_vali_models_valicast = s.select("/post_vali_models/valicast")

losses_postValiMods_overall = helperfn1(post_vali_models_valicast)

## at the page level
losses_postValiMods_byPage = post_vali_models_valicast.groupby("ts_id").apply(helperfn1)

### distributions of results at the page level (where no model has a missing projection)
losses_postValiMods_byPage_distrib = losses_postValiMods_byPage.dropna(how="any").describe()

print(losses_postValiMods_byPage_distrib)

## save
losses_postValiMods_overall.to_csv(outputs_nb_fldrpath + "/losses_postValiMods_overall.csv")
losses_postValiMods_byPage_distrib.to_csv(outputs_nb_fldrpath + "/losses_postValiMods_byPage_distrib.csv")

savelist = [
    losses_postValiMods_overall,
    losses_postValiMods_byPage,
    losses_postValiMods_byPage_distrib
]

pickle.dump(savelist, open(data_intermed_nb_fldrpath + "/ensemble_choice_postValiMods.p", "wb"))
