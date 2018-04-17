
# setup
## import libraries
import pandas as pd
import dask.dataframe as dd
import pickle

## set paths
data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"
output_fldrpath = "/media/disk1/forecast_wiki_traffic/output"

## set up dask dataframes
ts_daily_long_dd = dd.read_hdf(data_intermed_nb_fldrpath + "/ts_store.h5", key="/ts_daily_long")


# set up dictionary that will hold data to save
d = dict()

# pull tables
with pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r") as s:
    page_to_id_map = s.select("/page_to_id_map")
    ts_stn_params = s.select("/ts_stn_params")

d["page_to_id_map"] = page_to_id_map
d["ts_stn_params"] = ts_stn_params

with pd.HDFStore(data_intermed_nb_fldrpath + "/summary_stats_store.h5", mode="r") as s:
    agg_stat_dayofweek_sum = s.select("/agg_stat_dayofweek", columns=["sum"])
    pageview_sum = s.select("/ts_stat_overall", columns=["sum"])
    pageview_median = s.select("/ts_stat_overall", columns=["median"])

d["agg_stat_dayofweek_sum"] = agg_stat_dayofweek_sum
d["pageview_sum"] = pageview_sum
d["pageview_median"] = pageview_median

# pull modeling data example time series
## Abraham Lincoln, id 2968
with pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r") as s:
    ex1d_plotdata = s.select(key="ts_daily_long",
                             where="ts_id==2968")
    ex1dStn_plotdata = s.select(key="ts_daily_stn_long",
                             where="ts_id==2968")
    ex1w_plotdata = s.select(key="ts_weekly_long",
                             where="ts_id==2968")

ex1d_plotdata.index = ex1d_plotdata.index.droplevel(level=0)
ex1dStn_plotdata.index = ex1dStn_plotdata.index.droplevel(level=0)
ex1w_plotdata.index = ex1w_plotdata.index.droplevel(level=0)

d["ex1d_plotdata"] = ex1d_plotdata
d["ex1dStn_plotdata"] = ex1dStn_plotdata
d["ex1w_plotdata"] = ex1w_plotdata

## Aggregated: English Wiki, all access, all agents
with pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r") as s:
    ex2d_plotdata = s.select(key="agg_daily_long",
                             where=["project=='en.wikipedia.org'","access='all-access'","agent='all-agents'"])
    ex2w_plotdata = s.select(key="agg_weekly_long",
                             where=["project=='en.wikipedia.org'","access='all-access'","agent='all-agents'"])

ex2d_plotdata.index = ex2d_plotdata.index.droplevel(level=[0,1,2])
ex2w_plotdata.index = ex2w_plotdata.index.droplevel(level=[0,1,2])

d["ex2d_plotdata"] = ex2d_plotdata
d["ex2w_plotdata"] = ex2w_plotdata

## Aggregated: German Wiki, all access, all agents
with pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r") as s:
    ex9d_plotdata = s.select(key="agg_daily_long",
                             where=["project=='de.wikipedia.org'","access='all-access'","agent='all-agents'"])
    ex9w_plotdata = s.select(key="agg_weekly_long",
                             where=["project=='de.wikipedia.org'","access='all-access'","agent='all-agents'"])

ex9d_plotdata.index = ex9d_plotdata.index.droplevel(level=[0,1,2])
ex9w_plotdata.index = ex9w_plotdata.index.droplevel(level=[0,1,2])

d["ex9d_plotdata"] = ex9d_plotdata
d["ex9w_plotdata"] = ex9w_plotdata

## Halloween, id 41431 -- example for holiday adjustment
with pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r") as s:
    ex_halloween_d_plotdata = s.select(key="ts_daily_long",
                             where="ts_id==41431")
ex_halloween_d_plotdata.index = ex_halloween_d_plotdata.index.droplevel(level=0)

d["ex_halloween_d_plotdata"] = ex_halloween_d_plotdata

## 500 Days of Summer, id 31 -- example for stationarity adjustment
with pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r") as s:
    ex3d_plotdata = s.select(key="ts_daily_long",
                             where="ts_id==31")
    ex3dStn_plotdata = s.select(key="ts_daily_stn_long",
                             where="ts_id==31")

ex3d_plotdata.index = ex3d_plotdata.index.droplevel(level=0)
ex3dStn_plotdata.index = ex3dStn_plotdata.index.droplevel(level=0)

d["ex3d_plotdata"] = ex3d_plotdata
d["ex3dStn_plotdata"] = ex3dStn_plotdata

# pull fitted values example time series
example_model_time_series_d = dict()

with pd.HDFStore(data_intermed_nb_fldrpath + "/all_models_store.h5", mode="r") as s:
    ## Abraham Lincoln, id 2968
    ## java, en, all access, all agents, id 48258
    ## main page, en, all access, all agents, id 58760
    ## queen mary, en, all access, all agents, id 61083
    ## python, en, desktop, all agents, id 72937
    ## Marathonlauf, de, mobile, all agents, id 59948
    for my_ts_id in [2968, 48258, 58760, 61083, 72937, 59948]:
        my_models_vali = s.select("/all_models_Rehol_NotPlugged/valicast", where=["ts_id=={}".format(str(my_ts_id))])
        my_models_vali.index = my_models_vali.index.droplevel(level=0)

        my_models_fore = s.select("/all_models_Rehol_NotPlugged/forecast", where=["ts_id=={}".format(str(my_ts_id))])
        my_models_fore.index = my_models_fore.index.droplevel(level=0)

        my_postvali_models_vali = s.select("/post_vali_models/valicast", where=["ts_id=={}".format(str(my_ts_id))])
        my_postvali_models_vali.index = my_postvali_models_vali.index.droplevel(level=0)

        my_postvali_models_fore = s.select("/post_vali_models/forecast", where=["ts_id=={}".format(str(my_ts_id))])
        my_postvali_models_fore.index = my_postvali_models_fore.index.droplevel(level=0)

        my_fore_solution = s.select("/solution", where=["ts_id=={}".format(str(my_ts_id))])
        my_fore_solution.index = my_fore_solution.index.droplevel(level=["ts_id","Id"])
        my_fore_solution = my_fore_solution[["Visits"]]
        my_fore_solution.columns = ["daily_untouched"]

        example_model_time_series_d[my_ts_id] = {"comp_vali":my_models_vali, "comp_fore":my_models_fore,
            "postvali_vali":my_postvali_models_vali, "postvali_fore":my_postvali_models_fore,
            "fore_solution": my_fore_solution}

d["example_model_time_series_d"] = example_model_time_series_d


# calculate missing value statistics
missing_counts_by_obs = ts_daily_long_dd.loc[:,["missing_type"]]\
    .groupby("missing_type").size().compute()
missing_counts_by_ts = ts_daily_long_dd.loc[:,["missing_type"]]\
    .groupby(["ts_id","missing_type"]).size().compute()

d["missing_counts_by_obs"] = missing_counts_by_obs
d["missing_counts_by_ts"] = missing_counts_by_ts

# most visited pages in English locale, all access, all agents, excluding special pages
pageview_sum = pd.merge(pageview_sum, page_to_id_map, left_index=True, right_index=True)

pageview_sum.sort_values(["sum"], ascending=False, inplace=True)

pageview_sum_subset = pageview_sum[
    (pageview_sum.project=="en.wikipedia.org") &
    (pageview_sum.access=="all-access") &
    (pageview_sum.agent=="all-agents") &
    ~(pageview_sum.name.str.contains("Special:")) &
    ~(pageview_sum.name.str.startswith("X"))
    ]

most_visited_200 = pageview_sum_subset[:200]

d["most_visited_200"] = most_visited_200


# create copies of model ranks and ensemble weights in the summary dict
losses_byPage_modelRanked = pickle.load(open(data_intermed_nb_fldrpath + "/losses_byPage_modelRanked.p", "rb"))
d["losses_byPage_modelRanked"] = losses_byPage_modelRanked

d1 = pickle.load(open(data_intermed_nb_fldrpath + "/postvali_ensemble_weights.p", "rb"))
d["ranked1_ensemble_weights"] = d1["ranked1_ensemble_weights"]
d["ranked5_ensemble_weights"] = d1["ranked5_ensemble_weights"]
del d1


# save
pickle.dump(d, open(output_fldrpath + "/summary_report_data.p", "wb"))
