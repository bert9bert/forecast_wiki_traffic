
import pandas as pd
import numpy as np
import more_itertools

import projutils_backtrans


def create_mapper_input(chunk_y_hist, chunk_y_hist_weekly,
    chunk_intraweek_seasonal_df,
    chunk_y_daily_level_forecast, chunk_y_daily_wowGr_forecast, chunk_y_weekly_level_forecast, chunk_y_weekly_wowGr_forecast,
    chunk_y_daily_level_valicast, chunk_y_daily_wowGr_valicast, chunk_y_weekly_level_valicast, chunk_y_weekly_wowGr_valicast,
    chunk_y_daily_level_fitted, chunk_y_daily_wowGr_fitted, chunk_y_weekly_level_fitted, chunk_y_weekly_wowGr_fitted):

    # get the pages in this chunk
    pages_list = chunk_y_daily_level_forecast.index.values
    num_levels = len(pages_list[0]) - 1
    pages_list = [x[:num_levels] for x in pages_list]

    pages_list = list(more_itertools.unique_everseen(pages_list))

    # create vector of inputs for mapper function
    mapper_inputs = []

    for k in pages_list:
        ## create blank dict to store inputs for this page
        inputs_this = dict()

        ## store key
        inputs_this["k"] = k

        ## store historical data
        inputs_this["y_hist"] = chunk_y_hist.xs(key=k, level=list(range(num_levels)), drop_level=False)
        inputs_this["y_hist_weekly"] = chunk_y_hist_weekly.xs(key=k, level=list(range(num_levels)), drop_level=False)

        ## store seasonal distribution
        intraweek_seasonal_df = chunk_intraweek_seasonal_df.xs(key=k, level=list(range(num_levels)), drop_level=False)

        intraweek_seasonal_df = np.array(intraweek_seasonal_df)
        intraweek_seasonal_df = intraweek_seasonal_df[:,0]

        if not all(intraweek_seasonal_df==0):
            ### (if not all zeros then distribute seasonality according to historical trend)
            intraweek_seasonal_df = intraweek_seasonal_df/np.sum(intraweek_seasonal_df)
        else:
            ### (otherwise distribute evenly)
            intraweek_seasonal_df = np.repeat(1/7,7)


        inputs_this["intraweek_seasonal_dist"]  = intraweek_seasonal_df

        del intraweek_seasonal_df

        ## store inputs for forecast
        inputs_this["y_daily_level_forecast"] = chunk_y_daily_level_forecast.xs(key=k, level=list(range(num_levels)), drop_level=False)
        inputs_this["y_daily_wowGr_forecast"] = chunk_y_daily_wowGr_forecast.xs(key=k, level=list(range(num_levels)), drop_level=False)
        inputs_this["y_weekly_level_forecast"] = chunk_y_weekly_level_forecast.xs(key=k, level=list(range(num_levels)), drop_level=False)
        inputs_this["y_weekly_wowGr_forecast"] = chunk_y_weekly_wowGr_forecast.xs(key=k, level=list(range(num_levels)), drop_level=False)

        ## store inputs for valicast
        inputs_this["y_daily_level_valicast"] = chunk_y_daily_level_valicast.xs(key=k, level=list(range(num_levels)), drop_level=False)
        inputs_this["y_daily_wowGr_valicast"] = chunk_y_daily_wowGr_valicast.xs(key=k, level=list(range(num_levels)), drop_level=False)
        inputs_this["y_weekly_level_valicast"] = chunk_y_weekly_level_valicast.xs(key=k, level=list(range(num_levels)), drop_level=False)
        inputs_this["y_weekly_wowGr_valicast"] = chunk_y_weekly_wowGr_valicast.xs(key=k, level=list(range(num_levels)), drop_level=False)

        ## store inputs for fitted
        inputs_this["y_daily_level_fitted"] = chunk_y_daily_level_fitted.xs(key=k, level=list(range(num_levels)), drop_level=False)
        inputs_this["y_daily_wowGr_fitted"] = chunk_y_daily_wowGr_fitted.xs(key=k, level=list(range(num_levels)), drop_level=False)
        inputs_this["y_weekly_level_fitted"] = chunk_y_weekly_level_fitted.xs(key=k, level=list(range(num_levels)), drop_level=False)
        inputs_this["y_weekly_wowGr_fitted"] = chunk_y_weekly_wowGr_fitted.xs(key=k, level=list(range(num_levels)), drop_level=False)

        ## put into mapper input vector
        mapper_inputs.append(inputs_this)

    del inputs_this

    return mapper_inputs


def mapperfn(v):
    z_forecast_this = projutils_backtrans.back_transformations(v["y_hist"], v["y_hist_weekly"],
        v["y_daily_level_forecast"], v["y_daily_wowGr_forecast"], v["y_weekly_level_forecast"], v["y_weekly_wowGr_forecast"],
        v["intraweek_seasonal_dist"])

    z_valicast_this = projutils_backtrans.back_transformations(v["y_hist"], v["y_hist_weekly"],
        v["y_daily_level_valicast"], v["y_daily_wowGr_valicast"], v["y_weekly_level_valicast"], v["y_weekly_wowGr_valicast"],
        v["intraweek_seasonal_dist"])

    z_fitted_this = projutils_backtrans.back_transformations(v["y_hist"], v["y_hist_weekly"],
        v["y_daily_level_fitted"], v["y_daily_wowGr_fitted"], v["y_weekly_level_fitted"], v["y_weekly_wowGr_fitted"],
        v["intraweek_seasonal_dist"])

    output = dict()
    output["k"] = v["k"]
    output["z_forecast"] = z_forecast_this
    output["z_valicast"] = z_valicast_this
    output["z_fitted"] = z_fitted_this

    return(output)



def reducerfn(mapper_output):
    z_forecast_reduced = pd.concat([d["z_forecast"] for d in mapper_output])
    z_valicast_reduced = pd.concat([d["z_valicast"] for d in mapper_output])
    z_fitted_reduced = pd.concat([d["z_fitted"] for d in mapper_output])

    output = dict()
    output["z_forecast"] = z_forecast_reduced
    output["z_valicast"] = z_valicast_reduced
    output["z_fitted"] = z_fitted_reduced

    return(output)
