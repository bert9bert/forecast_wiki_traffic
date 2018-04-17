
# import libraries
## pre-built libraries
import numpy as np
import pandas as pd
from collections import Counter
import time
import more_itertools
from multiprocessing import Pool
import progressbar as pb

## custom and project specific libraries
import projutils_ts


NUM_VALIDATION_PERIODS_DAILY = 60
NUM_FORECAST_PERIODS_DAILY = 60 + 10
NUM_VALIDATION_PERIODS_WEEKLY = 10
NUM_FORECAST_PERIODS_WEEKLY = 10 + 2



def create_mapper_input(store, tbl_series, stn_params_df, var, freq, start=None, stop=None):
    """Creates list of input vectors for the mapper function"""

    # input checks
    assert(freq in ["D","W"])

    # pull out time series and stationary param chunks
    ts_chunk = store.select(tbl_series, columns=[var], start=start, stop=stop)

    stn_params_chunk = stn_params_df

    # create a list of all pages
    k_all = ts_chunk.index.values.tolist()
    k_num_idx_levels = len(k_all[0]) - 1
    k_all = [x[:k_num_idx_levels] for x in k_all]
    k_all = list(more_itertools.unique_everseen(k_all))

    # get index names
    index_names = list(ts_chunk.index.names)
    index_names = index_names[:k_num_idx_levels]

    # create mapper input vector of just the time series and the stationary params
    mapper_input_vec = []

    for k in k_all:
        ts_k = ts_chunk.xs(k, level=list(range(k_num_idx_levels)), drop_level=True)

        stn_params_k = stn_params_chunk.xs(k, level=list(range(k_num_idx_levels)), drop_level=True)
        stn_params_k = stn_params_k.loc[var]

        mapper_input_vec.append([ts_k, stn_params_k])

    # add other attributes to the mapper input vector
    if freq=="D":
        S_k = 7

        n_validation_k = NUM_VALIDATION_PERIODS_DAILY
        n_forecast_k = NUM_FORECAST_PERIODS_DAILY
    elif freq=="W":
        S_k = 1

        n_validation_k = NUM_VALIDATION_PERIODS_WEEKLY
        n_forecast_k = NUM_FORECAST_PERIODS_WEEKLY
    else:
        raise Exception

    mapper_input_vec = [(x[0], x[1], S_k, n_validation_k, n_forecast_k) for x in mapper_input_vec]

    # input
    return(mapper_input_vec, k_all, index_names)



def mapperfn(inputs):
    # input checks
    assert(isinstance(inputs, tuple))
    assert(len(inputs)==5)

    # pull out input tuple components
    ts_k, stn_params_k, S_k, n_validation_k, n_forecast_k = inputs


    boxcox_lambda_map = {"asis":1, "log":0}

    if stn_params_k["found_stn"]:
        # fit model if stationary params were found
        sarima_model_spec, y_fitted, y_forecast, y_valicast = projutils_ts.sarima_stepsearch(
            y = ts_k,
            d = stn_params_k["d"], D = stn_params_k["D"], S = S_k,
            max_p = 5, max_q = 5, max_P = 2, max_Q = 2,
            boxcox_lambda = boxcox_lambda_map[stn_params_k["func"]],
            n_validation = n_validation_k, n_forecast = n_forecast_k,
            cost_method = "validation_smape", method = "CSS",
            trace=False, verbose=False
        )
    else:
        # otherwise return missing values
        ## model spec should be None
        sarima_model_spec = None

        ## fitted values should be vector of nan
        y_fitted = pd.DataFrame(np.full(len(ts_k), np.nan), index=ts_k.index, columns=[ts_k.iloc[:,0].name+"_pred"])

        ## forecast values should be vector of nan
        ts_k_forecast = np.full(n_forecast_k, np.nan)

        days_btwn = (ts_k.index[-1] - ts_k.index[-2]).days
        if days_btwn==1:
            ts_k_freq = "D"
        elif days_btwn==7:
            ts_k_freq = "W"
        else:
            raise Exception("Unexpected frequency")
        del days_btwn

        ts_k_forecast_index = pd.date_range(start=ts_k.index.max(), periods=len(ts_k_forecast)+1, freq=ts_k_freq)[1:]

        y_forecast = pd.DataFrame(ts_k_forecast, index=ts_k_forecast_index, columns=[ts_k.iloc[:,0].name + "_pred"])

        del ts_k_forecast, ts_k_forecast_index

        ## validation-forecast values should be vector of nan
        y_valicast = np.full(n_validation_k, np.nan)
        y_valicast_index = ts_k[-n_validation_k:].index

        y_valicast = pd.DataFrame(y_valicast, index=y_valicast_index, columns=[ts_k.iloc[:,0].name + "_pred"])

        del y_valicast_index

    # return
    output = (sarima_model_spec, y_fitted, y_forecast, y_valicast)
    return(output)


def reducerfn(mapper_output_vec, k_all, index_names):
    # reduce to create dataframe of model specs
    sarima_model_spec_df = []

    for output in mapper_output_vec:
        if output[0] is not None:
            x = output[0].copy()

            row = []
            row.extend(x["nonseas_order"])
            row.extend(x["seas_order"])
            row.append(x["include_intercept"])

            sarima_model_spec_df.append(row)

            del x
        else:
            sarima_model_spec_df.append([np.nan]*8)

    sarima_model_spec_df = pd.DataFrame(sarima_model_spec_df,
                                       columns=["p","d","q","P","D","Q","S","include_intercept"])

    sarima_model_spec_df = pd.concat(
        [pd.DataFrame(k_all, columns=index_names), sarima_model_spec_df],
        axis=1
    )

    sarima_model_spec_df.set_index(index_names, inplace=True)

    # reduce to create dataframes of fitted and forecast values
    fitted_df = []
    forecast_df = []
    valicast_df = []

    i = 0
    for output in mapper_output_vec:
        y = output[1].copy()  # pull out fitted
        y = pd.concat([y, output[2]], axis=0)  # pull out forecast
        y = pd.concat([y, output[3]], axis=0)  # pull out  validation forecast

        n1 = len(output[1])
        n2 = len(output[2])

        for j in range(len(index_names)):
            y[index_names[j]] = k_all[i][j]
            j+=1

        y.set_index(index_names, inplace=True, append=True)

        odr_lvls = list(range(len(index_names)+1))[1:]
        odr_lvls.append(0)
        y = y.reorder_levels(odr_lvls)
        del odr_lvls

        fitted_df.append(y[:n1])
        forecast_df.append(y[n1:(n1+n2)])
        valicast_df.append(y[(n1+n2):])

        i+=1

    del y

    fitted_df = pd.concat(fitted_df, axis=0)
    forecast_df = pd.concat(forecast_df, axis=0)
    valicast_df = pd.concat(valicast_df, axis=0)

    # return
    return(sarima_model_spec_df, fitted_df, forecast_df, valicast_df)


def mapperfn_reducerfn(store, tbl_series, stn_params_df, var, freq,
                       start=None, stop=None,
                       parallel_pool=None, parallel_chunksize=None,
                       verbose = False):

    # create mapper input
    mapper_input_vec, k_all, index_names = create_mapper_input(
        store,
        tbl_series,
        stn_params_df,
        var,
        freq,
        start,
        stop
    )

    # map step
    t0 = time.time()

    # import pdb; pdb.set_trace()

    if parallel_pool is None:
        mapper_output_vec = [mapperfn(v) for v in mapper_input_vec]
    else:
        mapper_output_vec = parallel_pool.map(mapperfn, mapper_input_vec, chunksize=parallel_chunksize)

    tdelta = time.time() - t0

    if verbose:
        print("\nTOTAL TIME ELAPSED FOR MAP: {:.1f} seconds".format(tdelta))
        print("AVERAGE TIME ELAPSED FOR MAP PER TIME SERIES: {:.2f} seconds".format(tdelta/len(mapper_input_vec)))

    # reduce step
    t0 = time.time()

    sarima_model_spec_df, fitted_df, forecast_df, valicast_df = reducerfn(mapper_output_vec, k_all, index_names)

    tdelta = time.time() - t0

    if verbose:
        print("\nTOTAL TIME ELAPSED FOR REDUCE: {:.1f} seconds".format(tdelta))
        print("AVERAGE TIME ELAPSED FOR REDUCE PER TIME SERIES: {:.2f} seconds".format(tdelta/len(mapper_input_vec)))

    # return
    return sarima_model_spec_df, fitted_df, forecast_df, valicast_df




def fit_sarima_ts_parallel(
    store_modeling,
    tbl_series,
    stn_params_df,
    var,
    freq,
    store_fits,
    NUM_TS_PER_DATA_CHUNK = 25,
    NUM_PROCESSES = 4, PARALLEL_CHUNKSIZE = 5,
    debug_opts = None):

    # input checks
    if debug_opts is not None:
        assert(list(debug_opts.keys()) == ["TS_DATA_HARD_START", "TS_DATA_HARD_STOP"])

    # define constants for this function
    if freq=="D":
        n_validation = NUM_VALIDATION_PERIODS_DAILY
        n_forecast = NUM_FORECAST_PERIODS_DAILY
    elif freq=="W":
        n_validation = NUM_VALIDATION_PERIODS_WEEKLY
        n_forecast = NUM_FORECAST_PERIODS_WEEKLY
    else:
        raise Exception

    # start multiprocessing pool
    if NUM_PROCESSES > 1:
        p = Pool(processes=NUM_PROCESSES)
    elif NUM_PROCESSES == 1:
        p = None

    # set up hdf5 storage
    ## remove potentially stale data
    try:
        del store_fits["sarima_spec_ts_{}".format(var)]
        del store_fits["sarima_fitted_ts_{}".format(var)]
        del store_fits["sarima_forecast_ts_{}".format(var)]
        del store_fits["sarima_valicast_ts_{}".format(var)]
    except:
        pass

    ## get counts
    num_t = len(store_modeling.select(key=tbl_series, where="ts_id==%i" % store_modeling["page_to_id_map"].index.values[0]))
    num_dfrows = store_modeling.get_storer(tbl_series).nrows

    num_timeseries = num_dfrows / num_t
    assert(num_timeseries-int(num_timeseries)==0)
    num_timeseries = num_timeseries.astype(int)

    ## set iteration params
    datachunk_size = NUM_TS_PER_DATA_CHUNK * num_t

    # pull time series out of hdf5 storage by chunks,
    # fit models, and store
    if debug_opts is None:
        hard_start = 0
        hard_stop = num_dfrows
    else:
        hard_start = debug_opts["TS_DATA_HARD_START"] * num_t
        hard_stop = debug_opts["TS_DATA_HARD_STOP"] * num_t

    datachunk_start = hard_start
    datachunk_stop = min(hard_start + datachunk_size, hard_stop)

    # initialize progress bar
    widgets = ["Progress for {} time series: ".format(int(hard_stop/num_t)), pb.Percentage(), ' ',
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
    pbtimer = pb.ProgressBar(widgets=widgets, maxval=hard_stop/num_t).start()

    # ==== BEGIN LOOP THROUGH CHUNKS ====
    while datachunk_start < hard_stop:
        # print("----------------------------------------------")
        # print("hard start {}, hard stop {}, start {}, stop {}".format(hard_start, hard_stop, datachunk_start, datachunk_stop))

        # fit with parallel computing
        sarima_fits_ts_var = dict()

        x = mapperfn_reducerfn(
            store = store_modeling,
            tbl_series = tbl_series,
            stn_params_df = stn_params_df,
            var = var,
            freq = freq,
            start = datachunk_start,
            stop = datachunk_stop,
            parallel_pool = p,
            parallel_chunksize = PARALLEL_CHUNKSIZE
        )

        # format outputs
        for v in ["p","d","q","P","D","Q","S","include_intercept"]:
            x[0][v] = x[0][v].astype(np.float)
        del v

        x[0]["var"] = var
        x[0].set_index("var", append=True, inplace=True)
        sarima_fits_ts_var["sarima_model_spec_df"] = x[0]

        sarima_fits_ts_var["fitted_df"] = x[1]
        sarima_fits_ts_var["forecast_df"] = x[2]
        sarima_fits_ts_var["valicast_df"] = x[3]

        del x

        # check counts
        x1 = list(sarima_fits_ts_var["fitted_df"].index.get_level_values("ts_id"))
        x1_counted = Counter(x1)
        x1_counted_values = list(x1_counted.values())

        assert(all(np.array(x1_counted_values) == num_t))
        assert(len(list(set(x1_counted_values))) == 1)

        del x1, x1_counted, x1_counted_values

        # store the fitted models
        store_fits.append(
            "sarima_spec_ts_{}".format(var),
            sarima_fits_ts_var["sarima_model_spec_df"],
            format = "table",
            index = False,
            expectedrows = num_timeseries,
            complevel=9
        )

        store_fits.append(
            "sarima_fitted_ts_{}".format(var),
            sarima_fits_ts_var["fitted_df"],
            format = "table",
            index = False,
            expectedrows = num_dfrows,
            complevel=9
        )

        store_fits.append(
            "sarima_forecast_ts_{}".format(var),
            sarima_fits_ts_var["forecast_df"],
            format = "table",
            index = False,
            expectedrows = num_timeseries * n_forecast,
            complevel=9
        )

        store_fits.append(
            "sarima_valicast_ts_{}".format(var),
            sarima_fits_ts_var["valicast_df"],
            format = "table",
            index = False,
            expectedrows = num_timeseries * n_validation,
            complevel=9
        )

        del sarima_fits_ts_var

        # update progress bar
        pbtimer.update(datachunk_stop/num_t)

        # increment counters
        datachunk_start += datachunk_size
        datachunk_stop += datachunk_size

        if datachunk_start <= hard_stop:
            datachunk_stop = min(datachunk_stop, hard_stop)

    # ==== END LOOP THROUGH CHUNKS ====
    pbtimer.finish()

    # create indices
    store_fits.create_table_index("sarima_spec_ts_{}".format(var))
    store_fits.create_table_index("sarima_fitted_ts_{}".format(var))
    store_fits.create_table_index("sarima_forecast_ts_{}".format(var))
    store_fits.create_table_index("sarima_valicast_ts_{}".format(var))

    # check counts
    if debug_opts is None:
        assert(store_fits.get_storer("sarima_spec_ts_{}".format(var)).nrows == num_timeseries)
        assert(store_fits.get_storer("sarima_fitted_ts_{}".format(var)).nrows == num_dfrows)
        assert(store_fits.get_storer("sarima_forecast_ts_{}".format(var)).nrows == num_timeseries * n_forecast)
        assert(store_fits.get_storer("sarima_valicast_ts_{}".format(var)).nrows == num_timeseries * n_validation)

    # clean up
    store_modeling.close()
    store_fits.close()

    if NUM_PROCESSES > 1:
        p.close()
        p.join()
