# setup and paths
## prebuilt libraries
import pandas as pd
import progressbar as pb
from multiprocessing import Pool, cpu_count
from importlib import reload

## custom and project specific libraries
import helperfns_backtrans_mapreduce

if __name__ == "__main__":
    ## paths
    data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"

    # runtime params
    NUM_TS_PER_DATA_CHUNK = 4096

    NUM_PROCESSES = cpu_count()
    PARALLEL_CHUNKSIZE = 512


    # set up hdf5 storage
    ## set up read-only tables
    ts_store = pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r")
    sarima_ts_store = pd.HDFStore(data_intermed_nb_fldrpath + "/sarima_ts_store.h5", mode="r")
    summary_stats_store = pd.HDFStore(data_intermed_nb_fldrpath + "/summary_stats_store.h5", mode="r")

    ## set up write tables
    sarima_ts_backtrans_store = pd.HDFStore(data_intermed_nb_fldrpath + "/sarima_ts_backtrans_store.h5")

    ## remove potentially stale data
    try:
        del sarima_ts_backtrans_store["sarima_ts_backtrans_to_dailyts/forecast"]
        del sarima_ts_backtrans_store["sarima_ts_backtrans_to_dailyts/valicast"]
        del sarima_ts_backtrans_store["sarima_ts_backtrans_to_dailyts/fitted"]
    except:
        pass

    # get counts
    ## get number of time periods for each time series
    ts0 = ts_store["page_to_id_map"].index.values[0]

    num_t_daily_modeling = len(ts_store.select(key="ts_daily_long", where="ts_id==%i" % ts0))
    num_t_weekly_modeling = len(ts_store.select(key="ts_weekly_long", where="ts_id==%i" % ts0))
    num_t_daily_forecast = len(sarima_ts_store.select(key="sarima_forecast_ts_daily_level", where="ts_id==%i" % ts0))
    num_t_weekly_forecast = len(sarima_ts_store.select(key="sarima_forecast_ts_weekly_level", where="ts_id==%i" % ts0))
    num_t_daily_valicast = len(sarima_ts_store.select(key="sarima_valicast_ts_daily_level", where="ts_id==%i" % ts0))
    num_t_weekly_valicast = len(sarima_ts_store.select(key="sarima_valicast_ts_weekly_level", where="ts_id==%i" % ts0))

    del ts0

    ## get number of time series
    num_timeseries = ts_store.get_storer("ts_daily_long").nrows / num_t_daily_modeling
    assert(num_timeseries-int(num_timeseries)==0)
    num_timeseries = num_timeseries.astype(int)

    ## assertion checks
    assert(ts_store.get_storer("ts_daily_long").nrows == num_t_daily_modeling * num_timeseries)
    assert(ts_store.get_storer("ts_weekly_long").nrows == num_t_weekly_modeling * num_timeseries)

    assert(summary_stats_store.get_storer("ts_stat_dayofweek").nrows == 7 * num_timeseries)

    assert(sarima_ts_store.get_storer("sarima_forecast_ts_daily_level").nrows == num_t_daily_forecast * num_timeseries)
    assert(sarima_ts_store.get_storer("sarima_forecast_ts_daily_wowGr").nrows == num_t_daily_forecast * num_timeseries)
    assert(sarima_ts_store.get_storer("sarima_forecast_ts_weekly_level").nrows == num_t_weekly_forecast * num_timeseries)
    assert(sarima_ts_store.get_storer("sarima_forecast_ts_weekly_wowGr").nrows == num_t_weekly_forecast * num_timeseries)

    assert(sarima_ts_store.get_storer("sarima_valicast_ts_daily_level").nrows == num_t_daily_valicast * num_timeseries)
    assert(sarima_ts_store.get_storer("sarima_valicast_ts_daily_wowGr").nrows == num_t_daily_valicast * num_timeseries)
    assert(sarima_ts_store.get_storer("sarima_valicast_ts_weekly_level").nrows == num_t_weekly_valicast * num_timeseries)
    assert(sarima_ts_store.get_storer("sarima_valicast_ts_weekly_wowGr").nrows == num_t_weekly_valicast * num_timeseries)

    assert(sarima_ts_store.get_storer("sarima_fitted_ts_daily_level").nrows == num_t_daily_modeling * num_timeseries)
    assert(sarima_ts_store.get_storer("sarima_fitted_ts_daily_wowGr").nrows == num_t_daily_modeling * num_timeseries)
    assert(sarima_ts_store.get_storer("sarima_fitted_ts_weekly_level").nrows == num_t_weekly_modeling * num_timeseries)
    assert(sarima_ts_store.get_storer("sarima_fitted_ts_weekly_wowGr").nrows == num_t_weekly_modeling * num_timeseries)





    # create iterators for to go through data in chunks
    SEASONAL_VAR = "trunc60_sum60"

    ## set iteration params
    datachunk_size_daily_modeling = NUM_TS_PER_DATA_CHUNK * num_t_daily_modeling
    datachunk_size_weekly_modeling = NUM_TS_PER_DATA_CHUNK * num_t_weekly_modeling

    datachunk_size_seasonal_distribution = NUM_TS_PER_DATA_CHUNK * 7

    datachunk_size_daily_forecast = NUM_TS_PER_DATA_CHUNK * num_t_daily_forecast
    datachunk_size_weekly_forecast = NUM_TS_PER_DATA_CHUNK * num_t_weekly_forecast

    datachunk_size_daily_valicast = NUM_TS_PER_DATA_CHUNK * num_t_daily_valicast
    datachunk_size_weekly_valicast = NUM_TS_PER_DATA_CHUNK * num_t_weekly_valicast

    ## historical data iterators
    iter_chunk_y_hist = ts_store.select("ts_daily_long",
        columns=["daily_level"],
        iterator=True, chunksize=datachunk_size_daily_modeling).__iter__()
    iter_chunk_y_hist_weekly = ts_store.select("ts_weekly_long",
        columns=["weekly_level"],
        iterator=True, chunksize=datachunk_size_weekly_modeling).__iter__()

    ## intraweek seasonal distribution iterator
    iter_chunk_intraweek_seasonal_df = summary_stats_store.select("ts_stat_dayofweek",
        columns=[SEASONAL_VAR],
        iterator=True, chunksize=datachunk_size_seasonal_distribution).__iter__()

    ## inputs for forecast iterator

    iter_chunk_y_daily_level_forecast = sarima_ts_store.select("sarima_forecast_ts_daily_level",
        columns=["daily_level_pred"],
        iterator=True, chunksize=datachunk_size_daily_forecast).__iter__()
    iter_chunk_y_daily_wowGr_forecast = sarima_ts_store.select("sarima_forecast_ts_daily_wowGr",
        columns=["daily_wowGr_pred"],
        iterator=True, chunksize=datachunk_size_daily_forecast).__iter__()
    iter_chunk_y_weekly_level_forecast = sarima_ts_store.select("sarima_forecast_ts_weekly_level",
        columns=["weekly_level_pred"],
        iterator=True, chunksize=datachunk_size_weekly_forecast).__iter__()
    iter_chunk_y_weekly_wowGr_forecast = sarima_ts_store.select("sarima_forecast_ts_weekly_wowGr",
        columns=["weekly_wowGr_pred"],
        iterator=True, chunksize=datachunk_size_weekly_forecast).__iter__()

    ## inputs for valicast iterator

    iter_chunk_y_daily_level_valicast = sarima_ts_store.select("sarima_valicast_ts_daily_level",
        columns=["daily_level_pred"],
        iterator=True, chunksize=datachunk_size_daily_valicast).__iter__()
    iter_chunk_y_daily_wowGr_valicast = sarima_ts_store.select("sarima_valicast_ts_daily_wowGr",
        columns=["daily_wowGr_pred"],
        iterator=True, chunksize=datachunk_size_daily_valicast).__iter__()
    iter_chunk_y_weekly_level_valicast = sarima_ts_store.select("sarima_valicast_ts_weekly_level",
        columns=["weekly_level_pred"],
        iterator=True, chunksize=datachunk_size_weekly_valicast).__iter__()
    iter_chunk_y_weekly_wowGr_valicast = sarima_ts_store.select("sarima_valicast_ts_weekly_wowGr",
        columns=["weekly_wowGr_pred"],
        iterator=True, chunksize=datachunk_size_weekly_valicast).__iter__()

    ## inputs for fitted iterator

    iter_chunk_y_daily_level_fitted = sarima_ts_store.select("sarima_fitted_ts_daily_level",
        columns=["daily_level_pred"],
        iterator=True, chunksize=datachunk_size_daily_modeling).__iter__()
    iter_chunk_y_daily_wowGr_fitted = sarima_ts_store.select("sarima_fitted_ts_daily_wowGr",
        columns=["daily_wowGr_pred"],
        iterator=True, chunksize=datachunk_size_daily_modeling).__iter__()
    iter_chunk_y_weekly_level_fitted = sarima_ts_store.select("sarima_fitted_ts_weekly_level",
        columns=["weekly_level_pred"],
        iterator=True, chunksize=datachunk_size_weekly_modeling).__iter__()
    iter_chunk_y_weekly_wowGr_fitted = sarima_ts_store.select("sarima_fitted_ts_weekly_wowGr",
        columns=["weekly_wowGr_pred"],
        iterator=True, chunksize=datachunk_size_weekly_modeling).__iter__()


    # loop through chunks and process
    num_timeseries_processed = 0

    ## set up parallel pool
    if NUM_PROCESSES > 1:
        p = Pool(processes=NUM_PROCESSES)
    elif NUM_PROCESSES == 1:
        p = None


    ## initialize progress bar
    widgets = ["Progress for {} time series: ".format(int(num_timeseries)), pb.Percentage(), ' ',
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
    pbtimer = pb.ProgressBar(widgets=widgets, maxval=num_timeseries).start()

    while num_timeseries_processed < num_timeseries:
        ## get next in data chunk iterators
        chunk_y_hist = next(iter_chunk_y_hist)
        chunk_y_hist_weekly = next(iter_chunk_y_hist_weekly)

        chunk_intraweek_seasonal_df = next(iter_chunk_intraweek_seasonal_df)

        chunk_y_daily_level_forecast = next(iter_chunk_y_daily_level_forecast)
        chunk_y_daily_wowGr_forecast = next(iter_chunk_y_daily_wowGr_forecast)
        chunk_y_weekly_level_forecast = next(iter_chunk_y_weekly_level_forecast)
        chunk_y_weekly_wowGr_forecast = next(iter_chunk_y_weekly_wowGr_forecast)

        chunk_y_daily_level_valicast = next(iter_chunk_y_daily_level_valicast)
        chunk_y_daily_wowGr_valicast = next(iter_chunk_y_daily_wowGr_valicast)
        chunk_y_weekly_level_valicast = next(iter_chunk_y_weekly_level_valicast)
        chunk_y_weekly_wowGr_valicast = next(iter_chunk_y_weekly_wowGr_valicast)

        chunk_y_daily_level_fitted = next(iter_chunk_y_daily_level_fitted)
        chunk_y_daily_wowGr_fitted = next(iter_chunk_y_daily_wowGr_fitted)
        chunk_y_weekly_level_fitted = next(iter_chunk_y_weekly_level_fitted)
        chunk_y_weekly_wowGr_fitted = next(iter_chunk_y_weekly_wowGr_fitted)


        # apply back-transformations
        mapper_inputs = helperfns_backtrans_mapreduce.create_mapper_input(chunk_y_hist, chunk_y_hist_weekly,
            chunk_intraweek_seasonal_df,
            chunk_y_daily_level_forecast, chunk_y_daily_wowGr_forecast, chunk_y_weekly_level_forecast, chunk_y_weekly_wowGr_forecast,
            chunk_y_daily_level_valicast, chunk_y_daily_wowGr_valicast, chunk_y_weekly_level_valicast, chunk_y_weekly_wowGr_valicast,
            chunk_y_daily_level_fitted, chunk_y_daily_wowGr_fitted, chunk_y_weekly_level_fitted, chunk_y_weekly_wowGr_fitted)

        if NUM_PROCESSES is None:
            mapper_output = [helperfns_backtrans_mapreduce.mapperfn(v) for v in mapper_inputs]
        else:
            mapper_output = p.map(helperfns_backtrans_mapreduce.mapperfn, mapper_inputs, chunksize=PARALLEL_CHUNKSIZE)

        reducer_output = helperfns_backtrans_mapreduce.reducerfn(mapper_output)


        # store the processed data
        sarima_ts_backtrans_store.append(
            "sarima_ts_backtrans_to_dailyts/forecast",
            reducer_output["z_forecast"],
            format = "table",
            index = False,
            expectedrows = num_t_daily_forecast * num_timeseries,
            complevel=9
        )

        sarima_ts_backtrans_store.append(
            "sarima_ts_backtrans_to_dailyts/valicast",
            reducer_output["z_valicast"],
            format = "table",
            index = False,
            expectedrows = num_t_daily_valicast * num_timeseries,
            complevel=9
        )

        sarima_ts_backtrans_store.append(
            "sarima_ts_backtrans_to_dailyts/fitted",
            reducer_output["z_fitted"],
            format = "table",
            index = False,
            expectedrows = num_t_daily_modeling * num_timeseries,
            complevel=9
        )

        # increment counter
        num_timeseries_processed += NUM_TS_PER_DATA_CHUNK
        num_timeseries_processed = min(num_timeseries_processed, num_timeseries)

        # update progress bar
        pbtimer.update(num_timeseries_processed)

    pbtimer.finish()

    # create indices
    sarima_ts_backtrans_store.create_table_index("sarima_ts_backtrans_to_dailyts/forecast")
    sarima_ts_backtrans_store.create_table_index("sarima_ts_backtrans_to_dailyts/valicast")
    sarima_ts_backtrans_store.create_table_index("sarima_ts_backtrans_to_dailyts/fitted")

    # check counts
    assert(sarima_ts_backtrans_store.get_storer("sarima_ts_backtrans_to_dailyts/forecast").nrows \
        == num_t_daily_forecast * num_timeseries)
    assert(sarima_ts_backtrans_store.get_storer("sarima_ts_backtrans_to_dailyts/valicast").nrows \
        == num_t_daily_valicast * num_timeseries)
    assert(sarima_ts_backtrans_store.get_storer("sarima_ts_backtrans_to_dailyts/fitted").nrows \
        == num_t_daily_modeling * num_timeseries)

    # clean up
    ## hdf5 stores
    ts_store.close()
    sarima_ts_store.close()
    summary_stats_store.close()

    sarima_ts_backtrans_store.close()

    ## parallel pools
    if NUM_PROCESSES > 1:
        p.close()
        p.join()
