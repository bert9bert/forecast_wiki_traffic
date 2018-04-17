# setup and paths
## prebuilt libraries
import pandas as pd
import time
from multiprocessing import cpu_count

## custom and project specific libraries
import helperfns_sarima_mapreduce



if __name__ == "__main__":
    # set up
    data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"

    NUM_PROCESSES = cpu_count()

    ## load stationary params
    with pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r") as ts_store:
        ts_stn_params = ts_store.select("ts_stn_params")

    # fit DAILY LEVEL on component time series
    with pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r") as ts_store, \
        pd.HDFStore(data_intermed_nb_fldrpath + "/sarima_ts_store.h5") as sarima_ts_store:
        # fit models in parallel
        t0 = time.time()

        helperfns_sarima_mapreduce.fit_sarima_ts_parallel(
            store_modeling = ts_store,
            tbl_series = "ts_daily_long",
            stn_params_df = ts_stn_params,
            var = "daily_level",
            freq = "D",
            store_fits = sarima_ts_store,
            NUM_TS_PER_DATA_CHUNK = 256*NUM_PROCESSES,
            NUM_PROCESSES = NUM_PROCESSES, PARALLEL_CHUNKSIZE = 128)

        tdelta = time.time() - t0
        print("Run time for DAILY LEVEL was {:.2f} seconds.\n".format(tdelta))

    # fit DAILY W-o-W GROWTH on component time series
    with pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r") as ts_store, \
        pd.HDFStore(data_intermed_nb_fldrpath + "/sarima_ts_store.h5") as sarima_ts_store:
        # fit models in parallel
        t0 = time.time()

        helperfns_sarima_mapreduce.fit_sarima_ts_parallel(
            store_modeling = ts_store,
            tbl_series = "ts_daily_long",
            stn_params_df = ts_stn_params,
            var = "daily_wowGr",
            freq = "D",
            store_fits = sarima_ts_store,
            NUM_TS_PER_DATA_CHUNK = 256*NUM_PROCESSES,
            NUM_PROCESSES = NUM_PROCESSES, PARALLEL_CHUNKSIZE = 128)

        tdelta = time.time() - t0
        print("Run time for DAILY W-o-W Growth was {:.2f} seconds.\n".format(tdelta))

    # fit WEEKLY LEVEL on component time series
    with pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r") as ts_store, \
        pd.HDFStore(data_intermed_nb_fldrpath + "/sarima_ts_store.h5") as sarima_ts_store:
        # fit models in parallel
        t0 = time.time()

        helperfns_sarima_mapreduce.fit_sarima_ts_parallel(
            store_modeling = ts_store,
            tbl_series = "ts_weekly_long",
            stn_params_df = ts_stn_params,
            var = "weekly_level",
            freq = "W",
            store_fits = sarima_ts_store,
            NUM_TS_PER_DATA_CHUNK = 256*NUM_PROCESSES,
            NUM_PROCESSES = NUM_PROCESSES, PARALLEL_CHUNKSIZE = 128)

        tdelta = time.time() - t0
        print("Run time for WEEKLY LEVEL was {:.2f} seconds.\n".format(tdelta))

    # fit WEEKLY W-o-W GROWTH on component time series
    with pd.HDFStore(data_intermed_nb_fldrpath + "/ts_store.h5", mode="r") as ts_store, \
        pd.HDFStore(data_intermed_nb_fldrpath + "/sarima_ts_store.h5") as sarima_ts_store:
        # fit models in parallel
        t0 = time.time()

        helperfns_sarima_mapreduce.fit_sarima_ts_parallel(
            store_modeling = ts_store,
            tbl_series = "ts_weekly_long",
            stn_params_df = ts_stn_params,
            var = "weekly_wowGr",
            freq = "W",
            store_fits = sarima_ts_store,
            NUM_TS_PER_DATA_CHUNK = 256*NUM_PROCESSES,
            NUM_PROCESSES = NUM_PROCESSES, PARALLEL_CHUNKSIZE = 128)

        tdelta = time.time() - t0
        print("Run time for WEEKLY W-o-W Growth was {:.2f} seconds.\n".format(tdelta))
