
import pandas as pd
import numpy as np


def back_transformations(y_hist, y_hist_weekly,
    y_daily_level, y_daily_wowGr, y_weekly_level, y_weekly_wowGr,
    intraweek_seasonal_dist):

    # store daily level
    z_daily_level = y_daily_level
    z_daily_level.columns = y_daily_level.columns + "bt"

    # convert daily W-o-W growth to level
    ## get historical realized values needed to compute growth rates off of
    hist_stop = y_daily_wowGr.index.values[0][-1] + pd.DateOffset(-1)
    hist_stop_rownum = np.where(y_hist.index.get_level_values(-1) == hist_stop)[0]
    del hist_stop

    pad_beginning = False

    if len(hist_stop_rownum)==1:
        hist_stop_rownum = hist_stop_rownum[0]
        y_hist_needed = y_hist.iloc[list(range(hist_stop_rownum-7+1, hist_stop_rownum+1))]
    elif len(hist_stop_rownum)==0:
        y_hist_needed = np.full(7, np.nan)
        pad_beginning = True
    else:
        raise Exception

    del hist_stop_rownum

    v1= np.array(y_hist_needed, ndmin=2)
    if v1.shape[0]==1:
        v1 = v1.T

    v2 = np.array(y_daily_wowGr)
    if pad_beginning:
        padidx=y_daily_wowGr.index[:7]
        padding = y_daily_level.loc[y_daily_level.index.isin(padidx)]
        del padidx

        assert(all(np.isnan(v2[:7])))
        v2[:7] = np.array(padding)

    working_vec = np.concatenate((v1, v2))
    del v1, v2

    ## use the growth rate to calculate the projected level
    i_start = 7
    if pad_beginning:
        i_start = 14

    for i in range(i_start, len(working_vec)):
        working_vec[i] = (1 + working_vec[i]) * working_vec[i-7]
    del i_start

    working_vec = working_vec[7:,0]

    if pad_beginning:
        working_vec[:7] = np.nan

    z_daily_wowGr = pd.DataFrame(working_vec, columns=y_daily_wowGr.columns + "bt", index=y_daily_wowGr.index)

    # re-seasonalize weekly level to daily level

    working_mat = np.full((len(y_weekly_level),7), np.nan)

    for i in range(7):
        working_mat[:,i] = y_weekly_level.iloc[:,0] * intraweek_seasonal_dist[i]

    working_mat = np.reshape(working_mat, newshape=(-1,1))

    working_vec = working_mat[:,0]
    del working_mat

    idx = pd.date_range(y_weekly_level.index.values[0][-1] + pd.DateOffset(-6), y_weekly_level.index.values[-1][-1])

    z_weekly_level = pd.DataFrame(working_vec, columns=y_weekly_level.columns + "bt", index=idx)

    del working_vec, idx, i

    # for weekly growth, convert to weekly level and re-seasonalize
    ## weekly growth to weekly level
    ### get historical realized values needed to compute growth rates off of
    hist_stop = y_weekly_wowGr.index.values[0][-1] + pd.DateOffset(-7)

    pad_beginning = False

    if hist_stop >= y_hist_weekly.index[0][-1]:
        prev = y_hist_weekly.xs(hist_stop, level=-1)
    else:
        prev = np.nan
        pad_beginning = True

    v1 = np.array(prev, ndmin=2)
    v2 = 1 + np.array(y_weekly_wowGr)

    if pad_beginning:
        assert(np.isnan(v2[0]))
        padding = np.array(y_weekly_level.xs(y_weekly_wowGr.index[0][-1], level=-1))
        v2[0] = padding

    growth_vec = np.concatenate((v1, v2))
    del v1, v2

    ### compute projected level from projected growth
    if not pad_beginning:
        growth_vec = np.cumprod(growth_vec)
        growth_vec = growth_vec[1:]
    else:
        growth_vec = np.cumprod(growth_vec[1:])
        growth_vec[0] = np.nan


    idx = y_weekly_wowGr.index

    z1 = pd.DataFrame(growth_vec, columns=y_weekly_wowGr.columns + "bt", index=idx)

    ## weekly level to re-reseasonalized weekly level
    working_mat = np.full((len(z1),7), np.nan)

    for i in range(7):
        working_mat[:,i] = z1.iloc[:,0] * intraweek_seasonal_dist[i]

    working_mat = np.reshape(working_mat, newshape=(-1,1))

    working_vec = working_mat[:,0]
    del working_mat

    idx = pd.date_range(z1.index.values[0][-1] + pd.DateOffset(-6), z1.index.values[-1][-1])

    z_weekly_wowGr = pd.DataFrame(working_vec, columns=z1.columns, index=idx)

    del working_vec, idx, i

    # combine into one dataframe

    assert(len(z_daily_level)==len(z_daily_wowGr))
    assert(len(z_weekly_level)==len(z_weekly_wowGr))

    z1 = pd.concat([z_daily_level, z_daily_wowGr], axis=1)
    z1idxnames = list(z1.index.names)
    z1idxnames[-1] = "time_d"
    z1.index.names = z1idxnames
    del z1idxnames

    z2 = pd.concat([z_weekly_level, z_weekly_wowGr], axis=1)
    z2.index.names = ["time_d"]

    assert(len(z1)==len(z_daily_level))
    assert(len(z2)==len(z_weekly_level))

    z = z1.join(
        z2,
        how = "left"
    )

    z_vn = list(z.columns)

    # output assertions
    ## on size
    assert(len(z)==len(z1))

    ## on missing values
    ### did a model fail to fit on this time series?
    modAllNan_y_daily_level = (np.sum(np.isnan(y_daily_level))==len(y_daily_level)).bool()
    modAllNan_y_daily_wowGr = (np.sum(np.isnan(y_daily_wowGr))==len(y_daily_wowGr)).bool()
    modAllNan_y_weekly_level = (np.sum(np.isnan(y_weekly_level))==len(y_weekly_level)).bool()
    modAllNan_y_weekly_wowGr = (np.sum(np.isnan(y_weekly_wowGr))==len(y_weekly_wowGr)).bool()

    ### check that last row is filled
    if not (modAllNan_y_daily_level or modAllNan_y_daily_wowGr or modAllNan_y_weekly_level or modAllNan_y_weekly_wowGr):
        assert(all(~np.isnan(z.iloc[-1])))

    ### check that columns are filled as expected
    pad_beginning = y_weekly_wowGr.index.values[0][-1] + pd.DateOffset(-7) < y_hist_weekly.index[0][-1]

    if pad_beginning:
        weekly_offset = (y_weekly_level.index[0][-1] - y_daily_level.index[0][-1]).days + 1 - 7

        if not modAllNan_y_daily_level:
            assert(np.sum(np.isnan(z.loc[:,z_vn[0]]))==0)
        if not modAllNan_y_daily_wowGr:
            assert(np.sum(np.isnan(z.loc[:,z_vn[1]][7:]))==0)
        if not modAllNan_y_weekly_level:
            assert(np.sum(np.isnan(z.loc[:,z_vn[2]][weekly_offset:]))==0)
        if not modAllNan_y_weekly_wowGr:
            assert(np.sum(np.isnan(z.loc[:,z_vn[3]][(weekly_offset + 7):]))==0)

        del weekly_offset
    else:
        if not modAllNan_y_daily_level:
            assert(np.sum(np.isnan(z.loc[:,z_vn[0]]))==0)
        if not modAllNan_y_daily_wowGr:
            assert(np.sum(np.isnan(z.loc[:,z_vn[1]]))==0)
        if not modAllNan_y_weekly_level:
            assert(np.sum(np.isnan(z.loc[:,z_vn[2]]))==0)
        if not modAllNan_y_weekly_wowGr:
            assert(np.sum(np.isnan(z.loc[:,z_vn[3]]))==0)

    del pad_beginning


    if modAllNan_y_daily_level:
        assert(np.sum(np.isnan(z.loc[:,z_vn[0]]))==len(z))
    if modAllNan_y_daily_wowGr:
        assert(np.sum(np.isnan(z.loc[:,z_vn[1]]))==len(z))
    if modAllNan_y_weekly_level:
        assert(np.sum(np.isnan(z.loc[:,z_vn[2]]))==len(z))
    if modAllNan_y_weekly_wowGr:
        assert(np.sum(np.isnan(z.loc[:,z_vn[3]]))==len(z))


    # return
    return(z)
