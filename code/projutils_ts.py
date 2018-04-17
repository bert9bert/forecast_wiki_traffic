
import numpy as np
import pandas as pd
from collections import OrderedDict
import time
import warnings

from statsmodels.tools.eval_measures import mse
from projutils import smape

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# import r objects
ts = robjects.r("ts")
forecast = importr("forecast")

arroots = robjects.r("forecast:::arroots")
maroots = robjects.r("forecast:::maroots")

from rpy2.robjects import pandas2ri
pandas2ri.activate()


def sarima_stepsearch(y,
                      d, D, S,
                      max_p=5, max_q=5, max_P=2, max_Q=2,
                      boxcox_lambda=1,
                      n_validation=60, n_forecast=60,
                      cost_method="validation_smape", method="CSS",
                      trace=False, verbose=False):
    """Fit a seasonal ARIMA (SARIMA) model by stepwise search using validation (hold-out) data

    Uses a similar methodology to the stepwise search for SARIMA parameters in
    Hyndman and Khandakar (2008) section 3.2, which is implemented in
    R's forecast package as auto.arima, but allows for the use of a
    validation (hold-out, out-of-time) sample for parameter selection.

    Args:
        y (DataFrame): single-column time-indexed DataFrame with both the
            intended training and validation data
        d (int): number of non-seasonal differences to take
        D (int): number of seasonal differences to take
        S (int): frequency for seasonal component
        max_p (int, optional): max non-seasonal AR parameter allowed
        max_q (int, optional): max non-seasonal MA parameter allowed
        max_P (int, optional): max seasonal AR parameter allowed
        max_Q (int, optional): max seasonal MA parameter allowed
        boxcox_lambda (float, optional): Box-Cox Lambda transformation parameter
        n_validation (int, optional): number of observations to take off the
            end of `y` to be the validation sample, the remaining observations
            will be the training sample
        n_forecast (int, optional): number of periods to provide a forecast over
        cost_method (str, optional): cost method for choosing SARIMA parameters,
            can be ``validation_smape``, ``validation_mse``, or ``insample_aic``
        method (str, optional): method for SARIMA optimization,
            can be ``CSS``, ``CSS-ML``, or ``ML``
        trace (bool, optional): show trace info for finding SARIMA parameters
        verbose (bool, optional): print summary results of final model

    Returns:
        (OrderedDict, DataFrame, DataFrame)

        Returns an ordered dict with the final SARIMA parameters,
        DataFrame of fitted values after refitting with the final params,
        DataFrame of forecasts after refitting with the final params, and
        DataFrame of forecasts on the validation sample with the model fit on the training sample
    """


    # input checks
    ## check expected ranges
    assert(n_validation>=0)
    assert(n_forecast>=0)

    ## check that cost input and valid and consistent with other inputs
    insample_valid = ["insample_aic"]
    validation_valid = ["validation_smape", "validation_mse"]

    assert(any([cost_method==x for x in insample_valid+validation_valid]))

    if any([cost_method==x for x in insample_valid]):
        assert(n_validation==0)

    if any([cost_method==x for x in validation_valid]):
        assert(n_validation>0)

    # format input for Box-Cox lambda
    if boxcox_lambda==1:
        mykwargs = {}
    else:
        mykwargs = {"lambda": boxcox_lambda}

    # split train and validation sets
    if n_validation>0:
        y_train = y[:-n_validation]
        y_validation = y[-n_validation:]
    else:
        y_train = y
        y_validation = None

    # create R time series
    y_nomissing_rdata = ts(y_train.dropna(), frequency=S)

    # init variables to hold best values
    cost_best = np.inf
    full_order_best = None
    results_best = None
    y_valicast = None

    t0 = time.time()

    # define starting models
    # p,d,q P,D,Q, S, include c
    grid = [(2,d,2,1,D,1,S,1), (0,d,0,0,D,0,S,1), (1,d,0,1,D,0,S,1), (0,d,1,0,D,1,S,1)]

    if d+D>1:
        for i in range(len(grid)):
            grid[i] = tuple(np.concatenate((np.array(grid[i])[:7], np.array([0]))))

    if S<=1:
        grid = [(x[0],x[1],x[2],0,0,0,x[6],x[7]) for x in grid]
        grid = list(set(grid))

    # stepwise search based off Hyndman and Khandakar (2008) section 3.2, but with option for out of sample cost
    models_attempted = []
    models_attempted.extend(grid)

    keep_searching = True

    CTR_MAX = 1000
    ctr = 0
    while keep_searching and ctr<=CTR_MAX:
        ctr += 1

        keep_searching = False

        ## loop through grid
        for full_order_this in grid:
            try:
                # fit this model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # warnings ok, will produce exception later if cannot produce forecast
                    mod_r_step = forecast.Arima(
                        y = y_nomissing_rdata,
                        order = robjects.IntVector(full_order_this[:3]),
                        seasonal = robjects.IntVector(full_order_this[3:6]),
                        include_mean = full_order_this[7],
                        method = method,
                        **mykwargs
                    )

                # get cost for this model
                if cost_method=="insample_aic": # in-sample AIC
                    aic_this = mod_r_step.rx2("aic")[0]

                    if aic_this is robjects.NA_Logical:
                        aic_this = 2*(len(mod_r_step.rx2("coef"))+1) - 2*mod_r_step.rx2("loglik")[0]

                    cost_this = aic_this
                    del aic_this

                elif any([cost_method==x for x in validation_valid]):
                    validation_pred = np.array(forecast.forecast(mod_r_step, h=n_validation).rx2("mean"))
                    validation_pred = pd.DataFrame(validation_pred,
                                                   index=y_validation.index,
                                                   columns=[y_validation.iloc[:,0].name + "_pred"])

                    validation_pred = pd.concat([y_validation, validation_pred], axis=1)
                    assert(validation_pred.shape == (n_validation, 2))

                    if cost_method=="validation_smape":
                        cost_this = smape(validation_pred.iloc[:,0], validation_pred.iloc[:,1])
                    elif cost_method=="validation_mse":
                        cost_this = mse(validation_pred.iloc[:,0], validation_pred.iloc[:,1])
                    else:
                        raise Exception

                else:
                    raise Exception

                # reject this model if too close to non-invertible
                arroots_this = np.absolute(np.array([x for x in arroots(mod_r_step).rx2("roots")]))
                maroots_this = np.absolute(np.array([x for x in maroots(mod_r_step).rx2("roots")]))

                if (arroots_this<1.001).any() or (maroots_this<1.001).any():
                    cost_this = np.inf

                # store if this is the best model so far
                if cost_this < cost_best:
                    cost_best = cost_this
                    full_order_best = full_order_this
                    results_best = mod_r_step

                    keep_searching = True

                    if cost_method in validation_valid:
                        y_valicast = validation_pred.iloc[:,1].to_frame()

                if trace:
                    print('ARIMA{}x{}{} const {} - Cost:{}'\
                          .format(full_order_this[:3],
                                  full_order_this[3:6],
                                  full_order_this[6],
                                  full_order_this[7],
                                  cost_this))

                del cost_this, mod_r_step, validation_pred, arroots_this, maroots_this
            except:
                if trace:
                    print('ARIMA{}x{}{} const {} - Cost:{}'\
                          .format(full_order_this[:3],
                                  full_order_this[3:6],
                                  full_order_this[6],
                                  full_order_this[7],
                                  np.nan))

                continue

        ## if will keep searching, create new grid to search over
        if keep_searching:
            ### create full grid
            grid = []

            #### add one of p,q,P,Q +/- 1
            grid.append(tuple(np.array(full_order_best) + np.array([1,0,0,0,0,0,0,0])))
            grid.append(tuple(np.array(full_order_best) + np.array([0,0,1,0,0,0,0,0])))
            grid.append(tuple(np.array(full_order_best) + np.array([0,0,0,1,0,0,0,0])))
            grid.append(tuple(np.array(full_order_best) + np.array([0,0,0,0,0,1,0,0])))
            grid.append(tuple(np.array(full_order_best) - np.array([1,0,0,0,0,0,0,0])))
            grid.append(tuple(np.array(full_order_best) - np.array([0,0,1,0,0,0,0,0])))
            grid.append(tuple(np.array(full_order_best) - np.array([0,0,0,1,0,0,0,0])))
            grid.append(tuple(np.array(full_order_best) - np.array([0,0,0,0,0,1,0,0])))

            #### add both p,q together and P,Q together +/- 1
            grid.append(tuple(np.array(full_order_best) + np.array([1,0,1,0,0,0,0,0])))
            grid.append(tuple(np.array(full_order_best) - np.array([1,0,1,0,0,0,0,0])))

            grid.append(tuple(np.array(full_order_best) + np.array([0,0,0,1,0,1,0,0])))
            grid.append(tuple(np.array(full_order_best) - np.array([0,0,0,1,0,1,0,0])))

            #### change the mean term convention
            if full_order_best[7]==1:
                grid.append(tuple(np.array(full_order_best) - np.array([0,0,0,0,0,0,0,1])))
            else:
                grid.append(tuple(np.array(full_order_best) + np.array([0,0,0,0,0,0,0,1])))

            ### remove entries from grid
            #### remove seasonal parts if non-seasonal
            if S<=1:
                grid = [(x[0],x[1],x[2],0,0,0,x[6],x[7]) for x in grid]
                grid = list(set(grid))

            #### remove entries where p,q,P,Q exceed lower and upper bounds
            grid = [x for x in grid if x[0]>=0 and x[2]>=0 and x[3]>=0 and x[5]>=0]
            grid = [x for x in grid if x[0]<=max_p and x[2]<=max_q and x[3]<=max_P and x[5]<=max_Q]

            #### remove entries for models already attempted
            grid = [x for x in grid if x not in models_attempted]

            models_attempted.extend(grid)

        ## if on first iteration and none of the initial specs work, try again with no intercept
        ## if not already tried
        if ~keep_searching and ctr<2:
            if d+D<=1:
                for i in range(len(grid)):
                    grid[i] = tuple(np.concatenate((np.array(grid[i])[:7], np.array([0]))))
                keep_searching = True

    assert(ctr>=2)

    tdelta = time.time() - t0


    # refit on final model
    if n_validation>0:
        ## get final model fit, and if numerical error then reapply existing coefs
        y_all_nomissing_rdata = ts(y.dropna(), frequency=S)

        try:
            mod_final_r = forecast.Arima(
                y = y_all_nomissing_rdata,
                order = robjects.IntVector(full_order_best[:3]),
                seasonal = robjects.IntVector(full_order_best[3:6]),
                include_mean = full_order_best[7],
                method = method,
                **mykwargs
            )
        except:
            mod_final_r = forecast.Arima(
                y = y_all_nomissing_rdata,
                model = results_best
            )
    else:
        mod_final_r = results_best


    # produce fitted and forecast
    if n_forecast>0:
        ## get fitted value
        y_fitted = np.array(mod_final_r.rx2("fitted"))
        assert(y_fitted.shape[1] == 1)
        y_fitted = y_fitted[:,0]

        assert(len(y_fitted) <= len(y))

        if len(y_fitted) < len(y):
            y_fitted = np.concatenate([np.full(len(y)-len(y_fitted), np.nan), y_fitted])


        assert(len(y_fitted) == len(y))
        y_fitted = pd.DataFrame(y_fitted, index=y.index, columns=[y.iloc[:,0].name + "_pred"])

        ## get forecast values
        y_forecast = np.array(forecast.forecast(mod_final_r, h=n_forecast).rx2("mean"))

        days_btwn = (y.index[-1] - y.index[-2]).days
        if days_btwn==1:
            y_freq = "D"
        elif days_btwn==7:
            y_freq = "W"
        else:
            raise Exception("Unexpected frequency")
        del days_btwn

        y_forecast_index = pd.date_range(start=y.index.max(), periods=len(y_forecast)+1, freq=y_freq)[1:]

        y_forecast = pd.DataFrame(y_forecast, index=y_forecast_index, columns=[y.iloc[:,0].name + "_pred"])

        del y_forecast_index

    # store the model spec
    arima_model_spec = OrderedDict()
    arima_model_spec["nonseas_order"] = tuple(int(x) for x in full_order_best[:3])
    arima_model_spec["seas_order"] = tuple(int(x) for x in full_order_best[3:7])
    arima_model_spec["include_intercept"] = (full_order_best[7] == 1)

    # verbose post-fit info
    if verbose:
        print("\nSpec:")
        print('ARIMA{}x{}{} intercept={}'\
              .format(arima_model_spec["nonseas_order"],
                      arima_model_spec["seas_order"][:3],
                      arima_model_spec["seas_order"][3],
                      arima_model_spec["include_intercept"]))

        print("\nCoef:")
        print(results_best.rx2("coef"))

        print("\nCost (lower is better)")
        print(cost_best)

        print("\nLooping component to find best model ran in %8.2f seconds" % tdelta)

        plotdata = pd.concat([y, y_fitted], axis=1)
        plotdata = pd.concat([plotdata, y_forecast], axis=0)
        plotdata.plot(title="Actual vs Predicted and Forecast")
        plt.show()

    # return
    ret = [arima_model_spec, y_fitted, y_forecast]

    if cost_method in validation_valid:
        ret.append(y_valicast)

    return ret
