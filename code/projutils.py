
import pandas as pd
import numpy as np

def smape(F, A, dropna=True):
    """Compute the symmetric mean absolute percentage error (SMAPE) """
    ## input assertions
    assert(len(F)==len(A))

    ## if inputs are series, turn them to numpy arrays
    if isinstance(F, pd.Series):
        F = F.values
    if isinstance(A, pd.Series):
        A = A.values

    ## treat missings
    rows_notmissing_F = ~np.isnan(F)
    rows_notmissing_A = ~np.isnan(A)

    ### if input is to not drop missings but missings are present, them return nan
    if not dropna:
        if (np.sum(rows_notmissing_F)<len(F)) or (np.sum(rows_notmissing_A)<len(A)):
            return np.nan

    ### if one vector is all missing, then return nan
    if (np.sum(rows_notmissing_F)==0) or (np.sum(rows_notmissing_A)==0):
        return np.nan

    ### otherwise drop the pairwise missing values
    F = F[rows_notmissing_F & rows_notmissing_A]
    A = A[rows_notmissing_F & rows_notmissing_A]

    ## define length based off what's left after dropping missing values
    n = len(A)

    ## calculate smape
    ### modify intermediate vectors so that if F==0 and A==0 then smape==0
    both_zero_rows = ((F==0) & (A==0))
    F[both_zero_rows] = 1
    A[both_zero_rows] = 1

    ### calculate
    summands = np.absolute(F - A)/((np.absolute(A) + np.absolute(F))/2.)

    smape_calculated = np.sum(summands)
    smape_calculated = smape_calculated / n * 100

    return(smape_calculated)
