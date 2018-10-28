
# Project Overview

Kaggle hosted the [Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) competition, which provides a time series dataset of page visits to 145,063 articles on Wikipedia. Kaggle provided the number of visits to each of those pages for 803 days (approximately 26 months) of daily history. The goal was to forecast the daily visits to those same pages for the next 60 days. The visits for each page of the next 60 days was not known at the time of the forecast, and was necessary to wait 60 days after the competition ended to collect the observed values for the 60-day test data. Results were scored by SMAPE. Below is an example of one of these 145,063 time series.




![png](summary/Forecast_Wiki_Traffic_SummaryDetailed_files/Forecast_Wiki_Traffic_SummaryDetailed_4_0.png)


The approach for this project was to create several different models and combine them with an ensemble model.

# Data Exploration

## Page Distributions

In addition to the 803 days worth of page views, each page is also tagged with the Wikipedia project, the access method, and the agent.




![png](summary/Forecast_Wiki_Traffic_SummaryDetailed_files/Forecast_Wiki_Traffic_SummaryDetailed_9_0.png)


## Missing Values

There are missing values in the dataset, which could be either true missings because of data issues or they could be zero observations. Missing values that appear as a contiguous block from the beginning from the series are assumed to be from pages where the history does not exist far enough back, and are referred to as "short series." Missing values that occur between valid values are referred to as "holes."








Of the observations in the dataset, 2.0% are missing values that are holes and 4.1% are missing values at the beginning of a short series






<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Percent of Observations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Not Missing</th>
      <td>94</td>
    </tr>
    <tr>
      <th>Beginning of Short Series</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Hole</th>
      <td>2</td>
    </tr>
  </tbody>
</table>









Of the time series in the dataset, 13.1% have at least one missing value that is a hole and 14.3% are short series






<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Percent of Time Series</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Short Series</th>
      <td>14</td>
    </tr>
    <tr>
      <th>Series with Holes</th>
      <td>13</td>
    </tr>
  </tbody>
</table>


## Seasonality

The data likely has a seasonal pattern, as can be seen in the chart below when aggregated to an overall level. There are more page views at the beginning and end of the week, and the modeling will attempt to incorporate this seasonaility.




![png](summary/Forecast_Wiki_Traffic_SummaryDetailed_files/Forecast_Wiki_Traffic_SummaryDetailed_20_0.png)


## View Distributions

Much of the time series have modest daily visits, with more than half having a median of 122 or more views per day. For these series an ARIMA model may be worthwhile. For the those where there is not as much activity, such as the 25% that have a median of 16 or less visits per day, simple approaches such as means and medians may be useful.




<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>median</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min</th>
      <td>0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>16</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>122</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>531</td>
    </tr>
    <tr>
      <th>max</th>
      <td>19440903</td>
    </tr>
  </tbody>
</table>


# Modeling Approach

## Splitting into Training, Validation, and Test Samples

For hyperparameter tuning and model selection, the first 743 observations of the 803 observations for each time series were used as the **training sample** and the last 60 days were held out to be the **validation sample**. The 60 day window size of the validation sample was chosen to match the 60 day forecast window.

Once hyperparameters are chosen, the model is fit with all 803 observations for each time series and then used to forecast the following 60 days, which forms the **testing sample**.

## Creating the Modeling Dataset

### Data Cleaning

**Outliers** were identified as observations that are more than two overall standard deviations from the 30-day center rolling average for that time series. There were also **holes** in many of time series, where there is an NA value but valid observations before and after. It is unknown whether these NAs are data issues or if they are zero values, although zero values do appear in the data.

**Outliers and holes were cleaned by imputing** with the average of the prior day of week and following day of week. For example, if an outlier or hole was on a Wednesday, it would be imputed with the average value of the prior Wednesday and the following Wednesday. Imputing by the day of the week is done to preserve any weekly seasonality. If this was not possible because either those values were missing too, or if the outlier or hole occurs too close to the beginning or end of the series, then the outlier or hole was imputed with the rolling average.

Any negative page views were assumed to be data errors, and replaced with zeros.

The chart below shows the effect of cleaning up outliers for one time series.




![png](summary/Forecast_Wiki_Traffic_SummaryDetailed_files/Forecast_Wiki_Traffic_SummaryDetailed_29_0.png)


### Holiday Adjustments

A simple method was applied to account for holidays. The effect of the holiday is removed before models are fit, and the effect is re-applied after the model forecast.

A short list of holidays that always fall on the same day were defined, along with the locales that they are associated with. Future improvements could include holidays that may change days every year, for example, Labor Day which falls on the first Monday of September.

For holidays in the list where all the occurences of that holiday in the time series are outliers, the ratio between the cleaned figure to the untouched figure is used as the adjustment factor to apply for projection.

Below is an example of adjusting for Halloween. This adjustment will be un-applied after a forecast is made.




![png](summary/Forecast_Wiki_Traffic_SummaryDetailed_files/Forecast_Wiki_Traffic_SummaryDetailed_31_0.png)


### Data Transformations by Page

The cleaned and adjusted time series were then used to create four page-level data transformations for modeling. Models will be fit to each of these transformations.


| Transformation                    | Description                             |
| ---------------------------------:|:--------------------------------------- |
| Page Daily Level                  | Already have                            |
| Page Daily Week-over-Week Growth  | Convert daliy level to W-o-W            |
| Page Weekly Level                 | Sum daily level to weekly level         |
| Page Weekly Week-over-Week Growth | Convert weekly level to W-o-W           |




![png](summary/Forecast_Wiki_Traffic_SummaryDetailed_files/Forecast_Wiki_Traffic_SummaryDetailed_33_0.png)


### Data Aggregations

Models were also fit at a more aggregated level, and results distributed back to the page level. Aggregations were done by taking the sum of visits by project, access, and agent.

Some of the component time series are short. To account for the downward bias:
1. Take the global median of all time series.
2. At each date, scale the summed value of the available series up by multiplying by the ratio of the sum of all global medians to sum of available medians.






![png](summary/Forecast_Wiki_Traffic_SummaryDetailed_files/Forecast_Wiki_Traffic_SummaryDetailed_36_0.png)


| Transformation                          | Description           |
| ---------------------------------------:|:--------------------- |
| Aggregated Daily Level                  | Add up all the page visits for each combination of project, access, and agent for each day |
| Aggregated Daily Week-over-Week Growth  | Calculate the week-over-week growth rate of the above aggregation |
| Aggregated Weekly Level                 | Add up all the page visits for each combination of project, access, and agent for each week                      |
| Aggregated Weekly Week-over-Week Growth | Calculate the week-over-week growth rate of the above aggregation |

Models will be fit to these aggregated levels, and the forecasts will be re-distributed back to the individual page level.




![png](summary/Forecast_Wiki_Traffic_SummaryDetailed_files/Forecast_Wiki_Traffic_SummaryDetailed_38_0.png)


## Finding Stationary Transformations

ARMA models assume that the data is stationary, and since the raw time series may not be stationary they need to first be transformed so that they are. The procedure followed is based off the one outlined in Hyndman and Khandakar (2008), for which in particular the log transformation is added. The following steps were followed to find a stationary transformation where possible. This is done by performing transformations one by one until one transformation passes the KPSS test. The KPSS test was used instead of the Dickey-Fuller test because the null hypothesis of the KPSS test is stationary, so will will tend us toward less transformations.

The KPSS test was applied in waterfall way to determine the order of differencing, order of seasonal differencing, and transformations. Transformations were only considered if d+D<=2 and D<=1. Do S=7 for daily, but S=0 for weekly since the seasonality would be long if did 52.

The KPSS test was performed in this order until a stationary transformation was found (null hypothesis is not rejected). This is done for these eight transformations.

1. d=0, D=0, untransformed
2. d=1, D=0, untransformed
3. d=0, D=1, untransformed (daily only)
4. d=1, D=1, untransformed (daily only)
5. d=2, D=0, untransformed
6. repeat above with log transformation
7. if still no stationary transformation at this point, flag it


The two plots below show the time series before the stationary transformation and after the stationary transformation for an example page, which in this case is a first difference. After the transformation, the series mean becomes much closer to constant and the variance is much more stable.




![png](summary/Forecast_Wiki_Traffic_SummaryDetailed_files/Forecast_Wiki_Traffic_SummaryDetailed_42_0.png)


Of the series that were kept at the original daliy level, about half needed an additional difference transformation to be stationary, but only a small handful needed a seasonal difference. As the initial transformation becomes more aggressive, fewer additional transformations were needed. In particular, nearly all of the daily level series needed an additional difference transformation to be stationary. Most of the weekly level did not need an additional transformation compared to the daily level, and even less of the weekly growth transformed needed a transformation compared to the weekly level. None needed a log transformation.




<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th>Number of Time Series</th>
    </tr>
    <tr>
      <th>Variable</th>
      <th>Found Stationary</th>
      <th>Stationary Function</th>
      <th>d</th>
      <th>D</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Daily Level</th>
      <th rowspan="4" valign="top">True</th>
      <th rowspan="4" valign="top">asis</th>
      <th rowspan="2" valign="top">0.0</th>
      <th>0.0</th>
      <td>64762</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>7</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1.0</th>
      <th>0.0</th>
      <td>59476</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>61</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Daily W-o-W Growth</th>
      <th rowspan="2" valign="top">True</th>
      <th rowspan="2" valign="top">asis</th>
      <th>0.0</th>
      <th>0.0</th>
      <td>117062</td>
    </tr>
    <tr>
      <th>1.0</th>
      <th>0.0</th>
      <td>7243</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Weekly Level</th>
      <th rowspan="3" valign="top">True</th>
      <th rowspan="3" valign="top">asis</th>
      <th>0.0</th>
      <th>0.0</th>
      <td>114816</td>
    </tr>
    <tr>
      <th>1.0</th>
      <th>0.0</th>
      <td>30177</td>
    </tr>
    <tr>
      <th>2.0</th>
      <th>0.0</th>
      <td>38</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Weekly W-o-W Growth</th>
      <th rowspan="2" valign="top">True</th>
      <th rowspan="2" valign="top">asis</th>
      <th>0.0</th>
      <th>0.0</th>
      <td>143767</td>
    </tr>
    <tr>
      <th>1.0</th>
      <th>0.0</th>
      <td>1264</td>
    </tr>
  </tbody>
</table>


## Fitting Component Models

Twelve models were initially fit. Four models will be seasonal ARIMA models based off the four data transformations. Four more will be build off seasonal ARIMA models but on the aggregated data where page views are added up by project, access, and agent, and then re-allocated back to the time series level. The final four models will be based off summary statistics. These initial twelve models will be used later to construct an ensemble model which will produce a forecast.

| Model # | Model Type                           | Model                         | Necessary Back Transformations  |
| -------:| ------------------------------------:|:----------------------------- |:----- |
|  1      | Individual Time Series (SARIMA)      | Daily Level                   | None needed |
|  2      |  Individual Time Series (SARIMA)      | Daily W-o-W Growth            | Convert daily growth to daily level |
|  3      |  Individual Time Series (SARIMA)      | Weekly Level                  | Re-seasonalize to daily |
|  4      |  Individual Time Series (SARIMA)      | Weekly W-o-W Growth           | Convert weekly growth to weekly level, <br>then re-seasonalize to daily |
|  5      |  Aggregated Time Series (SARIMA)      | Daily Level                   | Same as for individual time series, <br>then reallocate the aggregated to <br>individual time series by fixed proportions |
|  6      |  Aggregated Time Series (SARIMA)      | Daily W-o-W Growth            | Same as for individual time series, <br>then reallocate the aggregated to <br>individual time series by fixed proportions  |
|  7      |  Aggregated Time Series (SARIMA)      | Weekly Level                  | Same as for individual time series, <br>then reallocate the aggregated to <br>individual time series by fixed proportions  |
|  8      |  Aggregated Time Series (SARIMA)      | Weekly W-o-W Growth           | Same as for individual time series, <br>then reallocate the aggregated to <br>individual time series by fixed proportions  |
|  9      |  Summary Statistics                   | Seasonal Mean, All Data       |  |
| 10      |  Summary Statistics                   | Seasonal Mean, Last 60 Days   |  |
| 11      |  Summary Statistics                   | Seasonal Median, All Data     |  |
| 12      |  Summary Statistics                   | Seasonal Median, Last 60 Days |  |




### Choosing Hyperparameters and Fitting ARIMA Models to Individual and Aggregated Time Series

An ARIMA model has three parameters, the autoregressive order (AR), the moving average order (MA), and the level of differencing. For a seasonal ARIMA model (SARIMA), three equivalent parameters exist for the seasonal component. Since the level of differencing for both the global and the seasonal components were already chosen when a stationary transformation was selected, only four parameters need to be chosen: the global AR order, the global MA order, the seasonal AR order, and the seasonal MA order. 

To choose these hyperparameters, various models over a grid of potential hyperparameters were fit on the training sample for each time series and the hyperparameters that lead to the best SMAPE in the hold-out validation sample were chosen as the hyperparameters for that time series. The grid is 1-5 for the global AR and MA orders, and the grid is 1-2 for the seasonal AR and MA orders. 

Since there are 145,063 time series and each time series need to be fit over a large grid of potential hyperparameters, the number of potential models that needs to be fit is in the millions. This would take a long time to fit on a laptop, so the calculation was instead done on AWS EC2 where the calculation is easy to scale up.

### Back Transformations for ARIMA Models

The forecast needs to be at the level of daily views, so depending on the level of aggregation, additional back transformations may be needed.

Time series modeled as growth rates are converted back to levels.

Time series modeled at the aggregated weekly level are disaggregated back to the daily level in a manner that incorporates seasonality. This is done by summing up the total visits in the past 60 days for each day of the week, and using those proportions to allocate the forecasted weekly estimates into daily. When mapping weekly back to daily, use the historical seasonal proportions calculated from the historical seasonal proportions without the validation data.

For time series that have been aggregated together into multiple time series, they are aggregated back to the individual time series by summing up all the historically observed page visits for each time series, and that ratio is proportionally applied to disaggregated the aggregated series back to the individual series.


### Controls on ARIMA Model Outputs

To prevent ARIMA models from giving extremely high estimates after transformations, a simple trimming method is applied at ten times the maximum historically observed value per time series.


### Summary Statistics Models
Means and medians were calculated for each of the seven days of the week, and are applied by forecasting the same mean or median for each day of the week. There are referred to as the seasonal mean and seasonal median models. In additional to computing seasonal mean and median models on all the historical observation, they were also computed using just the last 60 days of data since that is the forecast window.


### Example of Fitted Initial Models

Below is an example of the 12 models fit for the Wikipedia article about the Python programming language. The blue line is the time series as observed on the validation sample, and the orange line is the model prediction for a particular model on the validation sample. Some models clearly work better than others, which is accounted for in the process of creating the ensemble model, which accounts for the performance of each model on the validation sample.





![png](summary/Forecast_Wiki_Traffic_SummaryDetailed_files/Forecast_Wiki_Traffic_SummaryDetailed_53_0.png)


## Creating Ensemble Model

Since ensemble models combine multiple models, it can produce a model that is less noisy (has lower variance) than just choosing the model with the lowest SMAPE. Also, since the ensemble in this case will combine models built on individual time series and on aggregated time series, the ensemble models may capture correlated movements because the aggregated models are blended in.



For each candidate model, the SMAPE is calculated on the hold-out validation sample, which is used to determine the ensemble weights for the ensemble meta model. The holiday adjustment is reapplied before evaluating the validation sample SMAPE so that the model selection will be based off what would be forecasted.

For each time series, each component model was then ranked by its performance on the SMAPE. The chart below shows the distribution of SMAPE according to the model rank. As seen in the chart, there is not a large change in SMAPE performance for the first few ranked models. 

The methodology used for the ensemble model will be to take the top five models for each time series and the forecast of the ensemble model will be the average of those five models weighted by inverse SMAPE.




![png](summary/Forecast_Wiki_Traffic_SummaryDetailed_files/Forecast_Wiki_Traffic_SummaryDetailed_57_0.png)


The chart on the left below shows the emphasis that would be given to each model across all pages if the top model were chosen, and the chart on the right below shows the emphasis that would be given to each model across all pages with the ensemble methodology. Using the ensemble methodology makes use of more models where choosing the model with lowest SMAPE for each time series will choose the daily level SARIMA model much of the time.






![png](summary/Forecast_Wiki_Traffic_SummaryDetailed_files/Forecast_Wiki_Traffic_SummaryDetailed_60_0.png)


After the end of the competition and waiting 60 days for the testing sample to be collected, the performance between using the most with the lowest SMAPE versus using the ensemble methodology can be compared. As seen in the chart below, **using the ensemble model that blends the top five models weighted by inverse error results in a model with better SMAPE on the testing sample than using just the top model determined on the validation sample**.




<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMAPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Top Model</th>
      <td>49.53</td>
    </tr>
    <tr>
      <th>Ensemble of Top Five Models</th>
      <td>48.30</td>
    </tr>
  </tbody>
</table>


The two charts below show an example for one time series, which is the article for the Python programming language. It shows the observed (blue), the prediction when using the model with lowest SMAPE on the validation sample (orange), and the prediction when using the ensemble model (green). This is shown for both the validation sample and the forecast/test sample.










![png](summary/Forecast_Wiki_Traffic_SummaryDetailed_files/Forecast_Wiki_Traffic_SummaryDetailed_67_0.png)


# Computational Framework and Infrastructure

Since there were millions of candidate models that needed to be fit it would take a long time to fit all the models on my laptop. One way to reduce the computational time is to do the model fitting on AWS. To get a sense of what the savings in computational time could be, candidate models for approximately 100 pages were fit on my laptop and then on various AWS instances. The observed runtime to fit models on those 100 pages were extrapolated to all the pages by linearly multiplying the runtime up.

To reduce AWS costs, spot instances were used instead of on-demand instances. Also, the model fitting code was tested on a small subset of pages first before running on all the pages to minimize needing to re-run large jobs multiple times.

The table below shows estimated runtime and estimated cost from this sizing exercise. The estimated costs exclude the cost of AWS EBS storage since that cost is the same regardless of the chosen instance type and other fixed costs were excluded as well.

| Computational<br>Environment  | Estimated<br>Runtime  | Estimated<br>Cost Using<br>On-Demand<br>Instances  | Estimated<br>Cost Using<br>Spot<br>Instances  |
|:--|--:|--:|--:|
| Laptop (dual core)  | 51.8 hours  |   |   |
| AWS EC2 t2.micro (1 vCPU)  | 59.5 hours  | \$0.69  | \$0.21  |
| AWS EC2 c5.large (2 vCPU)  | 40.6 hours  | \$3.45  | \$1.22  |
| AWS EC2 c5.xlarge (4 vCPU)  | 24.5 hours  | \$4.16  | \$1.47  |
| AWS EC2 c5.2xlarge (8 vCPU) | 18.2 hours  | \$6.18  | \$2.25  |

# Kaggle Competition Results

Forecasts from this model for the next 60 days for each 145,063 time series were then submitted on Kaggle for scoring. To do so, the competition observed the next 60 days of actual traffic with the forecasts locked. This was then used to evaluate the model. This model placed in the top 18% out of 1095 entries.






![png](summary/Forecast_Wiki_Traffic_SummaryDetailed_files/Forecast_Wiki_Traffic_SummaryDetailed_73_0.png)



## Potential Enhancements

These could be potential enhancements to look into.

* Add further component models with different approachs to the ensemble process. These could be:
    * Neural network approach through LSTMs
    * ETS models
* The allocation method used to allocate the forecasted aggregated page visits back to to the individual pages is very simple, applying just the historical fixed proportion. It could be worth exploring allocating this based off modeled proportions instead of fixed proportions, such as building a model for each page's share of the total aggregated visits or repurposing one of the existing individual level models to do the allocation.
