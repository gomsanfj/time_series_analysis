import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller, kpss
from sktime.forecasting.model_selection import temporal_train_test_split, ForecastingGridSearchCV
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError


DATA_FOLDER = 'data'
TRAIN_INIT_DATE = '2015-01-01'
TRAIN_END_DATE = '2021-02-01'
TEST_INIT_DATE = '2021-03-01'
TEST_PERIODS = 12
PLOT_RAW_DATA = False
PLOT_FORECAST = True
SAVE_PREDS = False


def date_to_yyyy_mm(date: pd.Timestamp) -> pd.Timestamp:
    """
    Convert dates from format 'yyyy-dd-mm' to 'yyyy-mm'
    :param date: Date in format 'yyyy-dd-mm'
    :return: date in format 'yyyy-mm'
    """
    date_str = str(date)
    if date_str != 'NaT':
        year = date_str[:4]
        month = date_str[8:10]
        return pd.to_datetime(year+'-'+month+'-01')
    else:
        return date


def dates_to_orig_format(date: pd.Timestamp) -> pd.Timestamp:
    """
    Convert dates from timestamp format to original format
    :param date: timestamp date in format 'yyyy-mm-dd'
    :return: string type 'dd.mm.yyyy'
    """
    return date.strftime('%d.%m.%Y')


def preprocess_date(data_raw: pd.DataFrame, init_date: str, end_date: str) -> pd.DataFrame:
    """
    Preprocess original dataset. Modify dates and get only specified data
    :param data_raw: original data
    :param init_date: start of the series
    :param end_date: end of the series
    :return: processed data
    """
    data_raw.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

    data_raw['Date'] = pd.to_datetime(data_raw['Date'])

    data_raw['YearMonth'] = data_raw['Date'].apply(date_to_yyyy_mm)
    data_raw['YearMonth'] = pd.to_datetime(data_raw['YearMonth'], format='%Y-%m')

    # Create new date index
    data_raw['new_date'] = data_raw['YearMonth'].dt.to_period('M').astype(str)

    # Get only the points from init to end date
    data_raw_f = data_raw[(data_raw['YearMonth'] >= init_date) & (data_raw['YearMonth'] <= end_date)]

    # Set 'Date' as index
    data_raw_f.set_index('YearMonth', inplace=True)

    # Verify index frequency
    if data_raw_f.index.freq is None:
        data_raw_f.index = pd.date_range(start=data_raw_f.index[0], periods=len(data_raw_f), freq='M')

    return data_raw_f


def get_stationary(series: pd.Series, th: float = 0.05, mode='adf') -> None:
    """
    Check if the series is stationary
    :param series: series values
    :param th: threshold to reject or accept null hyphothesis
    :param mode: 'adf' or 'kpss' to select the method
    :return: None
    """
    # Augmented Dickey-Fuller
    if mode == 'adf':
        result = adfuller(series)
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        if result[1] > th:
            print("Non stationary series")
        else:
            print('Stationary series')
    # Kwiatkowski-Phillips-Schmidt-Shin (KPSS)
    elif mode == 'kpss':
        result = kpss(series)
        print('KPSS Statistic:', result[0])
        print('p-value:', result[1])
        if result[1] < th:
            print("Non stationary series")
        else:
            print('Stationary series')
    else:
        print("Please, select 'adf' or 'kpss' as mode.")


def tranform_data(data: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Make some transformation to the series in case you need to make it stationary
    :param data: raw data
    :param value_col: name of the column with values of the series
    :return: data with transformations
    """
    # First order differentiation
    data['diff'] = data[value_col].diff()
    # Second order differentiation
    data['diff_2'] = data[value_col].diff(periods=2)
    # Logarithmic transformation
    data['log'] = np.log(data[value_col])
    # stationary_differentiation
    data['stationary_diff'] = data[value_col] - data[value_col].shift(12)
    # Box-Cox
    data['box-cox'], _ = boxcox(data['y'])

    return data


def plot_series(series_index: pd.DatetimeIndex, series_values: pd.Series, title: str = '') -> None:
    """
    Plot temporal series
    :param series_index: temporal index (x-axis)
    :param series_values: temporal series (y-axis)
    :param title: Title of the plot
    :return: None
    """
    plt.plot(series_index, series_values)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()


def plot_series_pred(series_values: pd.Series, series_index: pd.DatetimeIndex, fh: pd.DatetimeIndex, y_pred: pd.Series,
                     title: str = '') -> None:
    """
    Plot temporal series and new predictions
    :param series_values: temporal series (y-axis)
    :param series_index:  temporal index (x-axis)
    :param fh: index of the forecasted values
    :param y_pred: forecasted values
    :param title: title of the plot
    :return: None
    """
    plt.figure(figsize=(14, 7))
    plt.plot(series_index, series_values, label='Historical Data')
    plt.plot(fh, y_pred, label='Forecast', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # 1. Read and preprocess data
    data = pd.read_csv(os.path.join(DATA_FOLDER, 'train.csv'))
    data_f = preprocess_date(data, init_date=TRAIN_INIT_DATE, end_date=TRAIN_END_DATE)

    # Stationary test
    get_stationary(data_f['y'], mode='adf')

    # get data transformations
    data_f = tranform_data(data_f, value_col='y')

    # TODO: After some experiments, I've found that I got reasonable results using the second order differentiation
    #   series, but due to lack of time, I couldn't continue exploring this way.

    # Plot original dataset
    if PLOT_RAW_DATA:
        plot_series(data_f.index, data_f['y'], title='Original series')

    # split into train and test. I'll use last year as test
    y_train, y_test = temporal_train_test_split(data_f['y'], test_size=12)

    # 2. Set forecaster (ARIMA) and hyperparameters search
    forecaster = ARIMA()

    # hyperparameters set
    # p: for the Autoregressive component; # of lag observations
    # d: for the Integrated component; # of differences to make the series stationary
    # q: for the Moving Average component; size of the MA window
    param_grid = {'order': [(p, d, q) for p in range(0, 4) for d in range(0, 3) for q in range(0, 4)]}

    # Set cross-validation
    cv = ExpandingWindowSplitter(initial_window=36, step_length=12, fh=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # Set error metric
    scoring = MeanAbsolutePercentageError()

    # Hyperparameters search
    gs_cv = ForecastingGridSearchCV(forecaster, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)

    # 3. Train the forecaster with the grid search
    gs_cv.fit(y_train)

    print(f'Best params: {gs_cv.best_params_}')
    print(f'Best score: {gs_cv.best_score_}')

    # 4. Get the best model from the hyperparameter tuning and fit with all the dataset
    best_forecaster = gs_cv.best_forecaster_
    best_forecaster.fit(data_f['y'])

    # Get predictions
    fh_to_predict = pd.date_range(start=TEST_INIT_DATE, periods=TEST_PERIODS, freq='M').to_period('M')
    y_forecast = best_forecaster.predict(fh=[i for i in range(1, TEST_PERIODS+1)])

    if PLOT_FORECAST:
        plot_series_pred(series_values=data_f['y'], series_index=data_f.index,
                         fh=fh_to_predict.to_timestamp(), y_pred=y_forecast,
                         title='ARIMA Forecast from Mar 2021 to Feb 2022')

    # 5. Save predictions into the specified format
    if SAVE_PREDS:
        df_preds = pd.DataFrame.from_dict({'Date': fh_to_predict.to_timestamp(), 'y': y_forecast})

        df_preds['orig_date'] = df_preds['Date'].apply(dates_to_orig_format)
        df_preds['y'] = df_preds['y'].round(5)
        df_preds = df_preds.reset_index(drop=True)
        df_preds = df_preds.rename(columns={'orig_date': ''})
        df_preds[['', 'y']].to_csv(os.path.join(DATA_FOLDER, 'test.csv'), index=False)

    # TODO: Analyze residuals
