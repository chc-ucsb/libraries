import pandas as pd

def get_viewer_eo_ts(df, fnid, start_year, end_year, months, variable, operation="mean"):
    try:
        # Check if the input month is a single integer. If it is, wrap it in a list.
        if isinstance(months, int):
            months = [months]
        
        # Filter based on the list of months.
        ts = df.loc[
            (df['year'] >= start_year) & 
            (df['year'] <= end_year) & 
            (df['fnid'] == fnid) &
            (df['month'].isin(months)) &  
            (df['variable'] == variable) &
            (df['type'] == 'value'),
            ['year', 'value']
        ]

        # Group by year and then apply the desired aggregation function.
        if operation == "mean":
            ts = ts.groupby('year').mean()
        elif operation == "max":
            ts = ts.groupby('year').max()
        elif operation == "median":
            ts = ts.groupby('year').median()
        elif operation == "min":
            ts = ts.groupby('year').min()
        elif operation == "sum":
            ts = ts.groupby('year').sum()
        else:
            raise ValueError("Invalid operation. Choose from ['mean', 'max', 'median', 'min', 'sum']")

        return ts
    except Exception as e:
        return pd.DataFrame()


def get_viewer_data_ts(df, fnid, start_year, end_year, variable, product, season=None, months=None, model=None, oof=None, operation="mean"):
    try:
        conditions = [
            (df.index >= start_year), 
            (df.index <= end_year),
            (df['fnid'] == fnid),
            (df['variable'] == variable),
            (df['product'] == product),
        ]
        if season:
            conditions.append(df['season'] == season)
        if model:
            conditions.append(df['model'] == model)
        if oof:
            conditions.append(df['out-of-sample'] == oof)
        if months:
            if isinstance(months, int):
                months = [months]
            conditions.append(df['month'].isin(months))

        ts = df.loc[
            pd.np.logical_and.reduce(conditions),
            ['value']
        ]
        
        # Group by year and then apply the desired aggregation function.
        if operation == "mean":
            ts = ts.groupby(level=0).mean()
        elif operation == "max":
            ts = ts.groupby(level=0).max()
        elif operation == "median":
            ts = ts.groupby(level=0).median()
        elif operation == "min":
            ts = ts.groupby(level=0).min()
        elif operation == "sum":
            ts = ts.groupby('year').sum()
        else:
            raise ValueError("Invalid operation. Choose from ['mean', 'max', 'median', 'min', 'sum']")

        return ts
    except Exception as e:
        return pd.DataFrame()


def get_ts(df, fnid, start_year, end_year, variable, lag=False, months=None, season=None, product=None, model=None, oof=None, operation="mean"):
    time_series = get_viewer_eo_ts(df, fnid, start_year, end_year, months, variable, operation)
    
    if time_series.empty:
        time_series = get_viewer_data_ts(df, fnid, start_year, end_year, variable, product, season, months, model, oof, operation)

    if lag:
        time_series = time_series.shift(1)

    return time_series

'''
    To get time series for viewer_data:

        get_ts(df, fnid, start_year, end_year, variable, season, product)

        for prediction models:
        
        get_ts(df, fnid, start_year, end_year, variable, product, month, model, oof)

        When aggregating multiple months, pass in a list of months and also specify 'operation' (options: ['mean', 'max', 'median', 'min', 'sum'])


    To get time series for viewer_eo:

        get_ts(df, fnid, start_year, end_year, variable, month, lag (optional))

'''