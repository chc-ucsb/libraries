from pathlib import Path
import xarray as xr
import salem
import pandas as pd
import rioxarray as rxr
import gzip
from tempfile import NamedTemporaryFile
import numpy as np
import geopandas as gpd

def spatial_ts(data_input, shapefile_path, variable_label=None, ids=None, method='mean', index_col='FNID', freq = 'D'):
    """
    Aggregate gridded data values from an xarray dataset or GeoTIFF files by a specified polygons in a shapefile using salem.

    Parameters:
    - data_input: xarray dataset containing gridded data or path to a folder containing GeoTIFF files
    - shapefile_path: path to the shapefile containing the polygons and index values
    - variable_label (optional): label for the variable to be aggregated. Need this when passing in Xarray's Dataset and don't need it for DataArray
    - ids (optional): an ID/index value or list of ID/index values of polygons to aggregate. If not provided, all index values will be aggregated.
    - method (optional): aggregation method, either 'mean' or 'sum'. Default is 'mean'. (Uses Shrad's spatial mean & integral functions)
    - index_col (optional): column name in the shapefile that indexes each polygon. Default is 'FNID'.
    - freq (optional): when working with GeoTIFF, pass in 'D' if daily, 'M' if monthly, 'Y' if yearly

    Returns:
    - DataFrame with time series data values aggregated by the specified index column
    """

    # Load the shapefile with salem
    # shdf = salem.read_shapefile(shapefile_path)
    shdf = gpd.read_file(shapefile_path)# salem isn't compatible with GeoJSON files

    # If ids parameter is provided as a single string or integer, convert it to a list
    if isinstance(ids, (str, int)):
        ids = [ids]

    # If ids list is provided, filter the shapefile accordingly
    if ids is not None:
        shdf = shdf[shdf[index_col].isin(ids)]

    # Check if the provided aggregation method is valid
    if method not in ['mean', 'sum']:
        raise ValueError("Invalid method. Choose either 'mean' or 'sum'.")

    # Initialize an empty DataFrame to store results
    result_df = pd.DataFrame()

    # Check if data_input is a string (path to GeoTIFF folder) or xarray dataset
    if isinstance(data_input, (str, list)):
        # Assume data_input is a path to a folder containing GeoTIFF files
        ds = load_geotiff_files(data_input, freq)
        result_df = aggregate_data(ds, shdf, method, index_col)
    else:
        # Assume data_input is an xarray dataset
        ds = data_input

        if variable_label is None:
            result_df = aggregate_data(ds, shdf, method, index_col)
        else:
            result_df = aggregate_data(ds, shdf, method, index_col, variable_name = variable_label)
        

    return result_df

def load_geotiff_files(input_path, freq='D'):
    """
    Load GeoTIFF files from a given folder or a list of file paths into an xarray dataset.

    Parameters:
    - input_path: Path to the folder containing GeoTIFF files or a list of file paths to GeoTIFF files
    - freq (optional): Frequency of the data, 'D' for daily, 'M' for monthly, 'Y' for yearly. Default is 'D'.

    Returns:
    - xarray dataset containing concatenated data (across the time axis) from all GeoTIFF files
    """

    # Check if input_path is a folder or a list of file paths
    if isinstance(input_path, str) or isinstance(input_path, Path):
        folder = Path(input_path)
        geotiff_files = sorted(folder.glob('*.tif.gz'))
    elif isinstance(input_path, list):
        geotiff_files = [Path(file_path) for file_path in input_path]
    else:
        raise TypeError("input_path must be a folder path or a list of file paths")

    # Initialize a list to hold DataArrays
    data_arrays = []

    # Iterate through the GeoTIFF files and load them into DataArrays
    for file_path in geotiff_files:
        # Decompress the .gz file into a temporary file
        with gzip.open(file_path, 'rb') as gz_file:
            with NamedTemporaryFile(suffix='.tif') as tmp_file:
                tmp_file.write(gz_file.read())
                tmp_file.flush()

                # Open the temporary GeoTIFF file with rioxarray
                data_array = rxr.open_rasterio(tmp_file.name)

        # Extract the date from the filename based on the given pattern
        date_str = file_path.stem.split('.')[2:5]
        date_str = '-'.join(date_str)
        date = pd.to_datetime(date_str, infer_datetime_format=True)
        print(date)

        # Replace -9999 with NaN
        data_array = data_array.where(data_array != -9999)

        # Squeeze out the 'band' dimension, assuming it has size 1
        data_array = data_array.squeeze(dim='band', drop=True)

        # Expand the dimensions of the data_array to include a time dimension
        data_array = data_array.expand_dims(time=[date])

        # Append to the list of DataArrays
        data_arrays.append(data_array)

    # Concatenate the DataArrays along the time dimension to create an xarray dataset
    dataset = xr.concat(data_arrays, dim='time')

    return dataset


def aggregate_data(ds, shdf, method, index_col, variable_name=None):
    if isinstance(ds, xr.Dataset):
        if variable_name is None:
            raise ValueError("Must provide variable label for Dataset input")
        ds = ds[variable_name]

    # Initialize an empty DataFrame to store results for the current dataset
    aggregated_data_df = pd.DataFrame()

    # Determine the spatial dimensions for aggregation
    spatial_dims = None
    for possible_dims in [('Y', 'X'), ('y', 'x'), ('latitude', 'longitude'), ('lat', 'lon')]:
        if all(dim in ds.dims for dim in possible_dims):
            spatial_dims = possible_dims
            break

    if spatial_dims is None:
        raise ValueError("Spatial dimensions not found. Please ensure the dataset has one of the following dimension pairs: ('Y', 'X'), ('y', 'x'), ('latitude', 'longitude'), ('lat', 'lon').")

    # For each unique value in the specified index column of the (filtered) shapefile
    for idx_val in shdf[index_col].unique():
        # Extract the specific shape for the current index value
        shape_idx = shdf[shdf[index_col] == idx_val]

        # Crop the ds
        cropped_ds = crop_ds(shape_idx, ds, spatial_dims)

        # Check if the cropped dataset is empty
        if cropped_ds.size == 0:
            print(f"{idx_val} data is missing")
            continue

        # Create a mask using salem
        ds_roi = cropped_ds.salem.roi(shape=shape_idx)

        # Aggregate the data values over time for the current index value based on the specified method
        if method == 'mean':
            aggregated_data = calc_spatial_mean(ds_roi, lon_name = spatial_dims[1], lat_name = spatial_dims[0])
        else:  # method == 'sum'
            aggregated_data = calc_spatial_integral(ds_roi, lon_name = spatial_dims[1], lat_name = spatial_dims[0])

        # Set the name of the DataArray (this can be anything, as it will be overwritten later)
        aggregated_data.name = "value"

        # Convert to DataFrame
        aggregated_data_df[idx_val] = aggregated_data.to_dataframe()["value"]

    return aggregated_data_df

def crop_ds(gdf, ds, spatial_dims):
    # Calculate the total bounding box of all geometries
    minx, miny, maxx, maxy = gdf.total_bounds
    ds_cropped = ds.sel({spatial_dims[1]: slice(minx-.5, maxx+.5), spatial_dims[0]: slice(miny-.5, maxy+.5)})
    return ds_cropped

# Note: The function  supports both xarray datasets and paths to a folder with GeoTIFF files
#       The variable_label parameter allows the specification of the variable to be aggregated


########
# Below code is from Shard
# Source path: /home/shrad/Scripts/my_python_modules/Shrad_modules.py
EARTH_RADIUS = 6371000.0  # m

def _guess_bounds(points, bound_position=0.5):
    """
    Guess bounds of grid cells.
    
    Simplified function from iris.coord.Coord.
    
    Parameters
    ----------
    points: numpy.array
        Array of grid points of shape (N,).
    bound_position: float, optional
        Bounds offset relative to the grid cell centre.

    Returns
    -------
    Array of shape (N, 2).
    """
    diffs = np.diff(points)
    diffs = np.insert(diffs, 0, diffs[0])
    diffs = np.append(diffs, diffs[-1])

    min_bounds = points - diffs[:-1] * bound_position
    max_bounds = points + diffs[1:] * (1 - bound_position)

    return np.array([min_bounds, max_bounds]).transpose()


def _quadrant_area(radian_lat_bounds, radian_lon_bounds, radius_of_earth):
    """
    Calculate spherical segment areas.

    Taken from SciTools iris library.

    Area weights are calculated for each lat/lon cell as:
        .. math::
            r^2 (lon_1 - lon_0) ( sin(lat_1) - sin(lat_0))

    The resulting array will have a shape of
    *(radian_lat_bounds.shape[0], radian_lon_bounds.shape[0])*
    The calculations are done at 64 bit precision and the returned array
    will be of type numpy.float64.

    Parameters
    ----------
    radian_lat_bounds: numpy.array
        Array of latitude bounds (radians) of shape (M, 2)
    radian_lon_bounds: numpy.array
        Array of longitude bounds (radians) of shape (N, 2)
    radius_of_earth: float
        Radius of the Earth (currently assumed spherical)

    Returns
    -------
    Array of grid cell areas of shape (M, N).
    """
    # ensure pairs of bounds
    if (
        radian_lat_bounds.shape[-1] != 2
        or radian_lon_bounds.shape[-1] != 2
        or radian_lat_bounds.ndim != 2
        or radian_lon_bounds.ndim != 2
    ):
        raise ValueError("Bounds must be [n,2] array")

    # fill in a new array of areas
    radius_sqr = radius_of_earth ** 2
    radian_lat_64 = radian_lat_bounds.astype(np.float64)
    radian_lon_64 = radian_lon_bounds.astype(np.float64)

    ylen = np.sin(radian_lat_64[:, 1]) - np.sin(radian_lat_64[:, 0])
    xlen = radian_lon_64[:, 1] - radian_lon_64[:, 0]
    areas = radius_sqr * np.outer(ylen, xlen)

    # we use abs because backwards bounds (min > max) give negative areas.
    return np.abs(areas)

def grid_cell_areas(lon1d, lat1d, radius=EARTH_RADIUS):
    """
    Calculate grid cell areas given 1D arrays of longitudes and latitudes
    for a planet with the given radius.
    
    Parameters
    ----------
    lon1d: numpy.array
        Array of longitude points [degrees] of shape (M,)
    lat1d: numpy.array
        Array of latitude points [degrees] of shape (M,)
    radius: float, optional
        Radius of the planet [metres] (currently assumed spherical)

    Returns
    -------
    Array of grid cell areas [metres**2] of shape (M, N).
    """
    lon_bounds_radian = np.deg2rad(_guess_bounds(lon1d))
    lat_bounds_radian = np.deg2rad(_guess_bounds(lat1d))
    area = _quadrant_area(lat_bounds_radian, lon_bounds_radian, radius)
    return area

def calc_spatial_mean(
    xr_da, lon_name="longitude", lat_name="latitude", radius=EARTH_RADIUS
):
    """
    Calculate spatial mean of xarray.DataArray with grid cell weighting.
    
    Parameters
    ----------
    xr_da: xarray.DataArray
        Data to average
    lon_name: str, optional
        Name of x-coordinate
    lat_name: str, optional
        Name of y-coordinate
    radius: float
        Radius of the planet [metres], currently assumed spherical (not important anyway)

    Returns
    -------
    Spatially averaged xarray.DataArray.
    """
    lon = xr_da[lon_name].values
    lat = xr_da[lat_name].values

    area_weights = grid_cell_areas(lon, lat, radius=radius)
    aw_factor = area_weights / area_weights.max()

    return (xr_da * aw_factor).mean(dim=[lon_name, lat_name])


def calc_spatial_integral(
    xr_da, lon_name="longitude", lat_name="latitude", radius=EARTH_RADIUS
):
    """
    Calculate spatial integral of xarray.DataArray with grid cell weighting.
    
    Parameters
    ----------
    xr_da: xarray.DataArray
        Data to average
    lon_name: str, optional
        Name of x-coordinate
    lat_name: str, optional
        Name of y-coordinate
    radius: float
        Radius of the planet [metres], currently assumed spherical (not important anyway)

    Returns
    -------
    Spatially averaged xarray.DataArray.
    """
    lon = xr_da[lon_name].values
    lat = xr_da[lat_name].values

    area_weights = grid_cell_areas(lon, lat, radius=radius)

    return (xr_da * area_weights).sum(dim=[lon_name, lat_name])
