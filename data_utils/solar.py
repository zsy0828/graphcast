import pandas as pd
import numpy as np

import torch

# `integration_period` 参数的默认值，兼容 ERA5 数据。
_DEFAULT_INTEGRATION_PERIOD = pd.Timedelta(hours=1)

# `num_integration_bins` 参数的默认值，这为 ERA5 中的太阳辐射提供了良好的近似。
_DEFAULT_NUM_INTEGRATION_BINS = 360

# 儒略年（Julian Year）长度（以天为单位）。
# https://en.wikipedia.org/wiki/Julian_year_(astronomy)
_JULIAN_YEAR_LENGTH_IN_DAYS = 365.25

# J2000 纪元的儒略日期，这是天文学中的标准参考。
# https://en.wikipedia.org/wiki/Epoch_(astronomy)#Julian_years_and_J2000
_J2000_EPOCH = 2451545.0

# 每天的秒数。
_SECONDS_PER_DAY = 60 * 60 * 24

_TimestampLike = str | pd.Timestamp | np.datetime64
_TimedeltaLike = str | pd.Timedelta | np.timedelta64

# 用于加载总太阳辐照度（TSI）数据的接口。
# 返回一个包含每年平均 TSI 值的 xa.DataArray，其 `time` 坐标以自 0000-1-1 以来的年份为单位。
# 例如，2023.5 对应于 2023 年的年中。


# 总太阳辐照度（TSI）：以 W⋅m⁻² 为单位输入到地球大气层顶端的能量。
# TSI 随时间变化。当没有更精确的数据时，可以使用这个参考 TSI 值。
# https://www.ncei.noaa.gov/products/climate-data-records/total-solar-irradiance
# https://github.com/ecmwf-ifs/ecrad/blob/6db82f929fb75028cc20606a04da87c0abe9b642/radiation/radiation_ecckd.F90#L296
_REFERENCE_TSI = 1361.0


def era5_tsi_data():
    """A TsiDataProvider that returns ERA5 compatible TSI data."""
    # ECMWF provided the data used for ERA5, which was hardcoded in the IFS (cycle
    # 41r2). The values were scaled down to agree better with more recent
    # observations of the sun.
    time = np.arange(1951.5, 2035.5, 1.0)
    tsi = 0.9965 * np.array([
        # fmt: off
        # 1951-1995 (non-repeating sequence)
        1365.7765, 1365.7676, 1365.6284, 1365.6564, 1365.7773,
        1366.3109, 1366.6681, 1366.6328, 1366.3828, 1366.2767,
        1365.9199, 1365.7484, 1365.6963, 1365.6976, 1365.7341,
        1365.9178, 1366.1143, 1366.1644, 1366.2476, 1366.2426,
        1365.9580, 1366.0525, 1365.7991, 1365.7271, 1365.5345,
        1365.6453, 1365.8331, 1366.2747, 1366.6348, 1366.6482,
        1366.6951, 1366.2859, 1366.1992, 1365.8103, 1365.6416,
        1365.6379, 1365.7899, 1366.0826, 1366.6479, 1366.5533,
        1366.4457, 1366.3021, 1366.0286, 1365.7971, 1365.6996,
        # 1996-2008 (13 year cycle, repeated below)
        1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836,
        1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545,
        1365.8107, 1365.7240, 1365.6918,
        # 2009-2021
        1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836,
        1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545,
        1365.8107, 1365.7240, 1365.6918,
        # 2022-2034
        1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836,
        1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545,
        1365.8107, 1365.7240, 1365.6918,
        # fmt: on
    ])
    return time, tsi


def get_tsi(timestamps, time, tsi):
    """Returns TSI values for the given timestamps.

    TSI values are interpolated from the provided yearly TSI data.

    Args:
      timestamps: Timestamps for which to compute TSI values.
      tsi_data: A DataArray with a single dimension `time` that has coordinates in
        units of years since 0000-1-1. E.g. 2023.5 corresponds to the middle of
        the year 2023.

    Returns:
      An Array containing interpolated TSI data.
    """
    timestamps = pd.DatetimeIndex(timestamps)
    timestamps_date = pd.DatetimeIndex(timestamps.date)
    day_fraction = (timestamps - timestamps_date) / pd.Timedelta(days=1)
    year_length = 365 + timestamps.is_leap_year
    year_fraction = (timestamps.dayofyear - 1 + day_fraction) / year_length
    fractional_year = timestamps.year + year_fraction
    return np.interp(fractional_year, time, tsi)


def _get_orbital_parameters(j2000_days):
    """
    计算给定J2000日期的轨道参数。

    参数：
        j2000_days：从J2000时到目前时间的日期数。

    返回：
        轨道参数。每个属性都是一个数组，和输入类似。
    """
    # 计算从J2000日期开始的伦理年数（包括小数年）。
    theta = j2000_days / _JULIAN_YEAR_LENGTH_IN_DAYS
    # 地球轨道转动的相位置，用组合国际日期。
    rotational_phase = j2000_days % 1.0

    # REL(PTETA)：
    rel = 1.7535 + 6.283076 * theta
    # REM(PTETA)：
    rem = 6.240041 + 6.283020 * theta
    # RLLS(PTETA)：
    rlls = 4.8951 + 6.283076 * theta

    # 下面的三个多项式用到的变量
    one = np.ones_like(theta)
    sin_rel = np.sin(rel)
    cos_rel = np.cos(rel)
    sin_two_rel = np.sin(2.0 * rel)
    cos_two_rel = np.cos(2.0 * rel)
    sin_two_rlls = np.sin(2.0 * rlls)
    cos_two_rlls = np.cos(2.0 * rlls)
    sin_four_rlls = np.sin(4.0 * rlls)
    sin_rem = np.sin(rem)
    sin_two_rem = np.sin(2.0 * rem)

    # 轨道中的日长 - RLLLS(PTETA)：
    rllls = np.sum(
        np.stack(
            [one, theta, sin_rel, cos_rel, sin_two_rel, cos_two_rel], axis=-1
        ) * np.array([4.8952, 6.283320, -0.0075, -0.0326, -0.0003, 0.0002]),
        axis=-1,
    )

    # 地球的轨道转动与轨道起始的角度，应为23.4393°：
    repsm = 0.409093

    # 太阳的平星方位：
    sin_declination = np.sin(repsm) * np.sin(rllls)
    cos_declination = np.sqrt(1.0 - sin_declination ** 2)

    # 时间日线之间的公式：
    eq_of_time_seconds = np.sum(
        np.stack(
            [
                sin_two_rlls,
                sin_rem,
                sin_rem * cos_two_rlls,
                sin_four_rlls,
                sin_two_rem,
            ],
            axis=-1,
        ) * np.array([591.8, -459.4, 39.5, -12.7, -4.8]),
        axis=-1,
    )

    # 太阳距离。
    solar_distance_au = np.sum(
        np.stack([one, sin_rel, cos_rel], axis=-1)
        * np.array([1.0001, -0.0163, 0.0037]),
        axis=-1,
    )

    return (theta, rotational_phase, sin_declination, cos_declination, eq_of_time_seconds, solar_distance_au)


def _get_solar_sin_altitude(
        theta, rotational_phase, sin_declination, cos_declination, eq_of_time_seconds, solar_distance_au,
        sin_latitude,
        cos_latitude,
        longitude,
):
    """Returns the sine of the solar altitude angle.

    All computations are vectorized. Dimensions of all the inputs should be
    broadcastable using standard NumPy rules. For example, if `op` has shape
    `(T, 1, 1)`, `latitude` has shape `(1, H, 1)`, and `longitude` has shape
    `(1, H, W)`, the return value will have shape `(T, H, W)`.

    Args:
      op: Orbital parameters characterising Earth's position relative to the Sun.
      sin_latitude: Sine of latitude coordinates.
      cos_latitude: Cosine of latitude coordinates.
      longitude: Longitude coordinates in radians.

    Returns:
      Sine of the solar altitude angle for each set of orbital parameters and each
      geographical coordinates. The returned array has the shape resulting from
      broadcasting all the inputs together.
    """
    solar_time = rotational_phase + eq_of_time_seconds / _SECONDS_PER_DAY
    # https://en.wikipedia.org/wiki/Hour_angle#Solar_hour_angle
    hour_angle = 2.0 * np.pi * solar_time + longitude
    # https://en.wikipedia.org/wiki/Solar_zenith_angle
    sin_altitude = (
            cos_latitude * cos_declination * np.cos(hour_angle)
            + sin_latitude * sin_declination
    )
    return sin_altitude


def _get_radiation_flux(
        j2000_days,
        sin_latitude,
        cos_latitude,
        longitude,
        tsi,
):
    """Computes the instantaneous TOA incident solar radiation flux.

    Computes the instantanous Top-Of-the-Atmosphere (TOA) incident radiation flux
    in W⋅m⁻² for the given timestamps and locations on the surface of the Earth.
    See https://en.wikipedia.org/wiki/Solar_irradiance.

    All inputs are assumed to be broadcastable together using standard NumPy
    rules.

    Args:
      j2000_days: Timestamps represented as the number of days since the J2000
        epoch.
      sin_latitude: Sine of latitude coordinates.
      cos_latitude: Cosine of latitude coordinates.
      longitude: Longitude coordinates in radians.
      tsi: Total Solar Irradiance (TSI) in W⋅m⁻². This can be a scalar (default)
        to use the same TSI value for all the inputs, or an array to allow TSI to
        depend on the timestamps.

    Returns:
      The instataneous TOA incident solar radiation flux in W⋅m⁻² for the given
      timestamps and geographical coordinates. The returned array has the shape
      resulting from broadcasting all the inputs together.
    """
    theta, rotational_phase, sin_declination, cos_declination, eq_of_time_seconds, solar_distance_au = _get_orbital_parameters(
        j2000_days)
    # Attenuation of the solar radiation based on the solar distance.
    solar_factor = (1.0 / solar_distance_au) ** 2
    sin_altitude = _get_solar_sin_altitude(
        theta, rotational_phase, sin_declination, cos_declination, eq_of_time_seconds, solar_distance_au, sin_latitude,
        cos_latitude, longitude
    )
    return tsi * solar_factor * np.maximum(sin_altitude, 0.0)


def _get_integrated_radiation(
        j2000_days: np.ndarray,
        sin_latitude: np.ndarray,
        cos_latitude: np.ndarray,
        longitude: np.ndarray,
        tsi: np.ndarray,
        integration_period: pd.Timedelta,
        num_integration_bins: int,
) -> np.ndarray:
    """
    返回空间顶层太阳轻辉流程集成的结果。

    将当前时间中的空间顶层太阳轻辉流程进行时间求积。输入的时间标记表示每个集成时间幕的结束时间。

    参数：
        j2000_days：从J2000时到目前时间的日期数，表示每个集成时间幕的结束时间。
        sin_latitude：纬度的正弦值。
        cos_latitude：纬度的余弦值。
        longitude：经度，以弧度表示。
        tsi：总太阳轻辉（TSI），单位为 W·m⁻²。
        integration_period：集成时间。
        num_integration_bins：将`integration_period`拆分成的小块数，以用二次角形运算来近似积分。

    返回：
        在请求的时间幕内给定的时间空间和地理坐标下，空间顶层的太阳轻辉流积分，单位为J·m⁻²。
    """
    # 应用每个集成时间段的偏离值
    offsets = (
            pd.timedelta_range(
                start=-integration_period,
                end=pd.Timedelta(0),
                periods=num_integration_bins + 1,
            )
            / pd.Timedelta(days=1)
    ).to_numpy()

    # 在时间维度上进行集成，为了计算每个集成时间段的轻辉流，应用偏离值并向所有输入添加一个维度
    fluxes = _get_radiation_flux(
        j2000_days=np.expand_dims(j2000_days, axis=-1) + offsets,
        sin_latitude=np.expand_dims(sin_latitude, axis=-1),
        cos_latitude=np.expand_dims(cos_latitude, axis=-1),
        longitude=np.expand_dims(longitude, axis=-1),
        tsi=np.expand_dims(tsi, axis=-1),
    )

    # 每个小块的大小，单位为秒。当前的太阳轻辉流以 W·m⁻² 为单位。集成段进行积分并生成单位为 J·m⁻² 的数据。
    dx = (integration_period / num_integration_bins) / pd.Timedelta(seconds=1)
    return np.trapz(fluxes, dx=dx, axis=-1)


def _get_j2000_days(timestamp):
    return timestamp.to_julian_date() - _J2000_EPOCH


def get_toa_incident_solar_radiation(
        timestamps,
        latitude: np.ndarray,
        longitude: np.ndarray,
        time, tsi_data,
        integration_period,
        num_integration_bins,
):
    """
    计算空间顶层的太阳轻辉流。

    太阳轻辉流是在`timestamps`每个时间点为编结的，用`latitude`和`longitude`参数确定的网格上所有位置输出的。

    参数：
        timestamps：用于计算太阳轻辉流的时间点。
        latitude：网格纬度，单位是度。
        longitude：网格经度，单位是度。
        tsi_data：包含年度太阳轻辉(TSI)数据的DataArray，如果不输入，使用与ERA5相关的默认TSI数据。
        integration_period：用于集成轻辉的时间间隔。
        num_integration_bins：将`integration_period`分成的小块数。
        use_jit：设为True时，使用jitted实现，默认为False，使用非带jit的实现。

    返回：
        一个缺纳的三维数组，具有时间（time）、纬度（lat）和经度（lon）维度，表示了集成各种时间间隔和各时间点的空间顶层太阳轻辉。
    """
    # 添加纬度的尾部维度，使其达到网格的(lat, lon)维度。
    lat = np.radians(latitude).reshape((-1, 1))
    lon = np.radians(longitude)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    integration_period = pd.Timedelta(integration_period)

    tsi = get_tsi(timestamps, time, tsi_data)
    fn = _get_integrated_radiation

    # 为每个时间点单独计算集成。虽然这可以一步完成，但最峰内存使用量将与`len(timestamps) * num_integration_bins`成比例。每次单独计算时间点降低该值到`max(len(timestamps), num_integration_bins)`。
    # 例如，对于一个单独的时间点，配合全部 0.25° 的网格和 360 的集成块，内存使用量大约1.5 GB；计算 40 步预测步骤的集成也不会超过60 GB。
    results = []
    for idx, timestamp in enumerate(timestamps):
        results.append(
            fn(
                j2000_days=np.array(_get_j2000_days(pd.Timestamp(timestamp))),
                sin_latitude=sin_lat,
                cos_latitude=cos_lat,
                longitude=lon,
                tsi=tsi[idx],
                integration_period=integration_period,
                num_integration_bins=num_integration_bins,
            )
        )
    return np.stack(results, axis=0, dtype=np.float32)


def get_toa_incident_solar_radiation_for_torch(lat, lon, time, tsi,
                                               timestamps,
                                               integration_period=_DEFAULT_INTEGRATION_PERIOD,
                                               num_integration_bins=_DEFAULT_NUM_INTEGRATION_BINS):
    """
    计算空间顶层的太阳轻辉流。

    此方法是用于使用Xarray的坐标并返回一个Xarray的封装。

    参数：
        data_array_like：一个xa.Dataset或xa.DataArray，用于计算太阳轻辉流的时间和空间坐标。它必须包含`lat`和`lon`空间维度，且应包含对应的坐标。如果存在`time`维度，`datetime`坐标应是一个向量，包含这些时间点；否则，`datetime`应该是一个表示时间点的标量。
        tsi_data：包含年度TSI数据的DataArray，如果未输入，则使用ERA5兼容的TSI数据。
        integration_period：用于集成轻辉的时间，例如，当计算1989-11-08 21:00:00时，如果`integration_period`是"1h"，轻辉将从1989-11-08 20:00:00集成到1989-11-08 21:00:00。默认值("1h")与ERA5相匹配。
        num_integration_bins：将`integration_period`分成的平分小块数，用来近似使用二次角形规则进行积分。默认值(360)提供了一个良好的近似，但较低的值可能有助于提高性能和减少内存使用。
        use_jit：设为True使用jitted实现，设为False则使用非带jit的实现。

    返回：
        如果`data_array_like`具有`time`维度，则返回一个维度为`(time, lat, lon)`的xa.DataArray；否则返回维度为`(lat, lon)`。返回数组的坐标会存储并保存与原来的相匹配的坐标。这个数组包含集成轻辉的空间顶层太阳轻辉。

    强制性错误：
        ValueError：如果缺少的维度或坐标。
    """

    radiation = get_toa_incident_solar_radiation(
        timestamps=timestamps,
        latitude=lat,
        longitude=lon,
        time=time, tsi_data=tsi,
        integration_period=integration_period,
        num_integration_bins=num_integration_bins,
    )

    return torch.from_numpy(radiation)


if __name__ == '__main__':
    start_date = '2000-01-01 00:00:00'  # 使用2000年，因为它是一个闰年
    end_date = '2000-12-31 23:59:59'
    time_series = pd.date_range(start=start_date, end=end_date, freq='6h')
    lat = np.linspace(90, -90, 721)
    lon = np.linspace(0.0, 359.75, 1440)
    time, tsi = era5_tsi_data()
    time_series = time_series[:1460]
    time_array = time_series.to_numpy(dtype='datetime64[ns]')

    print(get_toa_incident_solar_radiation_for_torch(lat, lon, time, tsi, time_array[5:6]).shape)
