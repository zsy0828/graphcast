import numpy as np

def xyz2ball(x, y, z):
    phi = np.arctan2(y, x)
    with np.errstate(invalid="ignore"):  # circumventing b/253179568
        theta = np.arccos(z)  # Assuming unit radius.
    lon = np.mod(np.rad2deg(phi), 360)
    lat = 90 - np.rad2deg(theta)
    return lat, lon

def latlon2xyz(lat, lon):
    # Assuming unit radius.
    phi = np.deg2rad(lon)
    theta = np.deg2rad(90 - lat)
    return (np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(theta))

def get_phi_theta(lat, lon):
    return np.deg2rad(lon), np.deg2rad(90 - lat)

def spherical_to_cartesian(phi, theta):
    return (np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(theta))

# TODO: 了解这个函数是干啥的
def rotate_with_matrices(rotation_matrices: np.ndarray, positions: np.ndarray
                         ) -> np.ndarray:
    return np.einsum("bji,bi->bj", rotation_matrices, positions)