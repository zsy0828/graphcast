from typing import Mapping, Optional, Tuple
from tongyong_utils import *
from grid_mesh import *
from mesh import *


def get_graph_spatial_features(
        *, node_lat: np.ndarray, node_lon: np.ndarray,
        senders: np.ndarray, receivers: np.ndarray,
        add_node_positions: bool,
        add_node_latitude: bool,
        add_node_longitude: bool,
        add_relative_positions: bool,
        relative_longitude_local_coordinates: bool,
        relative_latitude_local_coordinates: bool,
        sine_cosine_encoding: bool = False,
        encoding_num_freqs: int = 10,
        encoding_multiplicative_factor: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray]:

    num_nodes = node_lat.shape[0]
    num_edges = senders.shape[0]
    dtype = node_lat.dtype
    node_phi, node_theta = get_phi_theta(node_lat, node_lon)

    node_features = []
    if add_node_positions:
        node_features.extend(spherical_to_cartesian(node_phi, node_theta))

    if add_node_latitude:
        node_features.append(np.cos(node_theta))

    if add_node_longitude:
        node_features.append(np.cos(node_phi))
        node_features.append(np.sin(node_phi))

    if not node_features:
        node_features = np.zeros([num_nodes, 0], dtype=dtype)
    else:
        node_features = np.stack(node_features, axis=-1)

    edge_features = []

    if add_relative_positions:
        relative_position = get_relative_position_in_receiver_local_coordinates(
            node_phi=node_phi,
            node_theta=node_theta,
            senders=senders,
            receivers=receivers,
            latitude_local_coordinates=relative_latitude_local_coordinates,
            longitude_local_coordinates=relative_longitude_local_coordinates
        )

        relative_edge_distances = np.linalg.norm(
            relative_position, axis=-1, keepdims=True)

        max_edge_distance = relative_edge_distances.max()
        edge_features.append(relative_edge_distances / max_edge_distance)
        edge_features.append(relative_position / max_edge_distance)

    if not edge_features:
        edge_features = np.zeros([num_edges, 0], dtype=dtype)
    else:
        edge_features = np.concatenate(edge_features, axis=-1)

    if sine_cosine_encoding:
        def sine_cosine_transform(x: np.ndarray) -> np.ndarray:
            freqs = encoding_multiplicative_factor ** np.arange(encoding_num_freqs)
            phases = freqs * x[..., None]
            x_sin = np.sin(phases)
            x_cos = np.cos(phases)
            x_cat = np.concatenate([x_sin, x_cos], axis=-1)
            return x_cat.reshape([x.shape[0], -1])

        node_features = sine_cosine_transform(node_features)
        edge_features = sine_cosine_transform(edge_features)

    return node_features, edge_features


def get_relative_position_in_receiver_local_coordinates(
        node_phi: np.ndarray,
        node_theta: np.ndarray,
        senders: np.ndarray,
        receivers: np.ndarray,
        latitude_local_coordinates: bool,
        longitude_local_coordinates: bool
) -> np.ndarray:

    node_pos = np.stack(spherical_to_cartesian(node_phi, node_theta), axis=-1)

    if not (latitude_local_coordinates or longitude_local_coordinates):
        return node_pos[senders] - node_pos[receivers]

    rotation_matrices = get_rotation_matrices_to_local_coordinates(
        reference_phi=node_phi,
        reference_theta=node_theta,
        rotate_latitude=latitude_local_coordinates,
        rotate_longitude=longitude_local_coordinates)

    edge_rotation_matrices = rotation_matrices[receivers]

    receiver_pos_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, node_pos[receivers])
    sender_pos_in_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, node_pos[senders])

    return sender_pos_in_in_rotated_space - receiver_pos_in_rotated_space




'''
-----------------------------------------------------------------------------------------------------------------------------------
'''
def get_bipartite_graph_spatial_features(
        *,
        senders_node_lat: np.ndarray,
        senders_node_lon: np.ndarray,
        senders: np.ndarray,
        receivers_node_lat: np.ndarray,
        receivers_node_lon: np.ndarray,
        receivers: np.ndarray,
        add_node_positions: bool,
        add_node_latitude: bool,
        add_node_longitude: bool,
        add_relative_positions: bool,
        edge_normalization_factor: Optional[float] = None,
        relative_longitude_local_coordinates: bool,
        relative_latitude_local_coordinates: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    num_senders = senders_node_lat.shape[0]
    num_receivers = receivers_node_lat.shape[0]
    num_edges = senders.shape[0]
    dtype = senders_node_lat.dtype
    assert receivers_node_lat.dtype == dtype

    senders_node_phi, senders_node_theta =  get_phi_theta(
        senders_node_lat, senders_node_lon)
    receivers_node_phi, receivers_node_theta =  get_phi_theta(
        receivers_node_lat, receivers_node_lon)

    senders_node_features = []
    receivers_node_features = []
    if add_node_positions:
        senders_node_features.extend(latlon2xyz(senders_node_lat, senders_node_lon))
        receivers_node_features.extend(latlon2xyz(receivers_node_lat, receivers_node_lon))

    if add_node_latitude:
        senders_node_features.append(np.cos(senders_node_theta))
        receivers_node_features.append(np.cos(receivers_node_theta))

    if add_node_longitude:
        senders_node_features.append(np.cos(senders_node_phi))
        senders_node_features.append(np.sin(senders_node_phi))

        receivers_node_features.append(np.cos(receivers_node_phi))
        receivers_node_features.append(np.sin(receivers_node_phi))

    if not senders_node_features:
        senders_node_features = np.zeros([num_senders, 0], dtype=dtype)
        receivers_node_features = np.zeros([num_receivers, 0], dtype=dtype)
    else:
        senders_node_features = np.stack(senders_node_features, axis=-1)
        receivers_node_features = np.stack(receivers_node_features, axis=-1)

    edge_features = []

    if add_relative_positions:

        relative_position = get_bipartite_relative_position_in_receiver_local_coordinates(
            # pylint: disable=line-too-long
            senders_node_phi=senders_node_phi,
            senders_node_theta=senders_node_theta,
            receivers_node_phi=receivers_node_phi,
            receivers_node_theta=receivers_node_theta,
            senders=senders,
            receivers=receivers,
            latitude_local_coordinates=relative_latitude_local_coordinates,
            longitude_local_coordinates=relative_longitude_local_coordinates)

        relative_edge_distances = np.linalg.norm(
            relative_position, axis=-1, keepdims=True)

        if edge_normalization_factor is None:
            edge_normalization_factor = relative_edge_distances.max()

        edge_features.append(relative_edge_distances / edge_normalization_factor)
        edge_features.append(relative_position / edge_normalization_factor)

    if not edge_features:
        edge_features = np.zeros([num_edges, 0], dtype=dtype)
    else:
        edge_features = np.concatenate(edge_features, axis=-1)

    return senders_node_features, receivers_node_features, edge_features


def get_bipartite_relative_position_in_receiver_local_coordinates(
        senders_node_phi: np.ndarray,
        senders_node_theta: np.ndarray,
        senders: np.ndarray,
        receivers_node_phi: np.ndarray,
        receivers_node_theta: np.ndarray,
        receivers: np.ndarray,
        latitude_local_coordinates: bool,
        longitude_local_coordinates: bool) -> np.ndarray:

    senders_node_pos = np.stack(
        spherical_to_cartesian(senders_node_phi, senders_node_theta), axis=-1)

    receivers_node_pos = np.stack(
        spherical_to_cartesian(receivers_node_phi, receivers_node_theta), axis=-1)


    if not (latitude_local_coordinates or longitude_local_coordinates):
        return senders_node_pos[senders] - receivers_node_pos[receivers]


    receiver_rotation_matrices = get_rotation_matrices_to_local_coordinates(
        reference_phi=receivers_node_phi,
        reference_theta=receivers_node_theta,
        rotate_latitude=latitude_local_coordinates,
        rotate_longitude=longitude_local_coordinates)

    edge_rotation_matrices = receiver_rotation_matrices[receivers]

    receiver_pos_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, receivers_node_pos[receivers])
    sender_pos_in_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, senders_node_pos[senders])
    return sender_pos_in_in_rotated_space - receiver_pos_in_rotated_space


def get_rotation_matrices_to_local_coordinates(
        reference_phi: np.ndarray,
        reference_theta: np.ndarray,
        rotate_latitude: bool,
        rotate_longitude: bool) -> np.ndarray:
    """Returns a rotation matrix to rotate to a point based on a reference vector.

    The rotation matrix is build such that, a vector in the
    same coordinate system at the reference point that points towards the pole
    before the rotation, continues to point towards the pole after the rotation.

    Args:
      reference_phi: [leading_axis] Polar angles of the reference.
      reference_theta: [leading_axis] Azimuthal angles of the reference.
      rotate_latitude: Whether to produce a rotation matrix that would rotate
          R^3 vectors to zero latitude.
      rotate_longitude: Whether to produce a rotation matrix that would rotate
          R^3 vectors to zero longitude.

    Returns:
      Matrices of shape [leading_axis] such that when applied to the reference
          position with `rotate_with_matrices(rotation_matrices, reference_pos)`

          * phi goes to 0. if "rotate_longitude" is True.

          * theta goes to np.pi / 2 if "rotate_latitude" is True.

          The rotation consists of:
          * rotate_latitude = False, rotate_longitude = True:
              Latitude preserving rotation.
          * rotate_latitude = True, rotate_longitude = True:
              Latitude preserving rotation, followed by longitude preserving
              rotation.
          * rotate_latitude = True, rotate_longitude = False:
              Latitude preserving rotation, followed by longitude preserving
              rotation, and the inverse of the latitude preserving rotation. Note
              this is computationally different from rotating the longitude only
              and is. We do it like this, so the polar geodesic curve, continues
              to be aligned with one of the axis after the rotation.

    """

    if rotate_longitude and rotate_latitude:

        # We first rotate around the z axis "minus the azimuthal angle", to get the
        # point with zero longitude
        azimuthal_rotation = - reference_phi

        # One then we will do a polar rotation (which can be done along the y
        # axis now that we are at longitude 0.), "minus the polar angle plus 2pi"
        # to get the point with zero latitude.
        polar_rotation = - reference_theta + np.pi / 2

        return transform.Rotation.from_euler(
            "zy", np.stack([azimuthal_rotation, polar_rotation],
                           axis=1)).as_matrix()
    elif rotate_longitude:
        # Just like the previous case, but applying only the azimuthal rotation.
        azimuthal_rotation = - reference_phi
        return transform.Rotation.from_euler("z", -reference_phi).as_matrix()
    elif rotate_latitude:
        # Just like the first case, but after doing the polar rotation, undoing
        # the azimuthal rotation.
        azimuthal_rotation = - reference_phi
        polar_rotation = - reference_theta + np.pi / 2

        return transform.Rotation.from_euler(
            "zyz", np.stack(
                [azimuthal_rotation, polar_rotation, -azimuthal_rotation]
                , axis=1)).as_matrix()
    else:
        raise ValueError(
            "At least one of longitude and latitude should be rotated.")




