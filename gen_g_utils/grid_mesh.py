import numpy as np
import trimesh
from scipy.spatial import cKDTree

from mesh import *


def grid_lat_lon_to_coordinates(
        grid_latitude: np.ndarray, grid_longitude: np.ndarray) -> np.ndarray:
    """Lat [num_lat] lon [num_lon] to 3d coordinates [num_lat, num_lon, 3]."""
    # Convert to spherical coordinates phi and theta defined in the grid.
    # Each [num_latitude_points, num_longitude_points]
    phi_grid, theta_grid = np.meshgrid(
        np.deg2rad(grid_longitude),
        np.deg2rad(90 - grid_latitude))

    # [num_latitude_points, num_longitude_points, 3]
    # Note this assumes unit radius, since for now we model the earth as a
    # sphere of unit radius, and keep any vertical dimension as a regular grid.
    return np.stack(
        [np.cos(phi_grid) * np.sin(theta_grid),
         np.sin(phi_grid) * np.sin(theta_grid),
         np.cos(theta_grid)], axis=-1)


def gen_grid2mesh(grid_coords_xyz, mesh_vertices, threshold):
    '''
    grid点链接mesh
    '''
    # [num_mesh_points, 3]
    # mesh_vertices
    tree = cKDTree(mesh_vertices)
    indices = tree.query_ball_point(grid_coords_xyz, r=threshold)
    grid_edge_indices = []
    mesh_edge_indices = []
    for grid_index, mesh_neighbors in enumerate(indices):
        grid_edge_indices.append(np.repeat(grid_index, len(mesh_neighbors)))
        mesh_edge_indices.append(mesh_neighbors)
    # [num_edges]
    grid_edge_indices = np.concatenate(grid_edge_indices, dtype=np.int64, axis=0)
    mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0, dtype=np.int64)
    return grid_edge_indices, mesh_edge_indices


def gen_mesh2grid(grid_coords_xyz, mesh_vertices, mesh_faces):
    '''
    mesh 链接 grid， 获取三个最邻近mesh的grid进行链接
    '''

    mesh_trimesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)

    _, _, query_face_indices = trimesh.proximity.closest_point(
        mesh_trimesh, grid_coords_xyz)

    mesh_edge_indices = mesh_faces[query_face_indices]
    grid_indices = np.arange(grid_coords_xyz.shape[0])
    grid_edge_indices = np.tile(grid_indices.reshape([-1, 1]), [1, 3])

    mesh_edge_indices = mesh_edge_indices.reshape([-1])
    grid_edge_indices = grid_edge_indices.reshape([-1])

    return mesh_edge_indices, grid_edge_indices

def flatten_grid(lat, lon):
    lat, lon = np.meshgrid(lat, lon)
    return lat.reshape([-1]).astype(np.float32), lon.reshape([-1]).astype(np.float32)

def get_max_edge_distance(mesh):
    senders = np.concatenate([mesh['faces'][:, 0], mesh['faces'][:, 1], mesh['faces'][:, 2]])
    receivers = np.concatenate([mesh['faces'][:, 1], mesh['faces'][:, 2], mesh['faces'][:, 0]])

    edge_distances = np.linalg.norm(
        mesh['vertices'][senders] - mesh['vertices'][receivers], axis=-1)
    return edge_distances.max()

def gen_grid_lat_lon():
    lats = np.linspace(90, -90, 721)
    lons = np.linspace(0, 359.75, 1440)
    grid_nodes_lat, grid_nodes_lon = np.meshgrid(lons, lats)
    return grid_nodes_lat.reshape([-1]).astype(np.float32), grid_nodes_lon.reshape([-1]).astype(np.float32)


if __name__ == '__main__':
    lats = np.linspace(90, -90, 721)
    lons = np.linspace(0, 359.75, 1440)
    flatten_grid(lats, lons)
    print()
    vertices, faces = gen_icosahedron_mesh()
    output_meshes = multi_xihua_icosahedron_mesh(vertices, faces)
    merge_mesh = merge_meshes(output_meshes)
    grid_coords_xyz = grid_lat_lon_to_coordinates(lats, lons).reshape(-1, 3)
    gen_mesh2grid(grid_coords_xyz, merge_mesh['vertices'], merge_mesh['faces'])

