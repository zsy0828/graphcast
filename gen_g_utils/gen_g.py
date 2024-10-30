from grid_mesh import *
from mesh import *
from relation import *


def geng_by_dgl(g2m, m2m, m2g):
    import torch
    import dgl
    g2m_senders, g2m_receivers, g2m_srcnode_feat, g2m_dstnode_feat, g2m_edge_feat = g2m
    m2m_senders, m2m_receivers, m2m_node_feat, m2m_edge_feat = m2m
    m2g_senders, m2g_receivers, m2g_srcnode_feat, m2g_dstnode_feat, m2g_edge_feat = m2g
    num_mesh = 40962
    num_grid = 721 * 1440

    g2m_dict = {
        ('grid', 'g2m', 'mesh'): (g2m_senders, g2m_receivers)
    }
    g2m_g = dgl.heterograph(data_dict=g2m_dict, num_nodes_dict={'mesh': num_mesh, 'grid': num_grid})
    g2m_g.nodes['grid'].data['h'] = torch.FloatTensor(g2m_srcnode_feat)
    g2m_g.nodes['mesh'].data['h'] = torch.FloatTensor(g2m_dstnode_feat)
    g2m_g.edges['g2m'].data['e_h'] = torch.FloatTensor(g2m_edge_feat)
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------
    m2m_dict = {
        ('mesh', 'm2m', 'mesh'): (m2m_senders, m2m_receivers)
    }
    m2m_g = dgl.heterograph(data_dict=m2m_dict, num_nodes_dict={'mesh': num_mesh})
    m2m_g.nodes['mesh'].data['h'] = torch.FloatTensor(m2m_node_feat)
    m2m_g.edges['m2m'].data['e_h'] = torch.FloatTensor(m2m_edge_feat)
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------
    m2g_dict = {
        ('mesh', 'm2g', 'grid'): (m2g_senders, m2g_receivers)
    }
    m2g_g = dgl.heterograph(data_dict=m2g_dict, num_nodes_dict={'mesh': num_mesh, 'grid': num_grid})
    m2g_g.nodes['mesh'].data['h'] = torch.FloatTensor(m2g_srcnode_feat)
    m2g_g.nodes['grid'].data['h'] = torch.FloatTensor(m2g_dstnode_feat)
    m2g_g.edges['m2g'].data['e_h'] = torch.FloatTensor(m2g_edge_feat)

    return g2m_g, m2m_g, m2g_g


if __name__ == '__main__':
    lats = np.linspace(90, -90, 721)
    lons = np.linspace(0, 359.75, 1440)
    flatten_grid(lats, lons)
    print()
    vertices, faces = gen_icosahedron_mesh()

    output_meshes = multi_xihua_icosahedron_mesh(vertices, faces)
    merge_mesh = merge_meshes(output_meshes)
    max_len = get_max_edge_distance(merge_mesh)
    grid_coords_xyz = grid_lat_lon_to_coordinates(lats, lons).reshape(-1, 3)
    g2m_senders, g2m_receivers = gen_grid2mesh(grid_coords_xyz, merge_mesh['vertices'], max_len * .6)
    spatial_features_kwargs = dict(
        add_node_positions=True,
        add_node_latitude=True,
        add_node_longitude=True,
        add_relative_positions=True,
        relative_longitude_local_coordinates=True,
        relative_latitude_local_coordinates=True,
    )
    grid_nodes_lat, grid_nodes_lon = gen_grid_lat_lon()

    mesh_nodes_lat, mesh_nodes_lon = xyz2ball(merge_mesh['vertices'][:, 0], merge_mesh['vertices'][:, 1],
                                              merge_mesh['vertices'][:, 2])

    (g2m_srcnode_feat, g2m_dstnode_feat,
     g2m_edge_feat) = get_bipartite_graph_spatial_features(
        senders_node_lat=grid_nodes_lat,
        senders_node_lon=grid_nodes_lon,
        receivers_node_lat=mesh_nodes_lat,
        receivers_node_lon=mesh_nodes_lon,
        senders=g2m_senders,
        receivers=g2m_receivers,
        edge_normalization_factor=None,
        **spatial_features_kwargs,
    )
    # 后续是处理 mesh2mesh
    m2m_senders, m2m_receivers = merge_mesh['edges'][0], merge_mesh['edges'][1]
    m2m_node_feat, m2m_edge_feat = get_graph_spatial_features(
        node_lat=mesh_nodes_lat,
        node_lon=mesh_nodes_lon,
        senders=m2m_senders,
        receivers=m2m_receivers,
        **spatial_features_kwargs,
    )

    # 最后是 mesh2grid

    m2g_senders, m2g_receivers = gen_mesh2grid(grid_coords_xyz, merge_mesh['vertices'], merge_mesh['faces'])
    (m2g_srcnode_feat, m2g_dstnode_feat,
     m2g_edge_feat) = get_bipartite_graph_spatial_features(
        senders_node_lat=mesh_nodes_lat,
        senders_node_lon=mesh_nodes_lon,
        receivers_node_lat=grid_nodes_lat,
        receivers_node_lon=grid_nodes_lon,
        senders=m2g_senders,
        receivers=m2g_receivers,
        edge_normalization_factor=True,  # 只有对这个可以进行处理，就是g2m是None
        **spatial_features_kwargs,
    )
    g2m_g, m2m_g, m2g_g = geng_by_dgl((g2m_senders, g2m_receivers, g2m_srcnode_feat, g2m_dstnode_feat, g2m_edge_feat),
                                      (m2m_senders, m2m_receivers, m2m_node_feat, m2m_edge_feat),
                                      (m2g_senders, m2g_receivers, m2g_srcnode_feat, m2g_dstnode_feat, m2g_edge_feat))

    import dgl
    dgl.save_graphs('graphcast.dgl',[g2m_g, m2m_g, m2g_g])
    # print(g2m_g)
    # print(m2m_g)
    # print(m2g_g)

