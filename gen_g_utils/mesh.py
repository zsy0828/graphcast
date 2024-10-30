import numpy as np
from scipy.spatial import transform

from tongyong_utils import xyz2ball


def gen_icosahedron_mesh():
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    for c1 in [1., -1.]:
        for c2 in [phi, -phi]:
            vertices.append((c1, c2, 0.))
            vertices.append((0., c1, c2))
            vertices.append((c2, 0., c1))

    vertices = np.array(vertices, dtype=np.float32)
    vertices /= np.linalg.norm([1., phi])
    faces = [(0, 1, 2),
             (0, 6, 1),
             (8, 0, 2),
             (8, 4, 0),
             (3, 8, 2),
             (3, 2, 7),
             (7, 2, 1),
             (0, 4, 6),
             (4, 11, 6),
             (6, 11, 5),
             (1, 5, 7),
             (4, 10, 11),
             (4, 8, 10),
             (10, 8, 3),
             (10, 3, 9),
             (11, 10, 9),
             (11, 9, 5),
             (5, 9, 7),
             (9, 3, 7),
             (1, 6, 5),
             ]
    angle_between_faces = 2 * np.arcsin(phi / np.sqrt(3))
    rotation_angle = (np.pi - angle_between_faces) / 2
    rotation = transform.Rotation.from_euler(seq="y", angles=rotation_angle)
    rotation_matrix = rotation.as_matrix()
    vertices = np.dot(vertices, rotation_matrix)
    # 节点的球面坐标; 正多面的由哪几个节点组成
    return vertices.astype(np.float32), np.array(faces, dtype=np.int64)


def xihua_icosahedron_mesh(parent_vertices, parent_faces):
    """将每个三角形面拆分为4个小三角形，保持方向不变。"""

    # 初始化新的顶点列表，包含原始顶点
    new_vertices_list = list(parent_vertices)
    # 创建一个字典来记录边与新顶点索引的映射
    edge_to_new_vertex_index = {}

    new_faces = []
    #                    ind3
    #                   /    \
    #                /          \
    #              /      #3       \
    #            /                  \
    #         ind31 -------------- ind23
    #         /   \                /   \
    #       /       \     #4     /      \
    #     /    #1     \        /    #2    \
    #   /               \    /              \
    # ind1 ------------ ind12 ------------ ind2
    for ind1, ind2, ind3 in parent_faces:
        # 处理每一条边，获取或创建中点顶点
        # 边 (ind1, ind2)
        edge = tuple(sorted((ind1, ind2)))
        if edge not in edge_to_new_vertex_index:
            # 创建中点顶点
            child_vertex_position = parent_vertices[list(edge)].mean(axis=0)
            # 投影到单位球面上
            child_vertex_position /= np.linalg.norm(child_vertex_position)
            # 添加到新的顶点列表
            edge_to_new_vertex_index[edge] = len(new_vertices_list)
            new_vertices_list.append(child_vertex_position)
        ind12 = edge_to_new_vertex_index[edge]

        # 边 (ind2, ind3)
        edge = tuple(sorted((ind2, ind3)))
        if edge not in edge_to_new_vertex_index:
            child_vertex_position = parent_vertices[list(edge)].mean(axis=0)
            child_vertex_position /= np.linalg.norm(child_vertex_position)
            edge_to_new_vertex_index[edge] = len(new_vertices_list)
            new_vertices_list.append(child_vertex_position)
        ind23 = edge_to_new_vertex_index[edge]

        # 边 (ind3, ind1)
        edge = tuple(sorted((ind3, ind1)))
        if edge not in edge_to_new_vertex_index:
            child_vertex_position = parent_vertices[list(edge)].mean(axis=0)
            child_vertex_position /= np.linalg.norm(child_vertex_position)
            edge_to_new_vertex_index[edge] = len(new_vertices_list)
            new_vertices_list.append(child_vertex_position)
        ind31 = edge_to_new_vertex_index[edge]

        # 创建4个新的三角形面，保持顶点的顺序以维持方向
        new_faces.extend([
            [ind1, ind12, ind31],  # 面1
            [ind12, ind2, ind23],  # 面2
            [ind31, ind23, ind3],  # 面3
            [ind12, ind23, ind31],  # 面4
        ])

    # 将新的顶点列表转换为numpy数组
    new_vertices = np.array(new_vertices_list)
    new_faces = np.array(new_faces, dtype=np.int64)

    return new_vertices, new_faces


def multi_xihua_icosahedron_mesh(parent_vertices, parent_faces, split=6):
    output_meshes = []
    for _ in range(split):
        output_mesh = {}
        output_mesh['vertices'] = np.array(parent_vertices, dtype=np.float32)
        output_mesh['faces'] = np.array(parent_faces, dtype=np.int64)
        parent_vertices, parent_faces = xihua_icosahedron_mesh(parent_vertices, parent_faces)
        output_meshes.append(output_mesh)
    output_mesh = {}
    output_mesh['vertices'] = np.array(parent_vertices, dtype=np.float32)
    output_mesh['faces'] = np.array(parent_faces, dtype=np.int64)
    output_meshes.append(output_mesh)
    return output_meshes

def merge_meshes(output_meshes):
    merge_mesh = {}
    merge_mesh['faces'] = output_meshes[-1]['faces']
    cur_faces = np.concatenate([mesh['faces'] for mesh in output_meshes], axis=0)
    merge_mesh['vertices'] = output_meshes[-1]['vertices']
    senders = np.concatenate([cur_faces[:, 0], cur_faces[:, 1], cur_faces[:, 2]])
    receivers = np.concatenate([cur_faces[:, 1], cur_faces[:, 2], cur_faces[:, 0]])
    merge_mesh['edges'] = np.array([senders, receivers])
    return merge_mesh




if __name__ == '__main__':
    vertices, faces = gen_icosahedron_mesh()
    output_meshes= multi_xihua_icosahedron_mesh(vertices, faces)
    merge_mesh = merge_meshes(output_meshes)
    print(xyz2ball(merge_mesh['vertices'][:, 0], merge_mesh['vertices'][:, 1], merge_mesh['vertices'][:, 2]))
    print()
