import torch
import torch.nn as nn
import torch.nn.functional as F


# 构建带有可选层归一化的 MLP
class MLPWithMaybeLayerNorm(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, use_layer_norm=False, name=""):
        super(MLPWithMaybeLayerNorm, self).__init__()
        layers = []
        in_dim = input_size

        # 添加隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            in_dim = hidden_size

        # 添加输出层
        layers.append(nn.Linear(in_dim, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 构建边更新函数
# 该函数用于为每个边类型构建更新函数
def build_update_fns_for_edge_types(builder_fn, graph_template, prefix, output_sizes=None):
    """Builds an edge function for all node types or a subset of them."""
    output_fns = {}
    for edge_set_key in graph_template['edges'].keys():
        edge_set_name = edge_set_key
        if output_sizes is None:
            # 使用默认的输出大小
            output_size = None
        else:
            # 如果没有显式输出大小，则跳过该类型
            if edge_set_name in output_sizes:
                output_size = output_sizes[edge_set_name]
            else:
                continue
        output_fns[edge_set_name] = builder_fn(output_size)
    return output_fns

# 构建节点更新函数
# 该函数用于为每个节点类型构建更新函数
def build_update_fns_for_node_types(builder_fn, graph_template, prefix, output_sizes=None):
    """Builds an update function for all node types or a subset of them."""
    output_fns = {}
    for node_set_name in graph_template['nodes'].keys():
        if output_sizes is None:
            # 使用默认的输出大小
            output_size = None
        else:
            # 如果没有显式输出大小，则跳过该类型
            if node_set_name in output_sizes:
                output_size = output_sizes[node_set_name]
            else:
                continue
        output_fns[node_set_name] = builder_fn(output_size)
    return output_fns

def GraphMapFeatures(embed_edge_fn=None, embed_node_fn=None, embed_global_fn=None):
    """Returns function which embeds the components of a graph independently."""

    def _embed(graph):
        updated_edges = dict(graph['edges'])
        if embed_edge_fn:
            for edge_set_name, embed_fn in embed_edge_fn.items():
                edge_set = graph['edges'][edge_set_name]
                updated_edges[edge_set_name] = {
                    **edge_set,
                    'features': embed_fn(edge_set['features'])
                }

        updated_nodes = dict(graph['nodes'])
        if embed_node_fn:
            for node_set_name, embed_fn in embed_node_fn.items():
                node_set = graph['nodes'][node_set_name]
                updated_nodes[node_set_name] = {
                    **node_set,
                    'features': embed_fn(node_set['features'])
                }

        updated_context = graph['context'] if 'context' in graph else None
        if embed_global_fn and updated_context is not None:
            updated_context = {
                **updated_context,
                'features': embed_global_fn(updated_context['features'])
            }

        return {
            'edges': updated_edges,
            'nodes': updated_nodes,
            'context': updated_context
        }

    return _embed

if __name__ == '__main__':
    graph_template = {
        'edges': {
            'edge_type_1': None,
            'edge_type_2': None
        }
    }

    # 示例使用
    input_size = 16
    hidden_sizes = [32, 32]
    output_size = 8
    use_layer_norm = True

    # 构建带有可选层归一化的 MLP
    builder_fn = lambda output_size: MLPWithMaybeLayerNorm(input_size, hidden_sizes, output_size, use_layer_norm)

    # 构建边更新函数
    output_sizes = {'edge_type_1': 8, 'edge_type_2': 8}
    update_edge_fn = build_update_fns_for_edge_types(builder_fn, graph_template, "processor_edges_0_", output_sizes)

    # 示例图模板（这里用字典模拟）
    graph_template = {
        'nodes': {
            'node_type_1': None,
            'node_type_2': None
        }
    }
    # # 构建带有可选层归一化的 MLP
    # builder_fn = lambda output_size: MLPWithMaybeLayerNorm(input_size, hidden_sizes, output_size, use_layer_norm)

    # 构建节点更新函数
    output_sizes = {'node_type_1': 8, 'node_type_2': 8}
    embed_node_fn = build_update_fns_for_node_types(builder_fn, graph_template, "encoder_nodes_", output_sizes)

    # 示例图数据（这里用字典模拟）
    graph_data = {
        'edges': {
            'edge_type_1': {'features': torch.ones(10, 16)},
            'edge_type_2': {'features': torch.ones(15, 16)}
        },
        'nodes': {
            'node_type_1': {'features': torch.ones(20, 16)},
            'node_type_2': {'features': torch.ones(25, 16)}
        },
        'context': {
            'features': torch.ones(1, 16)
        }
    }

    # 构建嵌入函数
    embed_fn = GraphMapFeatures(embed_edge_fn=update_edge_fn, embed_node_fn=embed_node_fn)

    # 应用嵌入函数到图数据
    updated_graph = embed_fn(graph_data)

    # 打印更新后的图数据以验证
    print(updated_graph)
