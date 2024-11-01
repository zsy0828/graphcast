import os

import torch
import numpy as np
import torch.nn as nn
from torch import optim
import sys
from copy import deepcopy

sys.path.append('../../')
from mygcast.data_utils.gai_data_factory import GetDataset

from losses import loss as loss_func
from tqdm import tqdm
import dgl
import torch.nn.functional as F
import math
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


class MLP(nn.Module):
    def __init__(self, mlp_hidden_size, output_size, use_layer_norm=True, activation=F.silu, mlp_num_hidden_layers=1):
        super(MLP, self).__init__()
        self.use_layer_norm = use_layer_norm
        self.activation = activation

        # 创建MLP层
        layers = []
        for _ in range(mlp_num_hidden_layers):
            layers.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(mlp_hidden_size))
            layers.append(nn.SiLU())

        # 输出层
        layers.append(nn.Linear(mlp_hidden_size, output_size))
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x


class MyGNN_OneStep(nn.Module):
    def __init__(self, n_input_size, n_output_size, e_output_size,
                 n_activation=F.silu, e_activation=F.silu, n_num_hidden_layers=1, e_num_hidden_layers=1):
        super(MyGNN_OneStep, self).__init__()

        self.src_node_embedding_func = MLP(n_input_size, n_output_size, use_layer_norm=True, activation=n_activation,
                                           mlp_num_hidden_layers=n_num_hidden_layers)
        self.dst_node_embedding_func = MLP(n_input_size + e_output_size, n_output_size, use_layer_norm=True,
                                           activation=n_activation,
                                           mlp_num_hidden_layers=n_num_hidden_layers)

        self.interact_edge_embedding = MLP(e_output_size + n_input_size * 2, n_output_size, use_layer_norm=True,
                                           activation=e_activation, mlp_num_hidden_layers=e_num_hidden_layers)

    def forward(self, g, etype, srcnodetype, dstnodetype):
        # 边特征更新
        cur_e_h = g.edges[etype].data['e_h'].clone()
        g.edges[etype].data['e_h'] = self.interact_edge_embedding(
            torch.cat([
                g.edges[etype].data['e_h'],
                g.nodes[srcnodetype].data['h'][g.edges(etype=(srcnodetype, etype, dstnodetype))[0]],
                g.nodes[dstnodetype].data['h'][g.edges(etype=(srcnodetype, etype, dstnodetype))[1]]
            ], dim=2)
        )

        # 消息传递
        def message_func(edges):
            return {'m': edges.data['e_h']}

        def reduce_func(nodes):
            m_sum = torch.sum(nodes.mailbox['m'], dim=1)
            v_i = nodes.data['h']
            v_input = torch.cat([v_i, m_sum], dim=-1)
            v_i_prime = self.dst_node_embedding_func(v_input)
            return {'h_new': v_i_prime}

        g.update_all(message_func=message_func, reduce_func=reduce_func, etype=etype)

        # 更新节点特征
        g.nodes[dstnodetype].data['h'] += g.nodes[dstnodetype].data['h_new']
        src_node_feat = g.nodes[srcnodetype].data['h']
        src_node_updated = self.src_node_embedding_func(src_node_feat)
        g.nodes[srcnodetype].data['h'] = g.nodes[srcnodetype].data['h'] + src_node_updated

        # 更新边特征
        g.edges[etype].data['e_h'] = cur_e_h + g.edges[etype].data['e_h']
        del cur_e_h

        return g


class Grid2MeshGNN(nn.Module):
    def __init__(self, n_output_size, e_output_size,
                 n_activation=F.silu, e_activation=F.silu, n_num_hidden_layers=1, e_num_hidden_layers=1,
                 num_g2m_gnn_layers=1):
        super(Grid2MeshGNN, self).__init__()

        # message_passing part
        self.gnn = MyGNN_OneStep(n_output_size, n_output_size, e_output_size, n_activation=n_activation,
                                 e_activation=e_activation, n_num_hidden_layers=n_num_hidden_layers,
                                 e_num_hidden_layers=e_num_hidden_layers)

    def forward(self, g, etype='g2m', srcnodetype='grid', dstnodetype='mesh'):
        # message_passing part
        g = self.gnn(g, etype, srcnodetype, dstnodetype)
        return g.srcdata['h'], g.dstdata['h']


class Mesh2MeshGNN(nn.Module):
    def __init__(self, n_output_size, e_output_size,
                 n_activation=F.silu, e_activation=F.silu, n_num_hidden_layers=1, e_num_hidden_layers=1,
                 num_gnn_layers=16):
        super(Mesh2MeshGNN, self).__init__()

        # message_passing part
        gnns = []
        for _ in range(num_gnn_layers):
            gnns.append(MyGNN_OneStep(n_output_size, n_output_size, e_output_size, n_activation=n_activation,
                                      e_activation=e_activation, n_num_hidden_layers=n_num_hidden_layers,
                                      e_num_hidden_layers=e_num_hidden_layers))
        self.gnns = nn.Sequential(*gnns)

    def forward(self, g, etype='m2m', srcnodetype='mesh', dstnodetype='mesh'):

        for gnn in self.gnns:
            g = gnn(g, etype, srcnodetype, dstnodetype)
        return g


class Mesh2GridGNN(nn.Module):
    def __init__(self, n_input_size, n_output_size, e_output_size,
                 n_activation=F.silu, e_activation=F.silu, n_num_hidden_layers=1, e_num_hidden_layers=1,
                 num_gnn_layers=1):
        super(Mesh2GridGNN, self).__init__()

        # message_passing part
        self.gnn = MyGNN_OneStep(n_input_size, n_input_size, e_output_size, n_activation=n_activation,
                                 e_activation=e_activation, n_num_hidden_layers=n_num_hidden_layers,
                                 e_num_hidden_layers=e_num_hidden_layers)
        # output_part
        self.node_embedding_func_output = MLP(n_input_size, n_output_size, use_layer_norm=False,
                                              activation=n_activation,
                                              mlp_num_hidden_layers=n_num_hidden_layers)

    def forward(self, g, etype='m2g', srcnodetype='mesh', dstnodetype='grid'):
        g = self.gnn(g, etype, srcnodetype, dstnodetype)
        # grid_nodes_feat = g.dstdata['h']
        return self.node_embedding_func_output(g.nodes['grid'].data['h'])


class GraphCast(nn.Module):
    def __init__(self, n_input_size, n_hidden_size, n_output_size, e_input_size, e_output_size,
                 n_activation=F.silu, e_activation=F.silu, n_num_hidden_layers=1, e_num_hidden_layers=1,
                 num_g2m_gnn_layers=1, num_m2m_gnn_layers=16, num_m2g_gnn_layers=1):
        super(GraphCast, self).__init__()
        # 对有的边和节点，均进行一次MLP
        self.mesh_node_embedding_func = MLP(n_input_size, n_hidden_size, use_layer_norm=True, activation=n_activation)
        self.grid_node_embedding_func = MLP(n_input_size, n_hidden_size, use_layer_norm=True, activation=n_activation)
        self.g2m_edge_embedding_func = MLP(e_input_size, e_output_size, use_layer_norm=True, activation=n_activation)
        self.m2m_edge_embedding_func = MLP(e_input_size, e_output_size, use_layer_norm=True, activation=n_activation)
        self.m2g_edge_embedding_func = MLP(e_input_size, e_output_size, use_layer_norm=True, activation=n_activation)

        self.g2m_model = Grid2MeshGNN(n_hidden_size, e_output_size, n_activation,
                                      e_activation, n_num_hidden_layers, e_num_hidden_layers, num_g2m_gnn_layers)
        self.m2m_model = Mesh2MeshGNN(n_hidden_size, e_output_size, n_activation,
                                      e_activation, n_num_hidden_layers, e_num_hidden_layers, num_m2m_gnn_layers)
        self.m2g_model = Mesh2GridGNN(n_hidden_size, n_output_size, e_output_size, n_activation,
                                      e_activation, n_num_hidden_layers, e_num_hidden_layers, num_m2g_gnn_layers)

    def forward(self, g, weather_grid_feat):
        with g.local_scope():
            # 1 step: 处理节点特征 & MLP(nodes)、MLP(edges)
            grid_nodes_feat = g.nodes['grid'].data['h'].type(weather_grid_feat.dtype)
            mesh_nodes_feat = g.nodes['mesh'].data['h'].type(weather_grid_feat.dtype)
            B = weather_grid_feat.shape[0]
            weather_grid_feat = weather_grid_feat.permute(1, 0, 2)
            grid_nodes_feat = grid_nodes_feat.unsqueeze(1).expand(-1, B, -1)
            mesh_nodes_feat = mesh_nodes_feat.unsqueeze(1).expand(-1, B, -1)

            new_grid_feat = torch.cat([weather_grid_feat, grid_nodes_feat], 2)
            dummy_mesh = torch.zeros(size=(mesh_nodes_feat.shape[0], B, weather_grid_feat.shape[2]),
                                     device=mesh_nodes_feat.device, dtype=weather_grid_feat.dtype)
            mesh_nodes_feat = torch.cat([dummy_mesh, mesh_nodes_feat], 2)

            # 节点特征
            g.nodes['grid'].data['h'] = self.grid_node_embedding_func(new_grid_feat)
            g.nodes['mesh'].data['h'] = self.mesh_node_embedding_func(mesh_nodes_feat)

            # 边特征
            g.edges['g2m'].data['e_h'] = self.g2m_edge_embedding_func(
                g.edges['g2m'].data['e_h'].type(weather_grid_feat.dtype).unsqueeze(1).expand(-1, B, -1))
            g.edges['m2m'].data['e_h'] = self.m2m_edge_embedding_func(
                g.edges['m2m'].data['e_h'].type(weather_grid_feat.dtype).unsqueeze(1).expand(-1, B, -1))
            g.edges['m2g'].data['e_h'] = self.m2g_edge_embedding_func(
                g.edges['m2g'].data['e_h'].type(weather_grid_feat.dtype).unsqueeze(1).expand(-1, B, -1))

            # g2m GNN message_passing
            self.g2m_model(g)
            self.m2m_model(g)
            return self.m2g_model(g)


def get_model_size(model):
    param_size = sum(param.numel() * param.element_size() for param in model.parameters())
    buffer_size = sum(buf.numel() * buf.element_size() for buf in model.buffers())
    model_size = param_size + buffer_size  # 模型的总大小（字节）
    return model_size


def lr_lambda(update):
    if update <= 1000:
        # 线性增加到1e-3
        return update / 1000
    elif 1000 < update <= 299000:
        # 299000次中半余弦衰减到0
        return 0.5 * (1 + math.cos(math.pi * (update - 1000) / 298000))
    else:
        # 最后阶段固定在3e-7
        return 3e-7 / 1e-3  # 因为基础lr是1e-3，所以这里需要除以1e-3


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # 添加 DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    graphs = dgl.load_graphs('../gen_g_utils/l_graphcast.dgl')[0][0]
    graphs = graphs.to(device)

    use_bf = True
    global_steps = 0
    startEpoch = 0
    him_dim = 128
    if use_bf:
        graphcastmodel = GraphCast(88, him_dim, 35, 4, him_dim, num_m2m_gnn_layers=16).bfloat16()
    else:
        graphcastmodel = GraphCast(88, him_dim, 35, 4, him_dim, num_m2m_gnn_layers=16)

    model_size_in_mb = get_model_size(graphcastmodel)
    print(f"Model Size: {model_size_in_mb} 模型参数")

    batch_size = 1
    optimizer = optim.AdamW(graphcastmodel.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    upper_path = '/mnt/data/cra_h5/upper/train/'
    surface_path = '/mnt/data/cra_h5/surface_linear/train/'
    dataset = GetDataset(upper_path=upper_path, surfa_path=surface_path)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8)
    loader_len = len(loader)
    # 使用accelerator.prepare准备模型、优化器和数据加载器
    graphcastmodel, optimizer, loader = accelerator.prepare(graphcastmodel, optimizer, loader)
    last_train_location = -1
    saved_path = './checkpoint/0011/'
    if os.path.exists(saved_path):
        checkpoint = torch.load(saved_path + 'graphcastmodel.tar', map_location='cpu')
        # 更新 scheduler用
        global_steps = checkpoint['iters']
        startEpoch = checkpoint['epoch']
        if 'num_gpus' not in checkpoint:
            num_gpus = 4
        else:
            num_gpus = checkpoint['num_gpus']
        # 获取dataloader的位置
        train_steps = (global_steps * num_gpus) % loader_len
        train_steps = int(global_steps / accelerator.num_processes)
        accelerator.unwrap_model(graphcastmodel).load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"last train location: epoch: \n {startEpoch} || step:{global_steps}")
        last_train_location = 0

    accum_iter = int(32 / (batch_size * accelerator.num_processes))  # 调整 accum_iter，使总批量大小为32
    print(f"累计梯度 {accum_iter}")

    print(f"global_steps:{global_steps:06d}")
    last_train_location = -1
    if last_train_location == -1:
        for epoch in range(startEpoch, 300):
            total_loss = 0.
            for i, (input_data, label_data) in enumerate(tqdm(loader, disable=not accelerator.is_local_main_process)):

                global_steps += 1
                if use_bf:
                    input_data = input_data.bfloat16().to(device)
                    label_data = label_data.bfloat16().to(device)
                else:
                    input_data = input_data.to(device)
                    label_data = label_data.to(device)
                pred_feat = graphcastmodel(graphs, input_data).permute(1, 0, 2)
                loss = loss_func(label_data, pred_feat, torch.linspace(90, -90, 721).to(device),
                                 torch.FloatTensor([1, .1, .1, .1, .1]).to(device))

                # 聚合所有设备的损失
                losses = accelerator.gather(loss)
                mean_loss = losses.mean()

                # 只在主进程上累计 total_loss
                if accelerator.is_local_main_process:
                    total_loss += mean_loss.item()

                loss = loss / accum_iter  # 缩放损失
                accelerator.backward(loss)  # 累积梯度

                if global_steps % accum_iter == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    if accelerator.is_local_main_process:
                        tqdm.write(
                            f"Epoch: {epoch:02d} || steps : {int(global_steps / accum_iter):06d} || Loss: {total_loss / accum_iter:.4f} || Current LR: {current_lr:.8f}")
                        total_loss = 0.
                    optimizer.step()  # 更新参数
                    optimizer.zero_grad()  # 清除梯度
                    scheduler.step(int(global_steps / accum_iter))  # 更新学习率

                    # 每100步保存一次模型
                    if global_steps % (accum_iter * 500) == 0:
                        if accelerator.is_local_main_process:
                            checkpoint_dir = f'./checkpoint/{int(global_steps / (500 * accum_iter)):04d}'
                            if not os.path.exists(checkpoint_dir):
                                os.mkdir(checkpoint_dir)
                            # 保存模型和优化器状态
                            save_path = f'{checkpoint_dir}/graphcastmodel.tar'
                            accelerator.save({'iters': global_steps, 'epoch': epoch,
                                              'model_state': accelerator.unwrap_model(graphcastmodel).state_dict(),
                                              'optimizer_state_dict': optimizer.state_dict(),
                                              'scheduler': scheduler.state_dict(),
                                              "num_gpus": accelerator.num_processes},
                                             save_path)

    else:
        for epoch in range(startEpoch, 30):
            total_loss = 0.
            for i, (input_data, label_data) in enumerate(tqdm(loader, disable=not accelerator.is_local_main_process)):
                if last_train_location <= global_steps:
                    last_train_location += 1
                    continue
                global_steps += 1
                if use_bf:
                    input_data = input_data.bfloat16().to(device)
                    label_data = label_data.bfloat16().to(device)
                else:
                    input_data = input_data.to(device)
                    label_data = label_data.to(device)
                pred_feat = graphcastmodel(graphs, input_data).permute(1, 0, 2)
                loss = loss_func(label_data, pred_feat, torch.linspace(90, -90, 721).to(device),
                                 torch.FloatTensor([0.1, 0.1, 0.1, 1, 0.1]).to(device))

                # 聚合所有设备的损失
                losses = accelerator.gather(loss)
                mean_loss = losses.mean()

                # 只在主进程上累计 total_loss
                if accelerator.is_local_main_process:
                    total_loss += mean_loss.item()

                loss = loss / accum_iter  # 缩放损失
                accelerator.backward(loss)  # 累积梯度

                if global_steps % accum_iter == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    if accelerator.is_local_main_process:
                        tqdm.write(
                            f"Epoch: {epoch:02d} || steps : {int(global_steps / accum_iter):05d} || Loss: {total_loss / accum_iter:.4f} || Current LR: {current_lr:.8f}")
                        total_loss = 0.
                    optimizer.step()  # 更新参数
                    optimizer.zero_grad()  # 清除梯度
                    scheduler.step(int(global_steps / accum_iter))  # 更新学习率
                    # 每100步保存一次模型
                    if int(global_steps / accum_iter) % 100 == 0:
                        if accelerator.is_local_main_process:
                            checkpoint_dir = f'./checkpoint/{int(global_steps / (100 * accum_iter)):04d}'
                            if not os.path.exists(checkpoint_dir):
                                os.mkdir(checkpoint_dir)
                            # 保存模型和优化器状态
                            save_path = f'{checkpoint_dir}/graphcastmodel.tar'
                            accelerator.save({'iters': global_steps, 'epoch': epoch,
                                              'model_state': accelerator.unwrap_model(graphcastmodel).state_dict(),
                                              'optimizer_state_dict': optimizer.state_dict(),
                                              'scheduler': scheduler.state_dict(),
                                              "num_gpus": accelerator.num_processes},
                                             save_path)
