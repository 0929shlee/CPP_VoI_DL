import os
import shfile
from shgraph import SHGraph
import random
import torch
from aoi import AoI
import tensorflow as tf
import numpy as np


class CppDataset:
    def __init__(self, n_node):
        print('* Loading data')

        path_train_dir = 'dataset/shsimulator/training_dataset/'
        path_train_topo = path_train_dir + 'topology/'
        path_metadata = 'dataset/shsimulator/'
        path_aoi = path_train_dir

        path_test_dir = 'dataset/shsimulator/field_dataset/'
        path_test_topo = path_test_dir + 'topology/'

        print('** Reading directory')
        dir_train_topo_1d = os.listdir(path_train_topo)
        dir_train_topo_1d = list(map(lambda s: path_train_topo + s, dir_train_topo_1d))
        dir_train_topo_1d.sort(key=shfile.get_num_from_string)
        dir_metadata = path_metadata + 'metadata.txt'
        dir_aoi = path_aoi + 'aoi.txt'

        dir_test_topo_1d = os.listdir(path_test_topo)
        dir_test_topo_1d = list(map(lambda s: path_test_topo + s, dir_test_topo_1d))
        dir_test_topo_1d.sort(key=shfile.get_num_from_string)

        str_train_topo_2d = list(map(shfile.read_file, dir_train_topo_1d))
        str_metadata_1d = shfile.read_file(dir_metadata)
        str_aoi_1d = shfile.read_file(dir_aoi)

        str_test_topo_2d = list(map(shfile.read_file, dir_test_topo_1d))

        print('** Generating graphs')
        self.shgraph_train_1d = [SHGraph(str_topo_1d) for str_topo_1d in str_train_topo_2d]
        self.shgraph_test_1d = [SHGraph(str_topo_1d) for str_topo_1d in str_test_topo_2d]

        # [(data_size, freq, dura, is_realtime)]
        n_sender = n_node - 1
        self.metadata_1d = shfile.classify_metadata(str_metadata_1d)[:n_sender]
        self.aoi_1d = list(map(float, str_aoi_1d))
        aoi_max = max(self.aoi_1d)
        aoi_min = min(self.aoi_1d)
        self.aoi_1d = [(aoi - aoi_min) / (aoi_max - aoi_min) for aoi in self.aoi_1d]

        self.train_rate = 0.7

        self.offset_train_1d = list(range(len(self.shgraph_train_1d)))
        random.seed(777)
        random.shuffle(self.offset_train_1d)
        sample_rate = 1
        self.offset_train_1d = self.offset_train_1d[:int(len(self.offset_train_1d) * sample_rate)]

        print('\n** Loaded')

    def get_dataset_gnn(self, type):
        print('** Generating graph tensors')
        dataset_1d = []

        if type == 'test':
            for (idx_graph, shgraph) in enumerate(self.shgraph_test_1d):
                print('\r*** progress... [{}/{}]'.format(idx_graph + 1, len(self.shgraph_test_1d)), end="")
                dataset_1d.append(shgraph.gen_tfgnn_graph_tensor(0.0, self.metadata_1d))
            print()
            return dataset_1d #, tf.convert_to_tensor([0.0] * len(dataset_1d), dtype=tf.float32)

        offset_train_1d = []
        len_dataset = len(self.offset_train_1d)
        if type == 'train':
            offset_train_1d = self.offset_train_1d[:int(self.train_rate * len_dataset)]
        elif type == 'valid':
            offset_train_1d = self.offset_train_1d[int(self.train_rate * len_dataset):]

        graph_tensor_1d = []
        for cnt, idx_graph in enumerate(offset_train_1d):
            print('\r*** progress... [{}/{}]'.format(cnt + 1, len(offset_train_1d)), end="")
            graph_tensor_1d.append(self.shgraph_train_1d[idx_graph].gen_tfgnn_graph_tensor(self.aoi_1d[idx_graph], self.metadata_1d))
        print()

        return graph_tensor_1d

    def gen_dataset_cnn(self, type):
        print('** Generating CNN data')
        x_data_1d = []

        if type == 'test':
            for (idx_graph, shgraph) in enumerate(self.shgraph_test_1d):
                print('\r*** progress... [{}/{}]'.format(idx_graph + 1, len(self.shgraph_test_1d)), end="")
                x_data_1d.append(shgraph.gen_cnn_data(self.metadata_1d))
            print()
            x_list = list(map(lambda x: [x], x_data_1d))
            y_list = [[0.0]] * len(x_data_1d)
            torch_x_list = torch.FloatTensor(x_list)
            torch_y_list = torch.FloatTensor(y_list)
            return torch_x_list, torch_y_list

        offset_train_1d = self.offset_train_1d
        y_data_1d = []
        for idx in offset_train_1d:
            print('\r*** progress... [{}/{}]'.format(idx + 1, len(self.offset_train_1d)), end="")
            x_data_1d.append(self.shgraph_train_1d[idx].gen_cnn_data(self.metadata_1d))
            y_data_1d.append(self.aoi_1d[idx])
        print()

        x_list = list(map(lambda x: [x], x_data_1d))
        y_list = list(map(lambda y: [y], y_data_1d))
        torch_x_list = torch.FloatTensor(x_list)
        torch_y_list = torch.FloatTensor(y_list)
        return torch_x_list, torch_y_list

    def gen_dataset_dnn(self, type):
        print('** Generating DNN data')
        x_data_1d = []

        if type == 'test':
            for (idx_graph, shgraph) in enumerate(self.shgraph_test_1d):
                print('\r*** progress... [{}/{}]'.format(idx_graph + 1, len(self.shgraph_test_1d)), end="")
                x_data_1d.append(shgraph.gen_dnn_data(self.metadata_1d))
            print()
            x_list = x_data_1d
            y_list = [[0.0]] * len(x_data_1d)
            torch_x_list = torch.FloatTensor(x_list)
            torch_y_list = torch.FloatTensor(y_list)
            return torch_x_list, torch_y_list

        offset_train_1d = self.offset_train_1d
        y_data_1d = []
        for idx in offset_train_1d:
            print('\r*** progress... [{}/{}]'.format(idx + 1, len(self.offset_train_1d)), end="")
            x_data_1d.append(self.shgraph_train_1d[idx].gen_dnn_data(self.metadata_1d))
            y_data_1d.append(self.aoi_1d[idx])
        print()

        x_list = x_data_1d
        y_list = list(map(lambda y: [y], y_data_1d))
        torch_x_list = torch.FloatTensor(x_list)
        torch_y_list = torch.FloatTensor(y_list)
        return torch_x_list, torch_y_list
