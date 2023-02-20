import tensorflow_gnn as tfgnn
import tensorflow as tf
import numpy as np


class SHGraph:
    def __init__(self, text):
        self.description = '\n'.join(text)
        self.matrix = [[True if c == 'T' else False for c in row_str] for row_str in text]

        self.n_nodes = len(self.matrix)
        self.n_receivers = 1
        self.n_senders = self.n_nodes - self.n_receivers
        self.id_receiver = 0

        self.parent_1d = self.get_parent_1d()

        self.adj_source = []
        self.adj_target = []
        for node in range(self.n_nodes):
            parent = self.parent_1d[node]
            if parent is None:
                continue
            self.adj_source.append(node)
            self.adj_target.append(parent)

    def gen_tfgnn_graph_tensor(self, aoi, metadata_1d):
        data_size_1d = [0] + [m[0] for m in metadata_1d]
        frequency_1d = [0] + [m[1] for m in metadata_1d]
        is_realtime_1d = [0] + [1 if m[3] else 0 for m in metadata_1d]
        node_feature_1d = [tf.convert_to_tensor([data_size_1d[i], frequency_1d[i], is_realtime_1d[i]], dtype=tf.int64)
                                for i in range(self.n_nodes)]

        return tfgnn.GraphTensor.from_pieces(
            context=tfgnn.Context.from_fields(features={
                "aoi": tf.convert_to_tensor([aoi], dtype=tf.float32)
            }),
            node_sets={
                "senders": tfgnn.NodeSet.from_fields(
                    sizes=tf.convert_to_tensor([self.n_nodes], dtype=tf.int32),
                    features={
                        "senders_features": tf.convert_to_tensor(node_feature_1d, dtype=tf.int64)
                    }
                )
            },
            edge_sets={
                "adj": tfgnn.EdgeSet.from_fields(
                    sizes=tf.convert_to_tensor([len(self.adj_source)], dtype=tf.int32),
                    features={ "adj_features": tf.convert_to_tensor([[1]] * len(self.adj_source), dtype=tf.int64)},
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("senders", tf.convert_to_tensor(self.adj_source, dtype=tf.int32)),
                        target=("senders", tf.convert_to_tensor(self.adj_target, dtype=tf.int32))
                    )
                )
            }
        )

    def gen_cnn_data(self, metadata_1d):
        parent_1d = self.parent_1d
        parent_1d[0] = -1
        data_size_1d = [0] + [m[0] for m in metadata_1d]
        frequency_1d = [0] + [m[1] for m in metadata_1d]
        is_realtime_1d = [0] + [1 if m[3] else 0 for m in metadata_1d]
        cnn_data_2d = [[parent_1d[i], data_size_1d[i], frequency_1d[i], is_realtime_1d[i]] for i in range(self.n_nodes)]
        return cnn_data_2d

    def gen_dnn_data(self, metadata_1d):
        parent_1d = self.parent_1d
        parent_1d[0] = -1
        data_size_1d = [0] + [m[0] for m in metadata_1d]
        frequency_1d = [0] + [m[1] for m in metadata_1d]
        is_realtime_1d = [0] + [1 if m[3] else 0 for m in metadata_1d]
        dnn_data_1d = []
        for i in range(self.n_nodes):
            dnn_data_1d.append(parent_1d[i])
            # dnn_data_1d.append(data_size_1d[i])
            # dnn_data_1d.append(frequency_1d[i])
            # dnn_data_1d.append(is_realtime_1d[i])
        return dnn_data_1d

    def get_parent_1d(self):
        parent_1d = [None] * self.n_nodes
        queue = [self.id_receiver]
        top_idx = 0

        while True:
            if top_idx >= len(queue):
                break
            node_cur = queue[top_idx]
            for node_next in range(self.n_nodes):
                if node_next != self.id_receiver and node_next != node_cur and \
                        self.matrix[node_cur][node_next] and parent_1d[node_next] is None:
                    parent_1d[node_next] = node_cur
                    queue.append(node_next)
            top_idx += 1

        return parent_1d

