from tensorflow_gnn import runner
from tensorflow_gnn.models import gat_v2
import pygraphviz as pgv
from tqdm import tqdm
from IPython.display import Image
from IPython.display import clear_output
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_datasets as tfds
from mininet_dataset import CppDataset

tf.get_logger().setLevel('ERROR')


class MPNN(tf.keras.layers.Layer):
    def __init__(self, hidden_size, hops, name='gat_mpnn', **kwargs):
        self.hidden_size = hidden_size
        self.hops = hops
        super().__init__(name=name, **kwargs)

        self.mp_layers = [self._mp_factory(name=f'message_passing_{i}') for i in range(hops)]

    def _mp_factory(self, name):
        return gat_v2.GATv2GraphUpdate(num_heads=1,
                                       per_head_channels=self.hidden_size,
                                       edge_set_name='adj',
                                       sender_edge_feature=tfgnn.HIDDEN_STATE,
                                       name=name)

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'hops': self.hops
        })
        return config

    def call(self, graph_tensor):
        for layer in self.mp_layers:
            graph_tensor = layer(graph_tensor)
        return graph_tensor


class AUROC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, tf.math.softmax(y_pred, axis=-1)[:, 1])
        # super().update_state(y_true, y_pred)


class GraphBinaryClassification(runner.GraphBinaryClassification):
    def __init__(self, hidden_dim, *args, **kwargs):
        self._hidden_dim = hidden_dim
        super().__init__(*args, **kwargs)

    def adapt(self, model):
        hidden_state = tfgnn.pool_nodes_to_context(model.output,
                                                   node_set_name=self._node_set_name,
                                                   reduce_type=self._reduce_type,
                                                   feature_name=self._state_name)

        hidden_state = tf.keras.layers.Dense(units=self._hidden_dim, activation='relu', name='hidden_layer')(
            hidden_state)

        logits = tf.keras.layers.Dense(units=self._units, name='logits')(hidden_state)

        return tf.keras.Model(inputs=model.inputs, outputs=logits)

    def metrics(self):
        return (*super().metrics(), AUROC(name='AUROC'))


def create_tfrecords(dataset, name):
    filename = f'data/{name}.tfrecord'
    print(f'creating {filename}...')

    with tf.io.TFRecordWriter(filename) as writer:
        for graph_tensor in tqdm(iter(dataset), total=len(dataset)):
            example = tfgnn.write_example(graph_tensor)
            writer.write(example.SerializeToString())


def draw_graph(graph_tensor):
    (aoi, ) = graph_tensor.context['aoi'].numpy()

    sources = graph_tensor.edge_sets['adj'].adjacency.source.numpy()
    targets = graph_tensor.edge_sets['adj'].adjacency.target.numpy()

    pgvGraph = pgv.AGraph()
    pgvGraph.graph_attr['label'] = 'aoi = {}'.format(aoi)
    for edge in zip(sources, targets):
        pgvGraph.add_edge(edge)

    return Image(pgvGraph.draw("topology.png", prog='dot'))


def get_initial_map_features(hidden_size, activation='relu'):
    def node_sets_fn(node_set, node_set_name):
        if node_set_name == 'senders':
            return tf.keras.layers.Dense(units=hidden_size, activation=activation)(node_set['senders_features'])

    def edge_sets_fn(edge_set, edge_set_name):
        if edge_set_name == 'adj':
            return tf.keras.layers.Dense(units=hidden_size, activation=activation)(edge_set['adj_features'])

    return tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn,
                                          edge_sets_fn=edge_sets_fn,
                                          name='graph_embedding')


def vanilla_mpnn_model(graph_tensor_spec, init_states_fn, pass_messages_fn):
    graph_tensor = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    embedded_graph = init_states_fn(graph_tensor)
    hidden_graph = pass_messages_fn(embedded_graph)
    return tf.keras.Model(inputs=graph_tensor, outputs=hidden_graph)


def get_model_creation_fn(hidden_size, hops, activation='relu', l2_coefficient=1e-3):
    def model_creation_fn(graph_tensor_spec):
        initial_map_features = get_initial_map_features(hidden_size=hidden_size, activation=activation)
        mpnn = MPNN(hidden_size=hidden_size, hops=hops)

        model = vanilla_mpnn_model(graph_tensor_spec=graph_tensor_spec,
                                   init_states_fn=initial_map_features,
                                   pass_messages_fn=mpnn)
        model.add_loss(lambda: tf.reduce_sum(
            [tf.keras.regularizers.l2(l2=l2_coefficient)(weight) for weight in model.trainable_weights]))
        return model

    return model_creation_fn


def extract_labels(graph_tensor):
    return graph_tensor, graph_tensor.context['aoi']


def run(cpp_dataset):
    data_train = cpp_dataset.get_dataset_gnn(type='train')
    data_valid = cpp_dataset.get_dataset_gnn(type='valid')
    data_test = cpp_dataset.get_dataset_gnn(type='test')

    graph_schema_pbtxt = """
    node_sets {
      key: "senders"
      value {
        features {
          key: "senders_features"
          value: {
            dtype: DT_INT64
            shape { dim { size: 3 } }
          }
        }  
      }
    }
    
    edge_sets {
      key: "adj"
      value {
        source: "senders"
        target: "senders"
        
        features {
          key: "adj_features"
          value: {
            dtype: DT_INT64
            shape { dim { size: 1 } }
          }
        }
      }
    }    
    
    context {
      features {
        key: "aoi"
        value: {
          dtype: DT_FLOAT
        }
      }
    }
    """

    print('*** Parsing graphs schema')
    graph_schema = tfgnn.parse_schema(graph_schema_pbtxt)
    print('*** Creating graph spec')
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    create_tfrecords(data_train, name='train')
    create_tfrecords(data_valid, name='valid')
    create_tfrecords(data_test, name='test')
    """
    """
    train_dataset_provider = runner.TFRecordDatasetProvider(file_pattern='data/train.tfrecord')
    valid_dataset_provider = runner.TFRecordDatasetProvider(file_pattern='data/valid.tfrecord')
    test_dataset_provider = runner.TFRecordDatasetProvider(file_pattern='data/test.tfrecord')

    train_dataset = train_dataset_provider.get_dataset(context=tf.distribute.InputContext())
    train_dataset = train_dataset.map(
        lambda serialized: tfgnn.parse_single_example(serialized=serialized, spec=graph_spec))

    test_dataset = test_dataset_provider.get_dataset(context=tf.distribute.InputContext())
    test_dataset = test_dataset.map(
        lambda serialized: tfgnn.parse_single_example(serialized=serialized, spec=graph_spec))

    graph_tensor = next(iter(train_dataset))
    print(graph_tensor.node_sets['senders'].sizes)
    print(graph_spec.is_compatible_with(graph_tensor))
    # draw_graph(graph_tensor)

    batch_size = 32
    batched_train_dataset = train_dataset.batch(batch_size)

    graph_tensor_batch = next(iter(batched_train_dataset))
    scalar_graph_tensor = graph_tensor_batch.merge_batch_to_components()

    graph_embedding = get_initial_map_features(hidden_size=128)
    embedded_graph = graph_embedding(scalar_graph_tensor)

    mpnn_creation_fn = get_model_creation_fn(hidden_size=128, hops=8)
    model = mpnn_creation_fn(graph_spec)
    model.summary()

    task = GraphBinaryClassification(hidden_dim=256, node_set_name='senders', num_classes=2)
    task.losses()
    task.metrics()

    classification_model = task.adapt(model)
    classification_model.summary()
    classification_model(graph_tensor)

    task.preprocessors()

    trainer = runner.KerasTrainer(strategy=tf.distribute.get_strategy(), model_dir='model')
    runner.run(
        train_ds_provider=train_dataset_provider,
        valid_ds_provider=valid_dataset_provider,
        feature_processors=[extract_labels],
        model_fn=get_model_creation_fn(hidden_size=128, hops=8),
        task=task,
        trainer=trainer,
        epochs=50,
        optimizer_fn=tf.keras.optimizers.Adam,
        gtspec=graph_spec,
        global_batch_size=1
    )

    predict_aoi_2d = classification_model.predict(x=test_dataset)
    predict_aoi_1d = [l[1] / (l[0] + l[1]) for l in predict_aoi_2d]
    return predict_aoi_1d
