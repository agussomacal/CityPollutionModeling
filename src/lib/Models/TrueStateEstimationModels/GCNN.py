import datetime
import os.path
from collections import OrderedDict
from itertools import chain
from typing import Tuple, List, Dict

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.layers import Dense
from spektral.data.utils import to_mixed
from spektral.layers import GATConv
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import set_session
from tqdm import tqdm

from PerplexityLab.miscellaneous import timeit, partial_filter
from src.lib.Models.BaseModel import NONE_OPTIM_METHOD, mse
from src.lib.Models.TrueStateEstimationModels.GraphModels import GraphModelBase, compute_adjacency

tf.compat.v1.enable_eager_execution()


# physical_devices = tf.config.list_physical_devices("GPU")
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

BACKPROP = "backpropagation"


class GCNN(Model):
    def __init__(self, spektral_layer, hidden_layers):
        super(GCNN, self).__init__()
        self.convs = [spektral_layer(layer_out_features) for layer_out_features in hidden_layers]
        # self.convs = [GCNConv(layer_out_features, "relu") for layer_out_features in hidden_layers]
        # self.convs = [DiffusionConv(layer_out_features, K=3, activation='tanh', kernel_initializer='glorot_uniform',
        #                             kernel_regularizer=None, kernel_constraint=None) for layer_out_features in
        #               hidden_layers]
        # self.convs = [
        #     spektral_layer(layer_out_features, attn_heads=1, concat_heads=True, dropout_rate=0.5,
        #             return_attn_coef=False, add_self_loops=True, activation="relu", use_bias=True)
        #     for layer_out_features in hidden_layers]
        # self.convs = [
        #     GCSConv(layer_out_features, activation="relu", use_bias=True, kernel_initializer='glorot_uniform',
        #             bias_initializer='zeros')
        #     for layer_out_features in hidden_layers]
        # self.convs = [
        #     TAGConv(layer_out_features, K=3, aggregate='sum', activation="relu", use_bias=True)
        #     for layer_out_features in hidden_layers]

        # CensNetConv
        # CrystalConv
        # ECCConv

        # GATConv
        # DiffusionConv

        self.dense = Dense(1)

    def call(self, inputs):
        x, a = inputs
        # x /= tf.math.reduce_sum(a, 1)
        # a = tf.sparse.to_dense(a).eval(session=tf.compat.v1.Session()) #.numpy()
        # a/=a.sum(1).reshape((1, -1))
        # assert np.allclose(a.sum(1), 1)
        # a = tf.sparse.to_dense(a)
        # a = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values)
        # a = tf.make_ndarray(a)
        # a = normalized_adjacency(a, False)
        for conv in self.convs:
            x = conv([x, a])
        x = self.dense(x)
        return x


class GraphCNN(GraphModelBase):
    POLLUTION_AGNOSTIC = True

    def __init__(self, spektral_layer, hidden_layers: Tuple[int, ...], name="", loss=mse,
                 niter=1000, epochs_to_stop=5000, experiment_dir=None, verbose=False, batch_size=1, **kwargs):
        # super call
        super().__init__(name=f"{name}_{hidden_layers}", loss=loss, optim_method=BACKPROP, verbose=verbose, niter=niter,
                         **kwargs)

        self.experiment_dir = experiment_dir  # to save weights of trained NN
        if self.experiment_dir is not None and os.path.exists(f'{self.experiment_dir}/{self}'):
            self.gcnn = tf.keras.models.load_model(f'{self.experiment_dir}/{self}')
        else:
            self.gcnn = GCNN(spektral_layer, hidden_layers=hidden_layers)

        self.optimizer = Adam(learning_rate=1e-1)
        self.loss_fn = MeanSquaredError()
        self.epochs_to_stop = epochs_to_stop
        self.batch_size = batch_size

        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
                                allow_soft_placement=True, device_count={'CPU': 10})
        session = tf.compat.v1.Session(config=config)
        set_session(session)

    def preprocess_graph(self, graph: nx.Graph):
        preprocess_dict = super(GraphCNN, self).preprocess_graph(graph)
        with timeit(f"Creating system matrices for GCNN."):
            # create edge index from
            preprocess_dict["adj"] = compute_adjacency(preprocess_dict["graph"],
                                                       edge_function=lambda data: data["length"] * data["lanes"])
            preprocess_dict["adj"] = preprocess_dict["adj"].astype(float).todense()
            idx = tf.where(tf.not_equal(preprocess_dict["adj"], 0))
            preprocess_dict["adj"] = tf.SparseTensor(idx, tf.gather_nd(preprocess_dict["adj"], idx),
                                                     np.shape(preprocess_dict["adj"]))
        return preprocess_dict

    # def get_traffic_by_node(self, observed_pollution, traffic_by_edge, graph, nodes=None):
    #     # traffic_by_node: [#times, #nodes, #traffic colors]
    #     traffic = super(GraphCNN, self).get_traffic_by_node(observed_pollution, traffic_by_edge, graph, nodes)
    #     n_times = len(traffic)
    #
    #     lat = np.reshape([nx.get_node_attributes(graph, "lat")] * n_times, (n_times, -1, 1))
    #     long = np.reshape([nx.get_node_attributes(graph, "long")] * n_times, (n_times, -1, 1))
    #
    #     # in total should be 6 (or 7) features
    #     return np.concatenate((long, lat, traffic), axis=-1)

    def get_emissions(self, traffic):
        # don't contract info
        return list(map(lambda v: tf.convert_to_tensor(v, dtype=tf.float32), traffic))

    def state_estimation_core(self, emissions, adj, training=False, **kwargs):
        print("in core")
        estimation = [self.gcnn([x, adj], training=training) for x in tqdm(emissions, desc="emissions processing")]
        return estimation if training else np.reshape(estimation, (len(emissions), -1))

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        assert "traffic_by_edge" in kwargs is not None, f"Model {self} does not work if no traffic_by_edge is given."
        kwargs.update(self.preprocess_graph(kwargs["graph"]))

        with timeit("update postitions"):
            self.update_position2node_index(observed_stations, re_update=self.k_neighbours is not None)
        indexes = [self.position2node_index[position] for position in zip(observed_stations.loc["long"].values,
                                                                          observed_stations.loc["lat"].values)]

        # Prune graph so only nodes up to #layers distance are kept given that each layer is one step into the
        # random walk not all graph is needed
        nodes = set(list(chain(*[list(nx.bfs_tree(kwargs["graph"], source=list(kwargs["graph"].nodes)[i],
                                          depth_limit=len(self.gcnn.convs) + 1).nodes()) for i in indexes])))
        kwargs["graph"] = nx.subgraph(kwargs["graph"], nodes)
        kwargs.update(self.preprocess_graph(kwargs["graph"]))

        with timeit("update postitions"):
            self.update_position2node_index(observed_stations, re_update=self.k_neighbours is not None)
        indexes = [self.position2node_index[position] for position in zip(observed_stations.loc["long"].values,
                                                                          observed_stations.loc["lat"].values)]

        print("Number of nodes after pruning", len(kwargs["graph"]))

        with timeit("get emissions to calibrate"):
            traffic = self.get_emissions(
                self.get_traffic_by_node(observed_pollution, kwargs["traffic_by_edge"], kwargs["graph"]))
        target_values = tf.convert_to_tensor(observed_pollution.values, dtype=tf.float32)

        # Training step
        @tf.function
        def train():
            with tf.GradientTape() as tape:
                with timeit("predictions in train"):
                    predictions = self.state_estimation_core(traffic, kwargs["adj"], training=True)
                with timeit("mask"):
                    # predictions = self.state_estimation_core([t for i, t in enumerate(traffic) if i in idx], kwargs["adj"], training=True)
                    target_predictions = tf.squeeze(tf.concat([[[p[i] for i in indexes] for p in predictions]], 0))
                    mask = ~tf.math.is_nan(target_predictions)
                # loss = self.loss_fn(target_values[idx][mask], target_predictions[mask])
                with timeit("loss"):
                    loss = self.loss_fn(target_values[mask], target_predictions[mask])
                    loss += sum(self.gcnn.losses)
            with timeit("grad"):
                gradients = tape.gradient(loss, self.gcnn.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.gcnn.trainable_variables))
            return loss

        # initial_loss = train([0])
        min_loss = np.inf
        last_good_epoch = 0
        for i in range(1, self.niter):
            # times_shuffled = np.random.choice(len(observed_pollution), replace=False, size=len(observed_pollution))
            # for start in range(0, len(observed_pollution), self.batch_size):
            #     loss = train(times_shuffled[start:start+self.batch_size])
            #     print(f"\r{float(i) / self.niter * 100:2f}%: loss -> {loss:2f}", end="\r")
            loss = train()
            print(f"\r{float(i) / self.niter * 100:2f}%: loss -> {loss:2f}", end="\r")
            if loss < min_loss:
                min_loss = loss
                last_good_epoch = i
            if i - last_good_epoch > self.epochs_to_stop:
                break
        print(f"\nFinal train loss = {loss}")

        # save model params
        if self.experiment_dir is not None:
            self.gcnn.save(f'{self.experiment_dir}/{self}')


if __name__ == "__main__":
    from spektral.utils import tic, toc

    # https://hhaji.github.io/Deep-Learning/Graph-Neural-Networks/
    """
    This script is a proof of concept to train GCN as fast as possible and with as
    little lines of code as possible.
    It uses a custom training function instead of the standard Keras fit(), and
    can train GCN for 200 epochs in a few tenths of a second (~0.20 on a GTX 1050).
    """

    tf.random.set_seed(seed=0)  # make weight initialization reproducible

    # Load data
    # dataset = Cora(normalize_x=True, transforms=[LayerPreprocess(GCNConv), AdjToSpTensor()])
    # graph = dataset[0]
    # x, a, y = graph.x, graph.a, graph.y
    # mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te

    times = 100
    ntrain = 75
    nodes2train = [-1]

    graph = nx.path_graph(5)
    x, a = to_mixed(x_list=[np.random.uniform(size=(graph.number_of_nodes(), 1)) for _ in range(times)],
                    a=nx.adjacency_matrix(graph).astype(float).todense(), e_list=None)
    idx = tf.where(tf.not_equal(a, 0))
    # Make the sparse tensor
    # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape()
    # if tensor shape is dynamic
    a = tf.SparseTensor(idx, tf.gather_nd(a, idx), np.shape(a))
    y = np.cumsum(x, axis=1)

    # model = GCN(n_labels=dataset.n_labels, n_input_channels=dataset.n_node_features)
    model = GCNN(hidden_layers=(2, 2, 2,))
    optimizer = Adam(learning_rate=1e-2)
    loss_fn = MeanSquaredError()


    # Training step
    @tf.function
    def train():
        with tf.GradientTape() as tape:
            predictions = [model([x[i, :5, :1], a], training=True) for i in range(ntrain)]
            loss = loss_fn(y[:ntrain, nodes2train, :], [p[nodes2train] for p in predictions])
            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss


    # Time the execution of 200 epochs of training
    train()  # Warm up to ignore tracing times when timing
    tic()
    for epoch in range(1, 201):
        loss = train()
    toc("Spektral - GCN (200 epochs)")
    print(f"Final train loss = {loss}")

    predictions = [model([x[i, :5, :1], a], training=True) for i in range(ntrain, times)]
    print(f"Test loss = {loss_fn(y[ntrain:], predictions)}")

    y[ntrain:, :, :] - predictions
