from os.path import join, exists
from config import RESULT_PATH, REACTOM_PATHWAY_PATH, interpret, LOG_PATH
from data_access.data_access import Data
from custom.layer_custom import Diagonal
from custom import sankey

from captum.attr import LayerDeepLift, LayerIntegratedGradients
import pandas as pd
from torch import nn, tanh
import numpy as np
import torch
import copy


layer_names = ['inputs', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'l6']


class SubModel(nn.Module):

    def __init__(self, model, layer_names):
        super(SubModel, self).__init__()
        # self.dropout1 = nn.Dropout(p=models_params['params']['model_params']['dropout'][0])
        # self.dropout2 = nn.Dropout(p=models_params['params']['model_params']['dropout'][1])
        self.dropout1 = model.dropout1
        self.dropout2 = model.dropout2
        self.feature_names = model.feature_names
        for i, layer_name in enumerate(layer_names):
            self.__setattr__(layer_name, copy.deepcopy(model.get_submodule(layer_name)))

    def forward(self, input):
        out = tanh(self.h0(input))
        out = self.dropout1(out)

        out = tanh(self.h1(out))
        out = self.dropout2(out)

        out = tanh(self.h2(out))
        out = self.dropout2(out)

        out = tanh(self.h3(out))
        out = self.dropout2(out)

        out = tanh(self.h4(out))
        out = self.dropout2(out)

        out = tanh(self.h5(out))
        out = self.l6(out)
        return out


def save_gradient_importance(node_weights_):
    for i, k in enumerate(layer_names[:-1]):
        n = node_weights_[k]
        filename = join(RESULT_PATH, 'extracted/gradient_importance_{}.csv'.format(i))
        n.to_csv(filename)


def save_link_weights(link_weights_df, layers):
    for i, l in enumerate(layers):
        link = link_weights_df[l]
        filename = join(RESULT_PATH, 'extracted/link_weights_{}.csv'.format(i))
        link.to_csv(filename)


def save_graph_stats(degrees, fan_outs, fan_ins, layers):
    i = 1

    df = pd.concat([degrees[0], fan_outs[0]], axis=1)
    df.columns = ['degree', 'fan_out']
    df['fan_in'] = 0
    filename = join(RESULT_PATH, 'extracted/graph_stats_{}.csv'.format(i))
    df.to_csv(filename)

    for i, (d, fin, fout) in enumerate(zip(degrees[1:], fan_ins, fan_outs[1:])):
        df = pd.concat([d, fin, fout], axis=1)
        df.columns = ['degree', 'fan_in', 'fan_out']
        print(df.head())
        filename = join(RESULT_PATH, 'extracted/graph_stats_{}.csv'.format(i + 2))
        df.to_csv(filename)


def get_reactome_pathway_names():
    reactome_pathways_df = pd.read_csv(join(REACTOM_PATHWAY_PATH, 'ReactomePathways.txt'), sep='	', header=None)
    reactome_pathways_df.columns = ['id', 'name', 'species']
    reactome_pathways_df_human = reactome_pathways_df[reactome_pathways_df['species'] == 'Homo sapiens']
    reactome_pathways_df_human.reset_index(inplace=True)
    return reactome_pathways_df_human


def adjust_layer(df):
    # graph coef
    z1 = df.coef_graph
    z1 = (z1 - z1.mean()) / z1.std(ddof=0)

    # gradient coef
    z2 = df.coef
    z2 = (z2 - z2.mean()) / z2.std(ddof=0)

    z = z2 - z1

    z = (z - z.mean()) / z.std(ddof=0)
    x = np.arange(len(z))
    df['coef_combined2'] = z
    return df


def get_pathway_names(all_node_ids):
    #     pathways_names = get_reactome_pathway_names()
    #     all_node_labels = pd.Series(all_node_ids).replace(list(pathways_names['id']), list(pathways_names['name']))

    pathways_names = get_reactome_pathway_names()
    ids = list(pathways_names['id'])
    names = list(pathways_names['name'])
    ret_list = []
    for f in all_node_ids:
        # print f
        if f in ids:
            ind = ids.index(f)
            f = names[ind]
            ret_list.append(f)
        else:
            # print 'no'
            ret_list.append(f)

    return ret_list


def get_neuron_contribution(model, X, method_name, baseline=0):
    print(f"interpret model by {method_name}, and baseline is {baseline}")
    neuron_contribution = {}
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if baseline == 'mean':
        baseline = torch.mean(X, dim=-2, keepdim=True)
    elif baseline == 'zero':
        baseline = 0
    for i, layer_name in enumerate(layer_names[1:]):
        layer = model.get_submodule(layer_name)

        if method_name == 'deeplift':
            ld = LayerDeepLift(model, layer)
            contribution_layer_sample = ld.attribute(X,
                                                     baselines=baseline,
                                                     attribute_to_layer_input=True) # ensure neuron_contribution meat require about data_access format
        elif method_name == 'integratedgradients':
            ld = LayerIntegratedGradients(model, layer)
            contribution_layer_sample = ld.attribute(X,
                                                     baselines=baseline,
                                                     attribute_to_layer_input=True)
        neuron_contribution[layer_names[i]] = np.sum(contribution_layer_sample.detach().numpy(), axis=-2)
    return neuron_contribution


def get_node_importance(model, X, method_name, baseline):
    """

    :param model: nn.Model object
    :param X: numpy array
    :param method_name: name of method that interpret model
    :param baseline reference input
    :return: list of pandas dataframes with weights for each layer
    """
    # model = Model(nn_model.model.input, nn_model.model.outputs)
    # model.compile('sgd', 'mse')

    coef = get_neuron_contribution(model, X, method_name, baseline)
    node_weights_dfs = {}
    # layers = []
    # for i, (w, w_samples, name) in enumerate(zip(coef, coef_detailed, nn_model.feature_names)):
    for i, k in enumerate(model.feature_names.keys()):
        name = model.feature_names[k]
        w = coef[k]
        features = get_pathway_names(name)
        df = pd.DataFrame(abs(w.ravel()), index=name, columns=['coef'])
        layer = pd.DataFrame(index=name)
        layer['layer'] = i
        # node_weights_dfs.append(df)
        node_weights_dfs[k] = df
        # layers.append(layer)
    return node_weights_dfs


def get_layer_weights(layer):  # consider weather mask have
    w = layer.weight.T.detach()
    if isinstance(layer, Diagonal):
        w_list = []
        for i in range(w.shape[0]):
            w_list.append(w[i][i // 3])
        w_np = np.array(w_list)
    else:
        if hasattr(layer, 'mask'):
            w = torch.where(layer.mask.data.T == 1, w, 0)
        w_np = w.numpy()
    return w_np


def get_link_weights_df_(model, features, layer_names):
    # first layer
    # layer_name= layer_names[1]
    # layer= model.get_layer(layer_name)
    link_weights_df = {}
    # df = pd.DataFrame( layer.get_weights()[0], index=features[layer_names[0]])
    # link_weights_df[layer_name]=df

    for i, layer_name in enumerate(layer_names[1:]):
        # layer = model.get_layer(layer_name)
        # w = get_layer_weights(layer)
        w = get_layer_weights(model.get_submodule(layer_name))

        layer_ind = layer_names.index(layer_name)
        previous_layer_name = layer_names[layer_ind - 1]

        print (i, previous_layer_name, layer_name)
        if i == 0 or i == (len(layer_names) - 2):
            cols = ['root']
        else:
            cols = features[layer_name]
        rows = features[previous_layer_name]
        w_df = pd.DataFrame(w, index=rows, columns=cols)
        link_weights_df[layer_name] = w_df

    # last layer
    # layer_name = layer_names[-1]
    # layer = model.get_layer(layer_name)
    # link_weights_df = {}
    # df = pd.DataFrame(layer.get_weights()[0], index=features[layer_names[0]])
    # link_weights_df[layer_name] = df

    return link_weights_df


def get_degrees(maps, layers, model):
    stats = {}
    for i, (l1, l2) in enumerate(zip(layers[1:], layers[2:])):

        layer1 = maps[l1]
        layer2 = maps[l2]
        mask1 = model.get_submodule(l1).mask.data.t()
        if hasattr(model.get_submodule(l2), 'mask'):
            mask2 = model.get_submodule(l2).mask.data.t()
        else:
            mask2 = torch.tensor(np.where(np.array(layer2) != 0, 1., 0.))

        layer1[layer1 != 0] = 1.
        layer2[layer2 != 0] = 1.

        # fan_out1 = layer1.abs().sum(axis=1)
        # fan_in1 = layer1.abs().sum(axis=0)
        fan_out1 = pd.Series(data = torch.sum(mask1, dim=1), index=layer1.index)
        fan_in1 = pd.Series(data=torch.sum(mask1, dim=0), index=layer1.columns)

        # fan_out2 = layer2.abs().sum(axis=1)
        # fan_in2 = layer2.abs().sum(axis=0)
        fan_out2 = pd.Series(data=torch.sum(mask2, dim=1), index=layer2.index)
        fan_in2 = pd.Series(data=torch.sum(mask2, dim=0), index=layer2.columns)

        if i == 0:
            l = layers[0]
            df = pd.concat([fan_out1, fan_out1], keys=['degree', 'fanout'], axis=1)
            df['fanin'] = 1.
            stats[l] = df

        print('{}- layer {} :fan-in {}, fan-out {}'.format(i, l1, fan_in1.shape, fan_out2.shape))
        print('{}- layer {} :fan-in {}, fan-out {}'.format(i, l1, fan_in2.shape, fan_out1.shape))

        df = pd.concat([fan_in1, fan_out2], keys=['fanin', 'fanout'], axis=1)
        df['degree'] = df['fanin'] + df['fanout']
        stats[l1] = df

    return stats


def adjust_coef_with_graph_degree(node_importance_dfs, stats, layer_names):
    ret = []
    # for i, (grad, graph) in enumerate(zip(node_importance_dfs, degrees)):
    for i, l in enumerate(layer_names):
        grad = node_importance_dfs[l]
        graph = stats[l]['degree'].to_frame(name='coef_graph')

        graph.index = get_pathway_names(graph.index)
        grad.index = get_pathway_names(grad.index)
        d = grad.join(graph, how='inner')

        mean = d.coef_graph.mean()
        std = d.coef_graph.std()
        ind = d.coef_graph > mean + 5 * std
        divide = d.coef_graph.copy()
        divide[~ind] = divide[~ind] = 1.
        d['coef_combined'] = d.coef / divide
        z = d.coef_combined
        z = (z - z.mean()) / z.std(ddof=0)
        d['coef_combined_zscore'] = z
        d = adjust_layer(d)
        #         d['coef_combined'] = d['coef_combined']/sum(d['coef_combined'])
        filename = join(RESULT_PATH, 'extracted/layer_{}_graph_adjusted.csv'.format(i))
        d.to_csv(filename)
        d['layer'] = i + 1
        ret.append(d)
    node_importance = pd.concat(ret)
    node_importance = node_importance.groupby(node_importance.index).min()
    return node_importance


def run(model_name, X=None, method_name='deeplift', baseline='zero'):
    map_location = 'cpu'
    if model_name is not None:
        filename = join(RESULT_PATH, f'{model_name}_' + 'model.pt')
    else:
        filename = join(RESULT_PATH, 'model.pt')
    print(filename)
    if exists(filename):
        model = torch.load(filename, map_location=map_location)
    else:
        raise ValueError("model is not exist")
    sub_model = SubModel(model, layer_names[1:])
    # model.eval()

    #load data_access
    if interpret == True:
        X = torch.load(join(LOG_PATH, 'input.pt'))

    # get neuron contribution by deeplift rescale rule
    node_weights_ = get_node_importance(sub_model, X, method_name,  baseline)
    print("saving node weights")
    save_gradient_importance(node_weights_)

    # get link weights
    link_weights_df = get_link_weights_df_(model, model.feature_names, layer_names)
    print("saving link weights")
    save_link_weights(link_weights_df, layer_names[1:])

    if model.trainable_mask == True:
        trainable_mask = model.get_submodule('h1').trainable_mask.T.detach().numpy()
        trainable_mask_df = pd.DataFrame(
            trainable_mask,
            index=model.feature_names[layer_names[1]],
            columns=model.feature_names[layer_names[2]]
        )
        trainable_mask_df.to_csv(join(RESULT_PATH, 'extracted/trainable_mask_1.csv'))


    # get degree of genes
    deg_matrix = get_degrees(link_weights_df, layer_names[1:], model)
    print("saving degrees matrix weights")
    for k in layer_names[1:-1]:
        filename = join(RESULT_PATH, 'extracted/graph_stats_{}.csv'.format(k))
        deg_matrix[k].to_csv(filename)

    # get real contribution by considering edges and points
    node_importance = adjust_coef_with_graph_degree(node_weights_, deg_matrix, layer_names[1:-1])
    print("saving adjusted import with degree matrix")
    filename = join(RESULT_PATH, 'extracted/node_importance_graph_adjusted.csv')
    node_importance.to_csv(filename)

    sankey.run(model_name)


if __name__ == "__main__":
    run(model_name='pnet_deeplift', method_name='deeplift')