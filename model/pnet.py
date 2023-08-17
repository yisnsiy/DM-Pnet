from config import models_params
from model.builder_utils import *
from custom.layer_custom import CustomizedLinear, Diagonal
from data_access.data_access import Data

import torch
from torch import nn, tanh, sigmoid


class Pnet(nn.Module):

    def __init__(self, param):
        super(Pnet, self).__init__()

        self.model_params = param['model_params']
        self.data_params = self.model_params['data_params']
        self.use_bias = self.model_params['use_bias']
        self.trainable_mask = self.model_params['trainable_mask']
        self.full_train = self.model_params['full_train']

        print(f"model params: {self.model_params}")

        self.get_model_data()

        print("input dim: ", self.in_size)
        self.dropout1 = nn.Dropout(p=self.model_params['dropout'][0])
        self.dropout2 = nn.Dropout(p=self.model_params['dropout'][1])

        # each node in the next layer is connected to exactly three nodes of the input
        # layer representing mutations, copy number amplification and copy number deletions.
        # sparse layer 27687*9229
        self.h0 = Diagonal(self.in_size, self.maps[0].shape[0], self.use_bias, self.trainable_mask)  # sparse layer 27687*9229
        # sparse layer 9227*1387
        self.h1 = CustomizedLinear(self.maps[0].shape[0], self.maps[0].shape[1], self.use_bias,
                                    np.array(self.maps[0].values), self.trainable_mask, self.full_train)
        # sparse layer 1387*1066
        self.h2 = CustomizedLinear(self.maps[1].shape[0], self.maps[1].shape[1], self.use_bias,
                                    np.array(self.maps[1].values), self.trainable_mask, self.full_train)
        # sparse layer 1066*447
        self.h3 = CustomizedLinear(self.maps[2].shape[0], self.maps[2].shape[1], self.use_bias,
                                    np.array(self.maps[2].values), self.trainable_mask, self.full_train)
        # sparse layer 447*147
        self.h4 = CustomizedLinear(self.maps[3].shape[0], self.maps[3].shape[1], self.use_bias,
                                    np.array(self.maps[3].values), self.trainable_mask, self.full_train)
        # sparse layer 147*26
        self.h5 = CustomizedLinear(self.maps[4].shape[0], self.maps[4].shape[1], self.use_bias,
                                    np.array(self.maps[4].values), self.trainable_mask, self.full_train)
        # self.h0 = nn.Linear(self.in_size, self.maps[0].shape[0])  # sparse layer 27687*9229
        # self.h1 = nn.Linear(self.maps[0].shape[0], self.maps[0].shape[1])  # sparse layer 9227*1387
        # self.h2 = nn.Linear(self.maps[1].shape[0], self.maps[1].shape[1])  # sparse layer 1387*1066
        # self.h3 = nn.Linear(self.maps[2].shape[0], self.maps[2].shape[1])  # sparse layer 1066*447
        # self.h4 = nn.Linear(self.maps[3].shape[0], self.maps[3].shape[1])  # sparse layer 447*147
        # self.h5 = nn.Linear(self.maps[4].shape[0], self.maps[4].shape[1])  # sparse layer 147*26

        self.l1 = nn.Linear(self.maps[0].shape[0], 1)
        self.l2 = nn.Linear(self.maps[0].shape[1], 1)
        self.l3 = nn.Linear(self.maps[1].shape[1], 1)
        self.l4 = nn.Linear(self.maps[2].shape[1], 1)
        self.l5 = nn.Linear(self.maps[3].shape[1], 1)
        self.l6 = nn.Linear(self.maps[4].shape[1], 1)

    def forward(self, input):
        out = tanh(self.h0(input))
        o1 = sigmoid(self.l1(out))
        out = self.dropout1(out)

        out = tanh(self.h1(out))
        o2 = sigmoid(self.l2(out))
        out = self.dropout2(out)

        out = tanh(self.h2(out))
        o3 = sigmoid(self.l3(out))
        out = self.dropout2(out)

        out = tanh(self.h3(out))
        o4 = sigmoid(self.l4(out))
        out = self.dropout2(out)

        out = tanh(self.h4(out))
        o5 = sigmoid(self.l5(out))
        out = self.dropout2(out)

        out = tanh(self.h5(out))
        o6 = sigmoid(self.l6(out))
        out = self.dropout2(out)

        return torch.concat([o1, o2, o3, o4, o5, o6], dim=1)

    def get_model_data(self):
        """get mask matrix that sparseNn have."""
        data = Data(**self.data_params)
        x, y, info, cols = data.get_data()
        self.in_size = cols.shape[0]
        print('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

        if hasattr(cols, 'levels'):
            genes = cols.levels[0]
        else:
            genes = cols
        self.feature_names = {}
        self.feature_names['inputs'] = cols
        if self.model_params['n_hidden_layers'] > 0:
            maps = get_layer_maps(genes, self.model_params['n_hidden_layers'], 'root_to_leaf', self.model_params['add_unk_genes'])
            self.maps = maps
        for i, _maps in enumerate(maps):
            self.feature_names[f'h{i}'] = _maps.index
