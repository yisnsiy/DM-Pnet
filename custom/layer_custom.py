import torch.nn as nn
from custom.linearFunction import *
import math


class CustomizedLinear(nn.Module):
    def __init__(self,
                 input_features,
                 output_features,
                 bias=True,
                 mask=None,
                 trainable_mask=False,
                 full_train=False):
        """
        Argumens
        ------------------
        mask [numpy.array]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.full_train = full_train

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Initialize the above parameters (weight & bias).
        self.init_params()

        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.float).t()
            self.mask = nn.Parameter(mask, requires_grad=False)
            if trainable_mask is True:
                self.trainable_mask = nn.Parameter(mask.clone().detach(), requires_grad=True)
            else:
                self.register_parameter('trainable_mask', None)
        else:
            self.register_parameter('mask', None)
            self.register_parameter('trainable_mask', None)

    def init_params(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = math.sqrt(3. * 1. / self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            # self.bias.data_access.uniform_(-stdv, stdv)
            self.bias.data.zero_()

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if self.full_train is True and isinstance(self.trainable_mask, torch.Tensor):
            return input.mm((self.weight * self.trainable_mask).t())
        else:
            if isinstance(self.trainable_mask, torch.Tensor):
                return TrainableLinear.apply(input, self.weight, self.bias, self.mask, self.trainable_mask)
            else:
                return NotTrainableLinear.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}, mask={}'.format(
            self.input_features, self.output_features,
            self.bias is not None, self.mask is not None)


class Diagonal(nn.Module):
    def __init__(self,
                 input_features,
                 output_features,
                 bias=True,
                 trainable_mask=False,
                 full_train=False):
        """
        Argumens
        ------------------
        mask [numpy.array]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(Diagonal, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.full_train = full_train

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute.
        self.weight = nn.Parameter(torch.Tensor(
            self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Initialize the above parameters (weight & bias).
        self.init_params()

        mask = torch.zeros(output_features, input_features)
        for i in range(input_features):
            mask[math.floor(i / 3)][i] = 1.
        if mask is not None:
            mask = mask.float()
            self.mask = nn.Parameter(mask, requires_grad=False)
            if trainable_mask is True:
                # self.trainable_mask = nn.Parameter(mask, requires_grad=True)
                self.trainable_mask = nn.Parameter(mask.clone().detach(), requires_grad=True)
            else:
                self.register_parameter('trainable_mask', None)
            # print('\n[!] CustomizedLinear: \n', self.weight.data_access.t())
        else:
            self.register_parameter('mask', None)
            self.register_parameter('trainable_mask', None)

    def init_params(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = math.sqrt(3. * 1. / self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            # self.bias.data_access.uniform_(-stdv, stdv)
            self.bias.data.zero_()

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if self.full_train is True and isinstance(self.trainable_mask, torch.Tensor):
            return input.mm((self.weight * self.trainable_mask).t())
        else:
            if isinstance(self.trainable_mask, torch.Tensor):
                return TrainableLinear.apply(input, self.weight, self.bias, self.mask, self.trainable_mask)
            else:
                return NotTrainableLinear.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}, mask={}'.format(
            self.input_features, self.output_features,
            self.bias is not None, self.mask is not None)