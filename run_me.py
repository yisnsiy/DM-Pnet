from os.path import join

import pandas as pd

from data_access.pnet_data import PnetData
from model.pnet import Pnet
from model.train_utils import *
from config import RESULT_PATH, LOG_PATH, debug, save_res, parameters
from utils.general import try_gpu, create_data_iterator
from utils.metrics import Metrics
from custom import interpret_model
from sklearn.utils import class_weight

import torch
from torch import optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter


# class_weights = {0: 0.7458410351201479, 1: 1.5169172932330828}
# batch_size = 50
# epochs = 300
# penalty = 0.001

# """
device = try_gpu(7)
# device = torch.device('cpu')

results = None
for p in parameters:
    # get data_access
    model_params = p['model_params']
    data_params = model_params['data_params']

    model_name = p['id']
    methode_name = p['feature_importance']['method_name']
    baseline = p['feature_importance']['baseline']

    print("\n------------loading data------------\n")
    all_data = PnetData(data_params)

    # build neural network
    print("\n------------build model------------\n")
    if model_name.startswith('pnet'):
        model = Pnet(p)
    else:
        raise ValueError('no suitable model')



    # prepare data set
    print("\n------------prepare data set------------\n")
    fitting_params = p['fitting_params']
    batch_size = fitting_params['batch_size'] if not debug else 1
    train_iter = create_data_iterator(X=all_data.x_train,
                                      y=all_data.y_train,
                                      batch_size=batch_size,
                                      shuffle=fitting_params['shuffle'],
                                      data_type=torch.float32)
    valid_iter = create_data_iterator(X=all_data.x_validate_,
                                      y=all_data.y_validate_,
                                      batch_size=batch_size,
                                      shuffle=fitting_params['shuffle'],
                                      data_type=torch.float32)
    if fitting_params['class_weight'] == 'auto':
        classes = np.unique(all_data.y_train)
        class_weights = class_weight.compute_class_weight('balanced', classes, all_data.y_train.ravel())
        class_weights = dict(zip(classes, class_weights))
    else:
        class_weights = {0: 1, 1: 1}
    print(f'class_weights is {class_weights}')

    # writer = SummaryWriter('logs')
    # dummy_input = torch.rand(1, all_data.x_train.shape[-1])
    # with SummaryWriter(comment='pnet') as w:
    #     w.add_graph(model, (dummy_input,))

    # train model
    print("\n------------train model------------\n")
    loss_fn = get_loss_func(fitting_params['n_outputs'])
    optimizer = optim.Adam(model.parameters(),
                           lr=fitting_params['lr'],
                           weight_decay=model_params['penalty'])
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=fitting_params['reduce_lr_after_nepochs']['epochs_drop'],
                                    gamma=fitting_params['reduce_lr_after_nepochs']['drop'])
    net = model_train(model,
                      train_iter=train_iter,
                      loss=loss_fn,
                      optimizer=optimizer,
                      test_iter=valid_iter,
                      num_epochs=fitting_params['epoch'] if not debug else 2,
                      scheduler=scheduler,
                      loss_weights=fitting_params['loss_weights'],
                      class_weights=class_weights,
                      device=device)

    # evaluate performance
    print("\n------evaluate performance------\n")
    X_test = torch.tensor(all_data.x_test_, dtype=torch.float32)
    y_prob = model_predict(net, X_test, device)
    saving_dir = join(RESULT_PATH, 'metrics') if save_res else None
    res = Metrics.evaluate_classification_binary(y_prob.cpu().numpy(), all_data.y_test_, fitting_params['max_f1'], saving_dir, model_name)
    data = list([res.values()])
    if results is None:
        col = list(res.keys())
        results = pd.DataFrame(data=data, index=[model_name], columns=col)
    else:
        results.loc[model_name] = res

    # saving model
    if save_res is True:
        filename = join(RESULT_PATH, f'{model_name}_' + 'model.pt')
        torch.save(net, filename)  # save all parameters, feature names must be included, not generate again.

    # explain model
    if methode_name is not None:
        torch.save(X_test, join(LOG_PATH, 'input.pt'))
        interpret_model.run(model_name, X_test, methode_name, baseline)
    net.to('cpu')

print(results)

