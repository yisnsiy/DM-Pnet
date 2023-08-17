import sys
sys.path.append('/home/yis22/code/pnet_based_on_pytorch')
from os.path import join
import time
import pandas as pd
import logging

from data_access.pnet_data import PnetData
from model.pnet import Pnet
from model.train_utils import *
from config import LOG_PATH, debug, models_params
from utils.general import try_gpu, create_data_iterator
from utils.metrics import Metrics
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy

import torch
from torch import optim
from torch.optim import lr_scheduler

n_epochs = 10
n_splits = 10
device = try_gpu(3)

def run():
    logger = logging.getLogger(name='r')  # 不加名称设置root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 使用FileHandler输出到文件
    fh = logging.FileHandler(join(LOG_PATH, 'cross_validation_log.txt'), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

    results = None
    model_params = models_params['model_params']
    model_name = models_params['id']
    data_params = model_params['data_params']

    all_data = PnetData(data_params)
    X = np.concatenate((all_data.x_train, all_data.x_validate_, all_data.x_test_), axis=0)
    y = np.concatenate((all_data.y_train, all_data.y_validate_, all_data.y_test_), axis=0)

    pnet = Pnet(models_params)
    print('training on', device)
    for epoch in range(n_epochs):
        skf = StratifiedKFold(n_splits=n_splits, random_state=int(time.time()), shuffle=True)
        i = 0
        for train_index, test_index in skf.split(X, y.ravel()):
            state = f'epoch: {epoch}/{n_epochs} fold:{i}/{n_splits}'
            new_name = model_name + state
            model = deepcopy(pnet)
            logger.info(f'-------------------------{state}-------------------------')
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            fitting_params = models_params['fitting_params']
            batch_size = fitting_params['batch_size']
            train_iter = create_data_iterator(X=x_train,
                                              y=y_train,
                                              batch_size=batch_size,
                                              shuffle=fitting_params['shuffle'],
                                              data_type=torch.float32)
            if fitting_params['class_weight'] == 'auto':
                classes = np.unique(y_train)
                class_weights = class_weight.compute_class_weight('balanced', classes, y_train.ravel())
                class_weights = dict(zip(classes, class_weights))
            else:
                class_weights = {0: 1, 1: 1}
            # print(f'class_weights is {class_weights}')

            # train model
            print("\n------------train model------------\n")
            loss_fn = get_loss_func(fitting_params['n_outputs'])
            optimizer = optim.Adam(model.parameters(),
                                   lr=fitting_params['lr'],
                                   weight_decay=model_params['penalty'])
            scheduler = lr_scheduler.StepLR(optimizer,
                                            step_size=fitting_params['reduce_lr_after_nepochs']['epochs_drop'],
                                            gamma=fitting_params['reduce_lr_after_nepochs']['drop'])
            model = model_simply_train(model,
                              train_iter=train_iter,
                              loss=loss_fn,
                              optimizer=optimizer,
                              num_epochs=fitting_params['epoch'] if not debug else 2,
                              scheduler=scheduler,
                              loss_weights=fitting_params['loss_weights'],
                              class_weights=class_weights,
                              device=device)

            # evaluate performance
            print("\n------evaluate performance------\n")
            X_test = torch.tensor(x_test, dtype=torch.float32)
            y_prob = model_predict(model, X_test, device)
            res = Metrics.evaluate_classification_binary(y_prob.cpu().numpy(), y_test, None, new_name, False)
            logger.info(res)
            data = list([res.values()])
            if results is None:
                col = list(res.keys())
                results = pd.DataFrame(data=data, index=[new_name], columns=col)
            else:
                results.loc[new_name] = res
            model.to('cpu')
            i = i + 1
    results.to_csv(join(LOG_PATH, 'cross_valida_result_1.csv'))


if __name__ == "__main__":
    run()
