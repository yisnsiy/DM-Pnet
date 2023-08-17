import sys
import os
sys.path.append('/home/pnet_based_on_pytorch')
from os.path import join, abspath, dirname
import time
import pandas as pd
import logging

from data_access.pnet_data import PnetData
from model.pnet import Pnet
from model.train_utils import *
from config import debug, models_params
from utils.general import try_gpu, create_data_iterator
from utils.metrics import Metrics
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
import plotly.offline as pyo
import plotly.io as pio
import plotly.subplots as subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

import torch
from torch import optim
from torch.optim import lr_scheduler


device = try_gpu(7)
n_experiment_per_model = 20
out_dir = join(dirname(abspath(__file__)), 'experiment4')

def run():
    logger = logging.getLogger(name='r')  # 不加名称设置root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # 使用FileHandler输出到文件
    fh = logging.FileHandler(
        join(out_dir, 'experiment4.txt'), 
        mode='w'
    )
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
    # 将默认的参数修改成对应的local和global的参数
    local_dm_pnet = deepcopy(models_params)
    local_dm_pnet['model_params']['trainable_mask'] = True
    local_dm_pnet['model_params']['full_train'] = False
    local_dm_pnet['id'] = 'local_dm_pnet'

    global_dm_pnet = deepcopy(models_params)
    global_dm_pnet['model_params']['trainable_mask'] = True
    global_dm_pnet['model_params']['full_train'] = True
    global_dm_pnet['id'] = 'global_dm_pnet'

    list_p = []
    for i in range(n_experiment_per_model):
        tmp = deepcopy(local_dm_pnet)
        tmp['id'] = tmp['id'] + f'_{i}'
        list_p.append(tmp)
    
    for i in range(n_experiment_per_model):
        tmp = deepcopy(global_dm_pnet)
        tmp['id'] = tmp['id'] + f'_{i}'
        list_p.append(tmp)

    cnt = 1
    for p in list_p:
        print('cnt: ', cnt)
        cnt = cnt + 1
        model_params = p['model_params']
        model_name = p['id']
        data_params = model_params['data_params']

        all_data = PnetData(data_params)

        model = Pnet(p)
        print('training on', device)
        
        x_train, x_test = all_data.x_train, all_data.x_test_
        y_train, y_test = all_data.y_train, all_data.y_test_

        fitting_params = p['fitting_params']
        batch_size = fitting_params['batch_size']
        train_iter = create_data_iterator(
            X=x_train,
            y=y_train,
            batch_size=batch_size,
            shuffle=fitting_params['shuffle'],
            data_type=torch.float32
        )
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
        optimizer = optim.Adam(
            model.parameters(),
            lr=fitting_params['lr'],
            weight_decay=model_params['penalty']
        )
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=fitting_params['reduce_lr_after_nepochs']['epochs_drop'],
            gamma=fitting_params['reduce_lr_after_nepochs']['drop']
        )
        model = model_simply_train(
            model,
            train_iter=train_iter,
            loss=loss_fn,
            optimizer=optimizer,
            num_epochs=fitting_params['epoch'] if not debug else 2,
            scheduler=scheduler,
            loss_weights=fitting_params['loss_weights'],
            class_weights=class_weights,
            device=device
        )

        # evaluate performance
        print("\n------evaluate performance------\n")
        X_test = torch.tensor(x_test, dtype=torch.float32)
        y_prob = model_predict(model, X_test, device)
        res = Metrics.evaluate_classification_binary(
            y_prob.cpu().numpy(), 
            y_test,
            False,
            None, 
            model_name, 
            False
        )
        logger.info(res)
        data = list([res.values()])
        if results is None:
            col = list(res.keys())
            results = pd.DataFrame(data=data, index=[model_name], columns=col)
        else:
            results.loc[model_name] = res
        model.to('cpu')
    results.to_csv(join(out_dir, 'experiment4.csv'))

def get_figure():
    """
        x: ['Accuracy' * number_test, + [other metrics] * number_test]
        y: 
    """
    in_dir = join(dirname(abspath(__file__)), 'experiment4')
    df = pd.read_csv(join(in_dir, 'experiment4.csv'), index_col=0)

    metrics = df.columns.tolist()
    # 首字母大写,其余小写
    Metrics = [metric.capitalize() for metric in metrics]
    if 'Aupr' in Metrics:
        Metrics[Metrics.index('Aupr')] = 'AUPR'
    if 'Auc' in Metrics:
        Metrics[Metrics.index('Auc')] = 'AUC'
    # 新设一列的值设为index的值
    df['model_name'] = df.index

    df['training_style'] = df['model_name'].apply(
        lambda x: x.split('_')[0]
    )

    # 将index设为数字
    df.reset_index(drop=True, inplace=True)

    number_test = 20
    x = [_ for _ in Metrics for i in range(number_test)]
    print(x)

    accuracy_local = np.array(
        df[df['training_style'] == 'local']['accuracy']
    ).round(6)
    precision_local = np.array(
        df[df['training_style'] == 'local']['precision']
    ).round(6)
    f1_local = np.array(
        df[df['training_style'] == 'local']['f1']
    ).round(6)
    recall_local = np.array(
        df[df['training_style'] == 'local']['recall']
    ).round(6)
    auc_local = np.array(
        df[df['training_style'] == 'local']['auc']
    ).round(6)
    aupr_local = np.array(
        df[df['training_style'] == 'local']['aupr']
    ).round(6)

    accuracy_global = np.array(
        df[df['training_style'] == 'global']['accuracy']
    ).round(6)
    precision_global = np.array(
        df[df['training_style'] == 'global']['precision']
    ).round(6)
    f1_global = np.array(
        df[df['training_style'] == 'global']['f1']
    ).round(6)
    recall_global = np.array(
        df[df['training_style'] == 'global']['recall']
    ).round(6)
    auc_global = np.array(
        df[df['training_style'] == 'global']['auc']
    ).round(6)
    aupr_global = np.array(
        df[df['training_style'] == 'global']['aupr']
    ).round(6)

    metris_data_local = np.concatenate((
        accuracy_local, 
        precision_local, 
        f1_local, 
        recall_local, 
        aupr_local,
        auc_local, 
    ))
    metris_data_global = np.concatenate((
        accuracy_global, 
        precision_global, 
        f1_global, 
        recall_global, 
        aupr_global,
        auc_global, 
    ))

    fig = go.Figure()

    fig.add_trace(go.Box(
        y=metris_data_local,
        x=x,
        name='Local Training',
        marker_color='#ffa500',
        # boxpoints='all',
    ))
    fig.add_trace(go.Box(
        y=metris_data_global,
        x=x,
        name='Global Training',
        marker_color='#b30059',
        # boxpoints='all',
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        boxmode='group', # group together boxes of the different traces for each value of x
        legend=dict( # 左上
            x=0.01,
            y=0.99,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="black"
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
   )
    fig.update_xaxes(
        title_text="Metrics",
        ticks="inside",
        tickwidth=2,
        tickcolor='black',
    )
    fig.update_yaxes(
        title_text="Value",
        ticks="inside",
        tickwidth=2,
        tickcolor='black',
    )
    # fig.show()
    pyo.plot(fig, filename=join(in_dir, 'experiment4.html'), auto_open=False)
    # format = 'jpeg'
    # fig.write_image(
    #     join(in_dir, f'experiment4.{format}'),
    #     format=format, 
    #     width=1520, 
    #     height=739, 
    #     scale=10,
    # )


if __name__ == "__main__":
    # run()
    get_figure()
