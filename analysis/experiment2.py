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
from config import LOG_PATH, debug, models_params
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
name2number = {0:809, 3:581, 6:418, 9:300, 12:216, 15:155, 18:111}
n_experiment_per_sample = 5
out_dir = join(dirname(abspath(__file__)), 'experiment2')

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
        join(out_dir, 'experiment2.txt'), 
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
    # 将默认的参数修改成对应的pnet和dm-pnet的参数
    dm_pnet = deepcopy(models_params)
    dm_pnet['model_params']['trainable_mask'] = True
    dm_pnet['model_params']['full_train'] = False

    pnet = deepcopy(models_params)
    pnet['model_params']['trainable_mask'] = False
    pnet['model_params']['full_train'] = False

    list_p = []

    for i in range(7):
        dm_pnet_tmp = deepcopy(dm_pnet)
        dm_pnet_tmp['id'] = f'dm_pnet_{name2number[3 * i]}'
        dm_pnet_tmp['model_params']['data_params']['params']['training_split'] \
            = i * 3
        for j in range(n_experiment_per_sample):
            tmp = deepcopy(dm_pnet_tmp)
            tmp['id'] = tmp['id'] + f'_{j}'
            list_p.append(tmp)

    for i in range(7):
        pnet_tmp = deepcopy(pnet)
        pnet_tmp['id'] = f'pnet_{name2number[3 * i]}'
        pnet_tmp['model_params']['data_params']['params']['training_split'] \
            = i * 3
        for j in range(n_experiment_per_sample):
            tmp = deepcopy(pnet_tmp)
            tmp['id'] = tmp['id'] + f'_{j}'
            list_p.append(tmp)

    cnt = 0
    for p in list_p:
        print('cnt :', cnt + 1)
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
        print('results: ', results)
        model.to('cpu')
    results.to_csv(join(out_dir, 'experiment2.csv'))

def plot_filled_line(
        figs,
        x,
        order,
        title,
        y0, y0_upper, y0_lower, model0,
        y1, y1_upper, y1_lower, model1,
    ):
    x_rev = x[::-1]
    y0_lower = y0_lower[::-1]
    y1_lower = y1_lower[::-1]


    r = order // 2 + 1
    c = order % 2 + 1
    figs.add_trace(
        go.Scatter(
            x=x+x_rev,
            y=y0_upper+y0_lower,
            fill='toself',
            fillcolor='rgba(191,178,204,0.5)',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            name=model0
        ),
        row = r,
        col = c,
    )
    figs.add_trace(
        go.Scatter(
            x=x+x_rev,
            y=y1_upper+y1_lower,
            fill='toself',
            fillcolor='rgba(178,220,233,0.5)',
            line_color='rgba(255,255,255,0)',
            name=model1,
            showlegend=False,
        ),
        row = r,
        col = c
    )

    figs.add_trace(
        go.Scatter(
            x=x, y=y0,
            line_color='#220050',
            name=model0,
            showlegend=True if order == 0 else False,  # 只显示第一附图的图例
        ),
        row = r,
        col = c,
    )
    figs.add_trace(
        go.Scatter(
            x=x, y=y1,
            line_color='#0091a8',
            name=model1,
            showlegend=True if order == 0 else False, # 只显示第一附图的图例
            ),
        row = r,
        col = c,
    )

    # 美化figure内容
    if title == 'Auc':
        title = 'Area Under Curve'
    elif title == 'F1':
        title = 'F1 Score'
    elif title == 'Aupr':
        title = 'Area Under Precision-Recall'
    
    # y轴一定的最大值显示1，没有最小值要求
    figs.update_yaxes(dict(
        title=title,
        # showgrid=False,
        ticks='inside',  # 将刻度线放在坐标轴内部
        tickwidth=2,  # 刻度线的宽度
    ), row=r, col=c)
    figs.update_xaxes(dict(
        title='Number of samples',  # x轴标题
        showgrid=False,  # 不显示网格
        type='category',  # 将x轴的数字当作类别处理
        ticks='inside',  # 将刻度线放在坐标轴内部
        tickwidth=2,  # 刻度线的宽度
    ), row=r, col=c)

def get_figure():
    ROW, COL = 3, 2
    in_dir = join(dirname(abspath(__file__)), 'experiment2')
    df = pd.read_csv(join(in_dir, 'experiment2.csv'), index_col=0)

    metrics = df.columns.tolist()
    # 首字母大写,其余小写
    Metrics = [metric.capitalize() for metric in metrics]
    # 新设一列的值设为index的值
    df['model_name'] = df.index

    df['n_input'] = df['model_name'].apply(
        lambda x: int(x.split('_')[-2])
    )
    # 将model_name列的值按照_分割，去掉第倒数一个和倒数第二个，然后再合并
    df['model_name'] = df['model_name'].apply(
        lambda x: '_'.join(x.split('_')[:-2])
    )
    
    models = df['model_name'].unique().tolist()

    # 将index设为数字
    df.reset_index(drop=True, inplace=True)
    # df.set_index('model_name', inplace=True)

    # 去重并排序
    x = sorted(df['n_input'].unique().tolist())

    figs = subplots.make_subplots(
        rows=ROW, 
        cols=COL, 
        # subplot_titles=(Metrics),
    )

    for i, metric in enumerate(metrics):
        y_metric_max = df.groupby(['model_name', 'n_input'])[metric].max()
        y_metric_mean = df.groupby(['model_name', 'n_input'])[metric].mean() 
        y_metric_min = df.groupby(['model_name', 'n_input'])[metric].min()
        
        y0_upper, y0, y0_lower = [], [], []
        model0 = models[0]
        for number_samples in x:
            y0.append(y_metric_mean.loc[model0, number_samples])
            y0_upper.append(y_metric_max.loc[model0, number_samples])
            y0_lower.append(y_metric_min.loc[model0, number_samples])

        y1_upper, y1, y1_lower = [], [], []
        model1 = models[1]
        for number_samples in x:
            y1.append(y_metric_mean.loc[model1, number_samples])
            y1_upper.append(y_metric_max.loc[model1, number_samples])
            y1_lower.append(y_metric_min.loc[model1, number_samples])
        plot_filled_line(
            figs,
            x,
            i,
            Metrics[i],
            y0, y0_upper, y0_lower, model0, 
            y1, y1_upper, y1_lower, model1
        )
        
    
    figs.update_traces(mode='lines')
    # 背景色为透明
    figs.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
           color='black',  
        ),
        legend=dict(  # 图例放到正中间上方
            orientation='h',
            x=0.5,
            y=1.1,
            xanchor='center',
            yanchor='bottom',
        ),
    )
    # figs.show()
    pyo.plot(figs, filename=join(in_dir, 'experiment2.html'), auto_open=False)
    # format = 'jpeg'
    # figs.write_image(
    #     join(in_dir, f'experiment2.{format}'),
    #     format=format, 
    #     width=1520, 
    #     height=739, 
    #     scale=10,
    # )


if __name__ == "__main__":
    # 跑完了就不需要再用run生成实验数据
    # run()
    get_figure()
