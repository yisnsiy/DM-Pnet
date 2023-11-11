import sys
import os

from os.path import join, abspath, dirname
import pandas as pd
import numpy as np
import plotly.offline as pyo
import plotly.subplots as subplots
import plotly.graph_objects as go

from analysis.experiment3 import get_gene_and_pathway, get_pathway, get_dynamic_mask, get_mask, get_short_names
from config import RESULT_PATH, PATHWAY_PATH
from data_access.pathways.reactome import Reactome

out_dir = join(dirname(abspath(__file__)), 'experiment5')

def get_figure(
        top_trainable_mask_big, 
        top_trainable_mask, 
        new_connection
    ):

    figs = subplots.make_subplots(
        rows=1, 
        cols=3, 
        subplot_titles=("Overview", "Mixture", "Global training")
    )

    figs.add_trace(
        go.Heatmap(
            z=top_trainable_mask_big.values,
            # x=top_trainable_mask_big.columns,
            # y=top_trainable_mask_big.index,
            coloraxis="coloraxis",
        ),
        row=1,
        col=1,
    )
    figs.add_trace(
        go.Heatmap(
            z=top_trainable_mask.values,
            x=top_trainable_mask.columns,
            y=top_trainable_mask.index,
            coloraxis="coloraxis",
        ),
        row=1, 
        col=2,
    )
    figs.add_trace(
        go.Heatmap(
            z=new_connection.values,
            x=new_connection.columns,
            y=new_connection.index,
            coloraxis="coloraxis",
        ),
        row=1, 
        col=3,
    )

    figs.update_layout(
        coloraxis=dict(
            cmin=-1,
            cmax=1,
            colorscale="RdBu"
        ),
    )
    figs.update_xaxes(tickangle=70)

    # set style of subfigure 1 individually
    figs.update_xaxes(
        tickangle=60,
        row=1,
        col=1,
        tickvals=[i for i in range(20, 320, 20)]
    )
    figs.update_yaxes(
        row=1,
        col=1,
        tickvals=[i for i in range(20, 320, 20)]
    )
    counter_color = "rgb(121, 199, 88)"
    figs.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=20,
        y1=20,
        line=dict(
            color=counter_color,
            width=2,
            # dash='dash'
        ),
        row=1,
        col=1,
    )
    eps=0.5
    figs.add_shape(
        type="rect",
        x0=0-eps,
        y0=0-eps,
        x1=20-eps,
        y1=20-eps,
        line=dict(
            color=counter_color,
            width=2,
            # dash='dot'
        ),
        row=1,
        col=2,
    )
    pyo.plot(figs, filename=join(out_dir, 'experiment5.html'), auto_open=False)
    format = 'jpeg'
    figs.write_image(
        join(out_dir, f'experiment5.{format}'),
        format=format,
        width=1520,
        height=739,
        scale=10,
    )
    

def get_mixed_mask(dynamic_mask, mask):

    # make dynamic_mask's parts that equals maskde equals value of mask
    mixed_mask = dynamic_mask.mask(mask!=0, 0)
    mixed_mask = mixed_mask + mask
    return mixed_mask

def get_trainable_mask(genes, pathway):
    adjacent_matrix = pd.read_csv(join(RESULT_PATH, 'extracted/trainable_mask_1.csv'))
    adjacent_matrix.set_index(adjacent_matrix.columns[0], inplace=True)
    return adjacent_matrix.loc[genes, pathway]

def get_top_contribution_relationships(
        n_top, 
        trainable_mask:pd.DataFrame, 
        mask:pd.DataFrame
    ):
    
    # remove relation whose contribution is less.
    eps = 1e-2
    trainable_mask[
        (trainable_mask < eps) & 
        (trainable_mask > -eps)
    ] = 0

    tmp_mask = trainable_mask.mask(mask!=0, 0)  # remove found relationship.
    tmp_mask_np = tmp_mask.to_numpy()

    posi_nega = tmp_mask_np / np.abs(tmp_mask_np)  # keep positive and negative
    posi_nega[np.isnan(posi_nega)] = 1
    flattened = np.abs(tmp_mask_np).flatten()  # flatten matrix
    top_indices = np.argsort(flattened)[-n_top:]  # keep top n relations
    result = np.zeros_like(tmp_mask_np)
    result.flat[top_indices] = flattened[top_indices]  # restore matrix
    result  = result * posi_nega  # restore positive and negative
    
    top_mask_df = pd.DataFrame(
        data=result,
        index=mask.index,
        columns=mask.columns
    )

    return top_mask_df + trainable_mask.where(mask!=0, 0)


if __name__ == '__main__':
    n_top_gene, n_top_pathway = 305, 305
    n_top_contribution = 6000
    
    # get top 30 genes in lay1 and top 30 pathway in lay2.
    genes, tmp = get_gene_and_pathway(n_top_gene)
    tmp, pathway = get_gene_and_pathway(n_top_pathway)

    reactome = Reactome()

    # get mappint relation.
    name2id, id2name = get_pathway(reactome)

    # get dynamic mask.
    # dynamic_mask = get_dynamic_mask(genes, pathway)

    # get trainable mask
    trainable_mask_big = get_trainable_mask(genes, pathway)
    

    # get real mask by reactom dataset.
    mask_big = get_mask(trainable_mask_big, reactome)

    # test biggest values
    # tmp_mask_np = trainable_mask.mask(mask!=0, 0).to_numpy()
    # vector = tmp_mask_np.flatten()
    # print("Top 10 contribution of new relationship: ", np.sort(vector)[-10:])
    
    # tmp_mask_np = trainable_mask.where(mask!=0, 0).to_numpy()
    # vector = tmp_mask_np.flatten()
    # print("Minimum 10 contribution of existing relationship: ", np.sort(vector)[:10])
    # vector[vector < 0.21] = int(0)
    # vector[vector >= 0.21] = int(1)
    # vector = vector.astype(np.int32)
    # print(np.bincount(vector))

    # only keep new relationship between genne and pathway 
    # that has top n contribution.
    top_trainable_mask_big = get_top_contribution_relationships(
        n_top_contribution,
        trainable_mask_big,
        mask_big
    )
    n_small = 20
    trainable_mask = trainable_mask_big.iloc[0:n_small, 0:n_small]
    mask = mask_big.iloc[0:n_small, 0:n_small]

    top_trainable_mask = get_top_contribution_relationships(
        30,
        trainable_mask,
        mask,
    )

    mixed_mask = get_mixed_mask(top_trainable_mask, mask)

    # id to name.
    mixed_mask.rename(columns=id2name, inplace=True)
    short_name = get_short_names(mixed_mask.columns)
    mixed_mask.columns = short_name
    top_trainable_mask.columns = short_name
    mask.columns = short_name

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    trainable_mask.to_csv(join(out_dir, 'trainable_mask.csv'))
    mask.to_csv(join(out_dir, 'mask.csv'))
    top_trainable_mask.to_csv(join(out_dir, 'top_trainable_mask.csv'))
    mixed_mask.to_csv(join(out_dir, 'mixed_mask.csv'))

    # plot figure.
    get_figure(
        top_trainable_mask_big, 
        top_trainable_mask, 
        top_trainable_mask.mask(mask!=0, 0)
    )
    