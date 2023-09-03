import sys
import os

from os.path import join, abspath, dirname
import pandas as pd
import numpy as np
import plotly.offline as pyo
import plotly.graph_objects as go

from analysis.experiment3 import get_gene_and_pathway, get_pathway, get_dynamic_mask, get_mask, get_short_names
from config import RESULT_PATH, PATHWAY_PATH
from data_access.pathways.reactome import Reactome

out_dir = join(dirname(abspath(__file__)), 'experiment5')

def get_figure(mixed_mask):
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=mixed_mask.values,
        x=mixed_mask.columns,
        y=mixed_mask.index,
        coloraxis='coloraxis',
    ))
    fig.update_layout(coloraxis=dict(
        cmin=-1,
        cmax=1,
        colorscale="RdBu"
    ))
    fig.update_xaxes(tickangle=70)
    pyo.plot(fig, filename=join(out_dir, 'experiment5.html'), auto_open=False)
    format = 'jpeg'
    fig.write_image(
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



    # top_mask = tmp_mask.stack().nlargest(n_top)
    # top_mask = top_mask.unstack().reindex(mask.index)
    # top_mask.fillna(0, inplace=True)
    return top_mask_df + trainable_mask.where(mask!=0, 0)


if __name__ == '__main__':
    n_top_gene, n_top_pathway = 20, 30
    n_top_contribution = 20
    
    # get top 30 genes in lay1 and top 30 pathway in lay2.
    genes, tmp = get_gene_and_pathway(n_top_gene)
    tmp, pathway = get_gene_and_pathway(n_top_pathway)

    reactome = Reactome()

    # get mappint relation.
    name2id, id2name = get_pathway(reactome)

    # get dynamic mask.
    # dynamic_mask = get_dynamic_mask(genes, pathway)

    # get trainable mask
    trainable_mask = get_trainable_mask(genes, pathway)
    

    # get real mask by reactom dataset.
    mask = get_mask(trainable_mask, reactome)

    # only keep new relationship between genne and pathway 
    # that has top n contribution.
    top_trainable_mask = get_top_contribution_relationships(
        n_top_contribution, 
        trainable_mask, 
        mask
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
    get_figure(mixed_mask)
    