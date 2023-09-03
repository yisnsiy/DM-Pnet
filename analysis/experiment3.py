import sys
import os
sys.path.append('/home/pnet_based_on_pytorch')
from os.path import join, abspath, dirname
import pandas as pd
import numpy as np
from copy import deepcopy
import plotly.offline as pyo
import plotly.subplots as subplots
import plotly.graph_objects as go

from config import RESULT_PATH, PATHWAY_PATH
from data_access.pathways.reactome import Reactome

out_dir = join(dirname(abspath(__file__)), 'experiment3')


def get_gene_and_pathway(n_top):
    genes = pd.read_csv(join(RESULT_PATH, 'extracted/gradient_importance_1.csv'))
    pathway = pd.read_csv(join(RESULT_PATH, 'extracted/gradient_importance_2.csv'))

    genes.columns = ['gene', 'coef']
    pathway.columns = ['pathway', 'coef']
    genes = genes.sort_values(by='coef', ascending=False).head(n_top)
    top_genes =  genes['gene'].tolist()
    pathway =  pathway.sort_values(by='coef', ascending=False).head(n_top)
    top_pathway = pathway['pathway'].tolist()
    
    return top_genes, top_pathway

def get_short_names(all_node_labels):
        df = pd.read_excel(join(PATHWAY_PATH, 'pathways_short_names.xlsx'), index_col=0)
        mapping_dict = {}
        for k, v in zip(df['Full name'].values, df['Short name (Eli)'].values):
            mapping_dict[k] = str(v)

        all_node_labels_short = []
        for l in all_node_labels:
            short_name = l
            if l in mapping_dict.keys() and not mapping_dict[l] == 'nan':
                short_name = mapping_dict[l]

            if 'others' in short_name:
                short_name = 'residual'
            if 'root' in short_name:
                short_name = 'outcome'

            all_node_labels_short.append(short_name)
        return all_node_labels_short

def get_pathway(reactome):
    df = reactome.pathway_names
    df.drop_duplicates()
    id2name = df.set_index('reactome_id')['pathway_name'].to_dict()
    name2id = df.set_index('pathway_name')['reactome_id'].to_dict()
    return name2id, id2name

def get_dynamic_mask(genes, pathway):
    adjacent_matrix = pd.read_csv(join(RESULT_PATH, 'extracted/link_weights_1.csv'))
    adjacent_matrix.set_index(adjacent_matrix.columns[0], inplace=True)
    return adjacent_matrix.loc[genes, pathway]

def get_mask(dynamic_mask, reactome):
    mask = deepcopy(dynamic_mask)  # gene * pathway
    mask.loc[:, :] = 0

    # read gmt file containing relation between genen and pathway.
    df = reactome.pathway_genes  # pathway * gene
    genes = mask.index.tolist()
    pathway = mask.columns.tolist()
    df.set_index('group', inplace=True)
    df = df.loc[pathway]
    for index, row in df.iterrows():
        pathway = index
        gene = row['gene']
        if gene in genes:
            mask.loc[gene, pathway] += 1
    return mask


def get_figure(df_dm_pnet, df_pnet):
    df_connective = deepcopy(df_dm_pnet)
    df_connective.loc[:, :] = 1

    figs = subplots.make_subplots(
        rows=1, 
        cols=3, 
        subplot_titles=("Full connection", "Pnet", "DM-Pnet")
    )

    figs.add_trace(
        go.Heatmap(
            z=df_connective.values,
            x=df_connective.columns,
            y=df_connective.index,
            coloraxis="coloraxis",
        ),
        row=1,
        col=1,
    )
    figs.add_trace(
        go.Heatmap(
            z=df_pnet.values,
            x=df_pnet.columns,
            y=df_pnet.index,
            coloraxis="coloraxis",
        ),
        row=1, 
        col=2,
    )
    figs.add_trace(
        go.Heatmap(
            z=df_dm_pnet.values,
            x=df_dm_pnet.columns,
            y=df_dm_pnet.index,
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
    pyo.plot(figs, filename=join(out_dir, 'experiment3.html'), auto_open=False)
    # format = 'jpeg'
    # figs.write_image(
    #     join(out_dir, f'experiment3.{format}'),
    #     format=format, 
    #     width=1520, 
    #     height=739, 
    #     scale=10,
    # )
    

if __name__ == "__main__":
    n_top = 20

    # get top 20 genes in lay 1 and top 20 pathway in lay 2.
    genes, pathway = get_gene_and_pathway(n_top)

    reactome = Reactome()

    # get mapping relation.
    name2id, id2name = get_pathway(reactome)

    # get dynamic mask.
    dynamic_mask = get_dynamic_mask(genes, pathway)

    # get real mask by reactom dataset.
    mask = get_mask(dynamic_mask, reactome)

    # id to name.
    dynamic_mask.rename(columns=id2name, inplace=True)
    short_name = get_short_names(dynamic_mask.columns)
    dynamic_mask.columns = short_name
    # mask.rename(columns=id2name, inplace=True)
    mask.columns = short_name

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dynamic_mask.to_csv(join(out_dir, 'dynamic_mask.csv'))
    mask.to_csv(join(out_dir, 'mask.csv'))

    # plot figure.
    get_figure(dynamic_mask, mask)

