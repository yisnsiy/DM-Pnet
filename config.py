from os.path import join, realpath, dirname
from copy import deepcopy

# path param
BASE_PATH = dirname(realpath(__file__))
DATA_PATH = join(BASE_PATH, 'data')
GENE_PATH = join(DATA_PATH, 'genes')
PATHWAY_PATH = join(DATA_PATH, 'pathways')
PROSTATE_DATA_PATH = join(DATA_PATH, 'prostate')
REACTOM_PATHWAY_PATH = join(PATHWAY_PATH, 'Reactome')
LOG_PATH = join(BASE_PATH, 'logs')
RESULT_PATH = join(BASE_PATH, 'results')

# state param
debug = False  # is it debug mode
local = False  # is it remote running
save_res = True  # whether to save evaluation of model or not in result path
interpret = True
gpu_id = 0

# data param
selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes.csv'
data_params = {
    'id': 'ALL',
    'type': 'prostate_paper',
    'params': {
        'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
        'drop_AR': False,
        'cnv_levels': 3,
        'mut_binary': True,
        'balanced_data': False,
        'combine_type': 'union',  # intersection
        'use_coding_genes_only': True,
        'selected_genes': selected_genes,
        'training_split': 0,
    },
}

# nn param
n_hidden_layers = 5
base_dropout = 0.5
loss_weights = [2, 7, 20, 54, 148, 400]


# If the mask of model is trained locally, then set trainable_mask to true 
# and full_train to false.

# If the mask of model is trained globally, then set trainable_mask to true 
# and full_train to true.


models_params = {
    'id': 'pnet',
    'type': 'nn',
    'model_params': {
        'trainable_mask': True,
        'full_train': False,
        'use_bias': True,
        'penalty': 0.001,
        'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),
        'optimizer': 'Adam',
        'activation': 'tanh',
        'data_params': data_params,
        'add_unk_genes': False,
        'shuffle_genes': False,
        'n_hidden_layers': n_hidden_layers,
    },
    'fitting_params': dict(samples_per_epoch=10,
                           select_best_model=False,
                           verbose=2,
                           epoch=300,
                           shuffle=True,
                           batch_size=50,
                           loss_weights=loss_weights,
                           save_gradient=False,
                           class_weight='auto',
                           n_outputs=n_hidden_layers + 1,
                           reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
                           lr=0.001,
                           max_f1=False),
    'feature_importance': {
        'method_name': 'deeplift',  # integratedgradients
        'baseline': 'zero',  # zero or mean
    },
}

parameters = []
pnet_deeplift = deepcopy(models_params)
pnet_deeplift['id'] = 'pnet_deeplift'
parameters.append(pnet_deeplift)

# pnet_integratedgradients = deepcopy(models_params)
# pnet_integratedgradients['id'] = 'pnet_integratedgradients'
# pnet_integratedgradients['feature_importance']['method_name'] = 'integratedgradients'
# parameters.append(pnet_integratedgradients)

