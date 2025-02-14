import os

from pisa.constants import *


def process_params(params):
    dataset_params = params['dataset']
    training_params = params['training']
    eval_level = params['eval'].get('level', 'track')
    model_params = training_params['model']['params']

    # data specification
    dataset_spec = f'{dataset_params["name"]}_' \
                   f'minsess{dataset_params["min_sessions"]}_' \
                   f'samples_step{dataset_params["samples_step"]}'
    # model specification
    model_spec = gen_model_spec(training_params, model_params, eval_level)

    n_epoch = training_params["num_epochs"] if 'num_epochs' in training_params else 0
    training_params['model_dir'] = os.path.join(
        training_params['model_dir'],
        dataset_spec,
        f'nepoch{n_epoch}',
        model_spec)
    return training_params, model_params


def gen_model_spec(training_params, model_params,
                   eval_level='track'):
    model_name = training_params['model']['name']
    seqlen = model_params.get('seqlen', 0)
    # training specifications
    training_spec = f'lr{training_params["learning_rate"]}_' \
                    f'b{training_params["batch_size"]}_' \
                    f'op-{training_params["optimizer"].lower()}_' \
                    f'slen{seqlen}_' \
                    f'dim{training_params["embedding_dim"]}'
    dropout = model_params.get('dropout_rate', 0)
    if dropout > 0:
        training_spec = f'{training_spec}_dropout{dropout}'
    pretrained = model_params["pretrained"]
    if pretrained == "nopretrained":
        training_spec = f'{training_spec}_nopretrained'
    n_negatives = 1 if 'n_negatives' not in model_params \
        else model_params["n_negatives"]
    model_spec = f'{model_name}_{training_spec}_' \
                 f'l2emb{model_params["l2_emb"]}'
    # transformers blocks
    input_scale = model_params.get('input_scale', False)
    if input_scale is False:
        model_spec = f'{model_spec}_nonscale'
    causality = model_params["sab"].get('causality', True)
    if causality is not True:
        model_spec = f'{model_spec}_noncausal'
    model_spec = f'{model_spec}_nb{model_params["sab"]["num_blocks"]}_' \
                 f'nh{model_params["sab"]["num_heads"]}_' \
                 f'neg{n_negatives}'
    # ACTR
    model_spec = f'{model_spec}_ACTR-bll+{model_params["actr"]["bll"]["type"]}'
    activate_spread = model_params['actr']['spread']['activate']
    if activate_spread:
        model_spec = f'{model_spec}-spr'
        hop = model_params['actr']['spread'].get('hop', 1)
        if hop > 1:
            model_spec = f'{model_spec}+{hop}hop'
        n_last_sess = model_params['actr']['spread'].get(
            'n_last_sess', 1)
        model_spec = f'{model_spec}+last{n_last_sess}sess'
    activate_pm = model_params['actr']['pm']['activate']
    if activate_pm:
        model_spec = f'{model_spec}-pm'
    flatten_actr = model_params.get('flatten_actr', 1)
    if flatten_actr != 1:
        model_spec = f'{model_spec}-flat{flatten_actr}'

    lbda_pos = model_params.get('lbda_pos', 0.)
    if lbda_pos > 0:
        model_spec = f'{model_spec}_lbda-pos{lbda_pos}'
    # lambda multi task
    lbda_task = model_params.get('lbda_task', 0.)
    model_spec = f'{model_spec}_lbda-task{lbda_task}'
    # lambda long-short term
    lbda_ls = model_params.get('lbda_ls', 0.)
    model_spec = f'{model_spec}_lbdals{lbda_ls}'
    # num of favorites for long-term
    num_favs = model_params.get('num_favs', 0)
    model_spec = f'{model_spec}_numfavs{num_favs}'
    # negative sampling
    negsam_strategy = model_params.get('negsam_strategy', NEGSAM_UNIFORM)
    if negsam_strategy == NEGSAM_POP:
        neg_alpha = model_params.get('neg_alpha', 1.0)
        model_spec = f'{model_spec}_negsam-pop{neg_alpha}'
    if model_name == 'pisa_art':
        flatten_actr_art = model_params.get('flatten_actr_art', 1)
        num_favs_art = model_params.get('num_favs_art', 5)
        model_spec = f'{model_spec}_nartfavs{num_favs_art}'
        lbda_art_loss = model_params.get('lbda_art_loss', 0.5)
        lbda_narts = model_params.get('lbda_narts', 0)
        lbda_task_art = model_params.get('lbda_task_art', 0.)
        lbda_pos_art = model_params.get('lbda_pos_art', 0.)
        lbda_ls_art = model_params.get('lbda_ls_art', 0.)
        lbda_cross = model_params.get('lbda_cross', 0.)
        model_spec = (f'{model_spec}_lbdart-{flatten_actr_art}+{lbda_ls_art}+{lbda_task_art}+'
                      f'{lbda_pos_art}+{lbda_narts}+{lbda_art_loss}+{lbda_cross}')
        model_spec = f'{model_spec}_ev-{eval_level}_2br_+at'
    return model_spec
