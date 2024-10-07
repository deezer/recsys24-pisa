import os
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict

from pisa import PisaError
from pisa.logging import get_logger
from pisa.utils.params import process_params, gen_model_spec
from pisa.data.datasets import dataset_factory
from pisa.models import ModelFactory
from pisa.data.loaders import dataloader_factory
from pisa.eval.evaluator import Evaluator
from pisa.constants import *


def entrypoint(params):
    """ Command entrypoint
    :param params: Deserialized JSON configuration file
                   provided in CLI args.
    """
    logger = get_logger()
    tf.compat.v1.disable_eager_execution()
    # process params
    training_params, model_params = process_params(params)
    dataset_params = params['dataset']
    training_params = params['training']
    model_name = training_params['model']['name']
    cache_path = params['cache']['path']
    eval_params = params['eval']
    min_sessions = dataset_params.get('min_sessions', 500)
    cache_path = os.path.join(cache_path,
                              dataset_params['name'],
                              f'min{min_sessions}sess')
    embedding_dim = training_params.get('embedding_dim', 128)
    logger.info(training_params['model_dir'])
    params['command'] = 'eval'

    # load datasets
    data = dataset_factory(params=params)
    logger.info(f'Number of users: {data["n_users"]}')
    logger.info(f'Number of items: {data["n_items"]}')
    # get user artists
    userart_path = os.path.join(
        cache_path,
        f'user_artists_recenthist{dataset_params["recent_history"]}_'
        f'{data["data_split"]["valid"]}v_{data["data_split"]["test"]}t.pkl')
    data['user_artists'] = _get_user_artists(user_tracks=data['user_tracks'],
                                             track_art_dict=data['track_art'],
                                             userart_path=userart_path)
    training_params['n_artists'] = data['n_artists']

    # reco result path
    reco_parent_path = os.path.join(
        cache_path,
        f'result_recenthist{dataset_params["recent_history"]}_'
        f'{data["data_split"]["valid"]}v_{data["data_split"]["test"]}t')
    reco_parent_path = os.path.join(reco_parent_path, 'norm') \
        if training_params['normalize_embedding'] is True \
        else os.path.join(reco_parent_path, 'unnorm')
    if not os.path.exists(reco_parent_path):
        os.makedirs(reco_parent_path, exist_ok=True)

    pretrained = model_params["pretrained"]
    if pretrained == "nopretrained":
        pretrained_embs = None
    else:
        pretrained_embs = {
            'item_embeddings': np.array(list(data['track_embeddings'].values()))}
    num_favs = model_params.get('num_favs', 0)

    # start model eval
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    with tf.compat.v1.Session(config=sess_config) as sess:
        model = ModelFactory.generate_model(sess=sess,
                                            params=training_params,
                                            n_users=data['n_users'],
                                            n_items=data['n_items'],
                                            command='eval',
                                            pretrained_embs=pretrained_embs)
        # generate users for test
        scores = defaultdict(list)
        batch_size = eval_params.get('batch_size', 256)
        num_scored_users = eval_params.get('n_users')
        random_seeds = eval_params.get('random_seeds')
        seqlen = model_params.get('seqlen', 20)
        eval_mode = eval_params.get('mode', 'mean')
        for step, seed in enumerate(random_seeds):
            logger.info(f'EVALUATION for #{step + 1} COHORT')
            test_dataloader = dataloader_factory(
                data=data,
                batch_size=batch_size,
                seqlen=seqlen,
                mode='test',
                num_scored_users=num_scored_users,
                model_name=model_name,
                embedding_dim=embedding_dim,
                random_seed=seed,
                num_favs=num_favs,
                command='eval')
            # Evaluate
            ref_user_items = test_dataloader.get_ref_user_items()
            evaluator = Evaluator(config=eval_params,
                                  ref_user_items=ref_user_items,
                                  user_tracks=data['user_tracks']['train'],
                                  track_popularities=data['glob_track_popularities'],
                                  track_art=data['track_art'],
                                  user_artists=data['user_artists'],
                                  mode='test')
            # model specification
            model_spec = gen_model_spec(training_params, model_params)

            reco_outpath = os.path.join(reco_parent_path,
                                        f'{model_spec}_seed{seed}_top{evaluator.max_k}.pkl')
            reco_items = _recommend(dataloader=test_dataloader,
                                    model=model,
                                    top_n=evaluator.max_k,
                                    reco_outpath=reco_outpath,
                                    seed=seed)
            curr_scores = evaluator.eval(reco_items, mode=eval_mode)
            for metric, val in curr_scores.items():
                scores[metric].append(val)
        # Display final result
        message = ['RESULTS:']
        for metric, score_arr in scores.items():
            message.append(f'{metric}: {np.mean(score_arr):8.5f} +/- {np.std(score_arr):8.5f}')
        logger.info('\n'.join(message))


def _recommend(dataloader, model, reco_outpath,
               top_n=10, seed=0):
    """
    Args:
        dataloader:
        model:
        reco_outpath:
        top_n:
        seed:
    Returns:
    """
    logger = get_logger()
    if not os.path.exists(reco_outpath):
        n_batches = dataloader.get_num_batches()
        reco_items = {}
        # for each batch
        for _ in tqdm(range(1, n_batches), desc=f'Evaluating cohort generated '
                                                f'with random seed = {seed}...'):
            feed_dict = {}
            # get batch data
            batch_data = dataloader.next_batch()
            feed_dict['model_feed'] = model.build_feedict(batch_data,
                                                          is_training=False)
            feed_dict['item_ids'] = dataloader.item_ids
            feed_dict['user_ids'] = batch_data[-1]

            # get prediction from model
            batch_reco_items = model.predict(feed_dict, top_n=top_n)
            for uid, items in batch_reco_items.items():
                reco_items[uid] = items
        logger.info(f'Write prediction to {reco_outpath}')
        pickle.dump(reco_items, open(reco_outpath, 'wb'))
    else:
        logger.info(f'Load prediction from {reco_outpath}')
        reco_items = pickle.load(open(reco_outpath, 'rb'))
    return reco_items


def _get_user_artists(user_tracks, track_art_dict, userart_path):
    logger = get_logger()
    if not os.path.exists(userart_path):
        logger.info('Extract user artists...')
        res = defaultdict(set)
        train_user_tracks = user_tracks['train']
        valid_user_tracks = user_tracks['valid']
        logger.info(' + train user artists...')
        for uid, track_ids in train_user_tracks.items():
            for tid in track_ids:
                res[uid].add(track_art_dict[tid])
        logger.info(' + valid user artists...')
        for uid, track_ids_dict in valid_user_tracks.items():
            for track_ids in track_ids_dict.values():
                for tid in track_ids:
                    res[uid].add(track_art_dict[tid])
        pickle.dump(res, open(userart_path, 'wb'))
    else:
        logger.info(f'Load user artists {userart_path}')
        res = pickle.load(open(userart_path, 'rb'))
    return res
