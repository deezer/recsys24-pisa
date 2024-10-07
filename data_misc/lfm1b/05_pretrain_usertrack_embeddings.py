import os.path
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf

from utils import *
from triplet.model import Triplet
from triplet.trainer import Trainer


params = load_configuration('configs/lfm1b_pretrain_usertrack.json')
dataset_params = params['dataset']
training_params = params['training']
cache_params = params['cache']
model_params = training_params['model']
cache_path = os.path.join('cache', dataset_params['name'],
                          f'min{dataset_params["min_sessions"]}sess')
cache_path = os.path.join(cache_path, model_params['name'])
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

model_dir = training_params.get('model_dir', 'exp/model')
model_dir = os.path.join(model_dir, model_params['name'],
                         f'lr{training_params["learning_rate"]}_'
                         f'nepoch{training_params["num_epochs"]}_'
                         f'margin{model_params["params"]["margin"]}_'
                         f'clip{model_params["params"]["clip_norm"]}_'
                         f'neg{model_params["params"]["n_negatives"]}_noL2reg')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
params['training']['model_dir'] = model_dir

train_interactions_name = cache_params['train_interactions']
test_interactions_name = cache_params['test_interactions']
entities_name = cache_params.get('entities', 'entities')

train_interactions_path = os.path.join(cache_path,
                                       f'{train_interactions_name}.npz')
test_interactions_path = os.path.join(cache_path,
                                      f'{test_interactions_name}.npz')
entities_path = os.path.join(cache_path,
                             f'{entities_name}.npz')
logger = get_logger()
logger.info('Load interaction matrices')
train_interactions = sp.load_npz(train_interactions_path)
test_interactions = sp.load_npz(test_interactions_path)
entity_ids = np.load(entities_path, allow_pickle=True)
user_ids = entity_ids['user_ids']
item_ids = entity_ids['item_ids']
data = {
    'train_interactions': train_interactions,
    'test_interactions': test_interactions
}

COMMAND = 'eval'  # or 'eval' to get the perf & track embeddings

# start model training
sess_config = tf.compat.v1.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.allow_soft_placement = True
with tf.compat.v1.Session(config=sess_config) as sess:
    model = Triplet(sess=sess, params=params['training'],
                    n_users=len(user_ids), n_items=len(item_ids))
    if COMMAND == 'train':
        model.build_graph()
        sess.run(tf.compat.v1.global_variables_initializer())
        # create a trainer to train model
        trainer = Trainer(sess, model, params)
        logger.info('Start model training')
        trainer.fit(data=data)
        logger.info('Model training done')
    else:
        track_emb_path = os.path.join(
            'exp/data/lfm1b',
            'normalized_pretrained_user_track_embeddings_recent365days_'
            'histmin365days_unstreams1000_instreams1500_nsess50.pkl')
        if not os.path.exists(track_emb_path):
            model.restore()
            track_embeddings = model.get_embeddings()
            track_embeddings = np.array(track_embeddings)
            final_embeddings = {}
            for idx, tid in enumerate(item_ids):
                final_embeddings[tid] = track_embeddings[idx]
            pickle.dump(final_embeddings, open(track_emb_path, 'wb'))
        else:
            track_embeddings = pickle.load(open(track_emb_path, 'rb'))
        logger.info(f'Number of items: {len(track_embeddings)}')