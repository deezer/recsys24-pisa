import os
import numpy as np
import tensorflow as tf

from pisa.logging import get_logger
from pisa.utils.params import process_params
from pisa.models import ModelFactory
from pisa.data.datasets import dataset_factory
from pisa.train.trainer import Trainer


def entrypoint(params):
    """ Command entrypoint
    :param params: Deserialized JSON configuration file
                   provided in CLI args.
    """
    logger = get_logger()
    params['command'] = 'train'
    tf.compat.v1.disable_eager_execution()

    # process params
    training_params, model_params = process_params(params)

    # create model directory if not exist
    if not os.path.exists(training_params['model_dir']):
        os.makedirs(training_params['model_dir'], exist_ok=True)
    logger.info(training_params['model_dir'])

    params['command'] = 'train'
    # load datasets
    data = dataset_factory(params=params)
    training_params['n_artists'] = data['n_artists']
    logger.info(f'Number of users: {data["n_users"]}')
    logger.info(f'Number of items: {data["n_items"]}')
    logger.info(f'Number of artists: {data["n_artists"]}')

    if model_params["pretrained"] == "nopretrained":
        pretrained_embs = None
    else:
        pretrained_embs = {
            'item_embeddings': np.array(list(data['track_embeddings'].values())),
            'artist_embeddings': None
        }
    # start model training
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    with tf.compat.v1.Session(config=sess_config) as sess:
        # create model
        model = ModelFactory.generate_model(sess=sess,
                                            params=training_params,
                                            n_users=data['n_users'],
                                            n_items=data['n_items'],
                                            pretrained_embs=pretrained_embs)
        sess.run(tf.compat.v1.global_variables_initializer())
        # create a trainer to train model
        trainer = Trainer(sess, model, params)
        logger.info('Start model training')
        trainer.fit(data=data)
        logger.info('Model training done')
