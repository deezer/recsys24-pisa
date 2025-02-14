import os
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from pisa.logging import get_logger
from pisa.data.loaders import dataloader_factory
from pisa.eval.evaluator import Evaluator
from pisa.constants import *


class Trainer:
    """
    Trainer is responsible to estimate paramaters
    for a given model
    """
    def __init__(self, sess, model, params):
        """
        Initialization a trainer. The trainer will be responsible
        to train the model
        :param sess: global session
        :param model: model to be trained
        :param params: hyperparameters for training
        """
        self.sess = sess
        self.model = model
        self.params = params
        self.model_dir = params['training']['model_dir']
        self.n_epochs = self.params['training'].get('num_epochs', 20)
        self.n_valid_users = self.params['training'].get('n_valid_users',
                                                         1000)
        self.num_favs = params['training']['model']['params'].get(
            'num_favs', 20)
        self.num_favs_art = params['training']['model']['params'].get(
            'num_favs_art', 5)
        # primary metric used to optimize model
        self.prim_metric = self.params['eval']['metrics'].get('primary', 'ndcg')
        self.eval_level = self.params['eval'].get('level', 'track')
        self.logger = get_logger()

    def fit(self, data):
        """
        Training model
        :param data:
        :return:
        """
        # create data loaders for train & validation
        training_params = self.params['training']
        model_name = training_params['model']['name']
        model_params = training_params['model']['params']
        random_seed = self.params['dataset'].get('random_state', 2023)
        eval_params = self.params['eval']
        negsam_strategy = model_params.get('negsam_strategy', NEGSAM_UNIFORM)
        neg_alpha = model_params.get('neg_alpha', 1.0)
        metrics_path = '{}/metrics.csv'.format(self.model_dir)
        if os.path.isfile(metrics_path):
            os.remove(metrics_path)
        best_valid_score = -1.0
        best_ep = -1
        seqlen = model_params.get('seqlen', 50)
        art_in_ratio = model_params.get('art_in_ratio', 0.)
        if self.eval_level == 'track':
            user_tracks = data['user_tracks']['train']
        else:
            user_tracks = defaultdict(set)
            for uid, tracks in data['user_tracks']['train'].items():
                user_tracks[uid] = set([data['track_art'][tid] for tid in tracks])
        with open(metrics_path, 'w') as f:
            header = 'epoch,lr,train_loss,val_loss,ndcg_all@10,recall_all@10,'\
                     'ndcg_rep@10,recall_rep@10,ndcg_exp@10,recall_exp@10'
            f.write(f'{header}\n')
            # for each epoch
            for ep in range(self.n_epochs):
                start_time = time.time()
                train_dataloader = dataloader_factory(
                    data=data,
                    batch_size=training_params['batch_size'],
                    seqlen=seqlen,
                    embedding_dim=self.model.embedding_dim,
                    model_name=model_name,
                    mode='train',
                    random_seed=random_seed,
                    num_favs=self.num_favs,
                    num_favs_art=self.num_favs_art,
                    negsam_strategy=negsam_strategy,
                    neg_alpha=neg_alpha,
                    art_in_ratio=art_in_ratio)
                # calculate train loss
                train_loss = self._get_epoch_loss(
                    train_dataloader, ep)
                valid_dataloader = dataloader_factory(
                    data=data,
                    batch_size=eval_params['batch_size'],
                    seqlen=seqlen,
                    mode='valid',
                    num_scored_users=self.n_valid_users,
                    model_name=model_name,
                    embedding_dim=self.model.embedding_dim,
                    random_seed=random_seed,
                    num_favs=self.num_favs,
                    num_favs_art=self.num_favs_art,
                    eval_level=self.eval_level)
                ref_user_items = valid_dataloader.get_ref_user_items()
                evaluator = Evaluator(config=eval_params,
                                      ref_user_items=ref_user_items,
                                      user_tracks=user_tracks,
                                      track_popularities=data['glob_track_popularities'],
                                      track_art=data['track_art'],
                                      user_artists=None,
                                      num_sess_test=data['data_split']['valid'],
                                      mode='valid')
                # Get recommendation
                reco_items = self.recommend(dataloader=valid_dataloader,
                                            model=self.model,
                                            top_n=evaluator.max_k)
                # Evaluate results
                curr_scores = evaluator.eval(reco_items)
                score = curr_scores[f'{self.prim_metric}_all@{evaluator.min_k}']
                if best_valid_score < score or ep == 1:
                    save_model = True
                    best_valid_score = score
                    best_ep = ep
                else:
                    save_model = False
                logged_message = self._get_message(
                    ep, self.model.learning_rate,
                    train_loss, curr_scores, start_time)
                self.logger.info(', '.join(logged_message))
                metric_message = self._get_message(
                    ep, self.model.learning_rate,
                    train_loss, curr_scores, start_time,
                    logged=False)
                f.write(','.join(metric_message) + '\n')
                f.flush()
                if save_model:
                    save_path = f'{self.model_dir}/' \
                                f'{self.model.__class__.__name__.lower()}' \
                                f'-epoch_{ep}'
                    self.model.save(save_path=save_path, global_step=ep)
            self.logger.info(f'Best validation : {best_valid_score}, '
                             f'on epoch {best_ep}')

    @classmethod
    def recommend(cls, dataloader, model, top_n=10, seed=0):
        n_batches = dataloader.get_num_batches()
        reco_items = {}
        # for each batch
        for _ in tqdm(range(1, n_batches), desc=f'Evaluating '
                                                f'with random seed = {seed}...'):
            feed_dict = {}
            # get batch data
            batch_data = dataloader.next_batch()
            feed_dict['model_feed'] = model.build_feedict(batch_data,
                                                          is_training=False)
            feed_dict['user_ids'] = batch_data[-1]
            feed_dict['item_ids'] = dataloader.item_ids
            # get prediction from model
            batch_reco_items = model.predict(feed_dict, top_n=top_n)
            for uid, items in batch_reco_items.items():
                reco_items[uid] = items
        return reco_items

    def _get_epoch_loss(self, dataloader, epoch_id):
        """
        Forward pass for an epoch
        :param dataloader:
        :param epoch_id:
        :return:
        """
        n_batches = dataloader.get_num_batches()
        losses = []
        desc = f'Optimizing epoch #{epoch_id}'
        # for each batch
        for _ in tqdm(range(1, n_batches), desc=f'{desc}...'):
            # get batch data
            batch_data = dataloader.next_batch()
            batch_loss = self._get_batch_loss(batch=batch_data)
            if isinstance(batch_loss, np.ndarray):
                batch_loss = np.mean(batch_loss)
            if not np.isinf(batch_loss) and not np.isnan(batch_loss):
                losses.append(batch_loss)
        loss = np.mean(losses, axis=0)
        return loss

    def _get_batch_loss(self, batch):
        """
        Forward pass for a batch
        :param batch:
        :return:
        """
        feed_dict = self.model.build_feedict(batch, is_training=True)
        _, loss = self.sess.run(
            [self.model.train_ops, self.model.loss],
            feed_dict=feed_dict)
        return loss

    @classmethod
    def _get_message(cls, ep, learning_rate,
                     train_loss, score, start_time, logged=True):
        duration = int(time.time() - start_time)
        ss, duration = duration % 60, duration // 60
        mm, hh = duration % 60, duration // 60
        if logged is True:
            message = [f'Epoch #{ep}',
                       f'LR {learning_rate:6.5f}',
                       f'Tr-Loss {train_loss:7.5f}']
            for k, v in score.items():
                if 'all' in k:
                    message.append(f'Val {k} {v:7.5f}')
            message.append(f'Dur:{hh:0>2d}h{mm:0>2d}m{ss:0>2d}s')
        else:
            message = [f'{ep}:',
                       f'{learning_rate:6.7f}',
                       f'{train_loss:7.5f}']
            for _, v in score.items():
                message.append(f'{v:7.5f}')
        return message
