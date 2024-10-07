import os.path
import time
import sys
from tqdm import tqdm
import numpy as np
sys.path.append(".")
sys.path.append("..")

from .loaders.loader import DataLoader
from .logging import get_logger
from data_misc.lfm1b.utils import mat_to_dict


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
        self.logger = get_logger()

    def fit(self, data):
        """
        Training model
        :param data:
        :return:
        """
        # create data loaders for train & validation
        training_params = self.params['training']
        model_params = training_params['model']
        random_seed = self.params['dataset'].get('random_state', 2024)

        train_user_items = mat_to_dict(data['train_interactions'])
        valid_user_items = mat_to_dict(data['test_interactions'])
        total_user_items = {k: v.union(valid_user_items[k])
                            for k, v in train_user_items.items()}

        metrics_path = os.path.join(self.model_dir, 'metrics.csv')
        if os.path.isfile(metrics_path):
            os.remove(metrics_path)

        best_valid_loss = 1e100
        best_ep = -1
        with open(metrics_path, 'w') as f:
            header = 'epoch,lr,train_loss,val_loss'
            f.write(f'{header}\n')
        for ep in range(self.n_epochs):
            start_time = time.time()
            train_dataloader = DataLoader(interactions=data['train_interactions'],
                                          n_users=self.model.n_users,
                                          n_items=self.model.n_items,
                                          batch_size=training_params['batch_size'],
                                          n_negatives=model_params['params']['n_negatives'],
                                          random_seed=random_seed,
                                          user_items=None)
            # calculate train loss
            train_loss = self._get_epoch_loss(train_dataloader, ep)
            valid_dataloader = DataLoader(interactions=data['test_interactions'],
                                          n_users=self.model.n_users,
                                          n_items=self.model.n_items,
                                          batch_size=training_params['batch_size'],
                                          n_negatives=model_params['params']['n_negatives'],
                                          random_seed=random_seed,
                                          user_items=total_user_items)

            valid_loss = self._get_epoch_loss(valid_dataloader, ep)
            if valid_loss < best_valid_loss or ep == 1:
                save_model = True
                best_valid_loss = valid_loss
                best_ep = ep
            else:
                save_model = False
            logged_message = self._get_message(
                ep, self.model.learning_rate, train_loss, valid_loss, start_time)
            self.logger.info(', '.join(logged_message))
            metric_message = self._get_message(
                ep, self.model.learning_rate,
                train_loss, valid_loss, start_time,
                logged=False)
            # f.write(','.join(metric_message) + '\n')
            # f.flush()
            if save_model:
                save_path = f'{self.model_dir}/' \
                            f'{self.model.__class__.__name__.lower()}' \
                            f'-epoch_{ep}'
                self.model.save(save_path=save_path, global_step=ep)
        self.logger.info(f'Best validation : {best_valid_loss}, '
                         f'on epoch {best_ep}')

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
                     train_loss, valid_loss, start_time, logged=True):
        duration = int(time.time() - start_time)
        ss, duration = duration % 60, duration // 60
        mm, hh = duration % 60, duration // 60
        if logged is True:
            message = [f'Epoch #{ep}:',
                       f'LR {learning_rate:9.7f}',
                       f'Tr-loss {train_loss:8.5f}',
                       f'Val-loss {valid_loss:8.5f}',
                       f'Dur:{hh:0>2d}h{mm:0>2d}m{ss:0>2d}s']
        else:
            message = [f'{ep}:',
                       f'{learning_rate:9.7f}',
                       f'{train_loss:8.5f}',
                       f'{valid_loss:8.5f}']
        return message
