import numpy as np
import sys

from data_misc.lfm1b.utils import mat_to_dict
sys.path.append(".")
from .sampler import Sampler


class DataLoader:
    """
    DataLoader is responsible for train/valid
    batch data generation
    """
    def __init__(self, interactions, n_users, n_items, batch_size,
                 user_items=None, n_negatives=1, random_seed=2024, **kwargs):
        self.interactions = interactions
        if user_items is None:
            self.user_items = mat_to_dict(self.interactions, criteria=None)
        else:
            self.user_items = user_items
        self.n_users = n_users
        self.n_items = n_items
        self.batch_size = batch_size
        self.n_negatives = n_negatives
        self.random_seed = random_seed
        self.kwargs = kwargs
        n_interactions = self.interactions.count_nonzero()
        self.n_batches = int(n_interactions / self.batch_size)
        if self.n_batches * self.batch_size < n_interactions:
            self.n_batches += 1
        self.current_batch_idx = 0
        # positive user item pairs
        self.user_pos_item_pairs = np.asarray(self.interactions.nonzero()).T
        rng = np.random.RandomState(self.random_seed)
        rng.shuffle(self.user_pos_item_pairs)
        self.sampler = Sampler(user_items=self.user_items,
                               n_items=self.interactions.shape[1],
                               n_negatives=n_negatives)

    def next_batch(self):
        if self.current_batch_idx == self.n_batches:
            self.current_batch_idx = 0
        batch_samples = self._batch_sampling(self.current_batch_idx)
        self.current_batch_idx += 1
        return batch_samples

    def get_num_batches(self):
        return self.n_batches

    def _batch_sampling(self, batch_index):
        """
        Batch sampling
        :param batch_index:
        :return:
        """
        # user_ids, pos_ids
        batch_user_pos_items_pairs = self.user_pos_item_pairs[
                                     batch_index * self.batch_size:
                                     (batch_index + 1) * self.batch_size, :]
        # batch user_ids
        batch_user_ids = np.array(
            [uid for uid, _ in batch_user_pos_items_pairs])
        # batch positive item_ids
        batch_pos_ids = np.array([iid for _, iid in batch_user_pos_items_pairs])
        # batch negative item_ids
        batch_neg_ids = self.sampler.sampling(batch_user_ids)
        return batch_user_ids, batch_pos_ids, batch_neg_ids
