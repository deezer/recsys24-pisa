import numpy as np


class DataLoader:
    """
    DataLoader is responsible for train/valid
    batch data generation
    """
    def __init__(self, data, n_users, n_items,
                 batch_size, seqlen, random_seed=2022,
                 **kwargs):
        """
        Initialization
        :param data:
        :param n_users:
        :param n_items:
        :param batch_size:
        :param seqlen:
        :param random_seed:
        :param kwargs:
        """
        self.data = data
        self.n_users = n_users
        self.n_items = n_items
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.random_seed = random_seed
        self.kwargs = kwargs
        self.current_batch_idx = 0
        self.n_batches = 0
        self.rng = np.random.RandomState(self.random_seed)

    def next_batch(self):
        if self.current_batch_idx == self.n_batches:
            self.current_batch_idx = 0
        batch_samples = self._batch_sampling(self.current_batch_idx)
        self.current_batch_idx += 1
        return batch_samples

    def get_num_batches(self):
        return self.n_batches

    def get_ref_user_items(self):
        raise NotImplementedError('get_ref_user_items method should be '
                                  'implemented in concrete model')

    def _batch_sampling(self, batch_index):
        """
        Batch sampling
        :param batch_index:
        :return:
        """
        raise NotImplementedError('_batch_sampling method should be '
                                  'implemented in concrete model')
