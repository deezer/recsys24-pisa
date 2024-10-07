import numpy as np


class Sampler:
    """
    Negative sampler
    """
    def __init__(self, user_items, n_items,
                 n_negatives):
        """
        Initialize a new sampler
        :param n_items: total number of items in dataset
        :param n_negatives: number of negative
        """
        self.user_items = user_items
        self.n_items = n_items
        self.n_negatives = n_negatives

    def sampling(self, user_ids):
        """
        Negative sampling
        :param user_ids:
        :return:
        """
        neg_samples = np.random.choice(self.n_items,
                                       size=(len(user_ids), self.n_negatives))
        for i, uid, negatives in zip(range(len(user_ids)), user_ids,
                                     neg_samples):
            for j, neg in enumerate(negatives):
                while neg in self.user_items[uid]:
                    neg_samples[i, j] = neg = np.random.choice(self.n_items)
        return neg_samples
