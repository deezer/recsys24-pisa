from collections import defaultdict

from pisa.data.loaders.loader import DataLoader
from pisa.data.samplers import one_test_sample


class TestDataLoader(DataLoader):
    def __init__(self, data, n_users, n_items,
                 batch_size, seqlen, random_seed=2022, **kwargs):
        super(TestDataLoader, self).__init__(data, n_users, n_items,
                                             batch_size, seqlen, random_seed,
                                             **kwargs)
        self.item_ids = data['track_ids']
        user_sessions = data['user_sessions']
        self.user_ids = list(user_sessions.keys())
        if kwargs['num_scored_users'] > 0:
            self.rng.shuffle(self.user_ids)
            self.user_ids = self.user_ids[:kwargs['num_scored_users']]
        self.ref_user_items = defaultdict(list)
        self.n_valid_sess = data['data_split']['valid']
        self.n_test_sess = data['data_split']['test']
        self.ref_indices = slice(-self.n_test_sess, None) if kwargs['mode'] == 'test' \
            else slice(-self.n_test_sess - self.n_valid_sess, -self.n_test_sess)
        for uid in self.user_ids:
            ref_sessions = user_sessions[uid][self.ref_indices]
            ref_items = [set([tid for tid in ss['track_ids']]) for ss in ref_sessions]
            self.ref_user_items[uid] = ref_items
        self.item_pops = {
            'glob': data['glob_track_popularities'],
            'pers': data['pers_track_popularities']
        }
        self.n_batches = int(len(self.user_ids) / self.batch_size)
        if self.n_batches * self.batch_size < len(self.user_ids):
            self.n_batches += 1

    def get_ref_user_items(self):
        return self.ref_user_items

    def get_num_test_sessions(self):
        return self.n_test_sess

    def get_item_pops(self):
        return self.item_pops

    def _batch_sampling(self, batch_index):
        batch_user_ids = self.user_ids[batch_index * self.batch_size:
                                       (batch_index + 1) * self.batch_size]
        return self._batch_sampling_seq(batch_user_ids)

    def _batch_sampling_seq(self, batch_user_ids):
        """
        Batch sampling
        :param batch_user_ids:
        :return:
        """
        output = []
        stop = self.ref_indices.stop if self.ref_indices.stop is not None \
            else 0
        for uid in batch_user_ids:
            for nxt_idx in range(self.ref_indices.start, stop):
                one_sample = one_test_sample(uid, self.data, self.seqlen,
                                             nxt_idx, **self.kwargs)
                output.append(one_sample)
        return list(zip(*output))
