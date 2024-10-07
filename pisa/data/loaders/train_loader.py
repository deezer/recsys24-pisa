import numpy as np
import scipy.sparse as sp
from pisa.data.loaders.loader import DataLoader
from pisa.data.samplers import one_train_sample


class TrainDataLoader(DataLoader):
    def __init__(self, data, n_users, n_items,
                 batch_size, seqlen, random_seed=2022, **kwargs):
        super(TrainDataLoader, self).__init__(data, n_users, n_items,
                                              batch_size, seqlen, random_seed,
                                              **kwargs)
        self.train_user_tracks = data['user_tracks']['train']
        self.track_ids_map = data['track_ids_map']
        # # convert to internal ids
        # self.train_user_tracks = {uid: set([track_ids_map[tid] for tid in tracks])
        #                           for uid, tracks in train_user_tracks.items()}
        self.track_popularities = data['norm_track_popularities']
        self.negsam_strategy = kwargs['negsam_strategy']
        self.neg_alpha = kwargs['neg_alpha']
        self.interaction_indexes = \
            kwargs['train_session_indexes']
        self.rng.shuffle(self.interaction_indexes)
        self.n_batches = int(len(self.interaction_indexes) / batch_size)
        if self.n_batches * self.batch_size < len(self.interaction_indexes):
            self.n_batches += 1

    def _batch_sampling(self, batch_index):
        batch_interaction_indexes = self.interaction_indexes[
                                    batch_index * self.batch_size:
                                    (batch_index + 1) * self.batch_size]
        return self._batch_sampling_seq(batch_interaction_indexes)

    def _batch_sampling_seq(self, batch_interaction_indexes):
        """
        Batch sampling
        :param batch_interaction_indexes:
        :return:
        """
        output = []
        for uid, idx in batch_interaction_indexes:
            user_tracks = self.train_user_tracks[uid]
            one_sample = one_train_sample(uid, idx, self.data, self.seqlen,
                                          self.n_items, user_tracks=user_tracks,
                                          norm_track_popularities=self.track_popularities,
                                          **self.kwargs)
            output.append(one_sample)
        return list(zip(*output))

    @classmethod
    def _normalize_track_popularities(cls, popularities, alpha=1.0):
        if alpha != 1.0:
            popularities = {tid: np.power(freq, alpha)
                            for tid, freq in popularities.items()}
        total_count = np.sum(list(popularities.values()))
        item_popularities = {tid: freq / total_count for tid, freq in popularities.items()}
        return item_popularities
