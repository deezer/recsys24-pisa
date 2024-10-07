import os
import pickle
import numpy as np

from pisa.eval.metrics import get_metric


class Evaluator:
    """
    Evaluator for recommendation algorithms
    """
    def __init__(self, config, ref_user_items, **kwargs):
        """
        Initialize an evaluator
        :param config:
        :param ref_user_items: dictionary of user items
        """
        self.metrics = [get_metric(conf['name'], k, **kwargs)
                        for conf in config['metrics']['acc']
                        for k in conf['params']['k']]
        if kwargs['mode'] == 'test':
            non_acc_metrics = [get_metric(conf['name'], k, **kwargs)
                               for conf in config['metrics']['non_acc']
                               for k in conf['params']['k']]
            self.metrics = self.metrics + non_acc_metrics
        self.ref_user_items = ref_user_items
        self.max_k = np.max([k for conf in config['metrics']['acc']
                             for k in conf['params']['k']])
        self.min_k = np.min([k for conf in config['metrics']['acc']
                             for k in conf['params']['k']])

    def __str__(self):
        return 'Evaluator: ' + self.metric_str(sep='_')

    def metric_str(self, sep=','):
        return sep.join([str(m) for m in self.metrics])

    def eval(self, reco_items, mode='mean'):
        return {
            str(m): m.eval(reco_items, self.ref_user_items, mode)
            for m in self.metrics
        }
