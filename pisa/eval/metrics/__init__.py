from pisa import PisaError
from pisa.eval.metrics.ndcg import NDCG
from pisa.eval.metrics.recall import RECALL
from pisa.eval.metrics.repr import REPR
from pisa.eval.metrics.pop import POP


_SUPPORTED_METRICS = {
    'ndcg': NDCG,
    'ndcg_rep': NDCG,
    'ndcg_exp': NDCG,
    'recall': RECALL,
    'recall_rep': RECALL,
    'recall_exp': RECALL,
    'repr': REPR,
    'pop': POP
}


def get_metric(name, k, **kwargs):
    """
    Get metric object from configuration
    :param name:
    :param k:
    :return:
    """
    if name not in _SUPPORTED_METRICS:
        raise PisaError(f'Not supported metric `{name}`. '
                        f'Must one of {list(_SUPPORTED_METRICS.keys())}.')
    if 'rep' in name:
        kwargs['consumption_mode'] = 'rep'
    elif 'exp' in name:
        kwargs['consumption_mode'] = 'exp'
    else:
        kwargs['consumption_mode'] = 'all'
    return _SUPPORTED_METRICS[name](k=k, **kwargs)
