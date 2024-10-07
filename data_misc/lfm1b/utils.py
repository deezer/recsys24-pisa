import os
import logging
import json
from datetime import datetime
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

_FORMAT = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'


class _LoggerHolder(object):
    """
    Logger singleton instance holder.
    """
    INSTANCE = None


def get_logger():
    """
    Returns library scoped logger.
    :returns: Library logger.
    """
    if _LoggerHolder.INSTANCE is None:
        formatter = logging.Formatter(_FORMAT)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger('repeatflow')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        _LoggerHolder.INSTANCE = logger
    return _LoggerHolder.INSTANCE


def load_configuration(descriptor):
    """
    Load configuration from the given descriptor.
    Args:
        descriptor:
    Returns:
    """
    if not os.path.exists(descriptor):
        raise IOError(f'Configuration file {descriptor} '
                      f'not found')
    with open(descriptor, 'r') as stream:
        return json.load(stream)


def delta_ts(ts1, ts2):
    """
    Delta between two timestamps
    :param ts1:
    :param ts2:
    :return:
    """
    d1 = datetime.fromtimestamp(ts1)
    d2 = datetime.fromtimestamp(ts2)
    dt = d2 - d1
    return dt.days


def split_data(df, test_size=.2, random_state=42):
    train_set, test_set = train_test_split(df,
                                           test_size=test_size,
                                           random_state=random_state)
    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)
    return train_set, test_set


def df_to_mat(df, n_rows, n_cols, binary=True):
    """
    Convert dataframe to matrix
    :param df:
    :param n_rows:
    :param n_cols:
    :param binary:
    :return:
    """
    dtype = np.int32 if binary is True else np.float32
    interactions_mat = sp.dok_matrix((n_rows, n_cols),
                                     dtype=dtype)
    interactions_mat[
        df.user.tolist(), df.item.tolist()] = 1
    interactions_mat = interactions_mat.tocsr()
    return interactions_mat


def mat_to_dict(interactions, criteria=None):
    """
    Convert sparse matrix to dictionary of set
    :param interactions: scipy sparse matrix
    :param criteria:
    :return:
    """
    if not sp.isspmatrix_lil(interactions):
        interactions = sp.lil_matrix(interactions)
    n_rows = interactions.shape[0]
    res = {
        u: set(interactions.rows[u]) for u in range(n_rows)
        if criteria is None or
           (criteria is not None and criteria(interactions, u) is True)
    }
    return res
