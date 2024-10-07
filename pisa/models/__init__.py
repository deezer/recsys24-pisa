from pisa import PisaError
from pisa.models.net import PISA

_SUPPORTED_MODELS = {
    'pisa': PISA
}


class ModelFactory:
    @classmethod
    def generate_model(cls, sess, params, n_users, n_items,
                       pretrained_embs=None, command='train'):
        """
        Factory method to generate a model
        :param sess:
        :param params:
        :param n_users:
        :param n_items:
        :param pretrained_embs: dictionary of pretrained embeddings
        :param command:
        :return:
        """
        model_name = params['model']['name']
        try:
            # create a new model
            mdl = _SUPPORTED_MODELS[model_name](sess=sess,
                                                params=params,
                                                n_users=n_users,
                                                n_items=n_items,
                                                pretrained_embs=pretrained_embs)
            if command == 'train':
                # build computation graph
                mdl.build_graph(name=model_name)
            elif command == 'eval' or command == 'score':
                mdl.restore(name=model_name)
            return mdl
        except KeyError:
            raise PisaError(f'Currently not support model {model_name}')
