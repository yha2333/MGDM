from .MGDM import MGDMFullDP



def get_model(config):
    if config.network == 'MGDMFullDP':
        return MGDMFullDP(config)
    else:
        raise NotImplementedError('Unknown network: %s' % config.network)
