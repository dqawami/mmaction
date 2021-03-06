class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def _register_module(self, module_class):
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(module_name, self.name))

        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)

        return cls


BACKBONES = Registry('backbone')
FLOWNETS = Registry('flownet')
SPATIAL_TEMPORAL_MODULES = Registry('spatial_temporal_module')
SEGMENTAL_CONSENSUSES = Registry('segmental_consensus')
HEADS = Registry('head')
RECOGNIZERS = Registry('recognizer')
LOCALIZERS = Registry('localizer')
DETECTORS = Registry('detector')
ARCHITECTURES = Registry('architecture')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
CLASSIFICATION_LOSSES = Registry('cl_loss')
METRIC_LEARNING_LOSSES = Registry('ml_loss')
SCHEDULERS = Registry('schedulers')
