from torch.nn import Module, ModuleList

from components.featurenet import FeatureNet, ExUNet, ExpDiveNet


class NAMComponent(Module):

    def __init__(self, num_features):

        super(NAMComponent, self).__init__()

        self._feature_nets = ModuleList([FeatureNet() for _ in range(num_features)])
        self._out_units = self._feature_nets[0].get_out_units()

    def forward(self, features):
        features = [feature_net(features[:, i].unsqueeze(dim=1)) for i, feature_net in enumerate(self._feature_nets)]
        return features

    def get_out_units(self):
        return self._out_units

class ExUNAMComponent(Module):

    def __init__(self, num_features):

        super(ExUNAMComponent, self).__init__()

        self._feature_nets = ModuleList([ExUNet() for _ in range(num_features)])
        self._out_units = self._feature_nets[0].get_out_units()

    def forward(self, features):
        features = [feature_net(features[:, i].unsqueeze(dim=1)) for i, feature_net in enumerate(self._feature_nets)]
        return features

    def get_out_units(self):
        return self._out_units

class ExpDiveNAMComponent(Module):

    def __init__(self, num_features):

        super(ExpDiveNAMComponent, self).__init__()

        self._feature_nets = ModuleList([ExpDiveNet() for _ in range(num_features)])
        self._out_units = self._feature_nets[0].get_out_units()

    def forward(self, features):
        features = [feature_net(features[:, i].unsqueeze(dim=1)) for i, feature_net in enumerate(self._feature_nets)]
        return features

    def get_out_units(self):
        return self._out_units
