import torch

from torch.nn import Module
from torch.nn import Linear

from components.nam import NAMComponent, ExUNAMComponent, ExpDiveNAMComponent


class HONAM(Module):

    def __init__(self, num_features, out_size, **kwargs):

        super(HONAM, self).__init__()

        self._nam_component = NAMComponent(num_features)
        out_units = self._nam_component.get_out_units()

        self._order = kwargs["order"]

        self._output_layer = Linear(self._order * out_units, out_size)  # see FeatureNet

    def forward(self, features):

        features = self._nam_component(features)
        features = torch.stack(features, dim=1)

        # feature dropout to remove unexpected biases
        # features[:, 2:4] = 0
        # features[:, 4:10] = 0

        interactions = [1, features.sum(dim=1)]
        pow_interactions = [1, features.sum(dim=1)]

        for i in range(2, self._order + 1):

            sign = 1

            pow_interactions.append(features.pow(i).sum(dim=1))

            current_interaction = 0
            for j in range(1, i + 1):
                current_interaction = current_interaction + sign * pow_interactions[j] * interactions[i - j]
                sign *= -1

            interactions.append(current_interaction / i)

        interactions = torch.cat(interactions[1:], dim=1)

        outputs = self._output_layer(interactions)

        # re-scaling
        # outputs = outputs * (10 / 8)
        # outputs = outputs * (10 / 4)
        # outputs = outputs * (10 / 2)

        return outputs

    def local_first_order(self, features):

        features = self._nam_component(features)
        out_units = self._nam_component.get_out_units()
        weight = self._output_layer.weight.T[:out_units]

        interactions = []

        for feature in features:
            first_order_interaction = feature @ weight
            first_order_interaction = first_order_interaction.item()
            interactions.append(first_order_interaction)

        return interactions

    def local_second_order(self, features):

        features = self._nam_component(features)
        out_units = self._nam_component.get_out_units()
        weight = self._output_layer.weight.T[out_units: 2 * out_units]

        interactions = []

        for i, feature1 in enumerate(features):
            sub_interactions = []
            for j, feature2 in enumerate(features):
                if i == j:
                    second_order_interaction = 0
                else:
                    second_order_interaction = (feature1 * feature2) @ weight
                    second_order_interaction = second_order_interaction.item()
                sub_interactions.append(second_order_interaction)
            interactions.append(sub_interactions)

        return interactions

    def global_first_order(self, features, idx):

        features = self._nam_component(features)
        out_units = self._nam_component.get_out_units()
        weight = self._output_layer.weight.T[:out_units]

        first_order_interaction = features[idx] @ weight
        first_order_interaction = first_order_interaction.view(-1)
        first_order_interaction = first_order_interaction.detach().cpu().numpy()

        return first_order_interaction

    def global_second_order(self, features, idx1, idx2):

        features = self._nam_component(features)
        out_units = self._nam_component.get_out_units()
        weight = self._output_layer.weight.T[out_units: 2 * out_units]

        second_order_interaction = (features[idx1] * features[idx2]) @ weight
        second_order_interaction = second_order_interaction.view(-1)
        second_order_interaction = second_order_interaction.detach().cpu().numpy()

        return second_order_interaction

class ExUHONAM(Module):

    def __init__(self, num_features, out_size, **kwargs):

        super(ExUHONAM, self).__init__()

        self._nam_component = ExUNAMComponent(num_features)
        out_units = self._nam_component.get_out_units()

        self._order = kwargs["order"]

        self._output_layer = Linear(self._order * out_units, out_size)  # see FeatureNet

    def forward(self, features):

        features = self._nam_component(features)
        features = torch.stack(features, dim=1)

        interactions = [1, features.sum(dim=1)]
        pow_interactions = [1, features.sum(dim=1)]

        for i in range(2, self._order + 1):

            sign = 1

            pow_interactions.append(features.pow(i).sum(dim=1))

            current_interaction = 0
            for j in range(1, i + 1):
                current_interaction = current_interaction + sign * pow_interactions[j] * interactions[i - j]
                sign *= -1

            interactions.append(current_interaction / i)

        interactions = torch.cat(interactions[1:], dim=1)

        outputs = self._output_layer(interactions)

        return outputs

class ExpDiveHONAM(Module):

    def __init__(self, num_features, out_size, **kwargs):

        super(ExpDiveHONAM, self).__init__()

        self._nam_component = ExpDiveNAMComponent(num_features)
        out_units = self._nam_component.get_out_units()

        self._order = kwargs["order"]

        self._output_layer = Linear(self._order * out_units, out_size)  # see FeatureNet

    def forward(self, features):

        features = self._nam_component(features)
        features = torch.stack(features, dim=1)

        interactions = [1, features.sum(dim=1)]
        pow_interactions = [1, features.sum(dim=1)]

        for i in range(2, self._order + 1):

            sign = 1

            pow_interactions.append(features.pow(i).sum(dim=1))

            current_interaction = 0
            for j in range(1, i + 1):
                current_interaction = current_interaction + sign * pow_interactions[j] * interactions[i - j]
                sign *= -1

            interactions.append(current_interaction / i)

        interactions = torch.cat(interactions[1:], dim=1)

        outputs = self._output_layer(interactions)

        return outputs
