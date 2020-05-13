import pytest

import API

combinations_of_layer_calls = (
    ('add_conv', 'add_max_pooling', 'add_flatten', 'add_conv'),
    ('add_conv', 'add_max_pooling', 'add_conv', 'add_max_pooling', 'add_flatten', 'add_conv'),
    ('add_dense', 'add_dense', 'add_dense'),
    ('add_dense', 'add_dense', 'add_dropout')
)


@pytest.fixture(name='api')
def api_no_create_model_dir(mocker):
    mocker.patch('API.NNConstructorAPI._create_model_dir')
    return API.NNConstructorAPI()


@pytest.fixture()
def api_model_with_layers(api):
    api.model.layers = [API.Conv2D(32, (3, 3)), API.MaxPooling2D((2, 2)), API.Flatten(), API.Dense(32)]
    return api


@pytest.mark.parametrize('combination_of_layer_calls', combinations_of_layer_calls)
def test_api_add_multiple_layers(api, combination_of_layer_calls):
    layers = []
    for call in combination_of_layer_calls:
        result = getattr(api, call)()
        layers.append(result)

    assert api.model.layers == layers


def test_api_delete_layer(api_model_with_layers):
    model = api_model_with_layers.model
    layers = model.layers.copy()
    expected_layers = layers.copy()
    for layer in layers:
        expected_layers.remove(layer)

        api_model_with_layers.delete_layer(layer)

        assert expected_layers == model.layers
