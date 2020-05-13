import os
import shutil

import pytest
import API
import model


@pytest.fixture()
def models_dir(tmpdir):
    return tmpdir.mkdir('models')


@pytest.fixture()
def api_and_internal_model(mocker):
    api = API.NNConstructorAPI()
    internal_model = mocker.MagicMock()
    api.model._model = internal_model
    return api, internal_model


def test_api_save_model(models_dir, api_and_internal_model):
    api = api_and_internal_model[0]
    internal_model = api_and_internal_model[1]
    name = 'test_model'
    api.save_model(name)

    assert os.path.exists('models/%s/%s.pkl' % (name, name))
    shutil.rmtree('models')

    print(api.model._model)
    internal_model.save.assert_called_once_with('models/test_model/test_model.h5')
