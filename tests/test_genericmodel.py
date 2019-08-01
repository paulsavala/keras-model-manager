import pytest
from pathlib import Path
from utils import pathlib_utils, model_utils

from keras.models import Model

from models.models import GenericModel, FeedForward


class TestGenericModel(object):
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_mkdirs(self):
        test_model_dir = Path('models/weights/pytest')
        if test_model_dir.exists():
            pathlib_utils.delete_folder(test_model_dir)

        my_model = GenericModel(name='pytest_model')

        assert my_model.model_dir.exists()
        assert my_model.model_weights_dir.exists()
        assert my_model.model_tensorboard_dir.exists()

    def test_buildmodel(self):
        my_model = FeedForward(name='pytest_ff_model')
        my_model.build_model()
        assert isinstance(my_model.model, Model)

        pathlib_utils.delete_folder(my_model.model_dir)

    def test_trainmodel(self):
        my_model = model_utils.trained_binary_ff_model(name='pytest_ff_model')
        assert isinstance(my_model.trained_model, Model)

        pathlib_utils.delete_folder(my_model.model_dir)

    def test_savemodel(self):
        my_model = model_utils.trained_binary_ff_model(name='pytest_ff_model')
        assert my_model.version == 1

        my_model.save_model(update_version=False)
        assert my_model.version == 1
        weights_file = my_model.model_weights_dir / Path(f'v{my_model.version}.weights')
        assert weights_file.exists()

        my_model.save_model(update_version=True)
        assert my_model.version == 2
        weights_file = my_model.model_weights_dir / Path(f'v{my_model.version}.weights')
        assert weights_file.exists()

    def test_loadmodel(self):
        pass

    def test_deletemodel(self):
        my_model = model_utils.trained_binary_ff_model(name='pytest_ff_model')
        assert isinstance(my_model.trained_model, Model)

        my_model.delete_model()
        assert not my_model.model_dir.exists()
