from pathlib import Path
from utils import pathlib_utils, model_utils, data_utils

from keras.models import Model
from keras.optimizers import SGD

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
        assert my_model.weights_dir.exists()
        assert my_model.tensorboard_dir.exists()
        assert my_model.notes_dir.exists()

        my_model.delete_model()

    def test_buildmodel(self):
        my_model = FeedForward(name='pytest_ff_model')
        my_model.build_model()
        assert isinstance(my_model.model, Model)

        my_model.delete_model()

    def test_trainmodel(self):
        my_model = model_utils.trained_binary_ff_model(name='pytest_ff_model')
        assert isinstance(my_model.model, Model)

        my_model.delete_model()

    def test_savemodel(self):
        my_model = model_utils.trained_binary_ff_model(name='pytest_ff_model')
        assert my_model.version == 1

        my_model.save_model(update_version=False, notes='Test for v1')
        assert my_model.version == 1
        weights_file = my_model.weights_dir / Path(f'v{my_model.version}.weights')
        assert weights_file.exists()
        notes_file = my_model.notes_dir / Path(f'v{my_model.version}.txt')
        assert notes_file.exists()
        global_notes_file = my_model.notes_dir / Path(f'version_notes.txt')
        assert global_notes_file.exists()

        my_model.save_model(update_version=True, notes='Test for v2')
        assert my_model.version == 2
        weights_file = my_model.weights_dir / Path(f'v{my_model.version}.weights')
        assert weights_file.exists()
        notes_file = my_model.notes_dir / Path(f'v{my_model.version}.txt')
        assert notes_file.exists()
        global_notes_file = my_model.notes_dir / Path(f'version_notes.txt')
        assert global_notes_file.exists()

        my_model.delete_model()

    def test_loadmodel(self):
        my_model = model_utils.trained_binary_ff_model(name='pytest_ff_model')
        my_model.save_model()
        weights_file = my_model.weights_dir / Path(f'v{my_model.version}.weights')
        assert weights_file.exists()

        my_model2 = model_utils.binary_ff_model(name='pytest_ff_model_2')
        my_model2.load_model(version=my_model.version)
        assert isinstance(my_model2.model, Model)
        assert [a == b for a, b in zip(my_model2.model.get_weights(), my_model.model.get_weights())]

        my_model.delete_model()
        my_model2.delete_model()

    def test_deletemodel(self):
        my_model = model_utils.trained_binary_ff_model(name='pytest_ff_model_to_delete')
        assert isinstance(my_model.model, Model)

        my_model.delete_model()
        assert not my_model.model_dir.exists()
