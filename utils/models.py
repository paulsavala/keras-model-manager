from tensorflow.keras.models import load_model

from pathlib import Path

from models.generic import TensorflowModel
from utils.file_io import smart_load


def load_model(name, model_dir, version=1, attr_dir=None):
    model_file = model_dir / Path(f'v{version}.h5')
    assert model_file.exists(), f'Model file does not exist for version {version} in directory {self.model_dir}'
    raw_model = load_model(model_file.as_posix())

    model_obj = TensorflowModel(name, version)

    if attr_dir is not None:
        _load_attributes(attr_dir, model_obj)


def _load_attributes(attr_dir, model, version=1):
    attr_file = attr_dir / Path(f'v{version}.txt')
    with open(attr_file, 'r') as f:
        row = f.readline()
        while row:
            try:
                k, v = row.split(':')
            except ValueError:
                break
            v = smart_load(v, cleanup=True)
            setattr(model, k, v)
            row = f.readline()


def load_weights(self, target_model=None, weights_file=None, load_attributes=False):
    # Takes an already built model and loads weights from another (pretrained) model with the same architecture
    assert target_model is not None or weights_file is not None, 'You must specify either a target model to load weights from or a weights file'
    assert target_model is None or weights_file is None, 'Specify target_model or weights_file, not both'

    if target_model is not None:
        self.set_weights(target_model.model.get_weights())
    else:
        self.load_weights(weights_file)

    if load_attributes:
        self._load_attributes(attr_dir=target_model.attr_dir)