from keras.models import Model
from keras.optimizers import SGD

from models.models import FeedForward
from utils import data_utils


# These two methods are for quick testing when you just want a running model and you don't care what it is
def binary_ff_model(name='test_binary_ff'):
    bin_model = FeedForward(name='pytest_ff_model')
    bin_model.build_model()
    return bin_model


def trained_binary_ff_model(name='test_binary_ff', optimizer=SGD, loss='binary_crossentropy',
                            X_shape=(1000, 2), y_shape=(1000, 1)):
    X, y = data_utils.generate_binary_data(X_shape=X_shape, y_shape=y_shape)

    bin_model = binary_ff_model(name=name)
    bin_model.train_model(X=X, y=y, optimizer=optimizer, loss=loss)

    return bin_model
