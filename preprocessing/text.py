from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import numpy as np


class OneHot(object):
    def __init__(self, vocab, char_level=False, max_len=100):
        self.char_level = char_level
        self.max_len = max_len
        self.vocab = vocab

        self.tokenizer = Tokenizer(num_words=self.max_len, char_level=self.char_level)
        self.tokenizer.fit_on_texts(self.vocab)

    def encode(self, s):
        s_int = self.tokenizer.texts_to_sequences(s)
        s_int = [[x[0] - 1] for x in s_int]
        s_oh = to_categorical(s_int)
        return s_oh

    def decode(self, arr):
        s_int = np.argmax(arr, axis=1)
        s_int = [[x + 1] for x in s_int]
        s_list = self.tokenizer.sequences_to_texts(s_int)
        if self.char_level:
            sep = ''
        else:
            sep = ' '
        return sep.join(s_list)
