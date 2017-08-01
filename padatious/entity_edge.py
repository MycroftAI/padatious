# Copyright 2017 Mycroft AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fann2.libfann import neural_net, training_data as fann_data, GAUSSIAN

from padatious.id_object import IdObject
from padatious.util import StrEnum, resolve_conflicts


class Ids(StrEnum):
    unknown_tokens = ':0'


class EntityEdge(IdObject):
    """
    Represents the left or right side of an entity (a PosIntent)

    Args:
        token (str): token to attach to (something like {word})
        dir (int): -1 for left and +1 for right
    """
    def __init__(self, token, dir):
        IdObject.__init__(self, Ids)
        self.token = token
        self.dir = dir
        self.get_end = lambda x: len(x) if self.dir > 0 else -1
        self.net = None

    def vectorize(self, sent, pos):
        unknown = 0
        vector = self.create_tensor()
        for i in range(pos + self.dir, self.get_end(sent), self.dir):
            if self.has_id(sent[i]):
                self.set_id(vector, sent[i], 1.0 / abs(i - pos))
            else:
                unknown += 1
        self.set_id(vector, Ids.unknown_tokens, unknown / len(sent))
        return vector

    def match(self, sent, pos):
        return self.net.run(self.vectorize(sent, pos))[0]

    def configure_net(self):
        layers = [self.id_len, max(int(self.id_len / 2 + 0.5), 1), 1]
        self.net = neural_net()
        self.net.create_standard_array(layers)
        self.net.set_activation_function_hidden(GAUSSIAN)
        self.net.set_activation_function_output(GAUSSIAN)

    def save(self, prefix):
        prefix += '.' + {-1: 'l', +1: 'r'}[self.dir]
        self.net.save(str(prefix + '.net'))  # Must have str()
        self.save_ids(prefix)

    def load(self, prefix):
        prefix += '.' + {-1: 'l', +1: 'r'}[self.dir]
        self.net = neural_net()
        self.net.create_from_file(str(prefix + '.net'))  # Must have str()
        self.load_ids(prefix)

    def train(self, name, train_data):
        for sent in train_data.my_sents(name):
            if self.token in sent:
                for i in range(sent.index(self.token) + self.dir,
                               self.get_end(sent), self.dir):
                    self.register_token(sent[i])

        inputs, outputs = [], []

        def add_sents(sents, out_fn):
            for sent in sents:
                for i, token in enumerate(list(sent)):
                    inputs.append(self.vectorize(sent, i))
                    outputs.append([out_fn(token)])

        add_sents(train_data.my_sents(name), lambda x: float(x == self.token))
        add_sents(train_data.other_sents(name), lambda x: 0.0)
        inputs, outputs = resolve_conflicts(inputs, outputs)

        data = fann_data()
        data.set_train_data(inputs, outputs)

        self.configure_net()
        self.net.train_on_data(data, 10000, 0, 0.001)
