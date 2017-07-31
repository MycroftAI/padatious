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
from padatious.util import resolve_conflicts, StrEnum


class Ids(StrEnum):
    unknown_tokens = ':0'


class SimpleIntent(IdObject):
    """General intent used to match sentences or phrases"""
    HID_SIZE = 15
    NUM_HID = 2
    LENIENCE = 0.6

    def __init__(self):
        IdObject.__init__(self, Ids)
        self.net = None

    def match(self, sent):
        return self.net.run(self.vectorize(sent))[0]

    def vectorize(self, sent):
        vector = self.create_tensor()
        unknown = 0
        for token in sent:
            if self.has_id(token):
                self.set_id(vector, token, 1.0)
            else:
                unknown += 1
        if len(sent) > 0:
            self.set_id(vector, Ids.unknown_tokens, unknown / float(len(sent)))
        return vector

    def configure_net(self):
        self.net = neural_net()
        self.net.create_standard_array(
            [self.id_len] + [self.HID_SIZE] * self.NUM_HID + [1])
        self.net.set_activation_function_hidden(GAUSSIAN)
        self.net.set_activation_function_output(GAUSSIAN)

    def train(self, name, train_data):
        for sent in train_data.my_sents(name):
            self.register_sent(sent)

        inputs = []
        outputs = []

        def add(vec, out):
            inputs.append(self.vectorize(vec))
            outputs.append([out])

        def pollute(sent, p):
            sent = sent[:]
            for i in range(int((len(sent) + 2) / 3)):
                sent.insert(p, ':null:')
            add(sent, self.LENIENCE)

        def weight(sent):
            def calc_weight(w): return pow(len(w), 3.0)
            total_weight = 0.0
            for word in sent:
                total_weight += calc_weight(word)
            for word in sent:
                add([word], calc_weight(word) / total_weight)

        for sent in train_data.my_sents(name):
            add(sent, 1.0)
            pollute(sent, 0)
            pollute(sent, len(sent))
            weight(sent)

        for sent in train_data.other_sents(name):
            add(sent, 0.0)
        add([], 0.0)

        inputs, outputs = resolve_conflicts(inputs, outputs)

        train_data = fann_data()
        train_data.set_train_data(inputs, outputs)

        self.configure_net()
        self.net.train_on_data(train_data, 10000, 0, 0.001)

    def save(self, prefix):
        prefix += '.intent'
        self.net.save(prefix + '.net')
        self.save_ids(prefix)

    def load(self, prefix):
        prefix += '.intent'
        self.net = neural_net()
        self.net.create_from_file(prefix + '.net')
        self.load_ids(prefix)
