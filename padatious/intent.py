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

import json
from os.path import join

from padatious.match_data import MatchData
from padatious.pos_intent import PosIntent
from padatious.simple_intent import SimpleIntent


class Intent(object):
    """Full intent object to handle entity extraction and intent matching"""

    def __init__(self, name, hsh=b''):
        self.name = name
        self.hash = hsh
        self.simple_intent = SimpleIntent()
        self.pos_intents = []
        self.is_loaded = False

    def match(self, sent):
        possible_matches = [MatchData(self.name, sent)]
        for pi in self.pos_intents:
            for i in possible_matches:
                possible_matches += pi.match(i)

        data = max(possible_matches, key=lambda x: x.conf)
        data.conf = self.simple_intent.match(data.sent)
        return data

    def save(self, folder):
        prefix = join(folder, self.name)
        with open(prefix + '.hash', 'wb') as f:
            f.write(self.hash)
        self.simple_intent.save(prefix)
        prefix += '.pos'
        with open(prefix, 'w') as f:
            json.dump([i.token for i in self.pos_intents], f)
        for pos_intent in self.pos_intents:
            pos_intent.save(prefix)

    def load(self, folder):
        prefix = join(folder, self.name)
        with open(prefix + '.hash', 'rb') as f:
            self.hash = f.read()
        self.simple_intent.load(prefix)
        prefix += '.pos'
        with open(prefix, 'r') as f:
            tokens = json.load(f)
        for token in tokens:
            pi = PosIntent(token)
            pi.load(prefix)
            self.pos_intents.append(pi)
        self.is_loaded = True

    @classmethod
    def from_disk(cls, name, folder):
        i = cls(name)
        i.load(folder)
        return i

    def train(self, train_data):
        tokens = set([token for sent in train_data.my_sents(self.name)
                      for token in sent if token.startswith('{')])
        self.pos_intents = [PosIntent(i) for i in tokens]

        self.simple_intent.train(self.name, train_data)
        for i in self.pos_intents:
            i.train(self.name, train_data)
        self.is_loaded = True
