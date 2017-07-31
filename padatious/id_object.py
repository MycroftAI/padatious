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


class IdObject(object):
    """
    Gives inheriting object a set of methods
    for converting tokens to vectors
    """
    def __init__(self, id_cls, ids=None):
        if ids is not None:
            self.ids = ids
        else:
            self.ids = {}
            for i in id_cls.items():
                self.register_token(getattr(id_cls, i))

    @property
    def id_len(self):
        return len(self.ids)

    def create_tensor(self):
        return [0.0] * self.id_len

    def save_ids(self, prefix):
        with open(prefix + '.ids', 'w') as f:
            json.dump(self.ids, f)

    def load_ids(self, prefix):
        with open(prefix + '.ids', 'r') as f:
            self.ids = json.load(f)

    def set_id(self, vector, key, val):
        vector[self.ids[key]] = val

    def has_id(self, id):
        return id in self.ids

    def register_token(self, token):
        if token not in self.ids:
            self.ids[token] = len(self.ids)

    def register_sent(self, sent):
        for token in sent:
            self.register_token(token)
