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

from padatious.entity_edge import EntityEdge
from padatious.train_data import TrainData


class TestEntityEdge:
    def setup(self):
        self.data = TrainData()
        self.data.add_lines('', ['a {word} here', 'the {word} here'])
        self.le = EntityEdge(-1, '{word}', '')
        self.re = EntityEdge(+1, '{word}', '')

    def test_match(self):
        self.le.train(self.data)
        self.re.train(self.data)
        sent = ['a', '{word}', 'here']
        assert self.le.match(sent, 1) > self.le.match(sent, 0)
        assert self.le.match(sent, 1) > self.le.match(sent, 2)
        assert self.re.match(sent, 1) > self.re.match(sent, 0)
        assert self.re.match(sent, 1) > self.re.match(sent, 2)
