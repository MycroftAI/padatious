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
import os
from os.path import isfile

from padatious.train_data import TrainData


class TestTrainData:
    def setup(self):
        self.data = TrainData()
        with open('temp', 'w') as f:
            f.writelines(['hi'])

    def test_add_lines(self):
        self.data.add_file('hi', 'temp')
        self.data.add_lines('bye', ['bye'])
        self.data.add_lines('other', ['other'])

        def cmp(a, b):
            return set(' '.join(i) for i in a) == set(' '.join(i) for i in b)

        assert cmp(self.data.my_sents('hi'), [['hi']])
        assert cmp(self.data.other_sents('hi'), [['bye'], ['other']])
        assert cmp(self.data.all_sents(), [['hi'], ['bye'], ['other']])

    def teardown(self):
        if isfile('temp'):
            os.remove('temp')
