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

from os import mkdir
from os.path import isdir
from shutil import rmtree

from padatious.intent import Intent
from padatious.train_data import TrainData


class TestIntent:
    def setup(self):
        self.data = TrainData()
        self.data.add_lines('hi', ['hello', 'hi', 'hi there'])
        self.data.add_lines('bye', ['goodbye', 'bye', 'bye {person}', 'see you later'])
        self.i_hi = Intent('hi')
        self.i_bye = Intent('bye')
        self.i_hi.train(self.data)
        self.i_bye.train(self.data)

    def test_match(self):
        assert self.i_hi.match(['hi']).conf > self.i_hi.match(['bye']).conf
        assert self.i_hi.match(['hi']).conf > self.i_bye.match(['hi']).conf
        assert self.i_bye.match(['bye']).conf > self.i_bye.match(['hi']).conf
        assert self.i_bye.match(['bye']).conf > self.i_hi.match(['bye']).conf

        all = self.i_bye.match(['see', 'you', 'later']).conf
        assert all > self.i_hi.match(['see']).conf
        assert all > self.i_hi.match(['you']).conf
        assert all > self.i_hi.match(['later']).conf

        matches = self.i_bye.match(['bye', 'john']).matches
        assert len(matches) == 1
        assert '{person}' in matches
        assert matches['{person}'] == ['john']

    def test_save_load(self):
        if not isdir('temp'):
            mkdir('temp')
        self.i_hi.save('temp')
        self.i_bye.save('temp')

        self.i_hi = Intent.from_file('hi', 'temp')
        self.i_bye = Intent.from_file('bye', 'temp')

        self.test_match()

    def teardown(self):
        if isdir('temp'):
            rmtree('temp')
