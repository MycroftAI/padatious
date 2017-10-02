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
from os.path import isdir
from shutil import rmtree

from padatious.id_manager import IdManager


class TestIdManager:
    def test_add(self):
        ids = IdManager()
        assert 'a' not in ids
        ids.add_token('a')
        assert 'a' in ids
        ids.add_token('a')
        assert len(ids) == 1
        ids.add_sent(['b', 'c'])
        assert len(ids) == 3
        for i in ['b', 'c']:
            assert i in ids

    def test_vector(self):
        ids = IdManager()
        assert len(ids.vector()) == 0
        ids.add_token('a')
        assert len(ids.vector()) == 1
        ids.add_token('b')
        vec = ids.vector()
        ids.assign(vec, 'b', 0.5)
        assert vec == [0, 0.5]

    def test_assign(self):
        ids = IdManager(ids={'test': 0, 'word': 1})
        vec = ids.vector()
        ids.assign(vec, 'test', 0.7)
        ids.assign(vec, 'word', 0.2)
        assert vec == [0.7, 0.2]

    def test_save_load(self):
        ids1 = IdManager()
        ids1.add_token('hi')
        ids1.add_token('hello')

        if not isdir('temp'):
            os.mkdir('temp')
        ids1.save('temp/temp')

        ids2 = IdManager()
        ids2.load('temp/temp')

        vec1 = ids1.vector()
        vec2 = ids2.vector()
        ids1.assign(vec1, 'hello', 3)
        ids2.assign(vec2, 'hello', 3)

        assert vec1 == vec2

    def teardown(self):
        if isdir('temp'):
            rmtree('temp')
