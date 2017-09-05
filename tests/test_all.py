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

from os.path import isdir
from shutil import rmtree

from padatious.intent_container import IntentContainer


class TestAll:
    def setup(self):
        self.cont = IntentContainer('temp')

    def test_single_extraction(self):
        self.cont.add_intent('drive', [
            'drive to {place}',
            'driver over to {place}',
            'navigate to {place}'
        ])
        self.cont.add_intent('swim', [
            'swim to {island}',
            'swim across {ocean}'
        ])

        self.cont.train(False)

        data = self.cont.calc_intent('navigate to los angelos')
        assert data.name == 'drive'
        assert data.conf > 0.5
        assert data.matches == {'place': 'los angelos'}

        data = self.cont.calc_intent('swim to tahiti')
        assert data.name == 'swim'
        assert data.conf > 0.5
        assert data.matches == {'island': 'tahiti'}

    def test_multi_extraction_easy(self):
        self.cont.add_intent('search', [
            'search for {query} using {engine}',
            'find {query} using {engine}',
            'using {engine}, search for {query}',
            'using {engine}, find {query}'
        ])
        self.cont.add_intent('order', [
            'order some {food} from {store}',
            'place an order for {food} on {store}'
        ])

        self.cont.train(False)

        data = self.cont.calc_intent('search for bananas using random food search')
        assert data.name == 'search'
        assert data.matches == {'query': 'bananas', 'engine': 'random food search'}
        assert data.conf > 0.5

        data = self.cont.calc_intent('place an order for a loaf of bread on foodbuywebsite')
        assert data.name == 'order'
        assert data.matches == {'food': 'a loaf of bread', 'store': 'foodbuywebsite'}
        assert data.conf > 0.5

    def teardown(self):
        if isdir('temp'):
            rmtree('temp')
