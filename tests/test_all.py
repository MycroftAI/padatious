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

    def test_simple(self):
        self.cont.add_intent('hello', [
            'hello',
            'hi',
            'how are you',
            'whats up'
        ])
        self.cont.add_intent('goodbye', [
            'see you',
            'later',
            'bye',
            'goodbye',
            'another time'
        ])
        self.cont.train(False)

        data = self.cont.calc_intent('whats up')
        assert data.name == 'hello'
        assert data.conf > 0.5

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
            '(search for|find) {query}',
            '(search for|find) {query} (using|on) {engine}',
            '(using|on) {engine}, (search for|find) {query}'
        ])
        self.cont.add_intent('order', [
            'order some {food} from {store}',
            'place an order for {food} on {store}'
        ])

        self.cont.train(False)

        data = self.cont.calc_intent('search for funny dog videos')
        assert data.name == 'search'
        assert data.matches == {'query': 'funny dog videos'}
        assert data.conf > 0.5

        data = self.cont.calc_intent('search for bananas using random food search')
        assert data.name == 'search'
        assert data.matches == {'query': 'bananas', 'engine': 'random food search'}
        assert data.conf > 0.5

        data = self.cont.calc_intent('search for big furry cats using the best search engine')
        assert data.name == 'search'
        assert data.matches == {'query': 'big furry cats', 'engine': 'the best search engine'}
        assert data.conf > 0.5

        data = self.cont.calc_intent('place an order for a loaf of bread on foodbuywebsite')
        assert data.name == 'order'
        assert data.matches == {'food': 'a loaf of bread', 'store': 'foodbuywebsite'}
        assert data.conf > 0.5

    def test_extraction_dependence(self):
        self.cont.add_intent('search', [
            'wiki {query}'
        ])

        self.cont.train(False)

        data = self.cont.calc_intent('wiki')
        assert data.conf < 0.5

    def test_entity_recognition(self):
        self.cont.add_intent('weather', [
            'weather for {place} {time}'
        ], True)
        self.cont.add_intent('time', [
            'what time is it',
            'whats the time right now',
            'what time is it at the moment',
            'currently, what time is it'
        ], True)
        self.cont.add_entity('{place}', [
            'los angeles',
            'california',
            'new york',
            'chicago'
        ])
        self.cont.add_entity('{time}', [
            'right now',
            'currently',
            'at the moment',
        ])
        self.cont.train(False)
        data = self.cont.calc_intent('weather for los angeles right now')
        assert data.name == 'weather'
        assert data.matches == {'place': 'los angeles', 'time': 'right now'}
        assert data.conf > 0.5

    def teardown(self):
        if isdir('temp'):
            pass#rmtree('temp')
