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
from os.path import isdir, join
from shutil import rmtree

from padatious.intent_container import IntentContainer


class TestIntentContainer:
    test_lines = ['this is a test', 'another test']
    other_lines = ['something else', 'this is a different thing']

    def setup(self):
        self.cont = IntentContainer('temp')

    def test_add_intent(self):
        self.cont.add_intent('test', self.test_lines)
        self.cont.add_intent('other', self.other_lines)

    def test_load_intent(self):
        if not isdir('temp'):
            mkdir('temp')

        fn1 = join('temp', 'test.txt')
        with open(fn1, 'w') as f:
            f.writelines(self.test_lines)

        fn2 = join('temp', 'other.txt')
        with open(fn2, 'w') as f:
            f.writelines(self.other_lines)

        self.cont.load_intent('test', fn1)
        self.cont.load_intent('other', fn1)
        assert len(self.cont.intents.train_data.sent_lists) == 2

    def test_train(self):
        def test(a, b):
            self.setup()
            self.test_add_intent()
            self.cont.train(a, b)

        test(False, False)
        test(True, True)

    def test_calc_intents(self):
        self.test_add_intent()
        self.cont.train(False)

        intents = self.cont.calc_intents('this is another test')
        assert (intents[0].conf > intents[1].conf) == (intents[0].name == 'test')
        assert self.cont.calc_intent('this is another test').name == 'test'

    def teardown(self):
        if isdir('temp'):
            rmtree('temp')
