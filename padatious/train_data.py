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

import tokenize
from padatious.util import tokenize


class TrainData(object):
    """
    Training data used to access collections
    of tokenized sentences in intent files
    """

    def __init__(self):
        self.sent_lists = {}

    def add_lines(self, name, lines):
        self.sent_lists[name] = [tokenize(line)
                                 for line in lines if not line.isspace()]

    def add_file(self, name, file_name):
        with open(file_name, 'r') as f:
            self.add_lines(name, f.readlines())

    def all_sents(self):
        for _, sents in self.sent_lists.items():
            for i in sents:
                yield i

    def my_sents(self, my_name):
        for i in self.sent_lists[my_name]:
            yield i

    def other_sents(self, my_name):
        for name, sents in self.sent_lists.items():
            if name != my_name:
                for i in sents:
                    yield i
