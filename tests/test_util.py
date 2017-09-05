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

from padatious.util import lines_hash, tokenize, resolve_conflicts, StrEnum


def test_lines_hash():
    assert lines_hash(['word1', 'word2']) != lines_hash(['word2', 'word1'])
    assert lines_hash(['word1', 'word2']) != lines_hash(['word1', 'word1'])


def test_tokenize():
    assert tokenize('one two three') == ['one', 'two', 'three']
    assert tokenize('one1 two2') == ['one', '1', 'two', '2']
    assert tokenize('word {ent}') == ['word', '{ent}']
    assert tokenize('test:') == ['test', ':']


def test_resolve_conflicts():
    inputs = [[0, 1], [1, 1], [0, 1]]
    outputs = [[0.0], [0.5], [0.7]]
    inputs, outputs = resolve_conflicts(inputs, outputs)
    assert len(inputs) == 2
    assert len(outputs) == 2
    assert outputs[inputs.index([0, 1])] == [0.7]


def test_str_enum():
    class MyEnum(StrEnum):
        a = '1'
        b = '2'

    assert set(MyEnum.values()) == {'1', '2'}
