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

from padatious.match_data import MatchData


class TestMatchData:
    def setup(self):
        self.match = MatchData('name', ['one', 'two'], {'{word}': ['value', 'tokens']}, 0.5)

    def test_detokenize(self):
        self.match.detokenize()
        assert self.match.sent == 'one two'

        correct_match = MatchData('name', 'one two', {'word': 'value tokens'}, 0.5)
        assert self.match.__dict__ == correct_match.__dict__
