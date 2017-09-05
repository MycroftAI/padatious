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
from padatious.match_data import MatchData


class PosIntent(object):
    """
    Positional intent
    Used to extract entities

    Args:
        token (str): token to attach to (something like {word})
    """

    def __init__(self, token):
        self.token = token
        self.edges = [EntityEdge(token, -1), EntityEdge(token, +1)]

    def match(self, orig_data):
        l_matches = [(self.edges[0].match(orig_data.sent, pos), pos)
                     for pos in range(len(orig_data.sent))]
        r_matches = [(self.edges[1].match(orig_data.sent, pos), pos)
                     for pos in range(len(orig_data.sent))]

        def is_valid(l_pos, r_pos):
            if r_pos < l_pos:
                return False
            for p in range(l_pos, r_pos + 1):
                if orig_data.sent[p].startswith('{'):
                    return False
            return True

        possible_matches = []
        for l_conf, l_pos in l_matches:
            if l_conf < 0.05:
                continue
            for r_conf, r_pos in r_matches:
                if r_conf < 0.05:
                    continue
                if not is_valid(l_pos, r_pos):
                    continue
                extra_conf = (l_conf - 0.5 + r_conf - 0.5) / 2
                new_sent = orig_data.sent[:l_pos] + [self.token] + orig_data.sent[r_pos + 1:]
                new_matches = orig_data.matches.copy()
                new_matches[self.token] = orig_data.sent[l_pos:r_pos + 1]
                data = MatchData(orig_data.name, new_sent, new_matches,
                                 orig_data.conf + extra_conf)
                possible_matches.append(data)
        return possible_matches

    def save(self, prefix):
        prefix += '.' + self.token
        for i in self.edges:
            i.save(prefix)

    def load(self, prefix):
        prefix += '.' + self.token
        for i in self.edges:
            i.load(prefix)

    def train(self, name, train_data):
        for i in self.edges:
            i.train(name, train_data)
