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

import multiprocessing as mp
from os import mkdir
from os.path import join, isfile, isdir

from padatious.intent import Intent
from padatious.train_data import TrainData
from padatious.util import lines_hash, tokenize


def _train_and_save(intent, cache, data, print_updates):
    """Internal pickleable function used to train intents in another process"""
    intent.train(data)
    if print_updates:
        print('Regenerated ' + intent.name + '.')
    intent.save(cache)


class IntentContainer(object):
    """
    Creates an IntentContainer object used to load and match intents

    Args:
        cache_dir (str): Place to put all saved neural networks
    """
    def __init__(self, cache_dir):
        self.cache = cache_dir
        self.intents = []
        self.train_data = TrainData()

    def add_intent(self, name, lines, reload_cache=False):
        """
        Creates a new intent, optionally checking the cache first

        Args:
            name (str): The associated name of the intent
            lines (list<str>): All the sentences that should activate the intent
            reload_cache: Whether to ignore cached intent if exists
        """
        hash_fn = join(self.cache, name + '.hash')
        old_hsh = None
        if isfile(hash_fn):
            with open(hash_fn, 'rb') as g:
                old_hsh = g.read()
        new_hsh = lines_hash(lines)
        if reload_cache or old_hsh != new_hsh:
            self.intents.append(Intent(name, new_hsh))
        else:
            self.intents.append(Intent.from_disk(name, self.cache))
        self.train_data.add_lines(name, lines)

    def load_file(self, name, file_name, reload_cache=False):
        """
        Loads an intent, optionally checking the cache first

        Args:
            name (str): The associated name of the intent
            file_name (str): The location of the intent file
            reload_cache (bool): Whether to ignore cached intent if exists
        """
        with open(file_name, 'r') as f:
            self.add_intent(name, f.readlines(), reload_cache)

    def train(self, print_updates=True, single_thread=False):
        """
        Trains all the loaded intents that need to be updated
        If a cache file exists with the same hash as the intent file,
        the intent will not be trained and just loaded from file

        Args:
            print_updates (bool): Whether to print a message to stdout
                each time a new intent is trained
            single_thread (bool): Whether to force running in a single thread
        """
        if not isdir(self.cache):
            mkdir(self.cache)

        def args(i):
            return i, self.cache, self.train_data, print_updates

        if single_thread:
            for i in self.intents:
                _train_and_save(*args(i))
        else:
            # Train in multiple processes to disk
            pool = mp.Pool()
            try:
                results = [
                    pool.apply_async(_train_and_save, args(i))
                    for i in self.intents if not i.is_loaded
                ]

                for i in results:
                    i.get()
            finally:
                pool.close()

        # Load saved intents from disk
        for i, intent in enumerate(self.intents):
            if not intent.is_loaded:
                self.intents[i] = Intent.from_disk(intent.name, self.cache)

    def calc_intents(self, query):
        """
        Tests all the intents against the query and returns
        data on how well each one matched against the query

        Args:
            query (str): Input sentence to test against intents
        Returns:
            list<MatchData>: List of intent matches
        See calc_intent() for a description of the returned MatchData
        """
        sent = tokenize(query)
        matches = []
        for i in self.intents:
            match = i.match(sent)
            match.detokenize()
            matches.append(match)
        return matches

    def calc_intent(self, query):
        """
        Tests all the intents against the query and returns
        match data of the best intent

        Args:
            query (str): Input sentence to test against intents
        Returns:
            MatchData: Best intent match
        """
        matches = self.calc_intents(query)
        return max(matches, key=lambda x: x.conf)
