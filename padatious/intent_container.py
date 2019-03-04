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
from os import mkdir

from os.path import isdir
from time import sleep

import padaos
from threading import Thread

from padatious.match_data import MatchData
from padatious.entity import Entity
from padatious.entity_manager import EntityManager
from padatious.intent_manager import IntentManager
from padatious.util import tokenize


class IntentContainer(object):
    """
    Creates an IntentContainer object used to load and match intents

    Args:
        cache_dir (str): Place to put all saved neural networks
    """
    def __init__(self, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        self.must_train = False
        self.intents = IntentManager(cache_dir)
        self.entities = EntityManager(cache_dir)
        self.padaos = padaos.IntentContainer()
        self.train_thread = None  # type: Thread

    def add_intent(self, name, lines, reload_cache=False):
        """
        Creates a new intent, optionally checking the cache first

        Args:
            name (str): The associated name of the intent
            lines (list<str>): All the sentences that should activate the intent
            reload_cache: Whether to ignore cached intent if exists
        """
        self.intents.add(name, lines, reload_cache)
        self.padaos.add_intent(name, lines)
        self.must_train = True

    def add_entity(self, name, lines, reload_cache=False):
        """
        Adds an entity that matches the given lines.

        Example:
            self.add_intent('weather', ['will it rain on {weekday}?'])
            self.add_entity('{weekday}', ['monday', 'tuesday', 'wednesday'])  # ...

        Args:
            name (str): The name of the entity
            lines (list<str>): Lines of example extracted entities
            reload_cache (bool): Whether to refresh all of cache
        """
        Entity.verify_name(name)
        self.entities.add(Entity.wrap_name(name), lines, reload_cache)
        self.padaos.add_entity(name, lines)
        self.must_train = True

    def load_entity(self, name, file_name, reload_cache=False):
        """
       Loads an entity, optionally checking the cache first

       Args:
           name (str): The associated name of the entity
           file_name (str): The location of the entity file
           reload_cache (bool): Whether to refresh all of cache
       """
        Entity.verify_name(name)
        self.entities.load(Entity.wrap_name(name), file_name, reload_cache)
        with open(file_name) as f:
            self.padaos.add_entity(name, f.read().split('\n'))

    def load_file(self, *args, **kwargs):
        """Legacy. Use load_intent instead"""
        self.load_intent(*args, **kwargs)

    def load_intent(self, name, file_name, reload_cache=False):
        """
        Loads an intent, optionally checking the cache first

        Args:
            name (str): The associated name of the intent
            file_name (str): The location of the intent file
            reload_cache (bool): Whether to refresh all of cache
        """
        self.intents.load(name, file_name, reload_cache)
        with open(file_name) as f:
            self.padaos.add_intent(name, f.read().split('\n'))

    def remove_intent(self, name):
        """Unload an intent"""
        self.intents.remove(name)
        self.padaos.remove_intent(name)
        self.must_train = True

    def remove_entity(self, name):
        """Unload an entity"""
        self.entities.remove(name)
        self.padaos.remove_entity(name)

    def _train(self, *args, **kwargs):
        t1 = Thread(target=self.intents.train, args=args, kwargs=kwargs, daemon=True)
        t2 = Thread(target=self.entities.train,  args=args, kwargs=kwargs, daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.entities.calc_ent_dict()

    def train(self, *args, force=False, **kwargs):
        """
        Trains all the loaded intents that need to be updated
        If a cache file exists with the same hash as the intent file,
        the intent will not be trained and just loaded from file

        Args:
            print_updates (bool): Whether to print a message to stdout
                each time a new intent is trained
            force (bool): Whether to force training if already finished
            single_thread (bool): Whether to force running in a single thread
        """
        if not self.must_train and not force:
            return
        self.padaos.compile()

        timeout = kwargs.setdefault('timeout', 20)
        self.train_thread = Thread(target=self._train, args=args, kwargs=kwargs, daemon=True)
        self.train_thread.start()
        self.train_thread.join(timeout)

        self.must_train = False

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
        if self.must_train:
            self.train()
        intents = {} if self.train_thread and self.train_thread.is_alive() else {
            i.name: i for i in self.intents.calc_intents(query, self.entities)
        }
        sent = tokenize(query)
        for perfect_match in self.padaos.calc_intents(query):
            name = perfect_match['name']
            intents[name] = MatchData(name, sent, matches=perfect_match['entities'], conf=1.0)
        return list(intents.values())

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
        if len(matches) == 0:
            return MatchData('', '')
        best_match = max(matches, key=lambda x: x.conf)
        best_matches = (match for match in matches if match.conf == best_match.conf)
        return min(best_matches, key=lambda x: sum(map(len, x.matches.values())))
