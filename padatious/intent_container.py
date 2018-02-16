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
from padatious.entity import Entity
from padatious.entity_manager import EntityManager
from padatious.intent_manager import IntentManager


class IntentContainer(object):
    """
    Creates an IntentContainer object used to load and match intents

    Args:
        cache_dir (str): Place to put all saved neural networks
    """
    def __init__(self, cache_dir):
        self.intents = IntentManager(cache_dir)
        self.entities = EntityManager(cache_dir)

    def add_intent(self, *args, **kwargs):
        """
        Creates a new intent, optionally checking the cache first

        Args:
            name (str): The associated name of the intent
            lines (list<str>): All the sentences that should activate the intent
            reload_cache: Whether to ignore cached intent if exists
        """
        self.intents.add(*args, **kwargs)

    def add_entity(self, name, *args, **kwargs):
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
        self.entities.add(Entity.wrap_name(name), *args, **kwargs)

    def load_entity(self, name, *args, **kwargs):
        """
       Loads an entity, optionally checking the cache first

       Args:
           name (str): The associated name of the entity
           file_name (str): The location of the entity file
           reload_cache (bool): Whether to refresh all of cache
       """
        Entity.verify_name(name)
        self.entities.load(Entity.wrap_name(name), *args, **kwargs)

    def load_file(self, *args, **kwargs):
        """Legacy. Use load_intent instead"""
        self.load_intent(*args, **kwargs)

    def load_intent(self, *args, **kwargs):
        """
        Loads an intent, optionally checking the cache first

        Args:
            name (str): The associated name of the intent
            file_name (str): The location of the intent file
            reload_cache (bool): Whether to refresh all of cache
        """
        self.intents.load(*args, **kwargs)

    def remove_intent(self, name):
        """Unload an intent"""
        self.intents.remove(name)

    def remove_entity(self, name):
        """Unload an entity"""
        self.entities.remove(name)

    def train(self, *args, **kwargs):
        """
        Trains all the loaded intents that need to be updated
        If a cache file exists with the same hash as the intent file,
        the intent will not be trained and just loaded from file

        Args:
            print_updates (bool): Whether to print a message to stdout
                each time a new intent is trained
            single_thread (bool): Whether to force running in a single thread
        """
        self.intents.train(*args, **kwargs)
        self.entities.train(*args, **kwargs)
        self.entities.calc_ent_dict()

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
        return self.intents.calc_intents(query, self.entities)

    def calc_intent(self, query):
        """
        Tests all the intents against the query and returns
        match data of the best intent

        Args:
            query (str): Input sentence to test against intents
        Returns:
            MatchData: Best intent match
        """
        return self.intents.calc_intent(query, self.entities)
