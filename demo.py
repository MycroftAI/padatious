#!/usr/bin/env python3
# Sample Padatious program used for testing

import sys
from builtins import input
from glob import glob
from os.path import basename

from padatious import IntentContainer

reload_cache = len(sys.argv) > 1 and sys.argv[1] == '-r'
container = IntentContainer('intent_cache')

for file_name in glob('data/*.intent'):
    name = basename(file_name).replace('.intent', '')
    container.load_file(name, file_name, reload_cache=reload_cache)

for file_name in glob('data/*.entity'):
    name = basename(file_name).replace('.entity', '')
    container.load_entity(name, file_name, reload_cache=reload_cache)

container.train()

query = None
while query != 'q':
    try:
        query = input('> ')
    except (KeyboardInterrupt, EOFError):
        print()
        break
    data = container.calc_intent(query)
    print(data.name + ': ' + str(data.conf))
    for key, val in data.matches.items():
        print('\t' + key + ': ' + val)
