#!/usr/bin/env python3
# Sample Padatious program used for testing

import sys
from glob import glob
from os.path import basename

from padatious import IntentContainer

reload_cache = len(sys.argv) > 1 and sys.argv[1] == '-r'
container = IntentContainer('intent_cache')

for file_name in glob('data/*.intent'):
    name = basename(file_name).replace('.intent', '')
    container.load_file(name, file_name, reload_cache=reload_cache)
container.train()

while True:
    data = max(container.calc_intents(input('> ')), key=lambda x: x.conf)
    print(data.name + ': ' + str(data.conf))
    for key, val in data.matches.items():
        print('\t' + key + ': ' + val)
