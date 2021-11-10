[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md) [![CLA](https://img.shields.io/badge/CLA%3F-Required-blue.svg)](https://mycroft.ai/cla) [![Team](https://img.shields.io/badge/Team-Mycroft_Core-violetblue.svg)](https://github.com/MycroftAI/contributors/blob/master/team/Mycroft%20Core.md) ![Status](https://img.shields.io/badge/-Production_ready-green.svg)

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Join chat](https://img.shields.io/badge/Mattermost-join_chat-brightgreen.svg)](https://chat.mycroft.ai/community/channels/machine-learning)

# Padatious

An efficient and agile neural network intent parser. Padatious is a core component of [Mycroft AI](https://mycroft.ai). 

## Features

 - Intents are easy to create
 - Requires a relatively small amount of data
 - Intents run independent of each other
 - Easily extract entities (ie. Find the nearest *gas station* -> `place: gas station`)
 - Fast training with a modular approach to neural networks

## Getting Started

### Installing

Padatious requires the following native packages to be installed:

 - [`FANN`][fann] (with dev headers)
 - Python development headers
 - `pip3`
 - `swig`

Ubuntu:

```
sudo apt-get install libfann-dev python3-dev python3-pip swig libfann-dev python3-fann2
```

Next, install Padatious via `pip3`:

```
pip3 install padatious
```
Padatious also works in Python 2 if you are unable to upgrade.


[fann]:https://github.com/libfann/fann

### Example

Here's a simple example of how to use Padatious:

#### program.py
```Python
from padatious import IntentContainer

container = IntentContainer('intent_cache')
container.add_intent('hello', ['Hi there!', 'Hello.'])
container.add_intent('goodbye', ['See you!', 'Goodbye!'])
container.add_intent('search', ['Search for {query} (using|on) {engine}.'])
container.train()

print(container.calc_intent('Hello there!'))
print(container.calc_intent('Search for cats on CatTube.'))

container.remove_intent('goodbye')
```

Run with:

```bash
python3 program.py
```

## Learn More

Further documentation can be found at https://mycroft.ai/documentation/padatious/
