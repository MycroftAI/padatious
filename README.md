# Padatious #

An efficient and agile neural network intent parser

### Features ###

 - Intents are easy to create
 - Requires a relatively small amount of data
 - Intents run independent of each other
 - Easily extract entities (ie. Find the nearest *gas station* -> `place: gas station`)
 - Fast training

### API Example ###

Here's a simple example of how to use Padatious:

**program.py**:
```Python
from padatious.intent_container import IntentContainer

container = IntentContainer('intent_cache')
container.load_file('hello', 'hello.intent')
container.load_file('goodbye', 'goodbye.intent')
container.train()

data = container.calc_intent('Hello there!')
print(data.name)
```

**hello.intent**:
```
Hi there!
Hello.
```

**goodbye.intent**:
```
See you!
Goodbye!
```

Run with:

```bash
python3 program.py
```

### Installing ###

Padatious requires the following native packages to be installed:

 - [`FANN`][fann] (with dev headers)
 - Python development headers
 - `pip3`
 - `swig`

Ubuntu:

```
sudo apt-get install libfann-dev python3-dev python3-pip swig
```

Next, install Padatious via `pip3`:

```
pip3 install padatious
```
Padatious also works in Python 2 if you are unable to upgrade.


[fann]:https://github.com/libfann/fann
