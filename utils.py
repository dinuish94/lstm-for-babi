from __future__ import print_function

from random import sample
import numpy as np

import configparser
import os
from functools import reduce
import re
import tarfile

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences


def rand_batch_gen(x,xq,y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield [x[sample_idx],xq[sample_idx]], y[sample_idx]

def get_config(filename):
    parser = configparser.ConfigParser()
    parser.read(filename)
    conf_ints = [ (key, int(value)) for key,value in parser.items('int') ]
    conf_floats = [ (key, float(value)) for key,value in parser.items('float') ]
    conf_strings = [ (key, str(value)) for key,value in parser.items('str') ]
    return dict(conf_ints + conf_floats + conf_strings)


def assert_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def isEmpty(folder):
    return not (len(os.listdir(folder)) > 0)

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xqs, maxlen=query_maxlen), np.array(ys)

def get_train_data(x, xq, y, batch_size):
    train_data = rand_batch_gen(x, xq, y,batch_size=batch_size)
    return np.array(list(train_data))


