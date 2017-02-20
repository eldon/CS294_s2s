"""Tools for organizing, sanitizing, and tokenizing coffeescript.
<<<<<<< HEAD

Workflow:
>>> collect_from_dirs(<dir of repos>, <filetype>, <output path>)
>>> data = pickle.load(open(<output path>, 'rb'))
>>> in_tokens, out_tokens, stats = generate_vocab(data, save=<another output path>)
>>> export_dataset(data, in_tokens, out_tokens, <input path>, <output path>, <vocab limit>)
"""

from __future__ import print_function
from __future__ import unicode_literals

import codecs
import cPickle as pickle
import os
import re
import subprocess
import sys

from collections import defaultdict
from os.path import join

reload(sys)
sys.setdefaultencoding('UTF8')  # steamroll encoding errors...

# regex utilities

def _spaces_to_tab_token(instring):
    """Replaces tabs with the !TAB token.
    """
    return re.sub(re.compile(r'( {2}|\t)', re.MULTILINE), '!TAB', instring)


def _newline_to_ret_token(instring):
    """Replaces newlines with the !RET token.
    """
    return re.sub(r'\n', '!RET', instring)


def _split_on_tokens(instring):
    """Splits a file into tokens (vars, operators, spaces, others) once
    it has already been processed for tabs and newlines.
    """
    return tuple(
        re.findall(r'(!TAB|!RET|\d+\.?\d*|\w+|[-+<>\\\*=~\|&\^%:]+|\W)', instring),
    )


def untokenize_list(inlist):
    """Translates a tokenized file back to readable source.
    """
    s = ''.join(inlist)  # lol
    s = re.sub('!RET', r'\n', s)
    return re.sub('!TAB', '  ', s)


def tokenize_str(instring):
    """Tokenizes a file into vars, operators, spaces, and others. Use this one.
    """
    p = _spaces_to_tab_token(instring)
    p = _newline_to_ret_token(p)
    return _split_on_tokens(p)


def _coffeescript_compile(inpath):
    """Returns compiled coffeescript given a path to a .coffee file.
    """
    try:
        return subprocess.check_output((
            'coffee',
            '--compile',
            '--print',
            '--no-header',
            inpath,
        ))
    except subprocess.CalledProcessError as e:
        print(e)
        raise ValueError


def count_files(d, ftype):
    """Counts the number of files which need to be processed.

    This function is useful if you wish to gauge the total amount
    of work that needs to be done before starting.
    """
    num_files = 0
    for _, _, files in os.walk(d):
        num_files += reduce(lambda t, e: t + e.endswith(ftype), files, 0)
    return num_files


def save_progress(data, location, verbose=True):
    """Attempts to update a data file at location with new data, rewrites otherwise.

    This will iteratively append to location. If a datafile is not found,
    it will create a new one. If it cannot .extend to data (usually assuming
    data is in a list), it will fail loudly.
    """
    ret = True
    try:
        if verbose: print('Finding existing file...', end='')
        with open(location, 'rb') as f:
            existing_data = pickle.load(f)
            # assuming existing data is a list here!
            data.extend(existing_data)
            if verbose: print('Found')
    except (IOError, EOFError) as e:
        print('NOT FOUND. WRITING NEW FILE...')
        print(e)
        ret = False
    with open(location, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose: print('Wrote {}'.format(location))
    return ret


def _parse_source_files(d, ftype, compiler=_coffeescript_compile):
    """Find all files with extension ftype in a directory d, recursively.

    For each file, 'compile' it using compiler, and append to a list of tuples
    with (tokenized_uncompiled, tokenized_compiled). If compiler throws a ValueError,
    that file will be skipped.
    """
    data = []
    curdir = os.getcwd()
    num_files = count_files(d, ftype)
    i, skips = 0, 0
    try:
        os.chdir(d)
        for dirname, _, files in os.walk('.'):
            for f in filter(lambda f: f.endswith(ftype), files):
                with codecs.open(join(dirname, f), 'r') as ff:
                    coffee_in = ff.read()
                    try:
                        js_out = compiler(join(dirname, f))
                    except ValueError:
                        print('SKIPPING.')
                        skips += 1
                        continue
                    data.append((tokenize_str(coffee_in), tokenize_str(js_out)))
                    i += 1
                    if i % 10 == 0:
                        print('{}/{} processed, {} skipped'.format(i, num_files, skips))
        print('{}/{} processed, {} skipped'.format(i, num_files, skips))
    finally:
        os.chdir(curdir)
    return data


def _get_source_dirs(sources_dir):
    """Returns all directories in sources_dir
    """
    return [
        join(sources_dir, d) \
        for d in os.listdir(sources_dir) \
        if os.path.isdir(join(sources_dir, d))
    ]


def collect_from_dirs(sources_dir, ftype, save_loc, compiler=_coffeescript_compile)
    """Given a directory sources_dir of source directories, find all files with extension
    ftype and create build a dataset into save_loc.
    """
    collected_dirs = set()
    source_dirs = [None]

    # while we haven't processed everything in sources_dir
    while len([d for d in source_dirs if d not in collected_dirs]) > 0:
        source_dirs = _get_source_dirs(sources_dir)
        for source_dir in source_dirs:
            collected_dirs.add(source_dir)
            print('Working in {}'.format(source_dir))
            data = _parse_source_files(source_dir, ftyp, compiler=_coffeescript_compile)
            print('Saving work from {}'.format(source_dir))
            save_progress(data, save_loc)


def generate_vocab(data_in, save=False):
    """Given a list of (in, out) examples, construct lists of tokens
    arranged by frequency.
    """
    START_VOCAB = ['_PAD', '_GO', '_EOS', '_UNK']
    in_tokens_f = defaultdict(int)
    out_tokens_f = defaultdict(int)
    for (d_in, d_out) in data_in:
        for t in d_in:
            in_tokens_f[t] += 1
        for t in d_out:
            out_tokens_f[t] += 1
    in_tokens = START_VOCAB + sorted(in_tokens_f, key=in_tokens_f.get, reverse=True)
    out_tokens = START_VOCAB + sorted(out_tokens_f, key=out_tokens_f.get, reverse=True)
    if save:
        token_data = {
            'in_tokens': in_tokens,
            'out_tokens': out_tokens,
            'in_tokens_f': in_tokens_f,
            'out_tokens_f': out_tokens_f,
        }
        with open(save, 'wb') as f:
            pickle.dump(token_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('Wrote {}'.format(save))
    return in_tokens, out_tokens, (in_tokens_f, out_tokens_f)


def _token_to_int(t, token_list, token_cache, size_limit=float('inf')):
    """Return the int which represents a token, with caching.

    Throws a ValueError if token t is not in the token_list. There MUST
    be a _UNK token at the beginning of your vocab, or this may not halt.
    """
    if t not in token_cache:
        if t == '!RET':
            token_cache[t] = r'\n'
            return '\n'
        token = token_list.index(t)
        if token >= size_limit:  # too infrequent to include, >= for 0-index
            token = _token_to_int('_UNK', token_list, token_cache)
        token_cache[t] = token  # cache this token
    else:
        token = token_cache[t]
    return token


def _tokens_to_intfile(dataset, f, tokens, size_limit=float('inf'), cache=None):
    """Given a list, dataset, of lists of tokens in tokens, write ints to file f.
    """
    if cache is None:
        cache = {}
    with open(f, 'wb') as ff:
        for i, example in enumerate(dataset):
            intstream = (_token_to_int(t, tokens, cache, size_limit) for t in example)
            ff.write(' '.join(unicode(s) for s in intstream) + '\n')
            if i % 10 == 0:
                print('{}/{} processed'.format(i, len(dataset)))
        print('{}/{} processed'.format(i+1, len(dataset)))
    print('Wrote {}'.format(f))


def export_dataset(data, in_tokens, out_tokens, in_file, out_file, vocab_limit):
    """Given a list, data, of pairs (in_tokens, out_tokens), write each to files
    in_file and out_file.
    """
    data_in, data_out = zip(*data)  # actually unzip!
    _tokens_to_intfile(data_in, in_file, in_tokens, vocab_limit)
    _tokens_to_intfile(data_out, out_file, out_tokens, vocab_limit)
