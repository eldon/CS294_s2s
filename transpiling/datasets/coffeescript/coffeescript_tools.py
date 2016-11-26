"""Tools for organizing, sanitizing, and tokenizing coffeescript.
"""

from __future__ import print_function

import os
import cPickle as pickle
import re
import subprocess
import sys

from os.path import join


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


def parse_source_files(d, ftype, compiler=_coffeescript_compile):
    data = []
    curdir = os.getcwd()
    num_files = count_files(d, ftype)
    i, skips = 0, 0
    try:
        os.chdir(d)
        for dirname, dirs, files in os.walk('.'):
            for f in filter(lambda f: f.endswith(ftype), files):
                with open(join(dirname, f), 'rb') as ff:
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
    return [
        join(sources_dir, d) \
        for d in os.listdir(sources_dir) \
        if os.path.isdir(join(sources_dir, d))
    ]


def collect_from_dirs(sources_dir, ftype, save_loc):
    collected_dirs = set()
    source_dirs = [None]

    # while we haven't processed everything in sources_dir
    while len([d for d in source_dirs if d not in collected_dirs]) > 0:
        source_dirs = _get_source_dirs(sources_dir)
        for source_dir in source_dirs:
            collected_dirs.add(source_dir)
            print('Working in {}'.format(source_dir))
            data = parse_source_files(source_dir, ftype)
            print('Saving work from {}'.format(source_dir))
            save_progress(data, save_loc)
