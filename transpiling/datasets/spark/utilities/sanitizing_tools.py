import os
import re

from os.path import join

def sanitize_name(f):
    f = f[:f.rfind('.')]
    return ''.join(n.lower() for n in f if n.isalpha() or n == '.')


def sanitize_dir(d):
    from os.path import join
    for f in os.listdir(d):
        os.rename(join(d, f), join(d, sanitize_name(f)))


def find_matches(left, right):
    left, right = os.listdir(left), os.listdir(right)
    matches = []
    for l in left:
        l = l[:l.rfind('.')]
        matches.extend(filter(lambda x: l == x[:x.rfind('.')], right))
    return matches


def prune_dir(d, matches):
    from os.path import join
    for f in os.listdir(d):
        if f not in matches:
            os.remove(join(d, f))


def process_dataset(left, right):
    sanitize_dir(left)
    sanitize_dir(right)
    matches = find_matches(left, right)
    prune_dir(left, matches)
    prune_dir(right, matches)


# regex snippets

def spaces_to_tab_token(instring):
    return re.sub(re.compile(r'( {4}|\t)', re.MULTILINE), '!TAB', instring)

def newline_to_ret_token(instring):
    return re.sub(r'\n', '!RET', instring)

def split_on_tokens(instring):
    return re.findall(r'(!TAB|!RET|\d+\.?\d*|\w+|[-+<>\\\*=~\|&\^%:]+|\W)', instring)

def tokenize_str(instring):
    import re
    p = spaces_to_tab_token(instring)
    p = newline_to_ret_token(p)
    return split_on_tokens(p)


# do the sanitizing

def tokenize_dir(rootdir):
    matches = {}
    curdir = os.getcwd()
    os.chdir(rootdir)
    for dirname, dirs, files in os.walk('.'):
        for f in files:
            with open(join(dirname, f), 'rb') as ff:
                matches[join(dirname, f)] = tokenize_str(ff.read())
    os.chdir(curdir)
    return matches

def create_s2s_dataset(left, right):
    zipped = []
    left_matches = tokenize_dir(left)
    right_matches = tokenize_dir(right)
    for k in left_matches:
        if k in right_matches:
            zipped.append((left_matches[k], right_matches[k]))
    return zipped
