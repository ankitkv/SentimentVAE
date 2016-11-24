from pathlib import Path
import pickle
import random
import csv
import numpy as np
import tensorflow as tf
from operator import itemgetter
from config import cfg
import utils


class Vocab(object):

    '''Stores the vocab: forward and reverse mappings'''

    def __init__(self):
        self.init_special_tokens()

    def init_special_tokens(self):
        self.vocab = ['<pad>', '<sos>', '<eos>', '<unk>', '<drop>']
        self.vocab_lookup = {w: i for i, w in enumerate(self.vocab)}
        self.vocab_count = {w: 0 for w in self.vocab}
        self.unk_index = self.vocab_lookup.get('<unk>')
        self.sos_index = self.vocab_lookup.get('<sos>')
        self.eos_index = self.vocab_lookup.get('<eos>')
        self.drop_index = self.vocab_lookup.get('<drop>')  # for word dropout

    def load_by_csv(self, verbose=True):
        "Load vocabulary from csv files."
        fnames = Path(cfg.data_path).glob('*.csv')
        for fname in fnames:
            if verbose:
                print('reading csv:', fname)
            with fname.open('r') as f:
                for row in csv.reader(f):
                    line = row[1]
                    for word in line.split():
                        c = self.vocab_count.get(word, 0)
                        c += 1
                        self.vocab_count[word] = c

        if verbose:
            print('Read %d words' % len(self.vocab_count))

        self.prune_vocab(cfg.keep_fraction, verbose)

    def prune_vocab(self, keep_fraction, verbose=True):
        sorted_word_counts = sorted(self.vocab_count.items(), key=itemgetter(1),
                                        reverse=True)
        
        seen_count = 0
        total_count = sum(self.vocab_count.values())
        index = 0
        while seen_count < keep_fraction*total_count:
            seen_count += sorted_word_counts[index][1]
            index += 1
        sorted_word_counts = sorted_word_counts[:index]

        self.init_special_tokens()
        for word, count in sorted_word_counts:
            self.vocab_lookup[word] = len(self.vocab)
            self.vocab.append(word)

        if verbose:
            print('Keeping %d words after pruning' % len(self.vocab_lookup))

    def load_by_parsing(self, save=False, verbose=True):
        '''Read the vocab from the dataset'''
        if verbose:
            print('Loading vocabulary by parsing...')
        fnames = Path(cfg.data_path).glob('*.txt')
        for fname in fnames:
            if verbose:
                print(fname)
            with fname.open('r') as f:
                for line in f:
                    for word in utils.read_words(line):
                        if word not in self.vocab_lookup:
                            self.vocab_lookup[word] = len(self.vocab)
                            self.vocab.append(word)
        if verbose:
            print('Vocabulary loaded, size:', len(self.vocab))

    def load_from_pickle(self, verbose=True):
        '''Read the vocab from a pickled file'''
        pkfile = cfg.vocab_file
        try:
            if verbose:
                print('Loading vocabulary from pickle...')
            with open(pkfile, 'rb') as f:
                self.vocab, self.vocab_lookup = pickle.load(f)
            if verbose:
                print('Vocabulary loaded, size:', len(self.vocab))
        except IOError:
            if verbose:
                print('Error loading from pickle, attempting parsing.')
            self.load_by_csv(verbose=verbose)
            with open(pkfile, 'wb') as f:
                pickle.dump([self.vocab, self.vocab_lookup], f, -1)
                if verbose:
                    print('Saved pickle file.')

    def lookup(self, words):
        return [self.sos_index] + [self.vocab_lookup.get(w, self.unk_index)
                                   for w in words] + [self.eos_index]


class Reader(object):

    def __init__(self, vocab):
        self.vocab = vocab
        random.seed(0)  # deterministic random

    def read_lines(self, fnames):
        '''Read single lines from data'''
        for fname in fnames:
            with fname.open('r') as f:
                for line in f:
                    yield self.vocab.lookup([w for w in utils.read_words(line)])

    def buffered_read_sorted_lines(self, fnames, batches=50):
        '''Read and return a list of lines (length multiple of batch_size) worth at most
           $batches number of batches sorted in length'''
        buffer_size = cfg.batch_size * batches
        lines = []
        for line in self.read_lines(fnames):
            lines.append(line)
            if len(lines) == buffer_size:
                if cfg.bucket_data:
                    lines.sort(key=lambda x: len(x))
                else:
                    random.shuffle(lines)
                yield lines
                lines = []
        if lines:
            if cfg.bucket_data:
                lines.sort(key=lambda x: len(x))
            else:
                random.shuffle(lines)
            mod = len(lines) % cfg.batch_size
            if mod != 0:
                lines = [[self.vocab.sos_index, self.vocab.eos_index]
                         for _ in range(cfg.batch_size - mod)] + lines
            yield lines

    def buffered_read(self, fnames):
        '''Read packed batches from data with each batch having lines of similar lengths
        '''
        for line_collection in self.buffered_read_sorted_lines(fnames):
            batches = [b for b in utils.grouper(cfg.batch_size, line_collection)]
            random.shuffle(batches)
            for batch in batches:
                yield self.pack(batch)

    def training(self):
        '''Read batches from training data'''
        yield from self.buffered_read([Path(cfg.data_path) / 'train.txt'])

    def validation(self):
        '''Read batches from validation data'''
        yield from self.buffered_read([Path(cfg.data_path) / 'valid.txt'])

    def testing(self):
        '''Read batches from testing data'''
        yield from self.buffered_read([Path(cfg.data_path) / 'test.txt'])

    def _word_dropout(self, sent):
        ret = []
        for word in sent:
            if random.random() < cfg.word_dropout:
                ret.append(self.vocab.drop_index)
            else:
                ret.append(word)
        return ret

    def pack(self, batch):
        '''Pack python-list batches into numpy batches'''
        max_size = max(len(s) for s in batch)
        if len(batch) < cfg.batch_size:
            batch.extend([[] for _ in range(cfg.batch_size - len(batch))])
        leftalign_batch = np.zeros([cfg.batch_size, max_size], dtype=np.int32)
        leftalign_drop_batch = np.zeros([cfg.batch_size, max_size], dtype=np.int32)
        sent_lengths = np.zeros([cfg.batch_size], dtype=np.int32)
        for i, s in enumerate(batch):
            leftalign_batch[i, :len(s)] = s
            leftalign_drop_batch[i, :len(s)] = [s[0]] + self._word_dropout(s[1:-1]) + \
                                               [s[-1]]
            sent_lengths[i] = len(s)
        return (leftalign_batch, leftalign_drop_batch, sent_lengths)


def main(_):
    '''Reader tests'''
    vocab = Vocab()
    vocab.load_from_pickle()
    reader = csv.reader(open('data/yelp/train.csv'))
    for row in reader:
        words = row[1].split()
        words = [w if w in vocab.vocab_lookup else '<unk>:'+ w for w in words]
        print(' '. join(words))
        print()
        print()
        input()


if __name__ == '__main__':
    tf.app.run()
