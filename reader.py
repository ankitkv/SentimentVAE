from pathlib import Path
import pickle
import random
import csv
import numpy as np
import tensorflow as tf
from operator import itemgetter
from config import cfg
import utils


def read_all_csv_rows(name, vocab):
    pkfile = Path(cfg.data_path) / (name + '.%.3f.pk' % cfg.keep_fraction)
    try:
        print('Loading %s data from pickle...' % name)
        with pkfile.open('rb') as f:
            rows = pickle.load(f)
    except IOError:
        print('Error loading from pickle, preparing new pickle.')
        rows = []
        filename = Path(cfg.data_path) / (name + '.csv')
        with filename.open('r') as f:
            for row in csv.reader(f):
                line = vocab.lookup(row[1].split())
                line = line[:cfg.max_length]
                label = int(row[0])
                rows.append((line, label))
        with pkfile.open('wb') as f:
            pickle.dump(rows, f, -1)
        print('Saved', pkfile)

    return rows


def pack(batch):
    '''Pack python-list batches into numpy batches'''
    max_size = max(len(s[0]) for s in batch)
    leftalign_batch = np.zeros([cfg.batch_size, max_size], dtype=np.int32)
    sent_lengths = np.zeros([cfg.batch_size], dtype=np.int32)
    labels = np.zeros([cfg.batch_size], dtype=np.int32)
    for i, s in enumerate(batch):
        leftalign_batch[i, :len(s[0])] = s[0]
        sent_lengths[i] = len(s[0])
        labels[i] = s[1]
    return leftalign_batch, sent_lengths, labels


def is_batch_valid(batch):
    return batch[-1] is not None


def row_batch_iter(rows):
    if cfg.group_length:
        rows.sort(key=lambda row: len(row[0]))

    csv_batches = list(utils.grouper(cfg.batch_size, rows, None))
    random.shuffle(csv_batches)
    for batch in csv_batches:
        if is_batch_valid(batch):
            yield pack(batch)


class Vocab(object):

    '''Stores the vocab: forward and reverse mappings'''

    def __init__(self, verbose=True):
        self.init_special_tokens()
        self.verbose = verbose

    def init_special_tokens(self):
        self.vocab = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab_lookup = {w: i for i, w in enumerate(self.vocab)}
        self.labels = set([])
        self.unk_index = self.vocab_lookup.get('<unk>')
        self.sos_index = self.vocab_lookup.get('<sos>')
        self.eos_index = self.vocab_lookup.get('<eos>')

    def load_by_csv(self):
        "Load vocabulary from csv files."
        fnames = Path(cfg.data_path).glob('*.csv')
        vocab_count = {w: 0 for w in self.vocab}
        for fname in fnames:
            if self.verbose:
                print('reading csv:', fname)
            with fname.open('r') as f:
                for row in csv.reader(f):
                    line = row[1]
                    self.labels.add(int(row[0]))
                    for word in line.split():
                        c = vocab_count.get(word, 0)
                        c += 1
                        vocab_count[word] = c
        if self.verbose:
            print('Read %d words' % len(vocab_count))

        self.prune_vocab(vocab_count, cfg.keep_fraction)

    def prune_vocab(self, vocab_count, keep_fraction):
        sorted_word_counts = sorted(vocab_count.items(), key=itemgetter(1),
                                    reverse=True)

        seen_count = 0
        total_count = sum(vocab_count.values())
        index = 0
        while seen_count < keep_fraction*total_count:
            seen_count += sorted_word_counts[index][1]
            index += 1
        sorted_word_counts = sorted_word_counts[:index]

        for word, count in sorted_word_counts:
            self.vocab_lookup[word] = len(self.vocab)
            self.vocab.append(word)

        if self.verbose:
            print('Keeping %d words after pruning' % len(self.vocab_lookup))

    def load_from_pickle(self):
        '''Read the vocab from a pickled file'''
        pkfile = Path(cfg.data_path) / (cfg.vocab_file + '.%.3f.pk' % cfg.keep_fraction)
        try:
            if self.verbose:
                print('Loading vocabulary from pickle...')
            with pkfile.open('rb') as f:
                self.vocab, self.vocab_lookup, self.labels = pickle.load(f)
            if self.verbose:
                print('Vocabulary loaded, size:', len(self.vocab))
        except IOError:
            if self.verbose:
                print('Error loading from pickle, attempting parsing.')
            self.load_by_csv()
            with pkfile.open('wb') as f:
                pickle.dump([self.vocab, self.vocab_lookup, self.labels], f, -1)
                if self.verbose:
                    print('Saved pickle file.')

    def lookup(self, words):
        return [self.sos_index] + [self.vocab_lookup.get(w, self.unk_index)
                                   for w in words] + [self.eos_index]


class Reader(object):

    def __init__(self, vocab, verbose=True, load=['train', 'validation', 'test']):
        self.vocab = vocab
        random.seed(0)  # deterministic random
        self.verbose = verbose

        if 'train' in load:
            if self.verbose:
                print('Loading train csv')
            self.train_rows = read_all_csv_rows('train', vocab)
            if self.verbose:
                print('Training samples = %d' % len(self.train_rows))

        if 'validation' in load:
            if self.verbose:
                print('Loading validation csv')
            self.validation_rows = read_all_csv_rows('validation', vocab)
            if self.verbose:
                print('Validation samples = %d' % len(self.validation_rows))

        if 'test' in load:
            if self.verbose:
                print('Loading test csv')
            self.test_rows = read_all_csv_rows('test', vocab)
            if self.verbose:
                print('Testing samples = %d' % len(self.test_rows))

    def training(self):
        '''Read batches from training data'''
        return row_batch_iter(self.train_rows)

    def validation(self):
        '''Read batches from validation data'''
        return row_batch_iter(self.validation_rows)

    def testing(self):
        '''Read batches from testing data'''
        return row_batch_iter(self.test_rows)


def main(_):
    '''Reader tests'''
    vocab = Vocab()
    vocab.load_from_pickle()

    reader = Reader(vocab)
    for sents, lengths, labels in reader.testing():
        utils.display_sentences(sents, vocab)
        print()


if __name__ == '__main__':
    tf.app.run()
