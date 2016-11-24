"""Process the Yelp converted csv into proper form for the network."""

import sys
from nltk.tokenize import PunktSentenceTokenizer
import csv
from nltk import word_tokenize
import string
import re

tokenizer = PunktSentenceTokenizer()


fix_re = re.compile(r"[^a-z0-9.!,]+")
num_re = re.compile(r'[0-9]+')


def fix_word(word):
    word = word.lower()
    word = fix_re.sub('', word)
    word = num_re.sub('#', word)

    if not any((c.isalpha() or c in string.punctuation) for c in word):
        word = ''
    return word

stars_and_reviews = []

num_rows = 0
with open(sys.argv[1]) as inp_file, open(sys.argv[2], 'w') as out_file:
    reader = csv.reader(inp_file)
    writer = csv.writer(out_file)
    first = True
    for row in reader:
        if first:
            first = False
            continue

        review = row[2]
        stars = row[6]

        collected_words = []
        for sentence in tokenizer.tokenize(review):
            words = word_tokenize(sentence)
            words = [fix_word(word) for word in words]

            collected_words +=  words

        review = ' '.join(collected_words)
        writer.writerow((stars, review))
        num_rows += 1
        if num_rows % 1000 == 0:
            print('Rows %d' % num_rows)

