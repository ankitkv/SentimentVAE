import re
import sys
from matplotlib import pyplot as plt
import numpy as np


STAT_REGEX = re.compile(r'\d+:')
EVERY = 50
what = 'perplexity'
AVG = 1

with open(sys.argv[1]) as f:
    stats = {}
    stats['steps'] = []
    for line in f.readlines():
        if STAT_REGEX.match(line):
            words = line.split()
            words = words[2:]
            i = 0
            while i < len(words):
                next_ = i + 1
                if next_ >= len(words):
                    i += 2
                    continue
                key = words[i].rstrip(':')
                value = float(words[next_])
                stat_list = stats.get(key, [])
                stat_list.append(value)
                stats[key] = stat_list
                i += 2

l = len(stats[what])
steps = np.zeros(l, dtype=np.int)
kernel = np.ones(AVG)/AVG
steps += EVERY
steps = np.cumsum(steps)

stat = stats['perplexity']
stat = np.convolve(stat, kernel, 'same')
plt.plot(steps, stat)
plt.show()
