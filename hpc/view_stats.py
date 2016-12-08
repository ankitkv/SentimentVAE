import re
import sys
from matplotlib import pyplot as plt


STAT_REGEX = re.compile(r'\d+:')

with open(sys.argv[1]) as f:
    stats = {}
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
print(stats.keys())
plt.plot(stats['perplexity'])
plt.show()

