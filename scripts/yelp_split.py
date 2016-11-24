"Split the yelp dataset into train/validation/test."

import random
random.seed(13)
import sys

train = 0.91
valid = 0.03
test = 0.06

with open(sys.argv[1]) as inp_file:
    lines = list(inp_file.readlines())
    random.shuffle(lines)

    i = int(len(lines)*train)
    j = i + int(len(lines)*valid)
    train_lines = lines[:i]
    valid_lines = lines[i:j]
    test_lines = lines[j:]

    print('Train, Validation, Test')
    print(len(train_lines), len(valid_lines), len(test_lines))

    output_dir = sys.argv[2] + '/'
    with open(output_dir + 'train.csv', 'w') as f:
        f.writelines(train_lines)
    with open(output_dir + 'validation.csv', 'w') as f:
        f.writelines(valid_lines)
    with open(output_dir + 'test.csv', 'w') as f:
        f.writelines(test_lines)
