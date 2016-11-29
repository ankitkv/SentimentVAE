# flake8: noqa
import os
import tensorflow as tf

flags = tf.flags
cfg = flags.FLAGS


# command-line config
flags.DEFINE_string ("data_path",  "data/yelp/",              "Data path")
flags.DEFINE_string ("save_file",  "models/recent.dat", "Save file")
flags.DEFINE_string ("load_file",  "",                  "File to load model "
                                                        "from")
flags.DEFINE_string ("vocab_file", "data/yelp/vocab.pk",     "Vocab pickle file")
flags.DEFINE_string ("keep_fraction",   0.97,    "Percentage of vocab to keep.")


flags.DEFINE_integer("batch_size",      128,     "Batch size")
flags.DEFINE_integer("word_emb_size",   253,     "Number of learnable dimensions in "
                                                 "word embeddings")
flags.DEFINE_integer("label_emb_size",   3,      "Number of learnable dimensions in "
                                                 "label embeddings")
flags.DEFINE_integer("num_layers",      1,       "Number of RNN layers")
flags.DEFINE_integer("max_gen_length",  50,      "Maximum length of generated sentences")
flags.DEFINE_integer("beam_size",       15,      "Beam size for beam search")
flags.DEFINE_integer("hidden_size",     256,     "RNN hidden state size")
flags.DEFINE_integer("latent_size",     32,      "Latent representation size")
flags.DEFINE_float  ("word_dropout",    .75,     "Word dropout probability for decoder "
                                                 "input")
flags.DEFINE_integer("softmax_samples", 1000,    "Number of classes to sample for "
                                                 "softmax")
flags.DEFINE_float  ("max_grad_norm",   5.0,     "Gradient clipping")
flags.DEFINE_integer("anneal_bias",     3500,    "The step to reach 0.5 for KL "
                                                 "divergence weight annealing")
flags.DEFINE_bool   ("training",        True,    "Training mode, turn off for testing")
flags.DEFINE_bool   ("bucket_data",     False,   "Prepare batches by line lengths")
flags.DEFINE_string ("optimizer",       "adam",  "Optimizer to use (sgd, adam, adagrad, "
                                                 "adadelta)")
flags.DEFINE_float  ("learning_rate",   1e-3,    "Optimizer initial learning rate")
flags.DEFINE_integer("max_epoch",       10000,   "Maximum number of epochs to run for")
flags.DEFINE_integer("max_steps",       9999999, "Maximum number of steps to run for")

flags.DEFINE_integer("print_every",     50,      "Print every these many steps")
flags.DEFINE_integer("display_every",   500,     "Print generated sentences every these "
                                                 "many steps")
flags.DEFINE_integer("save_every",      -1,      "Save every these many steps (0 to "
                                                 "disable, -1 for each epoch)")
flags.DEFINE_bool   ("save_overwrite",  True,    "Overwrite the same file each time")
flags.DEFINE_integer("validate_every",  1,       "Validate every these many epochs (0 "
                                                 "to disable)")
flags.DEFINE_integer("gpu_id",          0,       "The GPU to use")

os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

print('Config:')
cfg._parse_flags()
cfg_dict = cfg.__dict__['__flags']
maxlen = max(len(k) for k in cfg_dict)
for k, v in sorted(cfg_dict.items(), key=lambda x: x[0]):
    print(k.ljust(maxlen + 2), v)
print()
