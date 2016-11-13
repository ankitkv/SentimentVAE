import sys
import time

import numpy as np
import tensorflow as tf

from config import cfg
from encdec import EncoderDecoderModel
from reader import Reader, Vocab


def call_mle_session(session, model, batch, summarize=False):
    '''Use the session to run the model on the batch data.'''
    f_dict = {model.data: batch[0],
              model.data_dropped: batch[1],
              model.lengths: batch[2]}

    if summarize:
        ops = [model.nll, model.kld, model.cost, model.summary(),
               model.global_step, model.train_op]
    else:
        ops = [model.nll, model.kld, model.cost, model.train_op]
    return session.run(ops, f_dict)[:-1]


def save_model(session, saver, perp, kld, cur_iters):
    '''Save model file.'''
    save_file = cfg.save_file
    if not cfg.save_overwrite:
        save_file = save_file + '.' + str(cur_iters)
    print("Saving model (epoch perplexity: %.3f, kl_divergence: %.3f) ..." % (perp, kld))
    save_file = saver.save(session, save_file)
    print("Saved to", save_file)


def run_epoch(epoch, session, model, batch_loader, vocab, saver, steps, max_steps,
              summary_writer=None):
    '''Runs the model on the given data for an epoch.'''
    start_time = time.time()
    word_count = 0.0
    nlls = 0.0
    klds = 0.0
    costs = 0.0
    iters = 0

    for step, batch in enumerate(batch_loader):
        if step % cfg.print_every == 0 and summary_writer:
            nll, kld, cost, summary_str, gstep = call_mle_session(session, model,batch,
                                                                  summarize=True)
        else:
            nll, kld, cost = call_mle_session(session, model, batch)
        sentence_length = batch[0].shape[1] - 1 
        word_count += sentence_length
        kld_weight = session.run(model.kld_weight)
        nlls += nll
        klds += nll
        costs += cost
        iters += sentence_length
        if step % cfg.print_every == 0:
            print("%d: %d  perplexity: %.3f  mle_loss: %.4f  kl_divergence: %.4f  "
                  "cost: %.4f  kld_weight: %.3f  speed: %.0f wps" % (epoch + 1, step,
                  np.exp(nll/sentence_length), nll, kld, cost, kld_weight,
                  word_count * cfg.batch_size / (time.time() - start_time)))

            if summary_writer:
                summary_writer.add_summary(summary_str, gstep)

        cur_iters = steps + step
        if saver is not None and cur_iters and cfg.save_every > 0 and \
                cur_iters % cfg.save_every == 0:
            save_model(session, saver, np.exp(nlls / iters), np.exp(klds / (step + 1)),   
                       cur_iters)
        if max_steps > 0 and cur_iters >= max_steps:
            break

    perp = np.exp(nlls / iters)
    kld = klds / step
    cur_iters = steps + step
    if saver is not None and cfg.save_every < 0:
        save_model(session, saver, perp, kld, cur_iters)
    return perp, kld, cur_iters


def main(_):
    vocab = Vocab()
    vocab.load_from_pickle()
    reader = Reader(vocab)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        with tf.variable_scope("Model") as scope:
            if cfg.training:
                with tf.name_scope('training'):
                    model = EncoderDecoderModel(vocab, True)
                with tf.name_scope('evaluation'):
                    scope.reuse_variables()
                    eval_model = EncoderDecoderModel(vocab, False)
            else:
                test_model = EncoderDecoderModel(vocab, False)
        saver = tf.train.Saver()
        summary_writer = tf.train.SummaryWriter('./summary', session.graph)
        try:
            # try to restore a saved model file
            saver.restore(session, cfg.load_file)
            print("Model restored from", cfg.load_file)
        except ValueError:
            if cfg.training:
                tf.initialize_all_variables().run()
                print("No loadable model file, new model initialized.")
            else:
                print("You need to provide a valid model file for testing!")
                sys.exit(1)

        if cfg.training:
            steps = 0
            train_losses = []
            valid_losses = []
            model.assign_lr(session, cfg.learning_rate)
            for i in range(cfg.max_epoch):
                print("\nEpoch: %d  Learning rate: %.5f" % (i + 1, session.run(model.lr)))
                perplexity, kld, steps = run_epoch(i, session, model, reader.training(),
                                                   vocab, saver, steps, cfg.max_steps,
                                                   summary_writer)
                print("Epoch: %d Train Perplexity: %.3f, KL Divergence: %.3f"
                      % (i + 1, perplexity, kld))
                train_losses.append((perplexity, kld))
                if cfg.validate_every > 0 and (i + 1) % cfg.validate_every == 0:
                    perplexity, kld, _ = run_epoch(i, session, eval_model,
                                                   reader.validation(), vocab, None, 0,
                                                   -1, None)
                    print("Epoch: %d Validation Perplexity: %.3f, KL Divergence: %.3f"
                          % (i + 1, perplexity, kld))
                    valid_losses.append((perplexity, kld))
                else:
                    valid_losses.append(None)
                print('Train:', train_losses)
                print('Valid:', valid_losses)
                if steps >= cfg.max_steps:
                    break
        else:
            print('\nTesting')
            perplexity, kld, _ = run_epoch(0, session, test_model, reader.testing(),
                                           vocab, None, 0, cfg.max_steps, None)
            print("Test Perplexity: %.3f, KL Divergence: %.3f" % (perplexity, kld))


if __name__ == "__main__":
    tf.app.run()
