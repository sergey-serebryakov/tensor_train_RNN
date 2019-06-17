"""Functions for downloading and reading time series data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from timeit import default_timer as timer
from reader import read_data_sets
from model_seq2seq import *     # Do not delete me
from trnn import *
import numpy
from train_config import *

# step 1: Flags for training configuration
flags = tf.flags

flags.DEFINE_string("model", "TLSTM", "Model used for learning.")
flags.DEFINE_string("data_path", "./data.npy", "Data input directory.")
flags.DEFINE_string("save_path", "./log/tlstm/", "Model output directory.")
flags.DEFINE_bool("use_sched_samp", False, "Use scheduled sampling in training")
flags.DEFINE_integer("inp_steps", 12, "burn in steps")
flags.DEFINE_integer("out_steps", None, "test steps")
flags.DEFINE_integer("hidden_size", 8, "hidden layer size")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("decay_rate", 0.8, "learning rate")
flags.DEFINE_integer("rank", 1, "rank for tt decomposition")
flags.DEFINE_integer("training_steps", 10000, "number of training batches")
flags.DEFINE_integer("display_step", 500, "Display and validate model interval")

FLAGS = flags.FLAGS
print('Flags configuration loaded ...')

# step 2: Read flags and data into memory
'''
To forecast time series using a recurrent neural network, we consider every 
row as a sequence of short time series.
'''

# Training Parameters
config = TrainConfig()
config.hidden_size = FLAGS.hidden_size
config.learning_rate = FLAGS.learning_rate
config.decay_rate = FLAGS.decay_rate
config.rank_vals = [FLAGS.rank]
config.training_steps = FLAGS.training_steps

# Scheduled sampling [optional]
if FLAGS.use_sched_samp:
    config.sample_prob = tf.get_variable("sample_prob", shape=(), initializer=tf.zeros_initializer())
sampling_burn_in = 400

# Training Parameters
training_steps = config.training_steps
batch_size = config.batch_size
display_step = FLAGS.display_step
inp_steps = config.inp_steps
out_steps = FLAGS.out_steps


# Read Dataset
dataset, stats = read_data_sets(FLAGS.data_path, True, inp_steps, out_steps)

# Network Parameters
num_input = stats['num_input']  # dataset data input (time series dimension: 3)
num_steps = stats['num_steps']

if out_steps is None:
    # Forecast for the rest if horizon is not set
    out_steps = num_steps - inp_steps

# Print training config
print('-'*100)
print('|input steps|', inp_steps, '|out steps|', out_steps, '|', 'hidden size|', config.hidden_size, '|learning rate|',
      config.learning_rate, '|rank val|', config.rank_vals)
print('-'*100)

# Step 3: Build neural network models
# tf Graph input
X = tf.placeholder("float", [None, inp_steps, num_input])
Y = tf.placeholder("float", [None, out_steps, num_input])

# Decoder output
Z = tf.placeholder("float", [None, out_steps, num_input])

Model = globals()[FLAGS.model]
with tf.name_scope("Train"):
    with tf.variable_scope("Model", reuse=None):
        train_pred = Model(X, Y, True,  config)
with tf.name_scope("Test"):
    with tf.variable_scope("Model", reuse=True):
        test_pred = Model(X, Y, False,  config)


# Define loss and optimizer
train_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(train_pred, Z)))
test_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(test_pred, Z)))

# Exponential learning rate decay
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = config.learning_rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           2000, config.decay_rate, staircase=True)
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_op = optimizer.minimize(train_loss, global_step=global_step)

# Scheduled sampling params
eps_min = 0.1  # minimal prob

# Write summary
train_summary = tf.summary.scalar('train_loss', train_loss)
valid_summary = tf.summary.scalar('valid_loss', test_loss)
lr_summary = tf.summary.scalar('learning_rate', learning_rate)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Saver for the model and loss
saver = tf.train.Saver()
hist_loss = []

# Training loop
# Start training
with tf.Session() as sess:
    # Merge all the summaries and write them out to /log/tlstm (by default)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.save_path, sess.graph)

    # Run the initializer
    sess.run(init)
    train_start = timer()
    iter_start = timer()
    iter_times = []
    for step in range(1, training_steps + 1):
        batch_x, batch_y, batch_z = dataset.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, Z: batch_z})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss
            summary, loss = sess.run([merged, train_loss], feed_dict={X: batch_x, Y: batch_y, Z: batch_z})
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)
            summary_writer.add_summary(summary, step)

            # Calculate validation
            valid_enc_inps = dataset.validation.enc_inps.reshape((-1, inp_steps, num_input))
            valid_dec_inps = dataset.validation.dec_inps.reshape((-1, out_steps, num_input))
            valid_dec_outs = dataset.validation.dec_outs.reshape((-1, out_steps, num_input))
            va_sum, va_loss = sess.run([valid_summary, test_loss],
                                       feed_dict={X: valid_enc_inps, Y: valid_dec_inps, Z: valid_dec_outs})
            summary_writer.add_summary(va_sum, step)

            # Overfitting
            hist_loss.append(va_loss)
            if len(hist_loss) > 20 and va_loss > np.mean(hist_loss):
                print("Early stopping: step ", step)
                break

            # Update sampling prob
            if FLAGS.use_sched_samp and step > sampling_burn_in:
                sample_prob = max(eps_min, 1.0 - step / (2 * training_steps))
                sess.run(tf.assign(config.sample_prob, sample_prob))
                print('Sampling prob:', sample_prob)

            iter_time = timer() - iter_start
            if step != 1:
                iter_times.append(iter_time)
            print("Step {:>5}, iter time = {:.4f} seconds, minibatch loss = {:.4f}, validation loss = {:.4f}".format(
                step, iter_time, loss, va_loss))
            iter_start = timer()

    average_iter_time = None
    throughput = None
    if len(iter_times) > 0:
        average_iter_time = sum(iter_times) / len(iter_times)
        throughput = float(display_step * batch_size) / average_iter_time
    print("[INFO]     Optimization finished in {:.4f} s, iter time {:.4f} s, throughput {:.2f} sequences/s".format(
        (timer() - train_start), average_iter_time, throughput))
    print("[INFO]     batch size={}, input length={}, output length={}".format(batch_size, inp_steps, out_steps))
    print("[WARNING]  Iteration time includes time to run model validation.")

    # Calculate accuracy for test datasets
    test_enc_inps = dataset.test.enc_inps.reshape((-1, inp_steps, num_input))
    test_dec_inps = dataset.test.dec_inps.reshape((-1, out_steps, num_input))
    test_dec_outs = dataset.test.dec_outs.reshape((-1, out_steps, num_input))

    # Fetch the predictions
    fetches = {
        "true": Z,
        "pred": test_pred,
        "loss": test_loss
    }
    test_vals = sess.run(fetches, feed_dict={X: test_enc_inps, Y: test_dec_inps, Z: test_dec_outs})
    print("Testing Loss:", test_vals["loss"])

    # Save the variables to disk.
    save_path = saver.save(sess, FLAGS.save_path)
    print("Model saved in file: %s" % save_path)
    # Save predictions
    numpy.save(save_path + "predict.npy", (test_vals["true"], test_vals["pred"]))
    # Save config file
    with open(save_path + "config.out", 'w') as f:
        f.write('hidden_size:' + str(config.hidden_size) + '\t' + 'learning_rate:' + str(config.learning_rate) + '\n')
        f.write('train_error:' + str(loss) + '\t' + 'valid_error:' + str(va_loss) + '\t' + 'test_error:' + str(
            test_vals["loss"]) + '\n')
