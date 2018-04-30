from __future__ import print_function

import argparse
import os, sys, math
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import model, utils
import time


def main(args):

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    with tf.Graph().as_default():

         global_step = tf.Variable(0, trainable=False)

         train_loc = np.fromfile(args.train_loc,'<f4').reshape(-1,5).astype(int)

         if args.valid_loc:
             valid_loc = np.fromfile(args.valid_loc,'<f4').reshape(-1,5).astype(int)

         locations = tf.placeholder(tf.string,[None,1,6],name='locations')

         y = tf.placeholder(tf.float32, (None,1, args.disparity_range), name='y')

         train_phase = tf.placeholder(tf.bool, name='phase_train')

         learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

         batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

         # Create a queue that produces indices from the locations array
         range_size = train_loc.shape[0]
         index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                 shuffle=True, seed=None, capacity=32)
         index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')

         # Create an input queue containing the locations and the corresponding targets

         input_queue = data_flow_ops.FIFOQueue(capacity=10000000,
                 dtypes=[tf.string, tf.float32],
                 shapes=[(1,6), (1,args.disparity_range)],
                 shared_name=None, name=None)

         enqueue_op = input_queue.enqueue_many([locations, y], name='enqueue_op')

         nrof_preprocess_threads = args.nrof_preprocess_threads

         # Get the next batch of left and right patches and the labels

         l_batch, r_batch, labels_batch = utils.create_input_pipeline(input_queue, args.patch_size, args.disparity_range, nrof_preprocess_threads, batch_size_placeholder)

         l_batch = tf.identity(l_batch,'l_batch')
         r_batch = tf.identity(r_batch, 'r_batch')

         # Compute logits for left and right patches

         l_branch = model.inference(l_batch,is_training=train_phase,weight_decay=args.weight_decay, reuse=False)
         r_branch = model.inference(r_batch,is_training=train_phase,weight_decay=args.weight_decay, reuse=True)

         l_branch = tf.identity(l_branch,'left_branch')
         r_branch = tf.identity(r_branch,'right_branch')

         l_branch = tf.squeeze(l_branch,[1])
         r_branch = tf.squeeze(r_branch,[1])

         # Compute the dot product of the two features
         final_emb = tf.matmul(l_branch,tf.transpose(r_branch,perm=(0,2,1)))

         final_emb = tf.contrib.layers.flatten(final_emb)

         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_batch, logits=final_emb), name='loss')

         # Calculate the total losses
         regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
         total_loss = tf.add_n([loss] + regularization_losses, name='total_loss')
         tf.summary.scalar("loss",total_loss)

         boundaries = [x
                 for x in np.array([2400, 3200, 4000], dtype=np.int32)]

         staged_lr = [learning_rate_placeholder * x for x in [1, 0.2, 0.04, 0.008]]

         learning_rate = tf.train.piecewise_constant(global_step,
                 boundaries,
                 staged_lr)


         tf.summary.scalar("learning_rate",learning_rate)

         optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

         grads = optimizer.compute_gradients(total_loss, tf.trainable_variables())

         # IMPORTANT : should execute the update ops for batch norm
         # variables. Otherwise they won't be updated

         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

         with tf.control_dependencies(update_ops):
             apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

         variable_averages = tf.train.ExponentialMovingAverage(
                 args.moving_average_decay,
                 global_step,
                 name="variable_averages_op")

         variables_averages_op = variable_averages.apply(tf.trainable_variables())

         with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
             train_op = tf.no_op(name='train')

         # Compute accuracy
         absolute_diff = tf.abs(tf.argmax(final_emb, 1)-tf.argmax(labels_batch,1))
         correct_prediction = tf.cast(tf.less_equal(absolute_diff,3), tf.float32)

         accuracy = tf.reduce_mean(correct_prediction)

         saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

         summary_op = tf.summary.merge_all()

         config = tf.ConfigProto()
         config.gpu_options.allow_growth = True

         sess = tf.Session(config=config)
         sess.run(tf.local_variables_initializer())
         sess.run(tf.global_variables_initializer())

         summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
         coord = tf.train.Coordinator()
         tf.train.start_queue_runners(coord=coord, sess=sess)

         with sess.as_default():

             if args.pretrained_model:
                print("Restoring pretrained model {}".format(args.pretrained_model))
                saver.restore(sess,args. pretrained_model)

             print('Launching training for {} samples'.format(len(train_loc)))

             nrof_steps = args.max_epochs*args.epoch_size
             nrof_val_samples = int(math.ceil(args.max_epochs / args.validation_epochs))

             for epoch in range(1,args.max_epochs):

                 step = sess.run(global_step,feed_dict=None)

                 cont = train(args, sess, epoch, train_loc, index_dequeue_op, enqueue_op, locations, y, learning_rate_placeholder, train_phase, batch_size_placeholder, global_step, loss, accuracy, train_op, summary_op, summary_writer)

                 if not cont:
                     print("Training loop terminated with error !please check")
                     break

                 if args.valid_loc and ((epoch-1) % args.validation_epochs == 0 or epoch==args.max_epochs):
                    validate(args, sess, epoch, global_step, valid_loc, enqueue_op, locations, y, train_phase, batch_size_placeholder,total_loss, accuracy, args.validation_epochs, summary_writer)


                 # Save variables and the metagraph if it doesn't exist already
                 save_variables_and_metagraph(sess, saver, summary_writer, args.log_dir, epoch)

    return args.log_dir

def train(args,
        sess,
        epoch,
        train_loc,
        index_dequeue_op,
        enqueue_op,
        locations,
        y,
        learning_rate_placeholder,
        train_phase,
        batch_size_placeholder,
        step,
        loss,
        accuracy,
        train_op,
        summary_op,
        summary_writer):

    batch_number = 0

    index_batch = sess.run(index_dequeue_op)
    batch_locations = train_loc[index_batch]

    # Change the locations data to add the images paths
    data = np.empty((batch_locations.shape[0],6),dtype="S100")
    for i,l in enumerate(batch_locations):
        data[i,0] = '{}/image_2/{:06d}_10.png'.format(args.data_dir, l[0])
        data[i,1] = '{}/image_3/{:06d}_10.png'.format(args.data_dir, l[0])
        data[i,2:] = l[1:].astype(str)

    data = np.expand_dims(data,1).astype(str)

    labels= get_labels((batch_locations.shape[0],args.disparity_range),args.dist)

    labels = np.expand_dims(np.array(labels),1)
    # Enqueue one epoch of locations and labels
    sess.run(enqueue_op,{locations: data, y:labels})

    # Training loop
    print("Starting Training for epoch {}".format(epoch))
    training_time = 0
    losses = []
    accuracies = []
    while batch_number < args.epoch_size:
        start = time.time()
        feed_dict = {learning_rate_placeholder:args.learning_rate, batch_size_placeholder: args.batch_size, train_phase:True}
        if batch_number % 200 ==0:
            train_loss,_,acc,_step, summary_str = sess.run([loss, train_op, accuracy, step]+[summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=_step)
        else:
            train_loss,_,acc, _step = sess.run([loss, train_op, accuracy, step], feed_dict=feed_dict)
            losses.append(train_loss)
            accuracies.append(acc)

        train_duration = time.time() - start


        print('[Epoch {:^2}][{}/{}] \tTime {:^2} \tLoss {:^2f} \tAccuracy {:^2f}'.format(epoch,batch_number+1,args.epoch_size,train_duration, train_loss, acc))

        batch_number += 1
        training_time += train_duration

    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=training_time)
    summary.value.add(tag='train/total_loss', simple_value=np.mean(losses))
    summary.value.add(tag='train/accuracy', simple_value=np.mean(accuracies))
    summary_writer.add_summary(summary, global_step=_step)
    return True


def validate(args,
        sess,
        epoch,
        step,
        valid_loc,
        enqueue_op,
        locations,
        y,
        train_phase,
        batch_size_placeholder,
        loss,
        accuracy,
        validate_every_n_epochs,
        summary_writer):

    print('Running forward pass on validation set')

    nrof_batches = (len(valid_loc)//100) // args.val_batch_size
    nrof_locations = nrof_batches * args.val_batch_size

    # Change the locations data to add the images paths
    data = np.empty((nrof_locations,6),dtype="S100")
    for i,l in enumerate(valid_loc[:nrof_locations]):
        data[i,0] = '{}/image_2/{:06d}_10.png'.format(args.data_dir, l[0])
        data[i,1] = '{}/image_3/{:06d}_10.png'.format(args.data_dir, l[0])
        data[i,2:] = l[1:].astype(str)
    data = np.expand_dims(data,1).astype(str)
    labels = get_labels((nrof_locations,args.disparity_range),args.dist)
    labels = np.expand_dims(np.array(labels),1)

    # Enqueue one epoch of locations and labels
    sess.run(enqueue_op,{locations: data, y:labels})

    loss_array = np.zeros((nrof_batches,), np.float32)
    accuracy_array = np.zeros((nrof_batches,), np.float32)

    # Training loop
    start_time = time.time()
    print("Starting validation for {} batches".format(nrof_batches))
    for i in range(nrof_batches):
        feed_dict = {train_phase:False, batch_size_placeholder:args.val_batch_size}
        _step, loss_, accuracy_ = sess.run([step, loss, accuracy], feed_dict=feed_dict)
        loss_array[i], accuracy_array[i] = (loss_, accuracy_)
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')

    duration = time.time() - start_time

    summary = tf.Summary()
    summary.value.add(tag='validation/time', simple_value=duration)
    summary.value.add(tag='validation/total_loss', simple_value=np.mean(loss_array))
    summary.value.add(tag='validation/accuracy', simple_value=np.mean(accuracy_array))
    summary_writer.add_summary(summary, global_step=_step)

    print('[Validation Epoch {:^2}] \tTime {:^2} \tLoss {:^2f} \tAccuracy {:^2f}'.format(epoch,duration, np.mean(loss_array), np.mean(accuracy_array)))


def get_labels(shape, dist):
    disparity_range = shape[1]
    gt = np.zeros((shape[1]))
    labels = []
    #dist = [0.05, 0.2, 0.5, 0.2, 0.05]
    half_dist = len(dist) // 2
    for j in range(shape[0]):
        count = 0
        for i in range(disparity_range //2 - half_dist, disparity_range // 2 + half_dist + 1):
           gt[i] = dist[count]
           count += 1

        labels.append(gt)
    return labels

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, step):
    # Save the model checkpoint
     print('Saving variables')
     start_time = time.time()
     checkpoint_path = os.path.join(model_dir, 'model.ckpt')
     saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
     save_time_variables = time.time() - start_time
     print('Variables saved in %.2f seconds' % save_time_variables)
     metagraph_filename = os.path.join(model_dir, 'model.meta')
     save_time_metagraph = 0
     if not os.path.exists(metagraph_filename):
         print('Saving metagraph')

     start_time = time.time()
     saver.export_meta_graph(metagraph_filename)
     save_time_metagraph = time.time() - start_time
     print('Metagraph saved in %.2f seconds' % save_time_metagraph)
     summary = tf.Summary()
     #pylint: disable=maybe-no-member
     summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
     summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
     summary_writer.add_summary(summary, step)

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='The directory where the model will be stored.')
    parser.add_argument('--log_dir', type=str, required=True, help='The directory where the log files and checkpoints will be stored.')
    parser.add_argument('--train_loc', type=str, required=True, help='The train locations.')
    parser.add_argument('--valid_loc', type=str, help='The validation locations.')
    parser.add_argument('--pretrained_model', type=str, help='Load a pretrained model before training')
    parser.add_argument('--batch_size', type=int, default=128, help='The training batch size.')
    parser.add_argument('--channels', type=int, default=3, help='The number of image channels.')
    parser.add_argument('--patch_size', type=int, default=37, help='The training patch size.')
    parser.add_argument('--disparity_range', type=int, default=201, help='The training disparity range.')

    parser.add_argument('--moving_average_decay', type=float,    help='Exponential decay for tracking of training parameters.', default=0.99)

    parser.add_argument('--max_epochs', type=int,
            help='Number of epochs to run.', default=50)
    parser.add_argument('--epoch_size', type=int,
            help='Number of batches in each epoch.', default=100)
    parser.add_argument('--validation_epochs', type=int,
            help='Number of iterations to launch validation', default=5)
    parser.add_argument('--val_batch_size', type=int,
            help='Batch size used for validation', default=1000)
    parser.add_argument('--learning_rate', type=float,
            help='Initial learning rate.', default=0.01)
    parser.add_argument('--weight_decay', type=float,
            help='Weight decay', default=0.0005)
    parser.add_argument('--dist', type=float,
            help='target_dist', nargs='+',default=[0.05,0.2,0.5,0.2,0.05])
    parser.add_argument('--nrof_preprocess_threads', type=int,
            help='Number of threads used for preprocessing',default=4)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
