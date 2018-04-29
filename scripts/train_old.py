
import argparse
import os, sys
import numpy as np
from scipy import misc
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import model
from time import time

def main(args):

     parser = argparse.ArgumentParser()
     parser.add_argument('--data_dir', type=str, required=True, help='The directory where the model will be stored.')
     parser.add_argument('--log_dir', type=str, required=True, help='The directory where the log files and checkpoints will be stored.')
     parser.add_argument('--train_loc', type=str, required=True, help='The train locations.')
     parser.add_argument('--batch_size', type=int, default=128, help='The training batch size.')
     parser.add_argument('--channels', type=int, default=3, help='The number of image channels.')
     parser.add_argument('--patch_size', type=int, default=37, help='The training patch size.')
     parser.add_argument('--disparity_range', type=int, default=201, help='The training disparity range.')

     parser.add_argument('--moving_average_decay', type=float,    help='Exponential decay for tracking of training parameters.', default=0.99)

     parser.add_argument('--max_iter', type=int,
                      help='Number of iterations to run.', default=40000)

     parser.add_argument('--learning_rate', type=float,
                  help='Initial learning rate.', default=0.01)

     args = parser.parse_args()

     if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

     with tf.Graph().as_default():

         train_loc = np.fromfile(args.train_loc,'<f4').reshape(-1,5).astype(int)

         #train_indices = [tr[i][0] for i in range(len(train_loc))]
         #locations = tf.placeholder(tf.int32,
         #                           [None,1,5],
         #                           name='locations')

         left_images = tf.placeholder(tf.float32,
                                     [None,args.patch_size,args.patch_size,args.channels],
                                     name='left_image')

         right_images = tf.placeholder(tf.float32,
                                      [None,args.patch_size,args.patch_size+args.disparity_range-1,args.channels],
                                      name='right_image')

         y = tf.placeholder(tf.float32, (None,1, args.disparity_range), name='y')
         train_phase = tf.placeholder(tf.bool, name='phase_train')

         learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

         batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

         # Create a queue that produces indices from the locations array
         range_size = train_loc.shape[0]
         index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                               shuffle=True, seed=None, capacity=32)
         index_dequeue_op = index_queue.dequeue_many(args.batch_size, 'index_dequeue')

         # Create an input queue containing the left and right patches and the corresponding targets
         l_shape = (args.patch_size,args.patch_size,args.channels)
         r_shape = (args.patch_size,args.patch_size+args.disparity_range-1,args.channels)

         input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                               dtypes=[tf.float32, tf.float32, tf.float32],
                                               shapes=[l_shape, r_shape, (1,args.disparity_range)],
                                               shared_name=None, name=None)

         enqueue_op = input_queue.enqueue_many([left_images,right_images, y], name='enqueue_op')

         l_batch, r_batch, labels_batch = input_queue.dequeue_many(batch_size_placeholder)

         tf.summary.image("right_patches",r_batch)
         tf.summary.image("left_patches",l_batch)

         l_branch = model.inference(l_batch,is_training=train_phase,reuse=False)
         r_branch = model.inference(r_batch,is_training=train_phase,reuse=True)

         l_branch = tf.squeeze(l_branch,[1])
         r_branch = tf.squeeze(r_branch,[1])

         final_emb = tf.matmul(l_branch,tf.transpose(r_branch,perm=[0,2,1]))

         final_emb = tf.contrib.layers.flatten(final_emb)

         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_batch, logits=final_emb), name='loss')

         tf.summary.scalar("loss",loss)

         global_step = tf.train.get_or_create_global_step()

         boundaries = [x
         for x in np.array([24000, 32000, 40000], dtype=np.int64)]

         staged_lr = [learning_rate_placeholder * x for x in [1, 0.2, 0.04, 0.008]]

         learning_rate = tf.train.piecewise_constant(global_step,
                                                     boundaries,
                                                     staged_lr)

         tf.summary.scalar("learning_rate",learning_rate)

         optimizer = tf.train.AdagradOptimizer(learning_rate)

         grads = optimizer.compute_gradients(loss, tf.global_variables())

         apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

         variable_averages = tf.train.ExponentialMovingAverage(
                 args.moving_average_decay,
                 global_step,
                 name="variable_averages_op")

         variables_averages_op = variable_averages.apply(tf.trainable_variables())

         with tf.control_dependencies([apply_gradient_op,variables_averages_op]):
             train_op = tf.no_op(name='train')

         saver = tf.train.Saver(tf.global_variables())

         summary_op = tf.summary.merge_all()

         sess = tf.Session()
         sess.run(tf.global_variables_initializer())
         sess.run(tf.local_variables_initializer())
         summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
         coord = tf.train.Coordinator()
         tf.train.start_queue_runners(coord=coord, sess=sess)

         with sess.as_default():

             it = 0

             while it < args.max_iter:

                index_batch = sess.run(index_dequeue_op)
                batch_locations = train_loc[index_batch]

                l_image_paths = ['{}/image_2/{:06d}_10.png'.format(args.data_dir, fn[0]) for fn in batch_locations]
                r_image_paths = ['{}/image_3/{:06d}_10.png'.format(args.data_dir, fn[0]) for fn in batch_locations]

                l_patches = np.zeros((len(l_image_paths),args.patch_size, args.patch_size, args.channels))
                r_patches = np.zeros((len(r_image_paths),args.patch_size, args.patch_size+args.disparity_range-1, args.channels))
                half_patch = args.patch_size // 2
                half_range = args.disparity_range // 2

                for i in range(len(l_image_paths)):

                    l_img = misc.imread(l_image_paths[i])
                    r_img = misc.imread(r_image_paths[i])
                    l_img = (l_img - l_img.mean()) / l_img.std()
                    r_img = (r_img - r_img.mean()) / r_img.std()

                    #print(l_img.shape)

                    loc_type = batch_locations[i][1]

                    center_x = batch_locations[i][2] - 1
                    center_y = batch_locations[i][3] - 1
                    r_center_x = batch_locations[i][4] - 1

                    if loc_type==1:
                        l_patches[i] = l_img[(center_y-half_patch):(center_y+half_patch+1), (center_x-half_patch):(center_x+half_patch+1),:]
                        r_patches[i] = r_img[(center_y-half_patch):(center_y+half_patch + 1), r_center_x-half_patch-half_range:r_center_x+half_patch+half_range + 1,:]
                    elif loc_type==2:
                        l_patches[i] = np.transpose(l_img[(center_y-half_patch):(center_y+half_patch +1), (center_x-half_patch):(center_x+half_patch + 1),:],(1,0,2))
                        r_patches[i] = np.transpose(r_img[(center_y-half_patch):(center_y+half_patch + 1), r_center_x-half_patch-half_range:r_center_x+half_patch+half_range + 1,:],(1,0,2))

                gt = np.zeros((args.disparity_range))
                labels = []
                dist = [0.05, 0.2, 0.5, 0.2, 0.05]
                half_dist = len(dist) // 2
                for j in range(batch_locations.shape[0]):
                   count = 0
                   for i in range(args.disparity_range //2 - half_dist, args.disparity_range // 2 + half_dist + 1):
                       gt[i] = dist[count]
                       count += 1
                   labels.append(gt)

                labels = np.expand_dims(np.array(labels),1)

                sess.run(enqueue_op,{left_images: l_patches, right_images: r_patches, y:labels})

                train_loss,_ = sess.run([loss, train_op], feed_dict={learning_rate_placeholder:args.learning_rate, batch_size_placeholder: args.batch_size, train_phase:True})


                if it % 100 == 0:
                    print("[Iteration {:^5}] Loss : {:^5}".format(it,train_loss))

                    saver.save(sess, os.path.join(args.log_dir, 'model.ckpt'), global_step=it)
                    print("Here")

                    #summary_str = sess.run(summary_op,feed_dict={train_phase:True,learning_rate_placeholder:args.learning_rate,batch_size_placeholder:args.batch_size})
                    print("Done")
                    #summary_writer.add_summary(summary_str,it)

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
     # Save the model checkpoint
     print('Saving variables')
     start_time = time.time()
     checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
     saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
     save_time_variables = time.time() - start_time
     print('Variables saved in %.2f seconds' % save_time_variables)
     metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
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

if __name__ == '__main__':
     main(sys.argv[1:])
