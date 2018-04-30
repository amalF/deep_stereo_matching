from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
from tensorflow.python.ops import data_flow_ops
import glob
import utils
import model
from scipy import misc
from tensorflow.python.tools import inspect_checkpoint as chkp

def main(args):
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, required=True, help='The directory where the model will be stored.')
	parser.add_argument('--model', type=str, required=True, help='The directory where the checkpoints are stored.')
	parser.add_argument('--out_dir', type=str, default='./outputs', help='The directory where the outputs will be stores.')
	parser.add_argument('--batch_size', type=int, default=128, help='The training batch size.')
	parser.add_argument('--patch_size', type=int, default=37, help='The training patch size.')
	parser.add_argument('--disparity_range', type=int, default=201, help='The training disparity range.')
	parser.add_argument('--image_size', type=int, nargs='+', default=(375,1242), help='The input image size')

	args = parser.parse_args()

	scale_factor = 255 / (args.disparity_range - 1)
	if not os.path.exists(args.out_dir):
        	os.makedirs(args.out_dir)

	sess = tf.Session()

	with sess.as_default():


	    # Get the paths for the corresponding images
	    paths = get_paths(os.path.expanduser(args.data_dir))

	    left_paths_placeholder = tf.placeholder(tf.float32, shape=(1,None,None,3), name='left_paths')
	    right_paths_placeholder = tf.placeholder(tf.float32, shape=(1,None,None,3), name='paths')

	    labels_placeholder = tf.placeholder(tf.int32, shape=(None,args.disparity_range), name='labels')

	    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

	    image_size = (args.image_size[0],args.image_size[1],3)
	    l_batch = left_paths_placeholder
	    r_batch = right_paths_placeholder

            #chkp.print_tensors_in_checkpoint_file(args.model, tensor_name='', all_tensors=True)


	    l_map = model.inference(l_batch, is_training=phase_train_placeholder)
	    r_map = model.inference(r_batch, is_training=phase_train_placeholder,reuse=True)

            prod = tf.reduce_sum(tf.multiply(l_map, r_map), axis=3, name='map_inner_product')
	    tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.model))

	    coord = tf.train.Coordinator()
	    tf.train.start_queue_runners(coord=coord, sess=sess)

	    nrof_images = len(paths)

	    nrof_batches = nrof_images // args.batch_size

	    print('Launching Evaluation for {} batches'.format(nrof_batches))


	    for i in range(nrof_images):
                idx = int(paths[i,0].split("/")[-1].split("_")[0])
	        l_image = misc.imread(paths[i,0])
	        r_image = misc.imread(paths[i,1])

	        l_image = (l_image - l_image.mean()) / l_image.std()
                r_image = (r_image - r_image.mean()) / r_image.std()

                l_image = l_image.reshape(1, l_image.shape[0], l_image.shape[1],3)
                r_image = r_image.reshape(1, r_image.shape[0], r_image.shape[1],3)

	        l_features, r_features = sess.run([l_map, r_map], feed_dict={left_paths_placeholder: l_image, right_paths_placeholder: r_image, phase_train_placeholder:False})
	     	map_width = l_features.shape[2]
             	unary_vol = np.zeros((l_features.shape[1], l_features.shape[2], args.disparity_range))
                start_id = 1
	        end_id = 1

	        while start_id <= map_width:

             		for loc in range(1,args.disparity_range):
	        		x_off = -loc+1
	        		if end_id+x_off >= 1 and map_width >= start_id+x_off:

             	   			l = l_features[:, :, max(start_id, -x_off+1): min(end_id, map_width-x_off),:]
             	   			r = r_features[:, :, max(1,x_off+start_id): min(map_width, end_id + x_off),:]
             	   			res = np.sum(np.multiply(l,r),axis=3)

             	   			unary_vol[:, max(1, start_id+x_off): min(map_width, end_id+x_off), loc] = res[0, :, :]
	        	start_id = end_id + 1
                	end_id = min(map_width, end_id+2000)
             	print('Image %s processed.' % (i + 1))
	     	pred = np.argmax(unary_vol, axis=2) * scale_factor
            	misc.imsave('%s/disp_map_%06d_10.png' % (args.out_dir,idx), pred)

def get_paths(data_dir):

	l_paths = glob.glob(data_dir+"/image_2/*_10.png")
        r_paths = glob.glob(data_dir+"/image_3/*_10.png")
	paths = np.empty((len(l_paths),2), dtype="S100")
	for i in range(len(l_paths)):
		paths[i,0] = l_paths[i]
		paths[i,1] = r_paths[i]
        return paths

if __name__ == '__main__':
     main(sys.argv[1:])

