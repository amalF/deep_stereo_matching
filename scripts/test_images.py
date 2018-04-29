import tensorflow as tf
import os
import model
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import glob

def get_test_data(data_dir):

	l_paths = glob.glob(data_dir+"/image_2/*_10.png")
	print(l_paths[:10])
	r_paths = glob.glob(data_dir+"/image_3/*_10.png")

	return l_paths, r_paths

def main(data_dir,model_dir):
	with tf.Session() as session:
	
	        limage = tf.placeholder(tf.float32, [None, None, None, num_channels], name='limage')
	        rimage = tf.placeholder(tf.float32, [None, None, None, num_channels], name='rimage')
	        targets = tf.placeholder(tf.float32, [None, FLAGS.disp_range], name='targets')
	
		l_branch = model.inference(limage,is_training=False)
		r_branch = model.inference(rimage, is_training=False)

	        lmap = tf.placeholder(tf.float32, [None, None, None, 64], name='lmap')
	        rmap = tf.placeholder(tf.float32, [None, None, None, 64], name='rmap')
	
	
	        map_prod = nf.map_inner_product(lmap, rmap)
	
	        saver = tf.train.Saver()
	        saver.restore(session, tf.train.latest_checkpoint(FLAGS.model_dir))
	
	        for i in range(FLAGS.start_id, FLAGS.start_id + FLAGS.num_imgs):
	            file_id = file_ids[i]
	
	            if FLAGS.data_version == 'kitti2015':
	                linput = misc.imread(('%s/image_2/%06d_10.png') % (FLAGS.data_root, file_id))
	                rinput = misc.imread(('%s/image_3/%06d_10.png') % (FLAGS.data_root, file_id))
	            
	            elif FLAGS.data_version == 'kitti2012':
	                linput = misc.imread(('%s/image_0/%06d_10.png') % (FLAGS.data_root, file_id))
	                rinput = misc.imread(('%s/image_1/%06d_10.png') % (FLAGS.data_root, file_id))
	         
	
	            linput = (linput - linput.mean()) / linput.std()
	            rinput = (rinput - rinput.mean()) / rinput.std()
	
	            linput = linput.reshape(1, linput.shape[0], linput.shape[1], num_channels)
	            rinput = rinput.reshape(1, rinput.shape[0], rinput.shape[1], num_channels)
	
	            test_dict = {limage:linput, rimage:rinput, snet['is_training']: False}
	            limage_map, rimage_map = session.run([snet['lbranch'], snet['rbranch']], feed_dict=test_dict)
	
	            map_width = limage_map.shape[2]
	            unary_vol = np.zeros((limage_map.shape[1], limage_map.shape[2], FLAGS.disp_range))
	
	            for loc in range(FLAGS.disp_range):
	                x_off = -loc
	                l = limage_map[:, :, max(0, -x_off): map_width,:]
	                r = rimage_map[:, :, 0: min(map_width, map_width + x_off),:]
	                res = session.run(map_prod, feed_dict={lmap: l, rmap: r})
	
	                unary_vol[:, max(0, -x_off): map_width, loc] = res[0, :, :]
	
	            print('Image %s processed.' % (i + 1))
	            pred = np.argmax(unary_vol, axis=2) * scale_factor
	
	
	            misc.imsave('%s/disp_map_%06d_10.png' % (FLAGS.out_dir, file_id), pred)
	

    

