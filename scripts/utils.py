import tensorflow as tf

def create_input_pipeline(input_queue, patch_size, disparity_range, nrof_preprocess_threads, batch_size_placeholder):


    images_and_labels = []
    half_patch = patch_size // 2
    half_range = disparity_range // 2

    for _ in range(nrof_preprocess_threads):

           # Dequeue elements from the input queue
           locs, label = input_queue.dequeue()
           l_images = []
           r_images = []

           for loc in tf.unstack(locs):

               # Read and preprocess left and right images

               l_image = read_and_process_image(loc[0])
               r_image = read_and_process_image(loc[1])

               loc_type = tf.string_to_number(loc[2], tf.int32)
               x_center = tf.string_to_number(loc[3], tf.int32)
               y_center = tf.string_to_number(loc[4], tf.int32)
               r_x_center =  tf.string_to_number(loc[5], tf.int32)

               left_patch = _crop_image(l_image, y_center-half_patch-1, x_center-half_patch-1, patch_size, patch_size)
               right_patch = _crop_image(r_image, y_center-half_patch-1, r_x_center-half_patch-half_range-1, patch_size, patch_size+disparity_range-1)


               l_images.append(left_patch)
               r_images.append(right_patch)

           images_and_labels.append([l_images, r_images, label])

    l_batch, r_batch, labels_batch = tf.train.batch_join(
                                   images_and_labels,
                                   batch_size=batch_size_placeholder,
                                   shapes=[(patch_size, patch_size, 3),(patch_size,patch_size+disparity_range-1,3), (disparity_range)],
                                   enqueue_many=True,
                                   capacity=4 * nrof_preprocess_threads * 100,
                                   allow_smaller_final_batch=True)

    return l_batch, r_batch, labels_batch

def read_and_process_image(filename, image_size=None, channels=3):

        file_content = tf.read_file(filename)
        image = tf.image.decode_png(file_content,channels=channels)

        if image_size!=None:
            image = tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1])

        image = tf.image.per_image_standardization(tf.to_float(image))
        return image

def _crop_image(image, offset_height, offset_width, crop_height, crop_width):

    original_shape = tf.shape(image)
    size_assertion = tf.Assert(tf.logical_and(
        	tf.greater_equal(original_shape[0], crop_height),
        	tf.greater_equal(original_shape[1], crop_width)),
        	['Crop size greater than the image size.'])
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])
    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, begin=offsets, size=cropped_shape)
    return tf.reshape(image, cropped_shape)
