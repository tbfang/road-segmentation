import gzip
import os
import re
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import code
from PIL import ImageFilter
import tensorflow.python.platform
from datetime import datetime
import matplotlib
import numpy
import tensorflow as tf

NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 100
TESTING_SIZE = 50
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 10
RESTORE_MODEL = True # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16 
# batch size of label image while extracting
LABEL_PARCH_SIZE = 4

# we store the event logs and checkpoint at 'mnist'
tf.app.flags.DEFINE_string('train_dir', 'mnist',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS

# Extract training image patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

# Extract label image patch from a given image
def img_crop_with_shift(im, w, h, shift_step):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,int(imgheight-(IMG_PATCH_SIZE-LABEL_PARCH_SIZE)),shift_step):
        for j in range(0,int(imgwidth-(IMG_PATCH_SIZE-LABEL_PARCH_SIZE)),shift_step):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        im = Image.open(image_filename)
        
        # perform image sharpen before we feed in CNN
        im_sharpen = im.filter(ImageFilter.SHARPEN)
        # zero pad the border, in order to obtain same length for train_data and train_label
        im_sharpen_zero_pad=ImageOps.expand(im_sharpen,border=int((IMG_PATCH_SIZE-LABEL_PARCH_SIZE)/2),fill='black')
        # save image since this image formate cannot be read by mpimg.imread
        im_sharpen_zero_pad.save('sharpen.png')
        # read shappened image
        img = mpimg.imread('sharpen.png')
        
        # color space changing, hsv color gives a better color
        img = matplotlib.colors.rgb_to_hsv(img) # img is a 400x400x3 array
        
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]

    img_patches = [img_crop_with_shift(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, LABEL_PARCH_SIZE) 
                   for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)

# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    # here, we crop the label image into a [LABEL_PARCH_SIZE, LABEL_PARCH_SIZE] patch
    gt_patches = [img_crop(gt_imgs[i], LABEL_PARCH_SIZE, LABEL_PARCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx][0] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

# convert the predicted image to an array with proper format but not concatenate with satImage
def test_images_non_concatenate(gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels != 3:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
    return gt_img_3c

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2) # using HSL instead of RGBA right now...
    return new_img



# assign a label to a patch
def patch_to_label(patch):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.mean(patch)
    if df > foreground_threshold:
        return 0 # changing this
    else:
        return 1


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


def main(argv=None):  # pylint: disable=unused-argument

    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/' 

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, TRAINING_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

    num_epochs = NUM_EPOCHS



    train_size = train_labels.shape[0]



   # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))
    
    train_all_data_node = tf.placeholder(tf.float32,shape = train_data.shape)
    #train_all_data_node = tf.constant(train_data)
    
    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(  # fully connected, depth 1024.
        tf.truncated_normal([int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 1024],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[1024]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([1024, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
    
    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(img, idx = 0):
        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(V)
        V = V - min_value
        max_value = tf.reduce_max(V)
        V = V / (max_value*PIXEL_DEPTH)
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V
    
    # Make an image summary for 3d tensor image with index idx
    def get_image_summary_3d(img):
        V = tf.slice(img, (0, 0, 0), (1, -1, -1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V
    
    # Get prediction for given input image 
    def get_prediction(img):
        data = numpy.asarray(img_crop_with_shift(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE, LABEL_PARCH_SIZE))
        #print('after crop the image',data.shape)
        data_node = tf.constant(data)
        output = tf.nn.softmax(model(data_node))
        output_prediction = s.run(output)
        #print(output_prediction.shape)
        imgwidth = img.shape[0] - (IMG_PATCH_SIZE-LABEL_PARCH_SIZE)
        imgheight = img.shape[1] - (IMG_PATCH_SIZE-LABEL_PARCH_SIZE)
        img_prediction = label_to_img(imgwidth, imgheight, LABEL_PARCH_SIZE, LABEL_PARCH_SIZE, output_prediction)
    
        return img_prediction
    
    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(filename, image_idx):
    
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)
    
        img_prediction = get_prediction(img)
        cimg = concatenate_images(img, img_prediction)
    
        return cimg
    # get prediction for the testing image but not concatenate with test image
    def get_prediction_with_groundtruth_non_concat(filename, image_idx):
    
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)
    
        img_prediction = get_prediction(img)
        cimg = test_images_non_concatenate(img_prediction)
    
        return cimg
    
    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(filename, image_idx):
    
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)
    
        img_prediction = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)
    
        return oimg
    
    # Get the prediction for given input testing file
    def get_prediction_for_testing_data(filename, image_idx, concate):
        
        folderid = "test_%d/" % image_idx
        imageid = "test_%d" % image_idx
        image_filename = filename + folderid + imageid + ".png"
    
        # load image that can be sharpen
        im = Image.open(image_filename)
        # perform image sharpen
        im_sharpen = im.filter(ImageFilter.SHARPEN)
        # zero pad the border
        im_sharpen_zero_pad=ImageOps.expand(im_sharpen,border=int((IMG_PATCH_SIZE-LABEL_PARCH_SIZE)/2),fill='black')
        im_sharpen_zero_pad.save('sharpen.png')
        img = mpimg.imread('sharpen.png')
    
        # changing the colorspace
        img = matplotlib.colors.rgb_to_hsv(img) # img is a 400x400x3 array
        
        #print('before feed in get_prediction',img.shape)    
        img_prediction = get_prediction(img)
        img_test = test_images_non_concatenate(img_prediction)
        if concate == 1:
            org_img = mpimg.imread(image_filename)
            img_test = concatenate_images(org_img, img_prediction)
        return img_test
    
    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
    
        conv2 = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
    
        # Uncomment these lines to check the size of each layer
        # print 'data ' + str(data.get_shape())
        # print 'conv ' + str(conv.get_shape())
        # print 'relu ' + str(relu.get_shape())
        # print 'pool ' + str(pool.get_shape())
        # print 'pool2 ' + str(pool2.get_shape())
    
    
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        #if train:
        #    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        out = tf.matmul(hidden, fc2_weights) + fc2_biases
    
        if train == True:
            summary_id = '_0'
            s_data = get_image_summary(data)
            filter_summary0 = tf.image_summary('summary_data' + summary_id, s_data)
            s_conv = get_image_summary(conv)
            filter_summary2 = tf.image_summary('summary_conv' + summary_id, s_conv)
            s_pool = get_image_summary(pool)
            filter_summary3 = tf.image_summary('summary_pool' + summary_id, s_pool)
            s_conv2 = get_image_summary(conv2)
            filter_summary4 = tf.image_summary('summary_conv2' + summary_id, s_conv2)
            s_pool2 = get_image_summary(pool2)
            filter_summary5 = tf.image_summary('summary_pool2' + summary_id, s_pool2)

        return out

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True) # BATCH_SIZE*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_labels_node))
    tf.scalar_summary('loss', loss)
    
    all_params_node = [conv1_weights, conv1_biases, conv2_weights, 
                       conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases]
    all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 
                        'conv2_biases', 'fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases']
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.scalar_summary(all_params_names[i], norm_grad_i)
    
    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers
    
    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    tf.scalar_summary('learning_rate', learning_rate)
    
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.0).minimize(loss,
                                                         global_step=batch)
    
    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    train_all_prediction = tf.nn.softmax(model(train_all_data_node))
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    # Create a local session to run this computation.
    with tf.Session(
                    
    config=tf.ConfigProto(
        intra_op_parallelism_threads=16,
        inter_op_parallelism_threads=16,
        use_per_session_threads=True
    )) as s:
    
    
        if RESTORE_MODEL:
            # Restore variables from disk.
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")
    
        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.initialize_all_variables().run()
    
            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                    graph_def=s.graph_def)
            print ('Initialized!')
            # Loop through training steps.
            print ('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))
    
            training_indices = range(train_size)
    
            for iepoch in range(num_epochs):
                
                print(datetime.now())
                # Permute training indices
                perm_indices = numpy.random.permutation(training_indices)
    
                for step in range (int(train_size / BATCH_SIZE)):
    
                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]
    
                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels,
                                 train_all_data_node: train_data}
    
                    if step % RECORDING_STEP == 0:
    
                        summary_str, _, l, lr, predictions = s.run(
                            [summary_op, optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)
                        #summary_str = s.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()
    
                        # print_predictions(predictions, batch_labels)
    
                        print ('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
                        print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))
    
                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)
    
                # Save the variables to disk.
                save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)
    
 
        
        
        # run testing data
        print ("Running prediction on testing set")
        prediction_test_dir = 'predictions_testing/'
        testing_data_dir = 'test_set_images/'

        if not os.path.isdir(prediction_test_dir):
            os.mkdir(prediction_test_dir)
        for i in range(1, TESTING_SIZE+1):
            print('processing image %d' %i)
            pimg = get_prediction_for_testing_data(testing_data_dir, i, 1)
            Image.fromarray(pimg).save(prediction_test_dir + "prediction_concat_" + str(i) + ".png")
            pimg_testing = get_prediction_for_testing_data(testing_data_dir, i, 0)
            Image.fromarray(pimg_testing).save(prediction_test_dir + "prediction_" + str(i) + ".png")
        print('testing image prediction finished!')  
        
        # creat the submission .csv file
        submission_filename = 'tf_submission.csv'
        image_filenames = []
        for i in range(1, TESTING_SIZE+1):
            image_filename = 'predictions_testing/prediction_%d' % i + '.png'
            print(image_filename)
            image_filenames.append(image_filename)
        masks_to_submission(submission_filename, *image_filenames)
        print('mask to submission finished!')

if __name__ == '__main__':
    tf.app.run()
