import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import os
from ml_params import max_nClass
from PSFitMLTools import checkLoopAtten
import matplotlib.pyplot as plt

def PCA(mlData):
    try:
        mlData.opt_iAttens()
    except AttributeError:
        mlData.loadRawTrainData()

    # train_ind = np.array(map(int,np.linspace(0,mlData.res_nums-1,mlData.res_nums*mlData.trainFrac)))

    res_nums = len(mlData.freqs)
    IQV_ratio = np.zeros((res_nums, max_nClass))
    # IQV_ratio = np.zeros((len(train_ind), max_nClass))
    for ir in range(res_nums):
    # for ir, r in enumerate(train_ind):
        for ia in range(max_nClass):
            loop_sat_cube = checkLoopAtten(mlData, res_num=ir, iAtten=ia, showFrames=False)
            max_ratio = loop_sat_cube[2]
            IQV_ratio[ir,ia] = max_ratio

    print np.shape(IQV_ratio)
    # max_iq = np.amax(trainImages[:,:,:,2], axis=2)
    # exit()
    sprite_images = np.zeros((res_nums,max_nClass,101))
    IQV_sprite = IQV_ratio
    # for r in range(res_nums):
    #     IQV_sprite[r] = IQV_sprite[r]*100./np.max(IQV_sprite[r])
    #     IQV_sprite[r] = map(int,IQV_sprite[r])
    #     for ia in range(max_nClass):
    #         sprite_images[r, ia, IQV_ratio[r,ia]] = 1
    
    plt.savefig('foo.png')

    LOG_DIR = '/tmp/emb_logs/'
    metadata = os.path.join(LOG_DIR, 'metadata.tsv')

    # mnist = input_data.read_data_sets('MNIST_data')

    #Variables
    images = tf.Variable(IQV_ratio[:200], name='images')

    with open(metadata, 'wb') as metadata_file:
        for i, row in enumerate(mlData.opt_iAttens[:200]):
            # row = np.argmax(row)
            metadata_file.write('%i\t%d\n' % (i,row))



    with tf.Session() as sess:
        saver = tf.train.Saver([images])

        sess.run(images.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = images.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = metadata
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
