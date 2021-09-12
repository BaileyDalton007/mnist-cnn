import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorboard.plugins import projector

test_df = pd.read_csv(r'data\fashion-mnist_test.csv')
test_data = np.array(test_df, dtype='float32')

embed_count = 1600
xTest = test_data[:embed_count, 1:] / 255
yTest = test_data[:embed_count, 0]

logdir = r'C:\Users\Bailey\Desktop\Projects\OSUSARC\mnist-cnn\logs\embed'

summary_writer = tf.summary.FileWriter(logdir)

embedding_var = tf.Variable(xTest, name='fmnist_embedding')

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

embedding.metadata_path = os.path.join(logdir, 'metadata.tsv')
embedding.sprite.image_path = os.path.join(logdir, 'sprite.png')
embedding.sprite.single_image_dim.extend([28,28])

# passed in logdir instead of summary_writer
projector.visualize_embeddings(summary_writer, config)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(logdir, 'model.ckpt'))

rows = 28
cols = 28

label = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']

sprite_dim = int(np.sqrt(xTest.shape[0]))

sprite_image = np.ones((cols * sprite_dim, rows * sprite_dim))

index = 0
labels = []
for i in range(sprite_dim):
    for j in range(sprite_dim):
        
        labels.append(label[int(yTest[index])])

        sprite_image[
            i * cols: (i + 1) * cols,
            j * rows: (j + 1) * rows
        ] = xTest[index].reshape(28, 28) * -1 + 1

        index += 1

# Metadata file column 1 is index and column 2 is label
with open(embedding.metadata_path, 'w') as meta:
    meta.write('Index\tLabel\n')
    for index, label in enumerate(labels):
        meta.write('{}\t{}\n'.format(index, label))

plt.imsave(embedding.sprite.image_path, sprite_image, cmap='gray')
plt.imshow(sprite_image, cmap='gray')
plt.show()