import os
import tensorflow as tf
from time import sleep

ID = os.environ['ID']

TRAIN_DIR = '/data/train'
LOG_DIR = '/data/volume/logs'

TRAINING_EPOCHS = 300

def train():
  print("Tensorflow: ", tf.__version__)

  print()
  print("[TRAIN FILES]")
  for file in sorted(os.listdir(TRAIN_DIR)):
    print("  ", file)
  print()
  print()

  tf.reset_default_graph()
  x_scalar = tf.get_variable('x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
  first_summary = tf.summary.scalar(name='My_first_scalar_summary', tensor=x_scalar)

  init = tf.global_variables_initializer()

  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    writer = tf.summary.FileWriter(os.path.join(LOG_DIR, ID), sess.graph)
    
    for epoch in range(TRAINING_EPOCHS):
      print("epoch: ", epoch)
      
      sess.run(init)
      summary = sess.run(first_summary)
      writer.add_summary(summary, epoch)

      sleep(1)

if __name__ == "__main__":
  train()