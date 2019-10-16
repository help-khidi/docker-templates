import os
import random
import torch
from tensorboardX import SummaryWriter
from time import sleep

ID = os.environ['ID']

TRAIN_DIR = '/data/train'
LOG_DIR = '/data/volume/logs'

TRAINING_EPOCHS = 300

def train():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Using device:', device)
  print()

  if device.type == 'cuda':
    for device_id in range(torch.cuda.device_count()):
      print(torch.cuda.get_device_name(device_id))
      print('Memory Usage:')
      print('Allocated:', round(torch.cuda.memory_allocated(device_id)/1024**3,1), 'GB')
      print('Cached:   ', round(torch.cuda.memory_cached(device_id)/1024**3,1), 'GB')
      print()

  print()
  print("[TRAIN FILES]")
  for file in sorted(os.listdir(TRAIN_DIR)):
    print("  ", file)
  print()
  print()

  # tf.reset_default_graph()
  # x_scalar = tf.get_variable('x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
  # first_summary = tf.summary.scalar(name='My_first_scalar_summary', tensor=x_scalar)

  # init = tf.global_variables_initializer()

  writer = SummaryWriter(os.path.join(LOG_DIR, ID))
    
  for epoch in range(TRAINING_EPOCHS):
    print("epoch: ", epoch)
    writer.add_scalar('My_first_scalar_summary', random.random(), epoch)

    # sleep(1)

if __name__ == "__main__":
  train()