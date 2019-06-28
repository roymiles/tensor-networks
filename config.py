""" Generic configuration file with no dependencies
    For example, learning rate, batch size etc """

import os

working_dir = "/media/roy/New Volume/"
print(f"Working directory: {working_dir}")

# Where all the tfds datasets are downloaded to
tfds_dir = working_dir + "tensorflow-datasets"

# The output directory
output_dir = working_dir + "output/"

# General logs
log_dir = output_dir + "logs"

# Profiling
profiling_dir = output_dir + "profiling"

# Tensorboard
tensorboard_dir = output_dir + "tensorboard"

# Where checkpoints are saved (.ckpt)
checkpoint_dir = output_dir + "checkpoints"

# Where tflite files are exported
tflite_dir = output_dir + "tflite"

# Where the standard models are saved (.pbtxt)
save_dir = output_dir + "models"

# Where the class labels are stored
labels_dir = output_dir + "labels"

# If the directories do not exist, make them
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(profiling_dir):
    os.makedirs(profiling_dir)

if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if not os.path.exists(tflite_dir):
    os.makedirs(tflite_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)
