""" Generic configuration file with no dependencies
    For example, learning rate, batch size etc """

working_dir = "/media/roy/New Volume/"

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

