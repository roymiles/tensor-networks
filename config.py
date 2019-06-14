""" Generic configuration file with no dependencies
    For example, learning rate, batch size etc """

working_dir = "/media/roy/New Volume/"

# Where all the tfds datasets are downloaded to
tfds_dir = working_dir + "tensorflow-datasets"

# Where tensorboard logs are stored
log_dir = working_dir + "Tensorboard"

# Where checkpoints are saved (.ckpt)
ckpt_dir = working_dir + "Checkpoints"

# Where tflite files are exported
tflite_dir = working_dir + "tflite"

# Where the standard models are saved (.pbtxt)
save_dir = working_dir + "Models"

# Where the class labels are stored
labels_dir = working_dir + "Labels"