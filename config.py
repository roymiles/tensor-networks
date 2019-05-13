""" Generic configuration file with no dependencies
    For example, learning rate, batch size etc """

# Show some pretty debugging messages
is_debugging = False

# Set random seeds
seed = 1234

# Number of epochs to run
epochs = 12

batch_size = 128

# This may be annealed over training steps
initial_learning_rate = 0.01

# Where all the tfds datasets are downloaded to
tfds_dir = "/media/roy/New Volume/tensorflow-datasets/"

# Dataset to load
# ['abstract_reasoning', 'bair_robot_pushing_small', 'caltech101', 'cats_vs_dogs', 'celeb_a', 'celeb_a_hq', 'chexpert',
# 'cifar10', 'cifar100', 'cifar10_corrupted', 'cnn_dailymail', 'coco2014', 'colorectal_histology',
# 'colorectal_histology_large', 'cycle_gan', 'diabetic_retinopathy_detection', 'dsprites', 'dtd',
# 'dummy_dataset_shared_generator', 'dummy_mnist', 'emnist', 'fashion_mnist', 'flores', 'glue', 'groove', 'higgs',
# 'horses_or_humans', 'image_label_folder', 'imagenet2012', 'imagenet2012_corrupted', 'imdb_reviews', 'iris', 'kmnist',
# 'lm1b', 'lsun', 'mnist', 'moving_mnist', 'multi_nli', 'nsynth', 'omniglot', 'open_images_v4', 'oxford_flowers102',
# 'oxford_iiit_pet', 'para_crawl', 'quickdraw_bitmap', 'rock_paper_scissors', 'shapes3d', 'smallnorb', 'squad',
# 'starcraft_video', 'sun397', 'svhn_cropped', 'ted_hrlr_translate', 'ted_multi_translate', 'tf_flowers', 'titanic',
# 'ucf101', 'voc2007', 'wikipedia', 'wmt15_translate', 'wmt16_translate', 'wmt17_translate', 'wmt18_translate',
# 'wmt19_translate', 'wmt_translate', 'xnli']
tfds_dataset_name = "imagenet2012"
num_classes = 100
