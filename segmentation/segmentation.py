import numpy as np
from tensorflow.keras.optimizers import Adam
from utils import read_img_nii, extract_dataset, create_path, save_best_model, scheduler, create_batch_of_path
from tensorflow.keras.callbacks import LearningRateScheduler
import time
import yaml
import argparse
from sklearn.model_selection import KFold
from model import build_model
from utils import My_Custom_Generator_segmentation, loss_gt, dice_coefficient


def training(model, checkpoint_best, reduce_lr, data_paths, batch_size=1, epochs_per_fold=4, nb_fold=5):
    cv = KFold(n_splits=nb_fold, shuffle=True)
    fold = 1
    start_time = time.time()
    paths, labels = create_batch_of_path(data_paths)

    for train, test in cv.split(data_paths):
        train_paths = []
        t_labels_paths = []
        val_paths = []
        v_labels_paths = []
        print('Fold:  ', fold)
        fold = fold + 1
        for i in list(train):
            path = paths[i]
            train_paths.append(path)
            label = labels[i]
            t_labels_paths.append(label)
        for i in list(test):
            path = paths[i]
            val_paths.append(path)
            label = labels[i]
            v_labels_paths.append(label)

    my_training_batch_generator = My_Custom_Generator_segmentation(train_paths, t_labels_paths, batch_size)
    validation_generator = My_Custom_Generator_segmentation(val_paths, v_labels_paths, batch_size)

    history = model.fit(my_training_batch_generator, validation_data=validation_generator,
                        steps_per_epoch=int(len(train_paths) // batch_size),
                        epochs=epochs_per_fold,
                        verbose=1,
                        callbacks=[checkpoint_best, reduce_lr])

    end_time = time.time() - start_time
    hours, rem = divmod(end_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Run time :")
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    return history


def main():
    dataset_path = config_file['path']['dataset']
    augmented_path = config_file['path']['augmented_data']
    checkpoint_path = config_file['path']['checkpoit_path']
    best_model = config_file['path']['best_model']
    save_model_path = config_file['path']['save_model_path']
    extract_dataset(dataset_path, 'Segmentation')
    extract_dataset(augmented_path, 'Augmented_Data')
    data_paths = create_path('./Segmentation', train=True) + create_path('./Augmented_Data', aug=True)
    input_shape = (4, 128, 128, 128)
    output_channels = 3
    resume_training = False
    number_of_images = len(data_paths)
    data = np.empty((len(data_paths[:number_of_images]),) + input_shape, dtype=np.float32)
    labels = np.empty((len(data_paths[:number_of_images]), input_shape[1], input_shape[2], input_shape[3]),
                      dtype=np.uint8)
    print('data shape : ', data.shape)
    print('labels shape : ', labels.shape)

    model = build_model(input_shape=input_shape, output_channels=output_channels)
    if bool(config_file['train_settings']['resume_training']):
        model.load_weights(best_model)
    checkpoint_best = save_best_model(checkpoint_path)
    reduce_lr = LearningRateScheduler(scheduler)
    model = build_model(input_shape=(4,128, 128, 128), output_channels=3, gradient_accumulation='True', n_gradients=8)
    model.compile(
        optimizer=Adam(lr=float(config_file['train_settings']['lr'])),
        loss=[loss_gt()],
        metrics=[dice_coefficient], experimental_run_tf_function=False
    )

    history = training(model, checkpoint_best, reduce_lr,
                       data_paths=data_paths, batch_size=int(config_file['train_settings']['train_batch_size']),
                       epochs_per_fold=int(config_file['train_settings']['epochs_per_fold']),
                       nb_fold=int(config_file['train_settings']['folds']))
    model.save_weights(save_model_path)
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", "-c", help="The location of config file", default='./config.yaml')
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path) as file:
        config_file = yaml.full_load(file)

    histoty = main()
