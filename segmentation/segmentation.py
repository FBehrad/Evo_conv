from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
import time
import yaml
from sklearn.model_selection import KFold
import datetime
from utils import create_batch_of_path, my_custom_generator_segmentation, create_path, save_best_model, scheduler, loss_gt, dice_coefficient
from model import build_model


def training(model, callbacks, data_paths, batch_size=1, epochs_per_fold=4, nb_fold=5):
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

        my_training_batch_generator = my_custom_generator_segmentation(train_paths, t_labels_paths, batch_size)
        validation_generator = my_custom_generator_segmentation(val_paths, v_labels_paths, batch_size)

        history = model.fit(my_training_batch_generator, validation_data=validation_generator,
                            steps_per_epoch=int(len(train_paths) // batch_size),
                            epochs=epochs_per_fold,
                            verbose=1,
                            callbacks=callbacks)

    end_time = time.time() - start_time
    hours, rem = divmod(end_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Run time :")
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    return history


def main():
    path = open('/content/config.yaml', 'r')
    config = yaml.safe_load(path)
    model_param = config['model']
    input_size = config['preprocessing_seg']['optimal_roi']
    input_size = (4, input_size[0], input_size[1], input_size[2])

    data_paths = create_path('/content/preprocessed_data', train=True) + create_path('/content/augmented_data', aug=True)
    model = build_model(input_shape=input_size,
                        gradient_accumulation=model_param['accumulated_grad']['enable'],
                        n_gradients=model_param['accumulated_grad']['num_batch'])

    if bool(config['training']['resume_training']):
        model.load_weights(config['path']['best_model']).expect_partial()

    checkpoint_best = save_best_model(config['path']['checkpoint'])
    reduce_lr = LearningRateScheduler(scheduler)
    log_dir = "/content/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(
        optimizer=Adam(lr=float(config['training']['lr'])),
        loss=[loss_gt()],
        metrics=[dice_coefficient], experimental_run_tf_function=False
    )

    history = training(model, [checkpoint_best, reduce_lr, tensorboard_callback],
                       data_paths=data_paths, batch_size=int(config['training']['batch_size']),
                       epochs_per_fold=int(config['training']['epochs_per_fold']),
                       nb_fold=int(config['training']['number_fold']))
    model.save_weights(config['path']['last_model'])
    return history


if __name__ == '__main__':
    histoty = main()

