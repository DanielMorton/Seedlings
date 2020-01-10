import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from argparse import ArgumentParser

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


CATEGORIES = 12
IMG_CHANNELS = 3        # Number of channels - RGB
IMG_HEIGHT = 299        # Height and width of the reshaped images.
IMG_WIDTH = IMG_HEIGHT
IMG_DIM = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
PATH = '/Users/dmorton/Plants/'
RANGE = 0.2
WEIGHTS = 'imagenet'


def main():
    parser = ArgumentParser()

    parser.add_argument("-bn", "--batch-normalization", action="store_true",
                        help="Apply batch normalization to input layer")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("-d", "--dropout", type=float, help="Dropout percentage to apply.")
    parser.add_argument("-e", "--epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument("-lr", "--learning-rate", type=float,
                        required=True,
                        help="Initial learning rate for training.")
    parser.add_argument("-r", "--reg", type=float, help="Regularization for final layer.")

    args = vars(parser.parse_args())

    trainBatches = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=RANGE,
        width_shift_range=RANGE,
        height_shift_range=RANGE,
        preprocessing_function=preprocess_input).flow_from_directory(PATH + 'train',
                                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                     class_mode='categorical', shuffle=True,
                                                                     batch_size=args["batch_size"])

    valBatches = ImageDataGenerator(
        preprocessing_function=preprocess_input).flow_from_directory(PATH + 'dev',
                                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                     class_mode='categorical', shuffle=False,
                                                                     batch_size=2 * args["batch_size"])

    best_model_file = PATH + f"IV3-AP_batch{args['batch_size']}_{'bn' if args['batch_normalization'] else ''}"
    if args["dropout"] is not None:
        best_model_file += f"_{args['dropout']}"
    if args["reg"] is not None:
        best_model_file += f"_reg{args['reg']}"
    best_model_file += f"_lr{args['learning_rate']}.h5"

    base_model = InceptionV3(weights=WEIGHTS,
                             include_top=False,
                             input_shape=IMG_DIM)

    input_tensor = Input(shape=IMG_DIM)
    if args["batch_normalization"]:
        bn = BatchNormalization()(input_tensor)
        x = base_model(bn)
    else:
        x = base_model(input_tensor)

    x = GlobalAveragePooling2D()(x)
    if args["dropout"] is not None:
        x = Dropout(args["dropout"])(x)
    output = Dense(CATEGORIES, activation="sigmoid",
                   kernel_regularizer=None if args["reg"] is None else l2(args["reg"]))(x)
    model = Model(input_tensor, output)

    callbacks = [CSVLogger("./training.log"),
                 EarlyStopping(monitor="val_loss", patience=8, verbose=1, min_delta=1e-4),
                 ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, cooldown=1,
                                   verbose=1, min_lr=1e-7),
                 ModelCheckpoint(filepath=best_model_file, verbose=1,
                                 save_best_only=True, save_weights_only=True, mode="auto")]

    model.compile(optimizer=Adam(lr=args["learning_rate"]),
                  loss="categorical_crossentropy",
                  metrics=["categorical_crossentropy", "accuracy"])

    model.fit_generator(generator=trainBatches,
                        verbose=1,
                        steps_per_epoch=trainBatches.n / trainBatches.batch_size,
                        epochs=args["epoch"],
                        validation_data=valBatches,
                        validation_steps=valBatches.n / valBatches.batch_size,
                        callbacks=callbacks)


if __name__ == '__main__':
    main()
