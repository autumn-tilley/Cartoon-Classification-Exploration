import os
import sys
import argparse
from datetime import datetime
import tensorflow as tf

from PIL import Image
import hyperparameters_sc as hp_sc

from model_sc import YourModel_sc, VGGModel_sc
from preprocess_sc import Datasets_sc

from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, CustomModelSaver
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        required=True,
        choices=['1','2' ,'3','4','5', '6'],
        help='''Which task of the assignment to run -
        training from scratch (1), or fine tuning VGG-16 (3).''')
    parser.add_argument(
        '--data',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--imagePath',
        default='..'+os.sep+'image'+os.sep,
        help='Location where the input image is stored.')
    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 3).''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')

    return parser.parse_args()

def train(model, datasets, checkpoint_path, logs_path, init_epoch, task):
    """ Training routine. """

    # Training using VGG for all Scenes
    if task == '1':
        callback_list = [
            tf.keras.callbacks.TensorBoard(
                log_dir=logs_path,
                update_freq='batch',
                profile_batch=0),
            ImageLabelingLogger(logs_path, datasets),
            CustomModelSaver(checkpoint_path, ARGS.task, hp_sc.max_num_weights)
        ]

        # Begin training
        model.fit(
            x=datasets.train_data,
            validation_data=datasets.test_data,
            epochs=hp_sc.num_epochs,
            batch_size=None,
            callbacks=callback_list,
            initial_epoch=init_epoch,
        )  

    elif task == '2':
        callback_list = [
            tf.keras.callbacks.TensorBoard(
                log_dir=logs_path,
                update_freq='batch',
                profile_batch=0),
            ImageLabelingLogger(logs_path, datasets),
            CustomModelSaver(checkpoint_path, ARGS.task, hp_sc.max_num_weights)
        ]

        # Begin training
        model.fit(
            x=datasets.train_data,
            validation_data=datasets.test_data,
            epochs=hp_sc.num_epochs,
            batch_size=None,
            callbacks=callback_list,
            initial_epoch=init_epoch,
        )
        # model.load_weights("your_weights.h5")   # What if I remove this  

    elif task == '3':
        # Keras callbacks for training
        callback_list = [
            tf.keras.callbacks.TensorBoard(
                log_dir=logs_path,
                update_freq='batch',
                profile_batch=0),
            ImageLabelingLogger(logs_path, datasets),
            CustomModelSaver(checkpoint_path, ARGS.task, hp_sc.max_num_weights)
        ]

        # Begin training
        model.fit(
            x=datasets.train_data,
            validation_data=datasets.test_data,
            epochs=hp_sc.num_epochs,
            batch_size=None,
            callbacks=callback_list,
            initial_epoch=init_epoch,
        )

    else: 
        pass

def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )

def predict(model, image_path, preprocess_fn):
    """Predict the catagory of the image using a pre-trained model."""

    # Load the image and resize it to the model's expected input size
    input_image = Image.open(image_path)
    input_image = np.array(input_image)
    
    image = resize(input_image, (hp_sc.img_size, hp_sc.img_size, 3), preserve_range=True, mode='reflect', anti_aliasing=True)

    # Preprocess the image and expand the dimensions to match the model's input shape
    input_image = preprocess_fn(image)
    input_image = np.expand_dims(input_image, axis=0)

    # Predict the scene using the model
    predictions = model(input_image)
    

    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)

    print(predicted_class)

    # Get the sorted class names from the test folders
    if ARGS.task == "5" or ARGS.task == "6":
        class_names = get_class_names("../cartoon-or-not/test")
    else:
        class_names = get_class_names("../15_Scene/test")

    # Convert the predicted class number to the class name
    predicted_class_name = scene_num_to_scene_name(predicted_class, class_names)

    # Return the predicted class name
    return predicted_class_name

def get_class_names(folder_path):
    """Get a sorted list of class names from the folder path."""

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Invalid folder path: {folder_path}")
        return None

    # Get the subdirectories in the folder
    class_dirs = [entry.name for entry in os.scandir(folder_path) if entry.is_dir()]

    # Sort the class names
    class_names = sorted(class_dirs)

    return class_names

def scene_num_to_scene_name(scene_num, class_names):
    """Convert a class number to the corresponding class name."""

    if scene_num < 0 or scene_num >= len(class_names):
        print(f"Invalid scene number: {scene_num}")
        return None

    return class_names[scene_num]


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)

    # Run script from location of run.py
    os.chdir(sys.path[0])

    datasets = None

    if ARGS.task == '1':
        datasets = Datasets_sc(ARGS.data, ARGS.task)
        model = VGGModel_sc(ARGS.task)
        checkpoint_path = "checkpoints" + os.sep + \
            "vgg_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "vgg_model" + \
            os.sep + timestamp + os.sep
        model(tf.keras.Input(shape=(224, 224, 3)))

        # Print summaries for both parts of the model
        model.vgg16.summary()
        model.head.summary()

        # Load base of VGG model
        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)

    elif ARGS.task == '2':
        datasets = Datasets_sc(ARGS.data, ARGS.task)
        model = YourModel_sc(ARGS.task)
        model(tf.keras.Input(shape=(hp_sc.img_size, hp_sc.img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "your_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "your_model" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()

    elif ARGS.task == '3':
        datasets = Datasets_sc(ARGS.data, ARGS.task)
        model = VGGModel_sc(ARGS.task)
        checkpoint_path = "checkpoints" + os.sep + \
            "vgg_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "vgg_model" + \
            os.sep + timestamp + os.sep
        model(tf.keras.Input(shape=(224, 224, 3)))

        # Print summaries for both parts of the model
        model.vgg16.summary()
        model.head.summary()

        # Load base of VGG model
        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)

    elif ARGS.task == '4':
        datasets = Datasets_sc(ARGS.data, ARGS.task)
        # Load the pre-trained model for scene classification
        model_path = "vgg_weights_scene.h5"
        model = VGGModel_sc(ARGS.task)
        model(tf.keras.Input(shape=(224, 224, 3)))

        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)
        model.head.load_weights(model_path, by_name=True)

        # Provide the path to the input image
        # input_image_path = "../data/15_Scene/test/Office/image_0001.jpg"
        input_image_path = ARGS.imagePath

        # Predict the scene in the input image using the pre-trained model
        predicted_class = predict(model, input_image_path, datasets.preprocess_fn)
        print("Predicted class: ", predicted_class)
        

    elif ARGS.task == '5':
        datasets = Datasets_sc(ARGS.data, ARGS.task)
        # Load the pre-trained model for scene classification
        model_path = "cartoon_or_not_your.h5"
        model = YourModel_sc(ARGS.task)
        model(tf.keras.Input(shape=(224, 224, 3)))

        model.load_weights(model_path, by_name=True)

        # Provide the path to the input image
        # input_image_path = "../data/15_Scene/test/Office/image_0001.jpg"
        input_image_path = ARGS.imagePath

        # Predict the scene in the input image using the pre-trained model
        predicted_class = predict(model, input_image_path, datasets.preprocess_fn)
        print("Predicted class: ", predicted_class)

    elif ARGS.task == '6':
        datasets = Datasets_sc(ARGS.data, ARGS.task)
        # Load the pre-trained model for scene classification
        model_path = "cartoon_or_not_vgg.h5"
        model = VGGModel_sc(ARGS.task)
        model(tf.keras.Input(shape=(224, 224, 3)))

        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)
        model.head.load_weights(model_path, by_name=True)

        # Provide the path to the input image
        # input_image_path = "../data/15_Scene/test/Office/image_0001.jpg"
        input_image_path = ARGS.imagePath

        # Predict the scene in the input image using the pre-trained model
        predicted_class = predict(model, input_image_path, datasets.preprocess_fn)
        print("Predicted class: ", predicted_class)

    else:
        model = None #REPLACE WITH BOTH
        checkpoint_path = "checkpoints" + os.sep + \
            "vgg_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "vgg_model" + \
            os.sep + timestamp + os.sep
        model(tf.keras.Input(shape=(224, 224, 3)))

        # Print summaries for both parts of the model
        model.vgg16.summary()
        model.head.summary()

        # Load base of VGG model
        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)

    # # Make checkpoint directory if needed
    if ARGS.task == '1':
        if not ARGS.evaluate and not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
    if ARGS.task == '2':
        if not ARGS.evaluate and not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
    if ARGS.task == '3':
        if not ARGS.evaluate and not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    if ARGS.task == '4':
        checkpoint_path = None
        logs_path = None
        train(model, datasets, checkpoint_path, logs_path, init_epoch, ARGS.task)
    if ARGS.task == '5':
        checkpoint_path = None
        logs_path = None
        train(model, datasets, checkpoint_path, logs_path, init_epoch, ARGS.task)
    if ARGS.task == '6':
        checkpoint_path = None
        logs_path = None
        train(model, datasets, checkpoint_path, logs_path, init_epoch, ARGS.task)
    else: 
        train(model, datasets, checkpoint_path, logs_path, init_epoch, ARGS.task)


# Make arguments global
ARGS = parse_args()

main()