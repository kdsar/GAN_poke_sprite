from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from random import sample


record_file_number = 0
RECORD_PATH = "./tfrecord/"
record_file_name = ("train-%.3d.tfrecords" % record_file_number)
IMAGE_COLOR_MODEL = 'RGB'
IMAGE_MODEL_CHANNELS = 3
IMAGE_SIZE_RESCALED = 128
BATCH_SIZE = 64


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_SPRITES = os.path.join(DIR_PATH, 'data/main-sprites')
GAME_SPRITES_DIRS = ['black-white', 'heartgold-soulsilver']
CSV_PATH = os.path.join(DIR_PATH, 'data/pokemon.csv')


pokemon_type_map = {
    'None': 0,
    'Normal': 1,
    'Fighting': 2,
    'Flying': 3,
    'Poison': 4,
    'Ground': 5,
    'Rock': 6,
    'Bug': 7,
    'Ghost': 8,
    'Steel': 9,
    'Fire': 10,
    'Water': 11,
    'Grass': 12,
    'Electric': 13,
    'Psychic': 14,
    'Ice': 15,
    'Dragon': 16,
    'Dark': 17,
    'Fairy': 18,
}
inv_pokemon_type_map = {v: k for k, v in pokemon_type_map.items()}
number_of_types = 19


def prep_tfrecords_from_csv(image_size=IMAGE_SIZE_RESCALED):
    '''
    Goes through the data paths and saves the necessary information to a list of tfrecord files

    Parameters:
    image_size (int): Size that the image data should ve rescaled to
    '''
    writer = tf.io.TFRecordWriter(RECORD_PATH + record_file_name)
    df = pd.read_csv(CSV_PATH)
    for index, row in df.iterrows():
        for game_sprites_dir in GAME_SPRITES_DIRS:
            single_game_path = os.path.join(DIR_SPRITES, game_sprites_dir)
            sprite_path = os.path.join(
                single_game_path, str(row['#']) + '.png')
            if not (os.path.isfile(sprite_path)):
                print(str(row['#']) + " " + row['Name'] +
                      " doesn't have a file in " + game_sprites_dir)
                continue
            img = Image.open(sprite_path).convert('RGBA')
            data = img.getdata()
            newData = []
            for item in data:
                # If the opacity is 0, set the color to white
                if item[3] == 0:
                    newData.append((255, 255, 255, 0))
                else:
                    newData.append(item)
            img.putdata(newData)
            img = img.convert(IMAGE_COLOR_MODEL)
            img = img.resize((image_size, image_size))

            writer.write(create_tf_example(img, row).SerializeToString())
            # Start on a new tfrecord if it reaches the maximum size
            if(os.path.getsize(RECORD_PATH + record_file_name) > 100000000):
                writer = update_record_file()
                print('New file after ' + row['Name'])
    writer.close()


def update_record_file():
    '''
    Set the next record file

    Returns:
    tf.io.TFRecordWriter: the writer class using the new file
    '''
    global record_file_number, record_file_name
    record_file_number += 1
    record_file_name = ("train-%.3d.tfrecords" % record_file_number)
    with open(RECORD_PATH + record_file_name, 'w'):
        pass
    return tf.io.TFRecordWriter(RECORD_PATH + record_file_name)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if type(value) != list:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_tf_example(img, data):
    '''
    Create the data for a single pokemon to be stored in the tfrecord

    Parameters:
    img (Image): the image data
    data (Series): all of the other data found in the csv

    Returns:
    (tf.Example): The mapping of the data
    '''
    img_array = np.array(img)
    img_raw = img.tobytes()
    # For monotype, add 'None' as the second type
    if(pd.isna(data['Type 2'])):
        data['Type 2'] = 'None'
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': _bytes_feature(img_raw),
        'name': _bytes_feature(bytes(data['Name'], 'utf8')),
        'type1': _int64_feature(_hot_encode_pokemon_type(data['Type 1'])),
        'type2': _int64_feature(_hot_encode_pokemon_type(data['Type 2'])),
        'type': _int64_feature(_multi_hot_encode_pokemon_type([data['Type 1'], data['Type 2']])),
        'hp': _int64_feature(data['HP']),
        'atk': _int64_feature(data['Attack']),
        'def': _int64_feature(data['Defense']),
        'spatk': _int64_feature(data['Sp. Atk']),
        'spdef': _int64_feature(data['Sp. Def']),
        'speed': _int64_feature(data['Speed']),
        'generation': _int64_feature(data['Generation']),
        'legendary': _int64_feature(data['Legendary']),
    }))
    return tf_example


def _multi_hot_encode_pokemon_type(types):
    '''Returns a list of hot encoded types'''
    hot_encoded = [0] * number_of_types
    for type in types:
        hot_encoded[pokemon_type_map.get(type, 0)] = 1
    return hot_encoded


def multi_decode_pokemon_type(one_hot_type):
    '''Decodes a hot encoded list to a list of strings for each type present'''
    type_list = []
    index = 0
    for type in one_hot_type:
        if type == 1:
            type_list.append(inv_pokemon_type_map[index])
        index += 1
    return type_list


def _hot_encode_pokemon_type(type):
    '''Return a one hot encoded list for a single type'''
    hot_encoded = [0] * number_of_types
    hot_encoded[pokemon_type_map.get(type, 0)] = 1
    return hot_encoded


def _decode_pokemon_type(one_hot_type):
    '''Deocde a hot encoded list with a single type and returns that type as a string'''
    index = tf.argmax(one_hot_type, axis=0)
    return inv_pokemon_type_map[index.numpy()]


def create_random_types(number_to_create):
    '''
    Create a number of fake random types
    '''
    return_seed = np.zeros([number_to_create, number_of_types], 'int')
    for i in range(number_to_create):
        indices = sample(range(0, number_of_types - 1), 2)
        indices = [0, 12]
        for index in indices:
            return_seed[i][index] = 1
    return_seed = tf.convert_to_tensor(return_seed)
    return return_seed


def read_tfrecord(batch_size=BATCH_SIZE):
    '''
    Read the tfrecord path to get the corresponding dataset

    Parameters:
    batch_size (int): Size used for batching

    Return:
    (Dataset): A batched and shuffled dataset
    '''
    total_dataset = None
    for tfrecord_file in os.listdir(os.path.join(DIR_PATH, RECORD_PATH)):
        # Skip dot files
        if tfrecord_file.startswith('.'):
            continue
        tfrecord_file = os.path.join(RECORD_PATH, tfrecord_file)
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
        if(total_dataset is None):
            total_dataset = raw_dataset
        else:
            total_dataset = total_dataset.concatenate(raw_dataset)
    parsed_dataset = total_dataset.map(
        _parse_data_function).shuffle(1176).batch(batch_size)
    return parsed_dataset


def _parse_data_function(example_proto):
    '''
    Function used for mapping the tfrecord information to usable data
    '''
    data_feature_description = {
        'img_raw': tf.io.FixedLenFeature([], tf.string),
        'name': tf.io.FixedLenFeature([], tf.string),
        'type1': tf.io.FixedLenFeature([number_of_types], tf.int64),
        'type2': tf.io.FixedLenFeature([number_of_types], tf.int64),
        'type': tf.io.FixedLenFeature([number_of_types], tf.int64),
        'hp': tf.io.FixedLenFeature([1], tf.int64),
        'atk': tf.io.FixedLenFeature([1], tf.int64),
        'def': tf.io.FixedLenFeature([1], tf.int64),
        'spatk': tf.io.FixedLenFeature([1], tf.int64),
        'spdef': tf.io.FixedLenFeature([1], tf.int64),
        'speed': tf.io.FixedLenFeature([1], tf.int64),
        'generation': tf.io.FixedLenFeature([1], tf.int64),
        'legendary': tf.io.FixedLenFeature([1], tf.int64),
    }
    # Parse the input tf.Example proto using the dictionary above.
    features = tf.io.parse_single_example(
        example_proto, data_feature_description)
    img = tf.io.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [IMAGE_SIZE_RESCALED,
                           IMAGE_SIZE_RESCALED, IMAGE_MODEL_CHANNELS])
    img = (tf.dtypes.cast(img, tf.float32) - 127.5) / 127.5
    features['img_raw'] = img
    return features


def plot_dataset_images(batched_dataset):
    '''Plot a number of images from the dataset for testing purposes'''
    counter = 0
    plt.figure(figsize=(15, 10))
    for batch in batched_dataset:
        for item in range(BATCH_SIZE):
            plt.subplot(10, 15, counter + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            img = batch['img_raw'][item].numpy()
            img = img * 127.5 + 127.5
            img = img.astype(int)

            plt.imshow(img)
            plt.xlabel(batch['name'][item].numpy().decode("utf-8"))
            counter += 1
            if(counter > 149):
                break
        if(counter > 149):
            break
    plt.show(block=False)
