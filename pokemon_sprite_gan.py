from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate

import matplotlib.pyplot as plt
import os

from pokemon_data_creator import *


IMAGE_SIZE = 128
EPOCHS = 1000
BATCH_SIZE = 64

ngf = 64  # Number of feature maps in generator
ndf = 64  # Number of feature maps in discriminator
nc = 3  # Number of channels

n_types = 19

NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16
SEED = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])
TYPE_SEED = create_random_types(NUM_EXAMPLES_TO_GENERATE)


CROSS_ENTROPY = tf.keras.losses.BinaryCrossentropy(from_logits=True)
CROSS_ENTROPY_SMOOTH = tf.keras.losses.BinaryCrossentropy(
    from_logits=True, label_smoothing=0.1)

generator_optimizer = Adam(2e-4, beta_1=0.5)
discriminator_optimizer = Adam(2e-4, beta_1=0.5)

CHECKPOINT_DIR = './training_checkpoints'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")


''' Wasserstein Loss
Discriminator loss: D(x) - D(G(z))
Generator loss: D(G(z))
For
D(x) the discriminator output
G(z) the generator output with noise z
Use smoothing on real output
'''


def discriminator_loss(disc_real_output, disc_fake_output):
    real_loss = CROSS_ENTROPY(
        tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = CROSS_ENTROPY(tf.zeros_like(
        disc_fake_output), disc_fake_output)
    return real_loss + fake_loss


def generator_loss(disc_fake_output):
    return CROSS_ENTROPY(tf.ones_like(disc_fake_output), disc_fake_output)


'''
Convolutional Layers
'''


def generator_conv2dT_layer(network, channels, kernel=(5, 5), strides=(2, 2), padding='same', alpha=0.3, dropout=0.3):
    '''Convolutional transpose layer for the generator'''
    network = Conv2DTranspose(
        channels, kernel, strides=strides, padding=padding, use_bias=False)(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=alpha)(network)
    return network


def discriminator_conv2d_layer(network, channels, kernel=(5, 5), strides=(2, 2), padding='same', alpha=0.3, dropout=0.3):
    '''Convolutional layer for the discriminator'''
    network = Conv2D(channels, (5, 5), strides=strides,
                     padding=padding)(network)
    network = LeakyReLU(alpha=alpha)(network)
    network = Dropout(dropout)(network)
    return network


'''
Models
'''


def make_generator_model():
    '''
    Generator with type and noise inputs
    '''
    first_layer_image_subdivision = 8
    layer_size = IMAGE_SIZE // first_layer_image_subdivision
    ngf_slide = ngf
    in_label = Input(shape=(n_types,))
    label_layer = Dense(layer_size * layer_size * n_types)(in_label)
    label_layer = Reshape((layer_size, layer_size, n_types))(label_layer)

    in_img_noise = Input(shape=(100,))
    image_layer = Dense(
        layer_size * layer_size * ngf, use_bias=False, input_shape=(100,))(in_img_noise)
    image_layer = BatchNormalization()(image_layer)
    image_layer = LeakyReLU()(image_layer)
    image_layer = Reshape(
        (layer_size, layer_size, ngf))(image_layer)
    # output_shape == (None, IMAGE_SIZE / 8, IMAGE_SIZE / 8, ngf)

    generator_network = Concatenate()([image_layer, label_layer])
    while(first_layer_image_subdivision > 2):
        ngf_slide //= 2
        generator_network = generator_conv2dT_layer(
            generator_network, ngf_slide)
        first_layer_image_subdivision //= 2
    # output_shape == (None, IMAGE_SIZE / 4, IMAGE_SIZE / 4, ngf / 2)
    # output_shape == (None, IMAGE_SIZE / 2, IMAGE_SIZE / 2, ngf / 4)

    out_layer = Conv2DTranspose(
        nc, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(generator_network)
    model = Model([in_img_noise, in_label], out_layer)
    # output_shape == (None, IMAGE_SIZE, IMAGE_SIZE, nc)

    return model


def make_discriminator_model():
    '''
    Discriminator with type and image inputs
    '''
    n_conv_layers = 3
    ndf_slide = ndf // (2 ** n_conv_layers)
    in_label = Input(shape=(n_types,))
    label_layer = Dense(IMAGE_SIZE * IMAGE_SIZE * n_types)(in_label)
    label_layer = Reshape((IMAGE_SIZE, IMAGE_SIZE, n_types))(label_layer)

    in_image = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, nc))

    discriminator_network = Concatenate()([in_image, label_layer])

    for layer in range(n_conv_layers):
        discriminator_network = discriminator_conv2d_layer(
            discriminator_network, ndf_slide)
        ndf_slide *= 2
    # output_shape == (None, IMAGE_SIZE / 2, IMAGE_SIZE / 2, ndf / 8)
    # output_shape == (None, IMAGE_SIZE / 4, IMAGE_SIZE / 4, ndf / 4)
    # output_shape == (None, IMAGE_SIZE / 8, IMAGE_SIZE / 8, ndf / 2)

    discriminator_network = Flatten()(discriminator_network)
    out_layer = Dense(1)(discriminator_network)

    model = Model([in_image, in_label], out_layer)
    return model


'''
Running
'''


def train_step(generator, discriminator, batch):
    input_image = batch['img_raw']
    input_types = batch['type']

    noise = tf.random.normal([BATCH_SIZE, 100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        if(batch['type'].shape[0] < BATCH_SIZE):
            noise = tf.slice(noise, [0, 0], [input_types.shape[0], 100])
        gen_output = generator([noise, input_types], training=True)

        disc_real_output = discriminator(
            [input_image, input_types], training=True)
        disc_generated_output = discriminator(
            [gen_output, input_types], training=True)

        gen_loss = generator_loss(disc_generated_output)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
    # print(f'Generator loss:{gen_loss} | Discriminator loss:{disc_loss}')


def train(generator_model, discriminator_model, dataset, checkpoint, epochs=EPOCHS):
    for epoch in range(epochs):
        # augmented_dataset = dataset.map(test_augment)
        augmented_dataset = dataset
        for batch in augmented_dataset:
            train_step(generator_model,
                       discriminator_model, batch)
        if (epoch + 1) % 100 == 0:
            generate_and_save_images(generator_model,
                                     epoch + 1,
                                     SEED,
                                     TYPE_SEED)

        if (epoch + 1) % 500 == 0:
            checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

    generate_and_save_images(generator_model,
                             epochs,
                             SEED,
                             TYPE_SEED)
    generator_model.save('cgan_generator.h5')


def run(dataset):
    generator_model = make_generator_model()
    discriminator_model = make_discriminator_model()
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator_model,
                                     discriminator=discriminator_model)
    train(generator_model, discriminator_model, dataset, checkpoint)


def generate_and_save_images(model, epoch, test_input_noise, test_input_types):
    predictions = model([test_input_noise, test_input_types], training=False)
    plt.figure(figsize=(8, 8))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = predictions[i].numpy()
        img = 1 / (img.max() - img.min()) * (img - img.min())
        img = (img * 255).astype(int)
        plt.imshow(img)
        plt.xlabel(multi_decode_pokemon_type(test_input_types[i].numpy()))
    plt.savefig('./training/3image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def print_one_image_from_dataset(batched_dataset):
    for batch in batched_dataset:
        plt.figure()
        img = batch['img_raw'][0].numpy()
        img = img * 127.5 + 127.5
        img = img.astype(int)
        plt.imshow(img)
        plt.show()
        break


prep_tfrecords_from_csv(image_size=IMAGE_SIZE)
run(read_tfrecord(batch_size=BATCH_SIZE))
