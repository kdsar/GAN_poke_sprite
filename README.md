# GAN_poke_sprite

Personal project using Tensorflow in Python to create a neural network that generates images based on pokemon data

## The data

The data used is a CSV containing pertinent information on each pokemon as well as sprites of these pokemons in different games.  
For now, typing is used and hot encoded. The typing is then paired with the sprites, which are rescaled to 128px.  

The data is then saved to a TFRecord file for quicker reading.  

## The architecture

The architecture uses a Conditional Deep Convolutional Generative Adversarial Network.  
Adam is used as the optimizer.  
The Generator takes as input the typing as well as a noise vector of size 100 and outputs an image.  
The discriminator takes the typing and the image and outputs if the image is real or generated.  

With the current settings, the Generator has layers:  
A dense layer for the type input which is reshaped to (16, 16, 19)  
A dense layer for the noise input of size 100 which is reshaped to (16, 16, 64)  
These layers are then concatenated and 2 convolutional transpose layers follow, using batch normalization.  
These layers have shape (32, 32, 32) and (64, 64, 16)  
A final convolutional transpose with tanh activation is used for output. Its shape is (128, 128, 3)  

The discriminator has layers:  
A dense layer for the type input which is reshaped to (128, 128, 19)  
A dense layer for the image input which is reshaped to (128, 128, 3)  
These layers are then concatenated and 3 convolutional layers follow, using dropout with rate 0.3.  
These layers have shape (64, 64, 8), (32, 32, 16) and (16, 16, 32)  
This last layer is flattened and a dense layer of shape 1 is used for ouput  

## Results

Here is an image of some outputs of the generator at the end of its runtime.

![Result Image](https://github.com/kdsar/GAN_poke_sprite/blob/master/result_images.png)

The NN quickly learns that the data should be centered and the outer parts of the result is white.  
A little while afterwards, the images become less blurry, with more defined lines.  
Sadly, it never completely stops putting a few pixels of color on the edges and has trouble with color overall, giving the same tone of color for every image generated in an epoch.
Overall the results are still satisfying for a first endeavor and can be likened to clouds, where you can seem to discern some patterns.  



## Future Work

PROGAN as well as StyleGAN are two future ameliorations for the architecture.  
Further testing on typing and the effect of the 'None' typing should be done.  
Add more pokemon data such as HP and Speed and see how it affects the results.  
Once it can learn well enough from the original data, seeing how data augmentation affects this by changing for example the sprite hues.  