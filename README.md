# Image-compression-and-reconstruction-using-conv-autoencoder
This project uses autoencoder to compress images particularly faces as it is trained on images of faces. The compression ratio 
is ```0.16``` and it is able to restore the compressed form to original image very clearly. Test set accuracy was 89%.
I used convolutional autoencoder as the alogorithm. Trained on 10000 face images. (250*250) .
I tried loss as 'mse'. Could try 'binary_crossentropy' too.

To compress the image (faces will give the best results) run ,

```autoencoder.py encode "path to image" "output path"```

To restore the compressed form to original image, run

```autoencoder.py decode "path of compressed form (.npy)" "output path "```

To see the resoration without saving the compressed form, just run ,

```autoencoder.py "path to i/p image" "output path"```

Requirements:
```
Tensorflow 2.0.0
matplotlib
numpy
```
