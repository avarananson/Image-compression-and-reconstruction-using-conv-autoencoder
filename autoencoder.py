import sys
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import warnings
warnings.filterwarnings("ignore")


try:
    code = sys.argv[1]
    inp_path = sys.argv[2]
    out_path = sys.argv[3]
except Exception as e:
    print("Improper arguments given")


if(str(code) == 'encode'):

    encoder = load_model("trained_models/encoder_functional_8.h5")
    if(plt.imread(inp_path).shape is not (250, 250, 3)):
        print("Shape is not in (250,250,3), resizing to the required format..")

    img = load_img(inp_path, target_size=(248, 248))
    base_arr = img_to_array(img) / 255
    base_arr = np.expand_dims(base_arr, axis=0)
    print("Shape of image array is", base_arr.shape)
    enc = encoder.predict(base_arr)
    print("Compressed.. ,compressed to 16%")
    np.save(out_path + "compressed.npy", enc)
    print("Saved the compressed form as .npy in the o/p directory")

elif(str(code) == 'decode'):
    if(not str(inp_path).endswith(".npy")):
        print("Please input the correct compressed file (.npy)")
        exit()
    else:
        decoder = load_model("trained_models/decoder_functional_8.h5")
        dec = np.load(inp_path)
        # dec = np.expand_dims(dec, axis=0)
        out = decoder.predict(dec)
       # plt.show()
        save_img(out_path + "restored_image.png", out[0])
        print("Restored")
else:
    encoder = load_model("trained_models/encoder_functional_8.h5")
    decoder = load_model("trained_models/decoder_functional_8.h5")
    inp_path = code
    out_path = inp_path
    print("Your input image path", inp_path)
    print("Your o/p path", out_path)
    img = load_img(inp_path, target_size=(248, 248))
    base_arr = img_to_array(img) / 255
    base_arr = np.expand_dims(base_arr, axis=0)
    print("Compression started..")
    out = decoder.predict(encoder.predict(base_arr))
    save_img(out_path + "restored_image.png", out[0])
    print("Restored")
