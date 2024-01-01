from astropy.io import fits
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score
import pandas as pd
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from photutils.segmentation import (detect_sources,
                                    make_2dgaussian_kernel)
import time
import matplotlib.pyplot as plt

for file in sorted(os.listdir("Star Images")):
    if file.endswith(".fits") and not file.endswith("_masked.fits"):
        print("Doing " + file)
        hdu = fits.open(f"Star Images/{file}")
        image = hdu[0].data

        mean, _, std = sigma_clipped_stats(image)

        threshold = 3. * std
        kernel = make_2dgaussian_kernel(3, size=3)
        convolved_data = convolve(image, kernel)
        segm = detect_sources(convolved_data, threshold, npixels=5)

        # fig, (ax1, ax2) = plt.subplots(2, 1)
        # norm = simple_norm(image, 'sqrt', percent=99.)
        # ax1.imshow(image, origin='lower', interpolation='nearest',
        #            norm=norm)
        # ax2.imshow(segm.data, origin='lower', interpolation='nearest',
        #            cmap=segm.make_cmap(seed=1234))
        # plt.tight_layout()
        # plt.show()
        #
        # # Show original image but with only the pixels that are part of a source
        # # shown.
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(image, origin='lower', interpolation='nearest',
        #           norm=norm)
        # ax.imshow(segm.data, origin='lower', interpolation='nearest',
        #             cmap=segm.make_cmap(seed=1234), alpha=0.5)
        # plt.show()

        # Create a mask from the segmentation data
        mask = segm.data.astype(bool)

        # Create a new image with background values outside the segmentation mask
        masked_image = np.where(mask, image, 0)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1)
        norm = simple_norm(image, 'sqrt', percent=99.)
        ax1.imshow(image, origin='lower', interpolation='nearest', norm=norm)
        ax1.set_title('Original Image')

        # Create a masked version of the original image using the segmentation mask
        masked_image = np.where(segm.data, image, np.nan)
        ax2.imshow(masked_image, origin='lower', interpolation='nearest', norm=norm)
        ax2.set_title('Original Image with Segmented Pixels')

        plt.tight_layout()
        plt.show()

        # save masked image
        hdu[0].data = masked_image
        hdu.writeto(f"Star Images/{file.replace('.fits', '')}_masked.fits", overwrite=True)
        time.sleep(3)