import numpy as np
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread

def SSIM(imageA, imageB, c1, c2):
    # Ensure the images have the same size
    assert imageA.shape == imageB.shape, "Input images must have the same dimensions."

    # Convert the images to float type
    imageA = img_as_float(imageA)
    imageB = img_as_float(imageB)

    # Compute SSIM between two images
    ssim_value, _ = ssim(imageA, imageB, data_range=imageB.max() - imageB.min(), full=True, K1=c1, K2=c2)

    return ssim_value

# Load two images (ensure they have the same size)
imageA = imread('img1.png', as_gray=True)
imageB = imread('img2.bmp', as_gray=True)

# Compute SSIM
ssim_value = SSIM(imageA, imageB, 0.01**2, 0.03**2)
print(f"SSIM: {ssim_value}")

