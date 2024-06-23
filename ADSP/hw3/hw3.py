import numpy as np
import cv2

def calculate_psnr(original_image, compressed_image):
    # Convert images to float32
    original_image = original_image.astype(np.float32)
    compressed_image = compressed_image.astype(np.float32)
    
    # Compute squared error image
    mse = np.mean((original_image - compressed_image) ** 2)
    
    # Maximum possible pixel value
    max_pixel = 255.0
    
    # Calculate PSNR
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def rgb_to_ycbcr(image):
    # Convert RGB image to YCbCr color space
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    Cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    
    return np.dstack((Y, Cb, Cr))

def ycbcr_to_rgb(image):
    # Convert YCbCr image to RGB color space
    Y = image[:,:,0]
    Cb = image[:,:,1]
    Cr = image[:,:,2]
    
    r = Y + 1.402 * (Cr - 128)
    g = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    b = Y + 1.772 * (Cb - 128)
    
    return np.clip(np.dstack((r, g, b)), 0, 255).astype(np.uint8)

def C420(image):
    # Convert the input image to YCbCr color space
    # ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycbcr_image = rgb_to_ycbcr(image)
    
    # Extract Y, Cb, and Cr channels
    Y = ycbcr_image[:,:,0]
    Cb = ycbcr_image[:,:,1]
    Cr = ycbcr_image[:,:,2]
    
    # Subsample Cb and Cr channels by a factor of 2 in both dimensions
    Cb_subsampled = Cb[::2, ::2]
    Cr_subsampled = Cr[::2, ::2]
    
    # Reconstruct the Cb and Cr channels by interpolation
    Cb_reconstructed = cv2.resize(Cb_subsampled, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    Cr_reconstructed = cv2.resize(Cr_subsampled, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Combine the reconstructed Y, Cb, and Cr channels
    reconstructed_image = np.zeros_like(ycbcr_image)
    reconstructed_image[:,:,0] = Y
    reconstructed_image[:,:,1] = Cb_reconstructed
    reconstructed_image[:,:,2] = Cr_reconstructed
    
    # Convert the reconstructed image back to BGR color space
    # reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_YCrCb2BGR)
    reconstructed_image = ycbcr_to_rgb(reconstructed_image)
    
    return reconstructed_image


# Example usage
input_image = cv2.imread('new_jeans.jpg')
compressed_image = C420(input_image)
print('PSNR:', calculate_psnr(input_image, compressed_image))

cv2.imshow('Original Image', input_image)
cv2.imshow('Compressed Image', compressed_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

