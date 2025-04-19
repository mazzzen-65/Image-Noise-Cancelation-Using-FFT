import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read and convert image to the gray Spectrum from the R G B Channels to Easily Represent it later
img = cv2.imread('Butterfly.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found or couldn't be loaded")

plt.figure()
plt.imshow(img, cmap='gray')
plt.title('Original Image')

# Compute FFT
ft = np.fft.fft2(img.astype(float))  # Apply the FFT Formula to the Image 
ft_shift = np.fft.fftshift(ft)  # Shift The low Frequancies to the center using the shift method from numpy 
magnitude = np.log(1 + np.abs(ft_shift)) # compute the absolute value for the Magnitude (Freqs) and applying the log for compressing the much larger values to smaller one

plt.figure() # to show a new window for the Frequancies After Shifting
plt.imshow(magnitude, cmap='gray') 
plt.title('Magnitude Spectrum')

# Create Low Filter
rows, cols = img.shape  # Getting the rows and the columns for the image
crow, ccol = rows//2, cols//2 # To Center the Mask
radius = 20 # The Radius for the Mask Applied

mask = np.zeros((rows, cols), dtype=np.float32)  # Use float32 to be more accurate
cv2.circle(mask, (ccol, crow), radius, 1, -1)  # Making a White circle in the middle for the Low Filter Pass (Mask , Center , Radius , 1=white)
mask_inv = 1 - mask  # The High Filter Mask (inverting the white to be black)

# Apply filters
low_pass = ft_shift * mask
high_pass = ft_shift * mask_inv

# Inverse FFT to return the filtered Photo
img_low = np.abs(np.fft.ifft2(np.fft.ifftshift(low_pass))) # Returning the shifted Freqs to it's original place
img_high = np.abs(np.fft.ifft2(np.fft.ifftshift(high_pass)))

# Normalize
img_low = cv2.normalize(img_low, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
img_high = cv2.normalize(img_high, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display results
plt.figure(figsize=(12,8))
plt.subplot(221), plt.imshow(mask, cmap='gray'), plt.title('Low-pass Filter')
plt.subplot(222), plt.imshow(mask_inv, cmap='gray'), plt.title('High-pass Filter')
plt.subplot(223), plt.imshow(img_low, cmap='gray'), plt.title('Low-pass Image')
plt.subplot(224), plt.imshow(img_high, cmap='gray'), plt.title('High-pass Image')
plt.tight_layout()
plt.show()