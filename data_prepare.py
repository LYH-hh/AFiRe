import os
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import random

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((filename, img))
    return images

def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size)

def otsu_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def get_noise(image, threshold):
    h, w, c = image.shape
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    '''for i in range (len(contours)):
        print(len(contours[i]))'''
    max_contour = max(contours, key=cv2.contourArea)
    ((x,y),r)=cv2.minEnclosingCircle(max_contour)
    x,y,r = int(x),int(y),int(r)

    left = max(x - 30, 0)
    top = max(y - 30, 0)
    right = min(x + 30, w)
    bottom = min(y + 30, h)

    noise = image[left:right, top:bottom, : ]
    
    return noise


def apply_gaussian_gradient_mask(mask):
    #print(mask.shape)
    h, w, c = mask.shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2

    # Creating an elliptical mask
    a = w / 2.0
    b = h / 2.0
    elliptical_mask = ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2
    elliptical_mask = np.exp(-elliptical_mask / 0.5)
    elliptical_mask = (elliptical_mask - elliptical_mask.min()) / (elliptical_mask.max() - elliptical_mask.min())*0.5

    noise = mask.astype(np.float32) * elliptical_mask[..., np.newaxis]
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    
    return noise

'''def apply_gaussian_gradient_mask(mask):
    h, w = mask.shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    gaussian_mask = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (1.0 * (min(h, w) / 2.0) ** 2))
    gaussian_mask = (gaussian_mask - gaussian_mask.min()) / (gaussian_mask.max() - gaussian_mask.min())
    
    noise = mask.astype(np.float32) * gaussian_mask[..., np.newaxis]
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    
    return noise'''

def generate_noise_mask(image, texture_noise):
    h, w, c = image.shape
    #print(texture_noise.shape)
    noise_mask = np.zeros_like(image)
    num_noises = random.randint(1, 4)
    for _ in range(num_noises):
        subregion_h = random.randint(16, 128)
        subregion_w = random.randint(16, 128)
        top_left_x = random.randint(0, w - subregion_w)
        top_left_y = random.randint(0, h - subregion_h)
        noise = cv2.resize(texture_noise, (subregion_w, subregion_h))
        #noise = apply_gaussian_gradient_mask(noise)
        noise_mask[top_left_y:top_left_y + subregion_h, top_left_x:top_left_x + subregion_w] += noise
    
    return noise_mask


def apply_noise_to_image(original_image, noise_mask):
    image = original_image.copy()
    h, w, c = image.shape
    image = image.astype(np.int32)
    noise_mask = noise_mask.astype(np.int32)
    
    max_pix = np.iinfo(original_image.dtype).max
    for i in range(h):
        for j in range(w):
            pix = image[i, j] + noise_mask[i, j]
            image[i, j] = np.clip(pix, 0, max_pix-20)
    
    image = image.astype(original_image.dtype)

    return image


def generate_synthtic_lesion_images(image_folder, output_folder_synthe, output_folder_mask, label_folder):
    
    images = load_images_from_folder(image_folder)
    for idx, (filename, image) in enumerate(images):
        resized_image = resize_image(image)
        for i in range(5):
            label_image_path = os.path.join(label_folder, f'{filename}_{i}.png')
            cv2.imwrite(label_image_path, resized_image)

            binary_image = otsu_threshold(resized_image)
            texture_noise = get_noise(resized_image, binary_image)
            noise_with_gaussian = apply_gaussian_gradient_mask(texture_noise)
            noise_mask = generate_noise_mask(resized_image, noise_with_gaussian)
            synthtic_lesion_image = apply_noise_to_image(resized_image, noise_mask)
            
            # Save synthtic lesion image
            synthtic_lesion_image_path = os.path.join(output_folder_synthe, f'{filename}_{i}.png')
            cv2.imwrite(synthtic_lesion_image_path, synthtic_lesion_image)
            
            # Save mask
            noise_mask_path = os.path.join(output_folder_mask, f'{filename}_{i}.png')
            cv2.imwrite(noise_mask_path, noise_mask)
        print(filename)

image_folder = '/lyh/dataset/wellPrepared/chestXray/train/normal'
output_folder_synthe = '/lyh/dataset/AFiRe/inputs'
output_folder_mask = '/lyh/dataset/AFiRe/masks'
label_folder = '/lyh/dataset/AFiRe/labels'
generate_synthtic_lesion_images(image_folder, output_folder_synthe, output_folder_mask, label_folder)
print("finish")