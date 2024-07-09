
# Registration via open3d tutorial: https://github.com/KuKuXia/OpenCV_for_Beginners/blob/master/48_Image_Alignment_based_on_ORB_Features.py

# Import the packages
from __future__ import print_function
import cv2
import numpy as np
import glob
import os
import time
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from PIL import Image
import argparse

import easyocr


MAX_MATCHES = 500
GOOD_MATCH_PERCENT = 0.15



def alignImages(im1, im2):

    # Convert images to grayscaleq
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)


    # Sort matches by score
    
    matches_distances = [x.distance for x in matches]
    indices = np.argsort(matches_distances)
    matches_list = [matches[ind] for ind in indices]

    
    # matches.sort(key=lambda x: x.distance, reverse=False)

    matches = matches_list
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(
        im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("./images/alignment/matches-example.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

class LineBuilder:

    def __init__(self, line, ax):
        self.line = line
        self.xs = []
        self.ys = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

class OCR():
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
    
    def __call__(self, image):

        return self.reader.readtext(image)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Align images and extract text from gas meter')
    parser.add_argument('--image_folder', default="gasmeter", help='Path to the image folder')
    parser.add_argument('--reference_image', default="gasmeter/20210322_120107.jpg", help='Path to the reference image')
    parser.add_argument('--show_image_estimates', default=0, help='Show the image estimates with numbers')

    args = parser.parse_args()
    image_folder = args.image_folder
    reference_image = args.reference_image
    show_image_estimates = int(args.show_image_estimates)

    resize_shape = (1024, 800)
    # Read reference image
    refFilename = reference_image
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    imReference = cv2.resize(imReference, resize_shape)

    image_paths = sorted(glob.glob(f"{image_folder}/*.jpg"))


    if not os.path.exists("polygon.npy"):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Click to separate the meter by rectangle')
        ax.imshow(imReference)
        line, = ax.plot([0], [0], 'r') 
        linebuilder = LineBuilder(line, ax)
        plt.show()

        line_points = np.array(list((zip(linebuilder.xs, linebuilder.ys))))
        np.save("polygon.npy", line_points)
    else:
        line_points = np.load("polygon.npy")

    # Define the polygon using the points from LineBuilder
    polygon = Polygon(line_points)
    
    TextModel = OCR()

    for idx, image_path in enumerate(image_paths):
        
        # Read image to be aligned
        im = np.asarray(Image.open(image_path))
        im = cv2.resize(im, resize_shape)

        # print("Aligning images ...")
        # Registered image will be resorted in imReg.
        # The estimated homography will be stored in h.
        imReg, h = alignImages(im, imReference)

        blended_img = 0.5 * imReg + 0.5 * imReference
        blended_img = blended_img.astype('uint8')
        
        # Write aligned image to disk.
        outFilename = "out_align.jpg"
        cv2.imwrite(outFilename, imReg)


        # Create a mask based on the polygon
        mask = np.zeros_like(imReference)
        cv2.fillPoly(mask, [np.array(polygon.exterior.coords, dtype=np.int32)], (255, 255, 255))

        # Apply the mask to the image
        cropped_image = cv2.bitwise_and(imReg, mask)
        cropped_image = cropped_image[int(np.min(line_points[:,1])): int(np.max(line_points[:,1])),
                                    int(np.min(line_points[:,0])) : int(np.max(line_points[:,0]))]

    
        ### Optional, just for testing methods
        # Apply Sobel-Feldman filter
        orig_cropped_image = cropped_image.copy()
        # sobel_x = cv2.Sobel(cropped_image, cv2.CV_64F, 1, 0, ksize=3)
        # sobel_y = cv2.Sobel(cropped_image, cv2.CV_64F, 0, 1, ksize=3)
        # sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        # sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # cropped_image = sobel_mag

        # Convert the cropped image to grayscale
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Set the thresholded image as the cropped image
        cropped_image = threshold
        
        ### Again, another test to enhance features
        # for i in range(10):
        #     kernel = np.ones((3,3),np.uint8)
        #     cropped_image = cv2.morphologyEx(cropped_image, cv2.MORPH_CLOSE, kernel)
        #     # cropped_image = cv2.morphologyEx(cropped_image, cv2.MORPH_OPEN, kernel)

        Image.fromarray(cropped_image).save("cropped_image.jpg")
        Image.fromarray(orig_cropped_image).save("orig_cropped_image.jpg")

        ### Apply OCR model
        results = TextModel(orig_cropped_image)
        results_sobel = TextModel(cropped_image)
        extracted_text = [result[1] for result in results]
        extracted_text_sobel = [result[1] for result in results_sobel]

        

        ### Gather timestamps for calculation of average m3
        timestamp = os.path.basename(image_path).split('_')[0]
        year = timestamp[:4]
        month = timestamp[4:6]
        day = timestamp[6:]
        current_time = f"{day}.{month} {year}"
        
        # Put value into suitable format
        repaired_value = ''.join(''.join(extracted_text).split(" "))
        
        # Check comma and decide if to trust the estimate
        comma_detected = False
        if len(repaired_value) > 5:
            comma_detected = ',' == repaired_value[5]

        correct_detection = len(repaired_value) == 9 and comma_detected


        if correct_detection:
            print(current_time, " --- " , repaired_value, "Trust Estimate: ", correct_detection)

        
        if show_image_estimates: # Show results?
            fig, ax = plt.subplots(1,2, figsize=(10,10))
            ax[0].imshow(orig_cropped_image)
            ax[0].set_title(" ".join(extracted_text))
            ax[1].imshow(cropped_image)
            ax[1].set_title(" ".join(extracted_text_sobel))
            
            
            for result in results:
                ax[0].add_patch(plt.Polygon(result[0], fill=None, edgecolor='r'))

            for result in results_sobel:
                ax[1].add_patch(plt.Polygon(result[0], fill=None, edgecolor='r'))

            
            plt.show()
            plt.close()
        