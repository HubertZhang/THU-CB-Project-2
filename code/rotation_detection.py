from PIL import Image
import os
import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt

class Detector:
	def __init__(self, data_root):
		self.image_set = []
		self.data_root = os.path.abspath(data_root)

		for item in os.listdir(data_root):
			file_type = item.split('.')[-1]
			data_name = '.'.join(item.split('.')[:-1])
			if file_type == 'jpg':
				self.image_set.append(data_name)

	def get_windows(self, data_name, window_size_x=180, window_size_y=180):
		def cut_mat():
			left_lim_x = pnt[1] - int(window_size_x/2)
			right_lim_x = pnt[1] + int((window_size_x+1)/2)
			left_lim_y = pnt[0] - int(window_size_y/2)
			right_lim_y = pnt[0] + int((window_size_y+1)/2)
			if left_lim_x < 0 or right_lim_x > img.shape[1] or left_lim_y < 0 or right_lim_y > img.shape[0]:
				return []
			return img[left_lim_y:right_lim_y,left_lim_x:right_lim_x]

		print("processing data {}".format(data_name))
		windows = []
		pnts = []
		with Image.open(os.path.join(self.data_root, '{}.jpg'.format(data_name))) as img_in:
			img = np.array(img_in)
		with open(os.path.join(self.data_root, '{}.star'.format(data_name))) as f_in:
			for line in f_in:
				line = line[:-1]
				if line in ['', 'data_', 'loop_', '_rlnCoordinateX #1', '_rlnCoordinateY #2']:
					continue
				pnt = line.split(' ')
				pnt = (int(pnt[1]), int(pnt[0]))
				pnts.append(pnt)
				mat = cut_mat()
				if len(mat) > 0:
					windows.append(cut_mat())

		# with open(os.path.join(self.data_root, '{}_windows.pkl'.format(data_name)), 'wb') as f_out:
		# 	pickle.dump(windows, f_out)
		return windows, pnts

	def generate_windows(self):
		windows = []
		for index, item in enumerate(self.image_set):
			print('{}/{}'.format(index+1, len(self.image_set)))
			windows += self.get_windows(item)
			if (index+1) % 100 == 0:
				print('saving {} windows...'.format(len(windows)))
				with open('data_{}.pkl'.format(index/100+1), 'wb') as f_out:
					pickle.dump(windows, f_out)
					windows = []
			# print('Number of windows: {}'.format(len(self.get_windows(item))))

	def convert_to_image(self, input_mat):
		vis = np.array(input_mat)
		vis = vis - np.min(vis)
		vis = vis / float(np.max(vis))
		vis = vis * 255.
		vis = vis.astype(np.uint8)
		# return vis
		return vis, cv2.adaptiveThreshold(vis, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 0)

def drawMatches(img1, kp1, img2, kp2, matches, img_name='matches.png'):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imwrite(img_name, out)

    # Also return the image if you'd like a copy
    return out

detector = Detector('../data/TRPV1/Particle2/')
windows = []
pnts = []
# img = cv2.imread('stx_1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# windows.append(img)
# img = cv2.imread('stx_2.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# windows.append(img)

for img in detector.image_set[:10]:
	window, pnt = detector.get_windows(img)
	windows += window
	pnts += pnt
# windows.append(np.rot90(windows[0]))
# windows = detector.get_windows(detector.image_set[0])
print('Num data: {}'.format(len(windows)))

img1, img1_bin = detector.convert_to_image(windows[0])        # queryImage
# img2 = detector.convert_to_image(np.rot90(windows[0])) # trainImage
surf = cv2.SURF(4000)
kp, des = surf.detectAndCompute(img1_bin, None)
img2 = cv2.drawKeypoints(img1, kp)
cv2.imwrite('surf.png', img2)
# fig = plt.figure()
# plt.imshow(img2)
# plt.savefig('surf.png')
# plt.close(fig)

# Initiate SIFT detector
sift = cv2.SIFT()
# find the keypoints and descriptors with SIFT
kp, des = sift.detectAndCompute(img1_bin, None)
img2 = cv2.drawKeypoints(img1, kp)
cv2.imwrite('sift.png', img2)
# fig = plt.figure()
# plt.imshow(img2)
# plt.savefig('sift.png')
# plt.close(fig)

orb = cv2.ORB(100, 1.2)
kp, des = orb.detectAndCompute(img1_bin, None)
img2 = cv2.drawKeypoints(img1, kp)
cv2.imwrite('orb.png', img2)

# BFMatcher with default params
bf = cv2.BFMatcher()

for index_1, mat_1 in enumerate(windows):
	img_1, img_1_bin = detector.convert_to_image(mat_1)
	kp1, des1 = surf.detectAndCompute(img_1_bin, None)
	if len(kp1) < 2:
		continue
	for index_2, mat_2 in enumerate(windows[index_1+1:]):
		img_2, img_2_bin = detector.convert_to_image(mat_2)
		kp2, des2 = surf.detectAndCompute(img_2_bin, None)
		if len(kp2) < 2:
			continue
		matches = bf.knnMatch(des1,des2, k=2)

		# Apply ratio test
		good = []
		for m,n in matches:
			# good.append(m)
			if m.distance < 0.85*n.distance:
				good.append(m)
		if len(good) > 15:
			print('Match {} and {}: {}'.format(index_1, index_2+index_1+1, len(good)))
			print('\tPoint: {}/{}'.format(pnts[index_1], pnts[index_1+index_2+1]))
			# fig = plt.figure(1)
			# plt.subplot(121)
			# plt.imshow(mat_1, cmap='gray')
			# plt.subplot(122)
			# plt.imshow(mat_2, cmap='gray')
			# plt.savefig('{}_{}'.format(index_1, index_2+index_1+1))
			# plt.close(fig)
			drawMatches(img_1, kp1, img_2, kp2, good, 'output/{}_{}.png'.format(index_1, index_2+index_1+1))