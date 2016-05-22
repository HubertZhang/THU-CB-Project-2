import struct
from PIL import Image
import os
import numpy as np
import pickle
import json

NUMBYTES1 = 56
NUMBYTES2 = 80 * 10
BATCH_SIZE = 100
class DataSet():
	def __init__(self, init_struct, img_set, ref_struct):
		with open(init_struct, 'rb') as f_in:
			self.init_img, self.init_img_dim, self.init_img_type = self.read_image(f_in)
		with open(ref_struct, 'rb') as f_in:
			self.ref_img, self.ref_img_dim, self.ref_img_type = self.read_image(f_in)
		self.data_set = []
		for item in os.listdir(img_set):
			if item.split('.')[-1] == 'jpg':
				self.data_set.append('.'.join(item.split('.')[:-1]))

	def read_tag(self, f_path):
		tags = []
		with open(f_path) as f_in:
			for line in f_in:
				line = line[:-1]
				if line in ['', 'data_', 'loop_', '_rlnCoordinateX #1', '_rlnCoordinateY #2']:
					continue
				pnt = line.split(' ')
				tags.append((int(pnt[1]), int(pnt[0])))
		return tags

	def extract_feature(self, window_size_x=180, window_size_y=180):
		def cut_mat():
			left_lim_x = pnt[1] - int(window_size_x/2)
			right_lim_x = pnt[1] + int((window_size_x+1)/2)
			left_lim_y = pnt[0] - int(window_size_y/2)
			right_lim_y = pnt[0] + int((window_size_y+1)/2)
			if left_lim_x < 0 or right_lim_x > img.shape[1] or left_lim_y < 0 or right_lim_y > img.shape[0]:
				return False
			return img[left_lim_y:right_lim_y,left_lim_x:right_lim_x]
		self.feature_set = []
		for index, img in enumerate(self.image_set):
			for pnt in self.image_tag[index]:
				self.feature_set.append(cut_mat())
		self.feature_set = filter(lambda x: x, self.feature_set)

	def read_image(self, input_image):
		img_header1 = input_image.read(NUMBYTES1 * 4)
		img_header2 = input_image.read(NUMBYTES2)

		byte_pattern = '=' + 'l' * NUMBYTES1  # '=' required to get machine independent standard size
		img_dim = struct.unpack(byte_pattern, img_header1)[:3]  # (dimx,dimy,dimz)
		img_type = struct.unpack(byte_pattern, img_header1)[3]  # 0: 8-bit signed, 1:16-bit signed, 2: 32-bit float, 6: unsigned 16-bit (non-std)
		if (img_type == 0):
			imtype = 'b'
		elif (img_type == 1):
			imtype = 'h'
		elif (img_type == 2):
			imtype = 'f4'
		elif (img_type == 6):
			imtype = 'H'
		else:
			type = 'unknown'  # should put a fail here
		input_image_dimension = (img_dim[2], img_dim[1], img_dim[0])  # 3D image

		image_data = np.fromfile(file=input_image, dtype=imtype, count=img_dim[0]*img_dim[1]*img_dim[2]).reshape(input_image_dimension)
		return image_data, img_dim, img_type
