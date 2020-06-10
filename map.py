import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.linalg as la
#todo: create map from a binary image
class Map():
	"""class for the map information in numpy matrices"""
	def __init__(self, x_lim = 10, y_lim = 10):
		super(Map, self).__init__()
		self.x_lim = x_lim
		self.y_lim = y_lim
		self.obst_centers = None
		self.obst_trans_mat = None

	def assign_obstacle(self, obs_center):
		self.obst_centers = obs_center
		self.obst_trans_mat = np.matrix([[16, 0],[0, 9]])

	def check_collision(self, q):
		pos = q[:2,:]
		val = np.random.rand(self.obst_centers.shape[1])
		for i in range(self.obst_centers.shape[1]):
			val[i] = np.matmul(pos.T, np.matmul(la.inv(self.obst_trans_mat), pos))

	def plot_map(self):
		pass