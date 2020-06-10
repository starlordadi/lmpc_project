import numpy as np
import os

class Double_integrator():
	"""double integrator simulator class"""
	def __init__(self, delt = 0.3): #todo: generalize for n dimension double integrator
		super(Double_integrator, self).__init__()
		self.A = np.matrix([[0, 0, 1, 0],
							[0, 0, 0, 1],
							[0, 0, 0, 0],
							[0, 0, 0, 0]])
		self.B = np.matrix([[0, 0],
							[0, 0],
							[1, 0],
							[0, 1]])
		self.delt = delt

	def forward_sim(self, q, u):
		n_dim = q.shape[0]
		Ad = (self.A * self.delt) + np.eye(n_dim)
		Bd = (self.B * self.delt)
		q_next = (Ad @ q) + (Bd @ u)
		self.Ad = Ad
		self.Bd = Bd
		return q_next

	def pred_horizon(self, q, u_arr):
		pred = np.random.random([q.shape[0], u_arr.shape[1]])
		for i in range(pred.shape[1]):
			pred[:,i] = self.forward_sim(q, u_arr[:,i])
			q = pred[:,i]
		return pred
		