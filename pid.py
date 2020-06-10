import numpy as np
from double_integrator import Double_integrator

class PID(Double_integrator):
	"""docstring for PID"""
	def __init__(self, goal, n_state, n_action):
		super(PID, self).__init__()
		self.goal = goal
		self.n_state = n_state
		self.n_action = n_action
		self.K = np.matrix([[0.1, 0, 0.5, 0],
							[0, 0.1, 0, 0.5]])

	def solve(self, q):
		u_pred = np.zeros(self.n_action).reshape([self.n_action, 1])
		u_pred = -np.matmul(self.K, (q - self.goal))
		for i in range(u_pred.shape[0]):
			u_pred[i,0] = max(-1,min(1,u_pred[i,0]))
		x_pred = self.forward_sim(q, u_pred)
		return x_pred, u_pred
		