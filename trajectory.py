import numpy as np

class Trajectory():
	"""docstring for trajectory"""
	def __init__(self, n_state, n_action, mode):
		super(Trajectory, self).__init__()
		self.n_state = n_state
		self.n_action = n_action
		self.state_arr = []
		self.action_arr = []
		self.mode = mode

	def append_data(self, q, u):
		self.state_arr.append(q)
		self.action_arr.append(u)

	def get_length(self):
		return len(self.state_arr)

	def compute_cost(self, Q, R, goal=None):
		n = len(self.state_arr)
		val = np.zeros(n)
		if self.mode == 'task':
			for i in range(len(self.state_arr)):
				if i==0:
					cost = 0
					val[n-1-i] = cost
				else:
					cost += np.matmul(self.state_arr[n-1-i].T, np.matmul(Q, self.state_arr[n-1-i])) + \
							np.matmul(self.action_arr[n-1-i].T, np.matmul(R, self.action_arr[n-1-i]))
					val[n-1-i] = cost
		elif self.mode == 'exp':
			for i in range(len(self.state_arr)):
				if i==0:
					cost=0
					val[i]=cost
				else:
					cost += np.matmul(self.state_arr[n-1-i].T, np.matmul(Q, self.state_arr[n-1-i])) + \
							np.matmul(self.action_arr[n-1-i].T, np.matmul(R, self.action_arr[n-1-i]))
					val[i] = cost

		return val.tolist()

