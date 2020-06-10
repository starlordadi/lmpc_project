import numpy as np

class Safe_set():
	"""docstring for safe_set"""
	def __init__(self, n_iter, n_state, n_action, goal):
		super(Safe_set, self).__init__()
		self.safe_set = [[] for i in range(n_iter+1)]
		self.value = [[] for i in range(n_iter+1)]
		# self.value = []
		self.n_action = n_action
		self.n_state = n_state
		self.n_iter = n_iter
		self.goal = goal

	def state_count(self, it):
		n_count = 0
		for i in range(it+1):
			n_count += len(self.safe_set[i])
		return n_count

	def update(self, traj, iter_val, Q, R, goal=None):
		for i in range(len(traj.state_arr)):
			self.safe_set[iter_val+1].append(traj.state_arr[i])
		self.value[iter_val+1]  = self.value[iter_val+1] + traj.compute_cost(Q=Q, R=R, goal=goal)


