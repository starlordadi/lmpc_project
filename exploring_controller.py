import numpy as np
import casadi as ca
from double_integrator import Double_integrator

class Explorer(Double_integrator):
	"""docstring for Explorer"""
	def __init__(self, goal, solver, n_state, n_action, horizon, Q, R):
		super(Explorer, self).__init__()
		self.solver = solver
		self.goal = goal
		self.n_state = n_state
		self.n_action = n_action
		self.horizon = horizon
		self.Q = Q
		self.R = R

	def solve(self, q0, safe_set, n_iter):
		""" Exploration by going towards the start state from the goal state"""
		goal = self.goal
		N = self.horizon
		n = self.n_state
		d = self.n_action
		# define variables
		x = ca.SX.sym('x', self.n_state*(self.horizon + 1))
		u = ca.SX.sym('u', self.n_action*self.horizon)
		ineq_slack = ca.SX.sym('slack', self.horizon-1)
		# lam = ca.SX.sym('lam', safe_set.)
		# get constraints
		constraints, lbg, ubg = self.get_constraints(x=x,u=u, slack= ineq_slack, q=q0)
		# get cost
		cost = self.get_cost(x=x, u=u, slack=ineq_slack)
		# form nlp
		opts = {'verbose':False, 'ipopt.print_level':0, 'print_time':0}
		nlp = {'x':ca.vertcat(x,u, ineq_slack), 'f':cost, 'g':constraints}
		solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
		# solve with box constraints
		lbx = q0.reshape(-1).tolist() + [-1000]*(self.horizon * self.n_state) + [-1]*(self.horizon * self.n_action) + [1.0]*(self.horizon - 1)
		ubx = q0.reshape(-1).tolist() + [1000]*(self.horizon * self.n_state) + [1]*(self.horizon * self.n_action) + [100]*(self.horizon - 1)

		sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
		# get solution values
		sol_x = np.array(sol['x'])
		pred_horizon = sol_x[:self.n_state * (self.horizon + 1)].reshape([self.horizon + 1, self.n_state]).T
		u_pred = sol_x[self.n_state * (self.horizon + 1):self.n_state * (self.horizon + 1) + self.n_action].reshape([self.n_action,1])

		return pred_horizon, u_pred

	def get_constraints(self, x, u, slack, q):
		constraints = []
		N = self.horizon
		n = self.n_state
		d = self.n_action
		# dynamics constraints
		q0 = q
		for i in range(N):
			q_next = self.forward_sim(q0,u[d*i:d*(i+1)])
			constraints = ca.vertcat(constraints, x[n*(i+1)+0] - q_next[0,0])
			constraints = ca.vertcat(constraints, x[n*(i+1)+1] - q_next[1,0])
			constraints = ca.vertcat(constraints, x[n*(i+1)+2] - q_next[2,0])
			constraints = ca.vertcat(constraints, x[n*(i+1)+3] - q_next[3,0])
			q0 = q_next
		# obstacle constraints
		for i in range(1,N):
			constraints = ca.vertcat(constraints, (x[n*i+0]+5)**2/4 + (x[n*i+1]-1)**2/1 - slack[i-1])
		lbg = [0]*(n*N) + [0]*(N-1)
		ubg = [0]*(n*N) + [0]*(N-1)

		return constraints, lbg, ubg

	def get_cost(self, x, u, slack):
		N = self.horizon
		n = self.n_state
		d = self.n_action
		goal = self.goal

		cost = 0
		for i in range(N):
			cost = cost + (x[n*i:n*(i+1)]-goal).T @ self.Q @ (x[n*i:n*(i+1)]-goal)
			cost = cost + u[d*i:d*(i+1)].T @ self.R @ u[d*i:d*(i+1)]

		return cost			

	def sample_informed_state(self):
		pass

	def check_collision(self):
		pass



