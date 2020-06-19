import numpy as np
import casadi as ca
from double_integrator import Double_integrator

class Mpc(Double_integrator):
	"""docstring for mpc"""
	def __init__(self, horizon, n_state, n_action, Q, R, goal, solver=None):
		super(Mpc, self).__init__()
		self.solver = solver
		self.horizon = horizon
		self.n_state = n_state
		self.n_action = n_action
		self.Q = Q
		self.R = R
		self.goal = goal

	def solve(self, q, safe_set, n_iter):
		slack_thresh = 1.5*1e-8
		#define variables
		x = ca.SX.sym('x', self.n_state*(self.horizon + 1))
		u = ca.SX.sym('u', self.n_action*self.horizon)
		lam = ca.SX.sym('lam', safe_set.state_count(n_iter))
		slack = ca.SX.sym('slack', self.n_state)
		slack_obst = ca.SX.sym('slack_obst', self.horizon-1)
		#get constraints
		constraints, lbg, ubg = self.get_constraints(x=x, u=u, slack=slack, slack_obst=slack_obst, lam=lam, q=q, safe_set=safe_set, n_iter=n_iter)

		#get cost
		cost = self.get_cost(x=x, u=u, slack=slack, slack_obst=slack_obst, lam=lam, q=q, safe_set=safe_set, n_iter=n_iter)

		#create nlp problem
		opts = {'verbose':False, 'ipopt.print_level':0, 'print_time':0}
		nlp = {'x':ca.vertcat(x,u,lam,slack,slack_obst), 'f':cost, 'g':constraints}
		solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

		#solve with box constraints
		lbx = q.reshape(-1).tolist() + [-1000]*(self.horizon * self.n_state) + [-1]*(self.horizon * self.n_action) + [0]*(lam.shape[0]) + \
			  [-1000]*(self.n_state) + [1]*(self.horizon - 1)
		ubx = q.reshape(-1).tolist() + [1000]*(self.horizon * self.n_state) + [1]*(self.horizon * self.n_action) + [1]*(lam.shape[0]) + \
			  [1000]*(self.n_state) + [1000]*(self.horizon - 1)

		sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

		#get solution values
		sol_x = np.array(sol['x'])
		pred_horizon = sol_x[:self.n_state * (self.horizon + 1)].reshape([self.horizon + 1, self.n_state]).T
		u_pred = sol_x[self.n_state * (self.horizon + 1):self.n_state * (self.horizon + 1) + self.n_action].reshape([self.n_action,1])

		#print debug values
		slack_val = sol_x[self.n_state*(self.horizon + 1) + self.n_action*self.horizon + lam.shape[0]: \
						self.n_state*(self.horizon + 1) + self.n_action*self.horizon + lam.shape[0] + self.n_state]
		slack_obs_val = sol_x[self.n_state*(self.horizon + 1) + self.n_action*self.horizon + lam.shape[0] + self.n_state:]
		lambd = sol_x[self.n_state*(self.horizon + 1) + self.n_action*self.horizon:\
					self.n_state*(self.horizon + 1) + self.n_action*self.horizon + lam.shape[0]]
		print(np.linalg.norm(slack_val))
		feasible = (np.linalg.norm(slack_val) <= slack_thresh)
		print(solver.stats()['success'])
		# print(lambd)
		return pred_horizon, u_pred, feasible

	def get_constraints(self, x, u, slack, slack_obst, lam, q, safe_set, n_iter):
		constraints = []
		n = self.n_state
		d = self.n_action
		N = self.horizon
		#dynamics constraints
		q0 = q
		for i in range(N):
			q_next = self.forward_sim(q0,u[d*i:d*(i+1)])
			constraints = ca.vertcat(constraints, x[n*(i+1)+0] - q_next[0,0])
			constraints = ca.vertcat(constraints, x[n*(i+1)+1] - q_next[1,0])
			constraints = ca.vertcat(constraints, x[n*(i+1)+2] - q_next[2,0])
			constraints = ca.vertcat(constraints, x[n*(i+1)+3] - q_next[3,0])
			q0 = q_next

		#obstacle constraints
		for i in range(1,N):
			constraints = ca.vertcat(constraints, (x[n*i+0]+5)**2/4 + (x[n*i+1]-1)**2/1 - slack_obst[i-1])

		#terminal constraint
		state_list = [val for sublist in safe_set.safe_set for val in sublist]
		state_list = np.hstack(state_list)
		xf = state_list @ lam		
		constraints = ca.vertcat(constraints, (x[n*(N)+0]) - xf[0] + slack[0])
		constraints = ca.vertcat(constraints, (x[n*(N)+1]) - xf[1] + slack[1])
		constraints = ca.vertcat(constraints, (x[n*(N)+2]) - xf[2] + slack[2])
		constraints = ca.vertcat(constraints, (x[n*(N)+3]) - xf[3] + slack[3])

		#lam constraint
		constraints = ca.vertcat(constraints, np.ones(lam.shape[0]).reshape([1,lam.shape[0]]) @ lam == 1)

		#lbg and ubg
		lbg = [0]*(n*N) + [0]*(N-1) + [0]*n + [0]
		ubg = [0]*(n*N) + [0]*(N-1) + [0]*n + [0]

		return constraints, lbg, ubg

	def get_cost(self, x, u, slack, slack_obst, lam, q, safe_set, n_iter):
		n = self.n_state
		d = self.n_action
		N = self.horizon
		goal = self.goal
		#terminal constraint cost
		cost = 10e8*ca.dot(slack, slack)

		#stage cost
		for i in range(N):
			cost = cost + (x[n*i:n*(i+1)]-goal).T @ self.Q @ (x[n*i:n*(i+1)]-goal)
			cost = cost + u[d*i:d*(i+1)].T @ self.R @ u[d*i:d*(i+1)]

		#terminal cost
		val_list = [item for sublist in safe_set.value for item in sublist]
		val_list = np.array(val_list).reshape([-1,1])
		cost = cost + lam.T @ val_list

		return cost


	# def solve(self, q, safe_set, n_iter):
	# 	# define variables
	# 	opti = ca.Opti()
	# 	x = opti.variable(self.n_state, self.horizon + 1)
	# 	u = opti.variable(self.n_action, self.horizon)
	# 	lam = opti.variable(safe_set.state_count(n_iter))
	# 	slack = opti.variable(self.horizon)
	# 	# get cost function
	# 	x_term, cost = self.get_cost(x=x, q=q, u=u, lam=lam, safe_set=safe_set, slack=slack)
	# 	# get constraints
	# 	constr = self.get_constraints(x=x, q=q, u=u, lam=lam, safe_set=safe_set, x_term=x_term, slack = slack)
	# 	# solve ocp
	# 	opts = dict()
	# 	opts['ipopt.print_level'] = 0
	# 	opts['verbose'] = False
	# 	opts['print_time'] = 0
	# 	opti.solver(self.solver, opts)
	# 	opti.minimize(cost)
	# 	opti.subject_to(constr)

	# 	sol = opti.solve()
	# 	u_pred = sol.value(u[:,0]).reshape([self.n_action,1])
	# 	x_pred = self.forward_sim(q, u_pred)
	# 	print(sol.value(slack))
	# 	return sol.value(x), u_pred

	# def get_cost(self, x, q, u, lam, safe_set, slack):
	# 	cost = 0
	# 	for i in range(self.horizon):
	# 		cost += self.norm(x[:,i], u[:,i])
	# 		# x = self.forward_sim(x, u[:,i])
	# 	# val_list = [val for sublist in safe_set.value for val in sublist]
	# 	# val_list = np.array(val_list).flatten()
	# 	val_list = np.vstack(safe_set.value)
	# 	# val_list = val_list.reshape([lam.shape[0],1])
	# 	# print(val_list.shape)
	# 	# print(lam.shape)
	# 	cost += lam.T @ val_list
	# 	cost += ca.dot(slack, slack)
	# 	# print(cost.shape)
	# 	return x, cost

	# def get_cost_time(self):
	# 	pass

	# def norm(self, q, u):
	# 	# print('1', q.shape)
	# 	# print('2', u.shape)
	# 	# print('3', self.Q.shape)
	# 	# print('4', self.R.shape)
	# 	# print('5', self.goal.shape)
	# 	# val = np.matmul((q-self.goal).T, np.matmul(self.Q, (q-self.goal))) + np.matmul(u.T, np.matmul(self.R, u))
	# 	val = (q-self.goal).T @ (self.Q @ (q - self.goal)) + u.T @ (self.R @ u)
	# 	# print(val)
	# 	return val

	# def get_constraints(self, x, q, u, lam, safe_set, x_term, slack):
	# 	constr = []
	# 	state_list = [val for sublist in safe_set.safe_set for val in sublist]
	# 	state_list = np.hstack(state_list)
	# 	for i in range(self.horizon):
	# 		constr.append(x[:,i+1] == self.forward_sim(x[:,i], u[:,i]))
	# 		constr.append((x[0,i+1] + 5)**2 /4 + (x[1,i+1]-1)**2 /1.0 == 1 +  slack[i])
	# 		# constr.append(slack[i] <= 0.1)
	# 	constr.append(x[:,0] == q)
	# 	constr.append(np.ones(lam.shape[0]).reshape([1,lam.shape[0]]) @ lam == 1)
	# 	constr.append(x[:,self.horizon] == (state_list @ lam))
	# 	constr.append(lam >= 0)
	# 	constr.append(ca.vec(u) >= -1)
	# 	constr.append(ca.vec(u) <= 1)
	# 	constr.append
	# 	return constr

