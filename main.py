import numpy as np
from map import Map
from safe_set import Safe_set
from mpc import Mpc
from trajectory import Trajectory
from double_integrator import Double_integrator
from pid import PID
from exploring_controller import Explorer 
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.linalg as la

def get_ellipse(xs, ys, a, b):
	step = 0.1
	z = np.arange(-a, a+step, step)
	x = np.round(np.array([z,z]).flatten(), decimals=3)
	# x = x + xs*np.ones(x.shape[0])
	y = []
	for i in range(int(x.shape[0]/2)):
		y.append(ys + b*np.sqrt(1-(np.round(x[i]/a, decimals=3))**2))
	for i in range(int(x.shape[0]/2)):
		y.append(ys - b*np.sqrt(1-(np.round(x[i]/a, decimals=3))**2))
	return x + xs*np.ones(x.shape[0]), np.array(y)

def get_lqr_heuristic(z_s, z_g, Q, R):
	#define A,B,Q,R
	A = np.matrix([[0,0,1,0],
				   [0,0,0,1],
				   [0,0,0,0],
				   [0,0,0,0]])
	B = np.matrix([[0,0],
				   [0,0],
				   [1,0],
				   [0,1]])
	Ad = (A * 0.3) + np.eye(4)
	Bd = B * 0.3
	#solve DARE
	P = la.solve_discrete_are(Ad, Bd, Q, R)
	#return value estimate
	val = (z_s).T @ P @ (z_s)
	return val

def main():
	n_iter = 5
	n_state = 4
	n_action = 2
	goal = [0, 0, 0, 0]
	goal_arr = [[-5, 5, 0, 0], [0, 0, 0, 0]]
	start = [-10, 0, 0, 0]
	Q = 10.0*np.eye(n_state)
	# Q[2,2] = 0
	# Q[3,3] = 0
	R = 1.0*np.eye(n_action)
	x_ellipse, y_ellipse = get_ellipse(xs = -5, ys=1, a=2,b=2)
	# initialize map, start point, goal point, no. of iterations, safe-set, value function, mpc parameters
	new_map = Map()
	start_config = np.array(start).reshape([n_state, 1])
	goal_config = np.array(goal).reshape([n_state, 1])
	# print(get_lqr_heuristic(start_config, np.array([-5, 2, 0, 0]).reshape([n_state,1]), Q, R) + \
	# 	get_lqr_heuristic(np.array([-5, 2, 0, 0]).reshape([n_state,1]), goal_config, Q, R))
	SS = Safe_set(n_iter=n_iter, n_state=n_state, n_action=n_action, goal = goal_config)
	mpc = Mpc(horizon=5, n_state=n_state, n_action=n_action, Q=Q, R=R, goal = goal_config, solver='ipopt')
	# initialize safe set with feasible solution
	init_traj = Trajectory(n_state=n_state, n_action=n_action, mode='task')
	pid = PID(goal = None, n_state = n_state, n_action = n_action)
	q_current = start_config
	goal_reached = False
	print('start PID')
	for i in range(len(goal_arr)):
		pid.goal = np.array(goal_arr[i]).reshape([n_state, 1])
		goal_reached = False
		while not goal_reached:
			q_pred, u_pred = pid.solve(q_current)
			init_traj.append_data(q_current, u_pred)
			q_current = q_pred
			if (np.sum(abs(q_current - pid.goal))) < 0.5:
				goal_reached = True
	SS.update(traj=init_traj, iter_val=-1, Q=Q, R=R)
	print('PID complete')
	x_pid = np.hstack(init_traj.state_arr)[0,:]
	y_pid = np.hstack(init_traj.state_arr)[1,:]
	# print(x.shape)
	# plt.plot(x_pid.T,y_pid.T)
	# plt.show()
	# print(SS.value[0])
	# 	run mpc with terminal constraint and get i-th trajectory
	plt.ylim([-2,10])
	plt.xlim([-10,0])
	print('start Mpc')
	# color = ['red', 'blue', 'green', 'black', 'brown']
	for i in range(n_iter):
		mpc.goal = goal_config
		goal_reached = False
		current_traj = Trajectory(n_state=n_state, n_action=n_action, mode='task')
		q0 = start_config.reshape([n_state,1])
		q_pred, u_pred, _ = mpc.solve(q0, SS, i)
		q_next = q_pred[:,1].reshape([n_state,1])
		# if i==n_iter-1:
			# plt.plot(q_pred[0,:], q_pred[1,:], marker='o')
		# plt.show()
		current_traj.append_data(q0, u_pred)
		if (np.sum(abs(q_next - goal_config))) < 0.5:
			goal_reached = True
		while not goal_reached:
			q0 = q_next
			q_pred, u_pred, _ = mpc.solve(q0, SS, i)
			q_next = q_pred[:,1].reshape([n_state,1])
			# plt.plot(q_pred[0,:], q_pred[1,:], 'ro')
			# plt.show()
			current_traj.append_data(q0, u_pred)
			if (np.sum(abs(q_next - goal_config))) < 0.5:
				goal_reached = True
			x = q_pred[0,:]
			y = q_pred[1,:]
			state_list = [val for sublist in SS.safe_set for val in sublist]
			state_list = np.hstack(state_list)
			x_SS = state_list[0,:]
			y_SS = state_list[1,:]
			plt.cla()
			plt.plot(x.T,y.T,'ro')
			plt.plot(x_pid.T, y_pid.T)
			plt.plot(x_ellipse, y_ellipse)
			plt.pause(0.001)
	# 	update safe-set and value function
		# print(current_traj.get_length())
		SS.update(traj=current_traj, iter_val=i, Q=Q, R=R, goal=goal_config)
		exploration_traj = Trajectory(n_state=n_state, n_action = n_action, mode='task')
		# run exploration controller and add to safe set
		print('exploration started')
		# exp = Explorer(goal=start_config, solver='ipopt', n_state=n_state, n_action=n_action, horizon=15, Q=Q, R=R)
		mpc.goal = np.array([-(i+1), -2.0, 0, 0]).reshape([-1,1])
		var = (i+1)*np.ones(n_state)
		var[-2:] = 0
		# sampled_goal = np.random.normal(goal_config.reshape(-1), var)
		# while (sampled_goal[0]+5)**2/4 + (sampled_goal[1]-1)**2/1 <= 1:
		# 	sampled_goal = np.random.normal(goal_config.reshape(-1), var)
		# mpc.goal = sampled_goal
		goal_reached = False
		feasible = True
		while not goal_reached:
			q0 = q_next
			q_pred, u_pred, feasible = mpc.solve(q0, SS, i+1)
			q_next = q_pred[:,1].reshape([n_state,1])
			# exploration_traj.append_data(q0, u_pred)
			if (np.linalg.norm(q_next-q0)) < 0.01:
				goal_reached = True
				# print(q_next[3,:])
				# print(q_next[2,:])
			x = q_pred[0,:]
			y = q_pred[1,:]
			plt.cla()
			plt.plot(x.T,y.T,'ro')
			plt.plot(x_pid.T, y_pid.T)
			plt.plot(x_ellipse, y_ellipse)
			plt.pause(0.001)	
		goal_reached = False
		mpc.goal = goal_config		
		while not goal_reached:
			q0 = q_next
			q_pred, u_pred, _ = mpc.solve(q0, SS, i+1)
			q_next = q_pred[:,1].reshape([n_state,1])
			# plt.plot(q_pred[0,:].T, q_pred[1,:].T, 'ro')
			# plt.plot(x_pid.T, y_pid.T)
			# plt.plot(x_ellipse, y_ellipse)
			# plt.show()
			exploration_traj.append_data(q0, u_pred)
			if (np.sum(abs(q_next - goal_config))) < 0.5:
				goal_reached = True


		SS.update(traj=exploration_traj, iter_val=i, Q=Q, R=R, goal=start_config)
		print('completed %d iteration' % (i))
		# print('task trajectory:', current_traj.get_length())
		# print('exploration trajectory:', exploration_traj.get_length())
		# x_val = np.arange(current_traj.get_length())
		# y_val = SS.value[i+1]
		# plt.plot(x_val, y_val)
		# plt.show()
		# x_task = np.arange(current_traj.get_length())
		# x_explore = np.arange(exploration_traj.get_length())
		# y_task = current_traj.compute_cost(Q=Q, R=R)
		# y_explore = exploration_traj.compute_cost(Q=Q, R=R)
		# plt.plot(x_task, y_task, color='blue')
		# plt.plot(x_explore, y_explore, color='red')
		x = np.hstack(current_traj.state_arr)[0,:]
		y = np.hstack(current_traj.state_arr)[1,:]
		# x = np.arange(len(current_traj.action_arr))
		state_list = [val for sublist in SS.safe_set for val in sublist]
		state_list = np.hstack(state_list)
		x_SS = state_list[0,:]
		y_SS = state_list[1,:]

		# plt.xlabel('x')
		# plt.ylabel('y')
		if i ==  n_iter-1:
			plt.cla()
			plt.plot(x_ellipse, y_ellipse, color='blue')
			plt.plot(x.T,y.T, color='black')
			plt.plot(x_pid.T, y_pid.T, color='red')
		plt.show()
		plt.cla()

	# plot trajectories
	# print(len(SS.value))
	# for i in range(n_iter+1):
	# 	print('no. of states:', len(SS.safe_set[i]))
	# 	print('no. of values:', len(SS.value[i]))
	plt.plot(x_ellipse, y_ellipse, color='black')
	# plt.plot(x_pid.T, y_pid.T)
	state_list = [val for sublist in SS.safe_set for val in sublist]
	state_list = np.hstack(state_list)
	x = state_list[0,:]
	y = state_list[1,:]
	plt.plot(x,y,marker='.',color='red')
	plt.show()
	# plt.show()
	x1 = np.arange(n_iter + 1)
	y1 = [arr[0] for arr in SS.value]
	plt.plot(x1,y1)
	plt.show()
	# print(SS.value)


if __name__ == '__main__':
	main()