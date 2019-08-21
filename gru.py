import torch
import torch.nn as nn
import numpy as np
import itertools

class Sigmoid:
	"""docstring for Sigmoid"""
	def __init__(self):
		pass
	def forward(self, x):
		self.res = 1/(1+np.exp(-x))
		return self.res
	def backward(self):
		return self.res * (1-self.res)
	def __call__(self, x):
		return self.forward(x)


class Tanh:
	def __init__(self):
		pass
	def forward(self, x):
		self.res = np.tanh(x)
		return self.res
	def backward(self):
		return 1 - (self.res**2)
	def __call__(self, x):
		return self.forward(x)


class GRU_Cell:
	"""docstring for GRU_Cell"""
	def __init__(self, in_dim, hidden_dim):
		self.d = in_dim
		self.h = hidden_dim
		h = self.h
		d = self.d

		self.Wzh = np.random.randn(h,h)
		self.Wrh = np.random.randn(h,h)
		self.Wh  = np.random.randn(h,h)

		self.Wzx = np.random.randn(h,d)
		self.Wrx = np.random.randn(h,d)
		self.Wx  = np.random.randn(h,d)



		self.dWzh = np.zeros((h,h))
		self.dWrh = np.zeros((h,h))
		self.dWh  = np.zeros((h,h))

		self.dWzx = np.zeros((h,d))
		self.dWrx = np.zeros((h,d))
		self.dWx  = np.zeros((h,d))

		self.z_act = Sigmoid()
		self.r_act = Sigmoid()
		self.h_act = Tanh()

		
	def forward(self, x, h):
		# input:
		# 	- x: shape(input dim),  observation at current time-step
		# 	- h: shape(hidden dim), hidden-state at previous time-step n
		# 
		# output:
		# 	- h_t: hidden state at current time-step

		# reset gate
		# r = self.r_act(np.matmul(self.Wrh, h) + np.matmul(self.Wrx, x))
		# # input gate
		# z = self.z_act(np.matmul(self.Wzh, h) + np.matmul(self.Wzx, x))
		# # update gate
		# g = self.h_act(np.matmul(self.Wh , (r * h)) + np.matmul(self.Wx, x))
		# # output gate
		# return (1 - z) * h + z * g
		self.x = x
		self.htminus1 = h

		self.rt = self.r_act(self.Wrh @ self.htminus1 + self.Wrx @ self.x)
		self.zt = self.z_act(self.Wzh @ self.htminus1 + self.Wzx @ self.x)
		self.htilde = self.h_act(self.Wh @ (self.rt * self.htminus1) + self.Wx @ self.x)
		self.h = (1 - self.zt) * self.htminus1 + self.zt * self.htilde
		return self.h

	def backward(self, delta):
		# input:
		# 	- delta: 	shape(hidden dim), summation of derivative wrt loss from next layer at 
		# 			same time-step and derivative wrt loss from same layer at
		# 			next time-step
		#
		# output:
		# 	- dx: 	Derivative of loss wrt the input x
		# 	- dh: 	Derivative of loss wrt the input hidden h

		# dLoss_dX
		dHt_dHtilde = delta * self.zt
		dHtilde_dX = self.h_act.backward()

		dHt_dZt = delta * (self.htilde - self.htminus1)
		dZt_dX = np.reshape(self.z_act.backward(), (1, self.h.size))

		dHtilde_dRt = np.reshape(self.h_act.backward(), (1, self.h.size))
		dRt_dX = np.reshape(self.r_act.backward(), (1, self.h.size))

		dLoss_dX = ((dHt_dHtilde * dHtilde_dX) @ self.Wx) + ((dHt_dZt * dZt_dX) @ self.Wzx) + ((dHt_dHtilde * dHtilde_dRt)  @ self.Wh) * self.htminus1 * dRt_dX @ self.Wrx

		# dLoss_dHtminus
		dHt_dHtminus1 = (1 - self.zt) * delta

		# dLoss_dHtminus1 = dHt_dHtminus1 + dHt_dHtilde * dHtilde_dHtminus1 @ self.Wh + dHt_dZt * dZt_dHtminus1 @ self.Wzh + ((dHt_dHtilde * dHtilde_dRt)  @ self.Wh) * dRt_dHtminus1 @ self.Wrh
		dLoss_dHtminus1 = ((dHt_dZt * dZt_dX) @ self.Wzh) + (((dHt_dHtilde * dHtilde_dRt)  @ self.Wh) * self.htminus1 * dRt_dX @ self.Wrh) + dHt_dHtminus1 + (((dHt_dHtilde * dHtilde_dRt) @ self.Wh) * self.rt)

		# weights
		dZt_dWzh = self.z_act.backward()
		dRt_dWrh = self.r_act.backward()
		dHtilde_dWh = self.h_act.backward()
		dRt_dWrx = self.r_act.backward()
		dZt_dWzx = self.z_act.backward()
		dHtilde_dWx = self.h_act.backward()

		# self.dWzh = (dHt_dZt * dZt_dWzh) @ self.htminus1
		self.dWzh = np.reshape((dHt_dZt * dZt_dWzh), (self.h.size, 1)) @ np.reshape(self.htminus1, (1, self.h.size))
		# self.dWrh = ((dHt_dHtilde * dHtilde_dRt)  @ self.Wh) * self.htminus1 * dRt_dWrh @ self.htminus1
		self.dWrh = (np.reshape(((dHt_dHtilde * dHtilde_dRt)  @ self.Wh) * self.htminus1 * dRt_dWrh, (self.h.size, 1)) @ np.reshape(self.htminus1, (1, self.h.size)))
		# self.dWh = dHt_dHtilde @ self.htminus1
		self.dWh = np.reshape(dHt_dHtilde * dHtilde_dWh, (self.h.size, 1)) @ np.reshape(self.rt * self.htminus1, (1, self.h.size))

		# self.dWzx = ((dHt_dZt * dZt_dWzx) @ self.x)
		self.dWzx = np.reshape(dHt_dZt * dZt_dWzx, (self.h.size, 1)) @ np.reshape(self.x, (1, self.x.size))
		# self.dWrx = ((dHt_dHtilde * dHtilde_dRt)  @ self.Wh) * self.htminus1 * dRt_dWrx @ self.x
		self.dWrx = np.reshape(((dHt_dHtilde * dHtilde_dRt)  @ self.Wh) * self.htminus1 * dRt_dWrx, (self.h.size, 1)) @ np.reshape(self.x, (1, self.x.size))
		# self.dWx = (dHt_dHtilde * dHtilde_dWx) @ self.x
		self.dWx = np.reshape((dHt_dHtilde * dHtilde_dWx), (self.h.size, 1)) @ np.reshape(self.x, (1, self.x.size))

		return dLoss_dX, dLoss_dHtminus1

def test():
	gru = GRU_Cell(3, 4)
	x = np.array([2, 4, 6])
	h = np.array([3, 5, 7, 9])
	gru.forward(x, h)
	delta = np.array([[0.52057634, -1.14434139, 1, 1]])
	gru.backward(delta)

if __name__ == '__main__':
	test()









