import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import os

class ElectricityMarket(gym.Env):
	wholesale_price = [2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,6,6,6,4,3,3,2,2]
	elasticity = [-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.5,-0.5,-0.5,-0.5,-0.7,-0.7,-0.7,-0.7,-0.7,-0.5,-0.5,-0.5]
	# curtailable load for 3 customers
	curt_demand = [[5,5,5,5,5,5,5,5,5,5,5,5,5,6,7,8,9,10,10,10,9,8,7,6],[5,5,5,5,5,5,5,5,5,5,5,5,5,6,7,8,10,11,11,11,10,8,7,6],[5,5,5,5,5,5,5,5,5,5,5,5,5,6,7,8,10,11,11,11,10,8,7,6]]
	num_customers = 3

	def __init__(self):
		self.wholesale_price = wholesale_price
		self.elasticity = elasticity
		self.curt_demand = curt_demand
		self.num_customers = num_customers
		# action space is the retail prices for customers
		# can use Discrete (i.e., discretize retail price)
		self.action_space = spaces.Box(low=0, high=10, shape=(num_customers,))
		# observation space is the consumed energy of customers
		# can use Discrete (i.e., discretize consumed energy of curtailable load)
		self.observation_space = spaces.Box(low=0, high=12,shape=(num_customers,))
		self.start = np.zeros(self.num_customers)
		self.state = self.start
		self.done = False

	def step(self, action, time):
		assert self.action_space.contains(action)
		new_state = self.take_action(action, time)
		# reward is defined as total net profit
		# for each customer, the profit is (retail price - wholesale price) * consumed energy
		reward = np.sum((action - np.ones(self.num_customers)*self.wholesale_price[time])*new_state)
		self.state = new_state
		# simulate 24 hrs
		if time == 24:
			self.done = True
		return self.state, reward, self.done, None

	def reset(self):
        self.done = False
        self.state = self.start
        return self.state

	def take_action(self, action, time):
		new_state = np.zeros(self.num_customers)
		for c in range(self.num_customers):
			curt_demand_c = self.curt_demand[c]
			retail_price_c = action[c]
			curt_consumption_c = curt_demand_c * (1 + self.elasticity[time]*(retail_price_c - self.wholesale_price[time])/self.wholesale_price[time])
			new_state[c] = curt_consumption_c
		return new_state



