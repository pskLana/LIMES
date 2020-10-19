# import gym

# env = gym.make('CartPole-v0')
# env.reset()

# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())

# env.close()

######################################

# %matplotlib inline
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T 

# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython: from IPython import display

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()

		# self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24)   
		# self.fc2 = nn.Linear(in_features=24, out_features=32)
		# self.out = nn.Linear(in_features=32, out_features=2)

		# weight_matrix = torch.tensor([
		#     [0,0],
		#     [0,2],
		#     [0,0],
		# 	[0,3],
		# 	[0,0],
		# ], dtype=torch.float32)
		weight_matrix = torch.zeros(5,2)
		self.fc = nn.Linear(in_features=2, out_features=5, bias=False)
		self.fc.weight = nn.Parameter(weight_matrix)
		self.out = nn.Linear(in_features=5, out_features=2, bias=False)

	def forward(self, t):
		# t = t.flatten(start_dim=1)
		# t = F.relu(self.fc1(t))
		# t = F.relu(self.fc2(t))
		# t = self.out(t)
		t = self.fc(t)
		t = F.relu(t)
		t = self.out(t)
		return t

Experience = namedtuple(
	'Experience',
	('state', 'action', 'next_state', 'reward')
)

# e = Experience(2,3,1,4)
# print(e)

class FrameRL:
	def __init__(self, source, target, similarity, sProp, tProp, result):
		self.source = source
		self.target = target
		self.similarity = similarity
		self.sProp = sProp
		self.tProp = tProp
		self.result = result
	
	def getSource(self):
		return self.source
	
	def getTarget(self):
		return self.target
	
	def getSimilarity(self):
		return self.similarity
	
	def getsProp(self):
		return self.sProp
	
	def gettProp(self):
		return self.tProp
	
	def getResult(self):
		return self.result
	


class ReplayMemory():
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.push_count = 0

	def push(self, experience):
		if len(self.memory) < self.capacity:
			self.memory.append(experience)
		else:
			self.memory[self.push_count % self.capacity] = experience
		self.push_count += 1

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def can_provide_sample(self, batch_size):
		return len(self.memory) >= batch_size

class EpsilonGreedyStrategy():
	def __init__(self, start, end, decay):
		self.start = start
		self.end = end
		self.decay = decay

	def get_exploration_rate(self, current_step):
		return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)

class Agent():
	def __init__(self, strategy, num_actions, device): # when we later create an agent object, need to get strategy from epsilon, num_actions = how many actions from a given state (2 for this game), device is the device in pytorch for tensor calculations CPU or GPU
		self.current_step = 0 # current step number in the environment
		self.strategy = strategy
		self.num_actions = num_actions
		self.device = device

	def select_action(self, state, policy_net): #policy_net is the policy trained by DQN
		rate = self.strategy.get_exploration_rate(self.current_step)
		self.current_step += 1

		if rate > random.random():
			action = random.randrange(self.num_actions)
			return torch.tensor([action]).to(self.device) # explore      
		else:
			with torch.no_grad(): #turn off gradient tracking
				return policy_net(state).argmax(dim=1).to(self.device) # exploit

class EnvManager():
	def __init__(self, device, current_state_examples):
		self.device = device
		self.state_num = 1
		self.current_state_examples = current_state_examples
		self.done = False
		self.num_actions = 2
		self.actions = {1: "take 3 best", 2: "take 3 nearest to 0.5"}
		self.oldFMeasure = 0.0
		self.selectedExamples = None
		self.endState = 5

	def reset(self):
		self.state_num = 1
		self.current_state = None

	# def close(self):
	# 	self.env.close()

	def num_actions_available(self):
		return self.num_actions    

	def take_action(self, action):
		reward = self.tryStep(action.item())     # if action is 0 or 1  
		self.state_num = self.state_num + 1
		return torch.tensor([reward], device=self.device)
	
	def tryStep(self, action):
		envRL = None
		if action == 0: # take 3 best
			envRL = WombatRLObject.get3Best(self.current_state_examples)
		else: # take 3 nearest to 0.5
			envRL = WombatRLObject.get3NearestToBoundary(self.current_state_examples)

		self.selectedExamples = envRL.getSelectedExamples()
		self.newFMeasure = envRL.getNewFMeasure()
		reward = envRL.getReward()
		self.oldFMeasure = self.newFMeasure
		return reward

	def just_starting(self):
		return self.state_num == 1

	def get_state(self): 
		if self.just_starting() or self.done:
			return self.current_state_examples
		else: # replace 3 predicted examples with 3 worst ones
			# sort self.current_state_examples by similarityMeasure and replace 3 first worse examples with self.selectedExamples
			self.current_state_examples = WombatRLObject.replaceWorstExamples(self.selectedExamples, self.current_state_examples)
			if self.state_num == self.endState:
				self.done = True
			return self.current_state_examples

def plot(values, moving_avg_period):
	plt.figure(2)
	plt.clf()        
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	plt.plot(values)

	moving_avg = get_moving_average(moving_avg_period, values)
	plt.plot(moving_avg)    
	plt.pause(0.001)
	print("Episode", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])
	# if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
	values = torch.tensor(values, dtype=torch.float)
	if len(values) >= period:
		moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
		moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
		return moving_avg.numpy()
	else:
		moving_avg = torch.zeros(len(values))
		return moving_avg.numpy()

# plot(np.random.rand(300), 100)    

def extract_tensors(experiences):
	# Convert batch of Experiences to Experience of batches
	batch = Experience(*zip(*experiences))

	t1 = torch.cat(batch.state)
	t2 = torch.cat(batch.action)
	t3 = torch.cat(batch.reward)
	t4 = torch.cat(batch.next_state)

	return (t1,t2,t3,t4)

class QValues():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	@staticmethod
	def get_current(policy_net, states, actions):
		return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

	@staticmethod        
	def get_next(target_net, next_states):                
		final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
		non_final_state_locations = (final_state_locations == False)
		non_final_states = next_states[non_final_state_locations]
		batch_size = next_states.shape[0]
		values = torch.zeros(batch_size).to(QValues.device)
		values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
		return values


dic = {'test':1,
'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")}
#### MAIN PART ########
def mainFun(newExamples):
	# newExamples = []
	# newExamples.append(FrameRL("s1", "t1", 0.1, "sProp", "tProp", -1))
	# newExamples.append(FrameRL("s2", "t2", 0.2, "sProp", "tProp", -1))
	# newExamples.append(FrameRL("s3", "t3", 0.3, "sProp", "tProp", -1))
	# newExamples.append(FrameRL("s4", "t4", 0.4, "sProp", "tProp", -1))
	# newExamples.append(FrameRL("s5", "t5", 0.6, "sProp", "tProp", 1))
	
	# t = torch.tensor(newExamples);
	# t = torch.as_tensor(newExamples);

	# t = torch.zeros(5,2)
	# t = torch.tensor([
	# 		[0,0],
	# 		[0,2],
	# 		[0,0],
	# 		[0,3],
	# 		[0,0],
	# 	], dtype=torch.float32)
	# # print(t)
	# examples = newExamples
	# aa = examples[0].getSource()

	# network = DQN()
	# pred = network(t)
	# print(pred)
	# return aa

	# dic['test']=dic['test']+1
	# return dic['test']

	batch_size = 256
	gamma = 0.999
	eps_start = 1
	eps_end = 0.01
	eps_decay = 0.001
	target_update = 10
	memory_size = 100000
	lr = 0.001
	num_episodes = 10000

	device = dic['device']#torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# em = CartPoleEnvManager(device)
	em = EnvManager(device, newExamples)
	strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
	agent = Agent(strategy, em.num_actions_available(), device) #change class -> Wombat in Java
	memory = ReplayMemory(memory_size) #change class -> move to java
	policy_net = DQN()
	target_net = DQN()
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()
	optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

	state = em.get_state() #starting state
	# for timestep in count():
	action = agent.select_action(state, policy_net) #change function
	reward = em.take_action(action) #change function
	next_state = em.get_state()
	return next_state
	memory.push(Experience(state, action, next_state, reward))
	state = next_state

	# if memory.can_provide_sample(batch_size):
	# 	experiences = memory.sample(batch_size)
	# 	states, actions, rewards, next_states = extract_tensors(experiences)

	# 	current_q_values = QValues.get_current(policy_net, states, actions)
	# 	next_q_values = QValues.get_next(target_net, next_states)
	# 	target_q_values = (next_q_values * gamma) + rewards

	# 	loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
	# 	optimizer.zero_grad()
	# 	loss.backward()
	# 	optimizer.step()

# 	if em.done:
# 		episode_durations.append(timestep)
# 		plot(episode_durations, 100)
# 		break

# if episode % target_update == 0:
# 	target_net.load_state_dict(policy_net.state_dict())





# episode_durations = []
# for episode in range(num_episodes):
# 	em.reset()
# 	state = em.get_state() #starting state
	# for timestep in count():
	# 	action = agent.select_action(state, policy_net) #change function
	# 	reward = em.take_action(action) #change function
	# 	next_state = em.get_state()
	# 	memory.push(Experience(state, action, next_state, reward))
	# 	state = next_state

		# if memory.can_provide_sample(batch_size):
		# 	experiences = memory.sample(batch_size)
		# 	states, actions, rewards, next_states = extract_tensors(experiences)

		# 	current_q_values = QValues.get_current(policy_net, states, actions)
		# 	next_q_values = QValues.get_next(target_net, next_states)
		# 	target_q_values = (next_q_values * gamma) + rewards

		# 	loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
		# 	optimizer.zero_grad()
		# 	loss.backward()
		# 	optimizer.step()

	# 	if em.done:
	# 		episode_durations.append(timestep)
	# 		plot(episode_durations, 100)
	# 		break

	# if episode % target_update == 0:
	# 	target_net.load_state_dict(policy_net.state_dict())

# em.close()

	# return dic['test']