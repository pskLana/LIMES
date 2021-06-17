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
from pathlib import Path

# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython: from IPython import display

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()

		# self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24)   
		# self.fc2 = nn.Linear(in_features=24, out_features=32)
		# self.out = nn.Linear(in_features=32, out_features=2)

		self.fc1 = nn.Linear(in_features=800, out_features=24)   
		self.fc2 = nn.Linear(in_features=24, out_features=32)
		self.out = nn.Linear(in_features=32, out_features=1)


	def forward(self, t):
		t = F.relu(self.fc1(t))
		t = F.relu(self.fc2(t))
		t = self.out(t).transpose(0,1)
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
 
# 		if rate > random.random():
# 			action = random.randrange(self.num_actions)
# 			return torch.tensor([action]).to(self.device) # explore      
# 		else:
# 			with torch.no_grad(): #turn off gradient tracking
# 				return policy_net(state).argmax(dim=1).to(self.device) # exploit
		return policy_net(state).argmax(dim=1).to(self.device)

class EnvManager():
	def __init__(self, device, current_state_examples):
		self.device = device
		self.state_num = 0
		self.current_state_examples = current_state_examples
		self.done = False
		self.num_actions = 1 # take one of 1 pairs
		self.stateTable = {}
		self.currentAction = None
		self.currentState = None
		self.currentReward = None

	def reset(self):
		self.state_num = 1
		self.current_state = None

	# def close(self):
	# 	self.env.close()
	
	def initializeCurrentState(self, current_state_examples):
		self.current_state_examples = current_state_examples

	def num_actions_available(self):
		return self.num_actions    

	def take_action(self, action, isLastIterationOfAL):
		if not isLastIterationOfAL:
			reward = 0.0
		else:
			reward = WombatRLObject.countFMeasure()
		self.state_num = self.state_num + 1
		self.currentAction = action
		self.currentReward = torch.tensor([reward], device=self.device)
		return torch.tensor([reward], device=self.device)

	def just_starting(self):
		return self.state_num == 1

	def get_state(self): 
		if self.just_starting() or self.done:
			# Add info about state_num and what is in this state(self.current_state_examples)
			# 		self.current_state_examples
			self.stateTable[self.state_num] = self.current_state_examples
			self.currentState = torch.tensor(self.current_state_examples, device=self.device, dtype=torch.float)
# 			torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0], device=self.device, dtype=torch.float)
			return self.currentState	
		else:
			# call Wombat, generate random examples and replace 3 worst ones
# 			if self.selectedExamples is not None:
# 				self.current_state_examples = WombatRLObject.replaceWorstExamples(self.selectedExamples)
# 			if self.state_num == self.endState:
# 				self.done = True
# 				
# 			self.stateTable[self.state_num] = self.current_state_examples
			
# 			t = self.currentState

# 			del self.current_state_examples[self.currentAction]
# 			t = torch.tensor(self.current_state_examples, device=self.device, dtype=torch.float)
# 			t[self.currentAction] = 1
			
# 			return torch.tensor(t, device=self.device, dtype=torch.float)

			self.currentState = torch.tensor(self.current_state_examples, device=self.device, dtype=torch.float)
# 			torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0], device=self.device, dtype=torch.float)
			return self.currentState

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

# 	t1 = torch.cat(batch.state)
	t1 = torch.stack(batch.state)
	t2 = torch.cat(batch.action)
	t3 = torch.cat(batch.reward)
# 	t4 = torch.cat(batch.next_state)
	t4 = torch.stack(batch.next_state)

	return (t1,t2,t3,t4)


class QValues():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	@staticmethod
	def get_current(policy_net, states, actions):
		# change states (list of objects to one dim array of code symbols of all source and target urls)
		return policy_net(states).mean(dim=1).gather(dim=0, index=actions.unsqueeze(-1)) #.unsqueeze(-1)
		
	@staticmethod        
	def get_next(target_net, next_states):                
# 		final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
# 		non_final_state_locations = (final_state_locations == False)
# 		non_final_states = next_states[non_final_state_locations]
# 		batch_size = next_states.shape[0]
# 		values = torch.zeros(batch_size).to(QValues.device)
# 		values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
# 		return values
		return target_net(next_states).max(dim=1)[0]

dic = {'test':1,
'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
'ema': None,
'memory': ReplayMemory(100000),
'strategy': EpsilonGreedyStrategy(1, 0.01, 0.001),
'policy_net': DQN(),
'target_net': DQN(),
'optimizer': None,
}

def initializeRL():
# 	pydevDebug()
	dic['ema'] = EnvManager(dic['device'], None)

	target_net = dic['target_net']
	target_net.load_state_dict(dic['policy_net'].state_dict())
	target_net.eval()
	
	lr = 0.001
	optimizer = optim.Adam(params=dic['policy_net'].parameters(), lr=lr)
	dic['optimizer'] = optimizer
	
def saveModel():
# 	pydevDebug()
	trainedModePath = "rl_policy_net.pt"
	nn_file = Path(trainedModePath)
	if not nn_file.is_file():
		torch.save(dic['policy_net'].state_dict(),"rl_policy_net.pt")

#### MAIN PART ########
def mainFun(newExamples, isLastIterationOfAL):
# 	pydevDebug()
	trainedModePath = "rl_policy_net.pt"
	trainedModelExists = False
	nn_file = Path(trainedModePath)
	if nn_file.is_file():
		trainedModelExists = True

	batch_size = 256
	gamma = 0.999
	eps_start = 1
	eps_end = 0.01
	eps_decay = 0.001
	target_update = 10
	memory_size = 100000
	lr = 0.001
	num_episodes = 2

	device = dic['device']
	em = dic['ema']
	em.initializeCurrentState(newExamples)
	strategy = dic['strategy']
	agent = Agent(strategy, em.num_actions_available(), device)
	memory = dic['memory']
	policy_net = dic['policy_net']
	target_net = dic['target_net']
	optimizer = dic['optimizer']
	
	if trainedModelExists:
		policy_net = DQN()
		policy_net.load_state_dict(torch.load(trainedModePath))
		policy_net.eval()

	if em.state_num == 0:
		state = em.get_state() #starting state
		action = agent.select_action(state, policy_net)
		reward = em.take_action(action, isLastIterationOfAL)
		return action.item()
	else:
		
# 		for timestep in count():
		for timestep in range(0, 6):
			state = em.currentState
			action = em.currentAction
			reward = em.currentReward
			next_state = em.get_state()
			next_action = agent.select_action(next_state, policy_net)
			next_reward = em.take_action(next_action, isLastIterationOfAL)
			memory.push(Experience(state, action, next_state, reward))
			
			if not trainedModelExists: 
				batch_size = 10 # at least 1 experiences should be done
				if memory.can_provide_sample(batch_size):
	# 				pydevDebug()
					experiences = memory.sample(batch_size)
					states, actions, rewards, next_states = extract_tensors(experiences)
					current_q_values = QValues.get_current(policy_net, states, actions)
					next_q_values = QValues.get_next(target_net, next_states)
					target_q_values = (next_q_values * gamma) + rewards
			
					loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					
				target_net.load_state_dict(policy_net.state_dict())
				
			return next_action.item()

# 	if em.done:
# 		episode_durations.append(timestep)
# 		plot(episode_durations, 100)
# 		break

# if episode % target_update == 0:
# 	target_net.load_state_dict(policy_net.state_dict())

	
def pydevDebug():
	import sys
	PYDEVD_PATH='the PYDEVD_PATH determined earlier'
	if sys.path.count(PYDEVD_PATH) < 1:
		sys.path.append(PYDEVD_PATH)
	import pydevd
	pydevd.settrace()
		