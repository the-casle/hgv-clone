
# %% Importing 
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

from sim_NeuralNet import plane

EPISODES = 100
TRAIN_START = 0.5
TRAIN_DUR = 0.5
EXPLORE_EPI_END = int(TRAIN_START*EPISODES) 
TEST_EPI_START = int((TRAIN_START + TRAIN_DUR)*EPISODES ) 
EPS_START = 1.0 
EPS_END = 0.05 
EPS_DECAY = 1+np.log(EPS_END)/(TRAIN_DUR*EPISODES) 
GAMMA = 0.9 
LR = 0.001 
MINIBATCH_SIZE = 25
ITERATIONS = 40 
REP_MEM_SIZE = 10000 

neqn = 5

use_cuda = torch.cuda.is_available() 
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

class QNet(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(state_space_dim, 24)
        self.l2 = nn.Linear(24, 24)
        self.l3 = nn.Linear(24, action_space_dim)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
class ReplayMemory: ## (D)
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    def push(self, s, a , r, ns):
        self.memory.append((FloatTensor([s]),
        a,
        FloatTensor([ns]),
        FloatTensor([r])))
        if len(self.memory) > self.capacity:
            del self.memory[0]
    def sample(self, MINIBATCH_SIZE):
        return random.sample(self.memory, MINIBATCH_SIZE)
    def __len__(self):
        return len(self.memory)

class QNetAgent: 
    def __init__(self, stateDim, actionDim):
        self.sDim = stateDim
        self.aDim = actionDim
        self.model = QNet(self.sDim, self.aDim) 
        if use_cuda:
            self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), LR)
        self.lossCriterion = torch.nn.MSELoss()
        self.memory = ReplayMemory(REP_MEM_SIZE) 
        
        self.steps_done = 0
        self.episode_durations = []
        self.avg_episode_duration = []
        self.epsilon = EPS_START
        self.epsilon_history = []
        self.mode = ""
        self.eps_done = 0

    def select_action(self, state):
        self.steps_done += 1
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.model(FloatTensor(state)).data.max(1)[1].view(1, 1)
        else:
            return LongTensor([[random.randrange(2)]])


    def run_episode(self, e, environment): 
        state = environment.render(self.eps_done)
        self.eps_done += 1
        done = False
        steps = 0
        
        if e < EXPLORE_EPI_END:
            self.epsilon = EPS_START
            self.mode = "Exploring"
        elif EXPLORE_EPI_END <= e <= TEST_EPI_START:
            self.epsilon = self.epsilon*EPS_DECAY
            self.mode = "Training"
        elif e > TEST_EPI_START:
            self.epsilon = 0.025
            self.mode = "Testing"
        self.epsilon_history.append(self.epsilon)
        while True: 
            action = self.select_action(FloatTensor([state]))
            c_action = action.data.cpu().numpy()[0,0]
            state, next_state, reward, done = environment.step(c_action)
            self.memory.push(state, action, reward, next_state)
            if EXPLORE_EPI_END <= e <= TEST_EPI_START:
                self.learn()
            state = next_state
            steps += 1
            if done:
                print("{2} Mode: {4} | Episode {0} Duration {1} steps | epsilon {3}"
                .format(e, steps, "\033[92m" if steps >= 195 else "\033[99m", self.epsilon, self.mode))
                self.episode_durations.append(steps)
                break
        return environment.cumreward

    def finaltest(self, eps, state):
        state = environment.render(eps=eps, state=state)
        self.epsilon = 0.025
        while True: 
            action = self.select_action(FloatTensor([state]))
            c_action = action.data.cpu().numpy()[0,0]
            state, next_state, reward, done = environment.step(c_action)
            state = next_state
            steps += 1
            if done:
                print("| Duration {0} steps | epsilon {1}"
                .format(steps, self.epsilon))
                break
        

    def learn(self): 
        if len(self.memory) < MINIBATCH_SIZE:
            return
        for i in range(ITERATIONS):
            
            experiences = self.memory.sample(MINIBATCH_SIZE)
            batch_state, batch_action, batch_next_state, batch_reward = zip(*experiences)
            
            batch_state = torch.cat(batch_state)
            batch_action = torch.cat(batch_action)
            batch_reward = torch.cat(batch_reward)
            batch_next_state = torch.cat(batch_next_state)
           
            current_q_values = self.model(batch_state).gather(1, batch_action)
            
            max_next_q_values = self.model(batch_next_state).detach().max(1)[0]
            expected_q_values = batch_reward + (GAMMA * max_next_q_values)
            
            loss = self.lossCriterion(current_q_values, expected_q_values.unsqueeze(1))
           
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

if __name__ == "__main__":
    environment = plane()
    state_size = 5
    action_size = 2
    rewards = np.zeros(EPISODES)
    
    agent = QNetAgent(state_size, action_size)
    for e in range(EPISODES): 
        reward = agent.run_episode(e, environment)
        rewards[e] = reward
    print('Complete')
    test_epi_duration = agent.episode_durations[TEST_EPI_START-EPISODES+1:]
    print("Average Test Episode Duration",np.mean(test_epi_duration))

    os.chdir(r'C:\Users\GreenFluids_VR\Documents\GitHub\CUDA__Project\code\data')
    torch.save(agent.model.state_dict(), "weights1.h5")
    with open('rewards.txt', 'w') as outputFile:
        for e in range(EPISODES):
            outputFile.write(f"{rewards[e]}\n")
    agent.finaltest(state=[0,6e3, 60e3, 0, 0.35])
    agent.finaltest(state=[0,6e3, 75e3, 0, 0.35])
    