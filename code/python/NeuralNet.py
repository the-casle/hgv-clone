
# %% Importing 
import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
## Set the hyperparameters for training:
EPISODES = 1000 # total number of episodes
EXPLORE_EPI_END = int(0.1*EPISODES) # initial exploration when agent will explore and no training
TEST_EPI_START = int(0.7*EPISODES ) # agent will be tested from this episode
EPS_START = 1.0 # e-greedy threshold start value
EPS_END = 0.05 # e-greedy threshold end value
EPS_DECAY = 1+np.log(EPS_END)/(0.6*EPISODES) # e-greedy threshold decay
GAMMA = 0.8 # Q-learning discount factor
LR = 0.001 # NN optimizer learning rate
MINIBATCH_SIZE = 64 # Q-learning batch size
ITERATIONS = 40 # Number of iterations for training
REP_MEM_SIZE = 10000 # Replay Memory size

# %% 
use_cuda = torch.cuda.is_available() 
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# %% Creation of QNet
class QNet(nn.Module):
    """
    Input to the network is a 4-dimensional state vector and the output a
    2-dimensional vector of two possible actions: move-left or move-right
    """
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
    """
    This class is used to store a large number, possibly say 10000, of the
    4-tuples "[State, Action, Next State, Reward]" at the output, meaning even
    before the nueral-network based learning kicks in. Subsequently, batches
    are constructed from this storage for training. The dynamically updated
    as each new 4-tuple becomes available.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    def push(self, s, a , r, ns):
        self.memory.append((FloatTensor([s]),
        a, # action is already a tensor
        FloatTensor([ns]),
        FloatTensor([r])))
        if len(self.memory) > self.capacity:
            del self.memory[0]
    def sample(self, MINIBATCH_SIZE):
        return random.sample(self.memory, MINIBATCH_SIZE)
    def __len__(self):
        return len(self.memory)

# %%
class QNetAgent: ## (E)
    def __init__(self, stateDim, actionDim):
        self.sDim = stateDim
        self.aDim = actionDim
        self.model = QNet(self.sDim, self.aDim) # Instantiate the NN model, loss and optimizer for training the agent
        if use_cuda:
            self.model.cuda()
            self.optimizer = optim.Adam(self.model.parameters(), LR)
            self.lossCriterion = torch.nn.MSELoss()
            self.memory = ReplayMemory(REP_MEM_SIZE) # Instantiate the Replay Memory for storing agent"s experiences
            # Initialize internal variables
            self.steps_done = 0
            self.episode_durations = []
            self.avg_episode_duration = []
            self.epsilon = EPS_START
            self.epsilon_history = []
            self.mode = ""

# %%
def select_action(self, state): ## (F)
    """ Select action based on epsilon-greedy approach """
    p = random.random() # generate a random number between 0 and 1
    self.steps_done += 1
    if p > self.epsilon:
        # if the agent is in ’exploitation mode’ select optimal action
        # based on the highest Q value returned by the trained NN
        with torch.no_grad():
            return self.model(FloatTensor(state)).data.max(1)[1].view(1, 1)
    else:
        # if the agent is in the ’exploration mode’ select a random action
        return LongTensor([[random.randrange(2)]])

# %%
def run_episode(self, e, environment): ## (G)
    state = environment.reset() # reset the environment at the beginning
    done = False
    steps = 0
    # Set the epsilon value for the episode
    if e < EXPLORE_EPI_END:
        self.epsilon = EPS_START
        self.mode = "Exploring"
    elif EXPLORE_EPI_END <= e <= TEST_EPI_START:
        self.epsilon = self.epsilon*EPS_DECAY
        self.mode = "Training"
    elif e > TEST_EPI_START:
        self.epsilon = 0.0
        self.mode = "Testing"
    self.epsilon_history.append(self.epsilon)
    while True: # Iterate until episode ends (i.e. a terminal state is reached)
        environment.render()
        action = self.select_action(FloatTensor([state])) # Select action based on epsilon-greedy approach
        c_action = action.data.cpu().numpy()[0,0]
        # Get next state and reward from environment based on current action
        next_state, reward, done, _ = environment.step(c_action)
        if done: # negative reward (punishment) if agent is in a terminal state
            reward = -10 # negative reward for failing
        # push experience into replay memory
        self.memory.push(state,action, reward, next_state)
        # if initial exploration is finished train the agent
        if EXPLORE_EPI_END <= e <= TEST_EPI_START:
            self.learn()
        state = next_state
        steps += 1
        if done: # Print information after every episode
            print("{2} Mode: {4} | Episode {0} Duration {1} steps | epsilon {3}"
            .format(e, steps, "\033[92m" if steps >= 195 else "\033[99m", self.epsilon, self.mode))
            self.episode_durations.append(steps)
            self. plot_durations(e)
            break

# %%
def learn(self): ## (H)
    """
    Train the neural newtwork using the randomly selected 4-tuples
    "[State, Action, Next State, Reward]" from the ReplayStore storage.
    """
    if len(self.memory) < MINIBATCH_SIZE:
        return
    for i in range(ITERATIONS):
        # minibatch is generated by random sampling from experience replay memory
        experiences = self.memory.sample(MINIBATCH_SIZE)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*experiences)
        # extract experience information for the entire minibatch
        batch_state = torch.cat(batch_state)
        batch_action = torch.cat(batch_action)
        batch_reward = torch.cat(batch_reward)
        batch_next_state = torch.cat(batch_next_state)
        # current Q values are estimated by NN for all actions
        current_q_values = self.model(batch_state).gather(1, batch_action)
        # expected Q values are estimated from actions which gives maximum Q value
        max_next_q_values = self.model(batch_next_state).detach().max(1)[0]
        expected_q_values = batch_reward + (GAMMA * max_next_q_values)
        # loss is measured from error between current and newly expected Q values
        loss = self.lossCriterion(current_q_values, expected_q_values.unsqueeze(1))
        # backpropagation of loss for NN training
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# %%
def plot_durations(self,epi): # Update the plot at the end of each episode ## (I)
    fig = plt.figure(1)
    fig.canvas.set_window_title("DQN Training Statistics")
    plt.clf()
    durations_t = torch.FloatTensor(self.episode_durations)
    plt.subplot(1,2,1)
    if epi<EXPLORE_EPI_END:
        plt.title("Agent Exploring Environment")
    elif EXPLORE_EPI_END <= e <= TEST_EPI_START:
        plt.title("Training Agent")
    else:
        plt.title("Testing Agent")
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(self.episode_durations)
        # Plot cumulative mean
    if len(durations_t) >= EXPLORE_EPI_END:
        means = durations_t.unfold(0, EXPLORE_EPI_END, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(EXPLORE_EPI_END-1), means))
        plt.plot(means.numpy())
    plt.subplot(1,2,2)
    plt.title("Epsilon per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.plot(self.epsilon_history)
    plt.show(block=False)
    plt.draw()
    plt.pause(0.0001)

# %%
if __name__ == "__main__":
    environment = gym.make('CartPole-v0') # creating the OpenAI Gym Cartpole environment
    state_size = environment.observation_space.shape[0]
    action_size = environment.action_space.n
    # Instantiate the RL Agent
    agent = QNetAgent(state_size, action_size)
    for e in range(EPISODES): # Train the agent
        agent.run_episode(e, environment)
    print('Complete')
    test_epi_duration = agent.episode_durations[TEST_EPI_START-EPISODES+1:]
    print("Average Test Episode Duration",np.mean(test_epi_duration))
    environment.close()
    plt.ioff()
    plt.show() 