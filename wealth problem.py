import numpy as np
import matplotlib.pyplot as plt

'''does the states and actions space need a uppper bound of 4.05 so that 4 is in the range?'''
states = np.arange(0, 4.0, 0.05)
actions = np.arange(0, 4.0, 0.05)

T = 2
Q = [np.zeros([round(4 / 0.05), round(4 / 0.05), 2])]
mu = [np.ones(len(states)) / len(states) for j in range(T)]
epsilon = 0.15
om_q = 0.55
om_mu = 0.85
gamma = 0.2
rho = 0.95
C = 3



# given white noise process with specific supports and probabilities
supp_W = [0.9, 1.3]
pmf_W = [0.75, 0.25]

# calculate expectation of white noise process 
exp_W_gamma = 0
for i in range(len(supp_W)):
    exp_W_gamma += np.pow(supp_W[i], gamma) * pmf_W[i]

# calculate rho * expectation [W ^ gamma]
pEWgamma = rho * exp_W_gamma

# white noise with two outcomes with probabilities established above
def W ():
    return np.random.choice(supp_W, p = pmf_W)

# given state/action (by index) and mean field investment
# return through G(mu, W()) * a (amount invested)
# the new state (rounded)
# and using formula for utility calculate reward
def env (state, action, mu):
    consump = states[state] - actions[action]
    wealth = actions[action] * W() * C / (pEWgamma * (1 + (C - 1) * np.pow(mu, 3)))
    newState = round(20 * wealth)
    utility = np.pow(consump, gamma) / gamma
    return { 'x': newState, 'u': utility }

# rho calculator, given we are on the kth episode and
# have visited the specific state/action/time pair count_txa times

def rhosCalc(count_txa, k):
    '''does this have a counter incrementing?'''
    rhoQ = 1 / np.pow(1 + count_txa, om_q)
    rhoMu = 1 / np.pow(2 + k, om_mu)
    return { 'q': rhoQ, 'mu': rhoMu }

# eps-greedy policy, takes in 1d array of Q matrix specified by state and time, and state (index)
# if Unif[0, 1] > epsilon, choose argmax on 1d array of Q matrix, limited by state
# if Unif[0, 1] < epsilon, choose random action, limited by state (unif distribution)
def epsAction (Q_x, state):
    if np.random.random() > epsilon:
        maxim = max(Q_x[:state + 1])
        ind = []
        for i in range(0, state + 1):
            if maxim == Q_x[i]:
                ind.append(i)
        return actions[ind[np.random.randint(0, len(ind))]]
    else:
        return np.random.choice(actions[:state + 1])

# initialize count for finding rho_Q (learning rate)
count_txa = np.zeros([T, len(states), len(actions)])


'''
C / (pEWgamma * (1 + (C - 1) * np.pow(0.5, 3)))
'''

num_episodes = 10000

# Learning loop
for k in range(num_episodes):

    # Sample initial state 
    x_idx = np.random.choice(len(states), p=mu[0])
    
    # Trajectory storage for mean field update
    mfg_update = []
    
    # Episode loop over time periods
    for t in range(T):
        
        # Ensure valid state
        if x_idx >= len(states) or x_idx < 0:
            break
            
        # Current Q values at t
        Q_xt = Q[0][x_idx, :, t]
        
        # Select action 
        a = epsAction(Q_xt, x_idx)
        a_idx = int(round(a / 0.05))  # Convert action value to index
        
        # Skip if action exceeds state 
        if a_idx > x_idx:
            a_idx = x_idx  #Invest everything instead of breaking
        
        # Store state-action pair for this time step
        mfg_update.append((t, x_idx, a_idx))
        
        # Get mean field investment at current time
        mu_t = np.dot(mu[t], actions)
        
        #call to environment
        result = env(x_idx, a_idx, mu_t)
        next_x_idx = result['x']
        reward = result['u']
        
        # Check next state is within the state space
        next_x_idx = min(max(next_x_idx, 0), len(states) - 1)
        
        # Calculate target
        if t < T - 1:
            if next_x_idx > 0:
                max_next_Q = np.max(Q[0][next_x_idx, :next_x_idx + 1, t + 1])
            else:
                max_next_Q = Q[0][0, 0, t + 1]
            td_target = reward + max_next_Q
        else:
            #end state
            td_target = reward
        
        # Update count for learning rate
        count_txa[t, x_idx, a_idx] += 1
        
        # Calculate learning rates
        rhos = rhosCalc(count_txa[t, x_idx, a_idx], k)
        rho_Q = rhos['q']
        
        # Q-learning update
        Q[0][x_idx, a_idx, t] = (1 - rho_Q) * Q[0][x_idx, a_idx, t] + rho_Q * td_target
        
        # Move to next state
        x_idx = next_x_idx
    
    # Mean field distribution update

    
    # Update mean field distribution for each time step 
    
    
    # update with learning rate
    

print(Q)