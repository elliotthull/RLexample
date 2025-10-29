import numpy as np
import matplotlib.pyplot as plt


states = np.arange(0, 4.0, 0.05)
actions = np.arange(0, 4.0, 0.05)

T = 2
Q = np.zeros([T, round(4 / 0.05), round(4 / 0.05)])
epsilon = 0.15
om_q = 0.55
om_mu = 0.85
gamma = 0.2
rho = 0.95
C = 3

mu = np.ones([T, len(actions)])

for t in range(T):
    for i in range(len(actions)):
        mu[t, i] = mu[t, i] / len(actions)

# idk about this factor (gamma in the pseudocode), a value was not mentioned in their description
discount = 0.95

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
        return ind[np.random.randint(0, len(ind))]
    else:
        return np.random.choice(list(range(state + 1)))

# initialize count for finding rho_Q (learning rate)

count_txa = np.zeros([T, len(states), len(actions)])


num_episodes = 200000
# Track mean field values for each time step
mean_mu_history = np.zeros((T, num_episodes))


# Learning loop
for k in range(num_episodes):

    # Sample initial state 
    x_idx = np.random.choice(list(range(0, 21)))
    
    # Episode loop over time periods
    for t in range(T):
        
        # Ensure valid state
        if x_idx >= len(states) or x_idx < 0:
            break
            
        # Current Q values at t
        Q_xt = Q[t, x_idx, :]
        
        # Select action
        a_idx = epsAction(Q_xt, x_idx)
        
        # Skip if action exceeds state 
        if a_idx > x_idx:
            print("act > stat")
            a_idx = x_idx  #Invest everything instead of breaking
        
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
                max_next_Q = discount * np.max(Q[t + 1, next_x_idx, :next_x_idx + 1])
            else:
                max_next_Q = discount * Q[t + 1, 0, 0]
            td_target = reward + max_next_Q
        else:
            #end state
            td_target = reward + rho * np.pow(states[next_x_idx], gamma) / gamma

        # target for mean field
        a_target = np.zeros(len(actions))
        a_target[a_idx] = 1
        
        # Update count for learning rate
        count_txa[t, x_idx, a_idx] += 1
        
        # Calculate learning rates
        rhos = rhosCalc(count_txa[t, x_idx, a_idx], k)
        rho_Q = rhos['q']
        rho_Mu = rhos['mu']
        
        # Q-learning update
        Q[t, x_idx, a_idx] = Q[t, x_idx, a_idx] + rho_Q * (td_target - Q[t, x_idx, a_idx])
        
        # Mean field distribution update
        mu[t, :] = mu[t, :] + rho_Mu * (a_target - mu[t, :])

        # Move to next state
        x_idx = next_x_idx
    # Record mean investment across actions at t=0 (for tracking)
    # Record the mean field for each time step after each episode
    for t in range(T):
        mean_mu_history[t, k] = np.dot(mu[t], actions)

'''
#for x in range(80):
    #print("%4.2f: " % (states[x]), end = '')
    #for a in range(x + 1):
       # print("%4.2f " % (Q[0, x, a]), end = '')

   # print("")
'''

#Plotting 
optimal = [[], []]
for x in range(80):
    optimal[0].append(actions[np.argmax(Q[0, x, :])])
    optimal[1].append(actions[np.argmax(Q[1, x, :])])

fig, (ax0, ax1) = plt.subplots(1, 2)

ax0.plot(states, optimal[0], 'k^')
ax0.set_xticks(np.arange(0, 1.5, 0.2))
ax0.set_yticks(np.arange(0,0.8,0.1))
ax0.set_xlim(0, 1.4)
ax0.set_xlabel('states')
ax0.set_ylabel('a(t, x)')
ax0.set_title('Learning Controls for MFG (t=0)')


# Plot for t = 1
ax1.plot(states, optimal[1], 'r^')
ax1.set_xticks(np.arange(0, 1.5, 0.2))
ax1.set_yticks(np.arange(0, 0.8, 0.1))
ax1.set_xlim(0, 1.4)
ax1.set_ylim(0, 0.7)
ax1.set_xlabel('states')
ax1.set_ylabel('a(t, x)') 
ax1.set_title('Learning Controls for MFG (t=1)')
plt.show()

#mean field plots
fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharey=True)
ax[0].plot(range(num_episodes), mean_mu_history[0, :], label='t = 0', color='blue')
ax[1].plot(range(num_episodes), mean_mu_history[1, :], label='t = 1', color='red')
ax[0].set_xlim(0,200000)
ax[0].set_xlabel('Episode')
ax[0].set_ylabel('Mean Investment μ(t)')
ax[0].set_title('Evolution of Mean Field Investments')
ax[0].legend()

ax[1].set_xlabel('Episode')
ax[1].set_xlim(0,200000)
ax[1].set_ylabel('Mean Investment μ(t)')
ax[1].set_title('Evolution of Mean Field Investments')
ax[1].legend()
plt.show()