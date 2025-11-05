import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# --- PARAMETERS ---
T = 4 
gamma = 0.2
rho = 0.95
C = 3
supp_W = [0.9, 1.3]
pmf_W = [0.75, 0.25]

exp_W_gamma = sum(np.power(w, gamma) * p for w, p in zip(supp_W, pmf_W))
pEWgamma = rho * exp_W_gamma
m0 = 0.5

states = np.arange(0, 4.05, 0.05)
actions = np.arange(0, 4.05, 0.05)
num_states = len(states)
num_actions = len(actions)

# --- THEORETICAL SOLUTION ---

def Phi(z):
    return pEWgamma / (1 + (C - 1) * np.power(z, 3))

def phi(z):
    return np.power(Phi(z), 1.0 / (gamma - 1))

def Psi(z):
    return np.power(Phi(z), 1.0 / gamma)

def fixed_point_equations(z_vec):
    z0, z1, z2, z3 = z_vec
    D = np.zeros(T + 1)
    D[T] = 1.0
    try:
        D[3] = phi(z3) * D[4] / (1 + phi(z3) * D[4])
        D[2] = phi(z2) * D[3] / (1 + phi(z2) * D[3])
        D[1] = phi(z1) * D[2] / (1 + phi(z1) * D[2])
        D[0] = phi(z0) * D[1] / (1 + phi(z0) * D[1])
    except:
        return np.full(T, 1e10)
    
    Psi_prod = [m0]
    for k in range(1, T):
        Psi_prod.append(Psi_prod[-1] * Psi(z_vec[k-1]))
    
    Lambda = []
    for k in range(T):
        numerator = 1.0
        for j in range(k + 1, T):
            numerator += phi(z_vec[j]) * D[j + 1]
        denominator = numerator + phi(z_vec[k]) * D[k + 1]
        Lambda_k = Psi_prod[k] * (numerator / denominator)
        Lambda.append(Lambda_k)
    
    residuals = [z_vec[k] - Lambda[k] for k in range(T)]
    return residuals

print("Solving theoretical solution...")
z_initial = np.full(T, m0)
z_theoretical, _, ier, _ = fsolve(fixed_point_equations, z_initial, full_output=True)

# Calculate D values
D_theoretical = np.zeros(T + 1)
D_theoretical[T] = 1.0
for t in range(T - 1, -1, -1):
    D_theoretical[t] = phi(z_theoretical[t]) * D_theoretical[t+1] / \
                       (1 + phi(z_theoretical[t]) * D_theoretical[t+1])

# Calculate theoretical policy coefficients
coefficients_theory = []
for t in range(T):
    if t < T - 1:
        coef = 1 / (1 + phi(z_theoretical[t]) * D_theoretical[t+1])
    else:
        coef = 0
    coefficients_theory.append(coef)

print("\nTheoretical Policy Coefficients α̂_t(x) = c_t * x:")
for t in range(T):
    print(f"  c_{t} = {coefficients_theory[t]:.6f}")

# --- Q-LEARNING (Simplified for speed) ---

print("\nRunning Q-Learning...")
Q = np.zeros([T, num_states, num_actions])
epsilon = 0.1
om_q = 0.6
om_mu = 0.75
discount = 0.95

mu = np.zeros([T, num_states, num_actions])
for t in range(T):
    for s in range(num_states):
        mu[t, s, :s+1] = 1.0 / (s + 1) if s > 0 else 1.0

count_txa = np.zeros([T, num_states, num_actions])
num_episodes = 50000  # Reduced for faster execution

def W():
    return np.random.choice(supp_W, p=pmf_W)

def env(state_idx, action_idx, mu_t):
    consump = states[state_idx] - actions[action_idx]
    G_mu_W = pEWgamma * (1 + (C - 1) * np.power(mu_t, 3))
    w_sample = W()
    wealth = actions[action_idx] * w_sample * C / G_mu_W
    new_state_idx = int(np.clip(round(wealth / 0.05), 0, num_states - 1))
    utility = np.power(max(consump, 1e-10), gamma) / gamma if consump > 0 else -1e10
    return {'x': new_state_idx, 'u': utility}

def rhosCalc(count_val, k):
    rhoQ = 1 / np.power(1 + count_val, om_q)
    rhoMu = 1 / np.power(2 + k, om_mu)
    return {'q': rhoQ, 'mu': rhoMu}

def epsAction(Q_x, state_idx):
    available_Q = Q_x[:state_idx + 1]
    if np.random.random() > epsilon:
        maxim = np.max(available_Q)
        ind = np.where(Q_x[:state_idx + 1] == maxim)[0]
        return np.random.choice(ind)
    else:
        return np.random.choice(list(range(state_idx + 1)))

# Training
for k in range(num_episodes):
    x_idx = np.random.choice(list(range(0, 21)))
    
    for t in range(T):
        if x_idx >= num_states or x_idx < 0:
            break
        
        a_idx = epsAction(Q[t, x_idx, :], x_idx)
        
        z_t = 0
        for s in range(min(21, num_states)):
            z_t += np.dot(mu[t, s, :], actions) / 21
        
        result = env(x_idx, a_idx, z_t)
        next_x_idx = result['x']
        reward = result['u']
        
        if t < T - 1:
            max_next_Q = np.max(Q[t + 1, next_x_idx, :next_x_idx + 1]) if next_x_idx > 0 else Q[t + 1, 0, 0]
            td_target = reward + discount * max_next_Q
        else:
            terminal_value = rho * np.power(states[next_x_idx], gamma) / gamma
            td_target = reward + terminal_value

        a_target = np.zeros(num_actions)
        a_target[a_idx] = 1
        
        count_txa[t, x_idx, a_idx] += 1
        rhos = rhosCalc(count_txa[t, x_idx, a_idx], k)
        
        Q[t, x_idx, a_idx] += rhos['q'] * (td_target - Q[t, x_idx, a_idx])
        mu[t, x_idx, :] += rhos['mu'] * (a_target - mu[t, x_idx, :])
        
        if np.sum(mu[t, x_idx, :x_idx+1]) > 0:
            mu[t, x_idx, :x_idx+1] /= np.sum(mu[t, x_idx, :x_idx+1])
        
        x_idx = next_x_idx
    
    if (k + 1) % 10000 == 0:
        print(f"  Episode {k+1}/{num_episodes}")

print("Q-Learning complete!\n")

# --- PLOTTING ---

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
colors = ['darkblue', 'darkred', 'darkgreen', 'darkorange']
time_labels = [f't={i}' for i in range(T)]

# Plot 1: Full policy functions
ax1 = axes[0, 0]
for t in range(T):
    # Theoretical policy
    theoretical_policy = coefficients_theory[t] * states
    ax1.plot(states, theoretical_policy, '-', linewidth=2.5, 
             color=colors[t], label=f'{time_labels[t]} Theory', alpha=0.7)
    
    # Q-Learning policy
    ql_policy = np.zeros(num_states)
    for s_idx in range(num_states):
        if s_idx > 0:
            optimal_action_idx = np.argmax(Q[t, s_idx, :s_idx + 1])
            ql_policy[s_idx] = actions[optimal_action_idx]
        else:
            ql_policy[s_idx] = 0
    
    ax1.plot(states, ql_policy, 'o', markersize=3, color=colors[t], 
             label=f'{time_labels[t]} Q-Learn', alpha=0.6, markevery=5)

ax1.plot([0, 4], [0, 4], 'k--', alpha=0.3, linewidth=1, label='x (max invest)')
ax1.set_xlabel('Wealth x', fontsize=13)
ax1.set_ylabel('Optimal Investment α̂(x)', fontsize=13)
ax1.set_title('Optimal Investment Policies: Theory vs Q-Learning', fontsize=15, fontweight='bold')
ax1.legend(fontsize=9, ncol=2, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 4])
ax1.set_ylim([0, 4])

# Plot 2: Investment fractions (α/x)
ax2 = axes[0, 1]
for t in range(T):
    # Theoretical fraction
    ax2.axhline(coefficients_theory[t], color=colors[t], linestyle='-', 
                linewidth=2.5, label=f'{time_labels[t]} Theory', alpha=0.7)
    
    # Q-Learning fractions
    ql_fractions = np.zeros(num_states)
    for s_idx in range(1, num_states):  # Skip x=0
        optimal_action_idx = np.argmax(Q[t, s_idx, :s_idx + 1])
        ql_fractions[s_idx] = actions[optimal_action_idx] / states[s_idx]
    
    ax2.plot(states[1:], ql_fractions[1:], 'o', markersize=3, color=colors[t], 
             label=f'{time_labels[t]} Q-Learn', alpha=0.6, markevery=5)

ax2.set_xlabel('Wealth x', fontsize=13)
ax2.set_ylabel('Investment Fraction α̂(x)/x', fontsize=13)
ax2.set_title('Investment Fraction Policies', fontsize=15, fontweight='bold')
ax2.legend(fontsize=9, ncol=2, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 4])
ax2.set_ylim([0, 1])

# Plot 3: Zoomed view for low wealth
ax3 = axes[1, 0]
zoom_range = states <= 1.0
for t in range(T):
    theoretical_policy = coefficients_theory[t] * states[zoom_range]
    ax3.plot(states[zoom_range], theoretical_policy, '-', linewidth=3, 
             color=colors[t], label=f'{time_labels[t]} Theory', alpha=0.7)
    
    ql_policy_zoom = np.zeros(sum(zoom_range))
    for i, s_idx in enumerate(np.where(zoom_range)[0]):
        if s_idx > 0:
            optimal_action_idx = np.argmax(Q[t, s_idx, :s_idx + 1])
            ql_policy_zoom[i] = actions[optimal_action_idx]
    
    ax3.plot(states[zoom_range], ql_policy_zoom, 'o', markersize=5, color=colors[t], 
             label=f'{time_labels[t]} Q-Learn', alpha=0.8, markevery=2)

ax3.set_xlabel('Wealth x', fontsize=13)
ax3.set_ylabel('Optimal Investment α̂(x)', fontsize=13)
ax3.set_title('Zoomed View: Low Wealth Region (x ≤ 1.0)', fontsize=15, fontweight='bold')
ax3.legend(fontsize=9, ncol=2)
ax3.grid(True, alpha=0.3)

# Plot 4: Policy coefficients comparison
ax4 = axes[1, 1]
x_pos = np.arange(T)
width = 0.35

bars1 = ax4.bar(x_pos - width/2, coefficients_theory, width, 
                label='Theoretical', color='steelblue', alpha=0.8, edgecolor='black')

# Calculate average Q-learning coefficients
coefficients_ql = []
for t in range(T):
    if t < T - 1:
        # Average over states 0.5 to 2.0
        sample_indices = np.where((states >= 0.5) & (states <= 2.0))[0]
        avg_coef = 0
        for s_idx in sample_indices:
            optimal_action_idx = np.argmax(Q[t, s_idx, :s_idx + 1])
            avg_coef += actions[optimal_action_idx] / states[s_idx]
        avg_coef /= len(sample_indices)
        coefficients_ql.append(avg_coef)
    else:
        coefficients_ql.append(0)

bars2 = ax4.bar(x_pos + width/2, coefficients_ql, width, 
                label='Q-Learning (avg)', color='coral', alpha=0.8, edgecolor='black')

ax4.set_xlabel('Time t', fontsize=13)
ax4.set_ylabel('Policy Coefficient c_t', fontsize=13)
ax4.set_title('Policy Coefficient Comparison: α̂_t(x) = c_t × x', fontsize=15, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f't={i}' for i in range(T)])
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([0, 1])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Optimal Investment Policy Comparison (T=4)', 
             fontsize=17, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('policy_comparison_detailed.png', dpi=150, bbox_inches='tight')
print("📊 Plot saved as 'policy_comparison_detailed.png'")
plt.show()

# Print comparison table
print("\n" + "="*70)
print("POLICY COEFFICIENT COMPARISON")
print("="*70)
print(f"{'Time':<8} {'Theory':<15} {'Q-Learning':<15} {'Difference':<15}")
print("-"*70)
for t in range(T):
    diff = abs(coefficients_theory[t] - coefficients_ql[t])
    print(f"t={t:<6} {coefficients_theory[t]:<15.6f} {coefficients_ql[t]:<15.6f} {diff:<15.6f}")
print("="*70)