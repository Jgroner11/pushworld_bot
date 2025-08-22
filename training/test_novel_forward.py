import numpy as np
import torch
import pickle


n_actions = 4
n = 150 # number of states
m = 6 # number of observations
t = 20 # num time steps of data

np.random.seed(42)
torch.manual_seed(42)

# initial distribution over states
pi = torch.ones(n, dtype=torch.float32) / n

a = torch.randint(low=0, high=n_actions, size=(t,))

with open('models/model3.pkl', 'rb') as f:
    chmm, _, _ = pickle.load(f)

T = torch.tensor(chmm.T, dtype=torch.float32)

# Make a emission matrix that has fixed emission probabilities depending on clone
E = torch.zeros((n, m), dtype=torch.float32)
state_loc = np.hstack(([0], chmm.n_clones)).cumsum(0)
for i in range(m):
    E[state_loc[i]:state_loc[i+1], i] = 1.0

# To test if our observation_probs implementation matches original, we actually need the probability of a certain observation to always be 1 in order to compare
o = np.random.randint(0, m, size=t, dtype=np.int64)
obs_probs = torch.zeros((t, m))
for t_ in range(t):
    obs_probs[t_, o[t_]] = 1.0


def forward_algorithm_with_action_dependent_transitions(T, E, pi, observation_probs, a):
    N = pi.shape[0]
    T_len = len(observation_probs)

    alpha = torch.zeros((T_len, N), dtype=pi.dtype, device=pi.device)

    alpha[0] = pi * torch.matmul(E, observation_probs[0])

    for t in range(1, T_len):
        T_a = T[a[t - 1]]
        alpha[t] = torch.matmul(alpha[t - 1], T_a) * torch.matmul(E, observation_probs[t])

    return alpha[-1].sum()

neg_log_prob_tensor = forward_algorithm_with_action_dependent_transitions(T, E, pi, obs_probs, a)
print("tensor forward:", neg_log_prob_tensor)

neg_log_prob_cscg = chmm.bps(o, np.array(a, dtype=np.int64))
neg_log_prob_cscg = chmm.forward()

print("cscg forward:", neg_log_prob_cscg)

print("")
