import torch

def encoder_loss(encoder, input, a, T, E, pi):
    """
    input: (T, C, H, W) sequence of input images (TODO I think the channel dim comes after the H, W)
    encoder: nn.Module that converts image to logits

    """
    logits = encoder(input)  # (T, O)
    observation_lls = torch.log_softmax(logits, dim=1)
    log_likelihood = log_forward(T, E, pi, observation_lls, a)
    return -log_likelihood


def forward_algorithm_with_action_dependent_transitions(T, E, pi, observation_probs, a):
    """
    T: (T_len, N, N) action-dependent transition matrix
    N: number of states
    O: number of observations
    E: emission matrix (N, O) E[s, o] gives 0/1 prob that state s emits observation o
    pi: (N, ) initial distribution of states
    observation_probs: (T_len, O) observation_probs[t, o] gives likelihood of being in observation o at time t
    a: action sequence
    """
    N = pi.shape[0]
    T_len = len(observation_probs)

    # alpha[t] is probability of observing sequence up to + including time t
    alpha = torch.zeros((T_len, N), dtype=pi.dtype, device=pi.device)

    alpha[0] = pi * torch.matmul(E, observation_probs[0])

    for t in range(1, T_len):
        T_a = T[a[t - 1]]
        alpha[t] = torch.matmul(alpha[t - 1], T_a) * torch.matmul(E, observation_probs[t])

    return alpha[-1].sum()

# TODO delete if not needed
def forward_with_actions_softobs(A_actions, E, pi, L, a):
    """
    A_actions: (A, N, N)  transition matrices for each action
    E:         (N, O)     emission probabilities P(o|s) (rows need not be one-hot)
    pi:        (N,)       initial state distribution
    L:         (T, O)     per-time observation likelihood/probability vectors
    a:         (T-1,)     action indices for transitions between t-1 -> t
    Returns: log_likelihood, alpha, scales
    """
    N = pi.shape[0]
    T_len = L.shape[0]

    # precompute soft emission likelihoods b_t(s) = sum_o E[s,o] * L[t,o]
    # works whether rows of E are one-hot or probabilistic
    b = (E @ L.T).T              # shape (T, N)

    alpha = torch.zeros((T_len, N), dtype=pi.dtype, device=pi.device)
    scales = torch.zeros(T_len, dtype=pi.dtype, device=pi.device)

    alpha[0] = pi * b[0]
    c0 = alpha[0].sum()
    # handle impossible evidence at t=0
    if c0 <= 0:
        return -float('inf'), alpha, scales
    alpha[0] = alpha[0] / c0
    scales[0] = c0

    for t in range(1, T_len):
        A = A_actions[a[t-1]]    # (N, N)
        pred = alpha[t-1] @ A    # (N,)
        alpha[t] = pred * b[t]
        ct = alpha[t].sum()
        if ct <= 0:
            return -float('inf'), alpha, scales
        alpha[t] = alpha[t] / ct
        scales[t] = ct

    log_likelihood = torch.log(scales).sum()
    return log_likelihood, alpha, scales


def log_forward(T, E, pi, observation_lls, a):
    """
    T: (A, N, N) action-dependent transition matrix
    N: number of states
    O: number of observations
    E: emission matrix (N, O) E[s, o] gives 0/1 prob that state s emits observation o
    pi: (N, ) initial distribution of states
    observation_probs: (T_len, O) observation_probs[t, o] gives likelihood of being in observation o at time t
    a: action sequence
    """

    N = pi.shape[0]
    T_len = len(observation_lls)
    log_T = torch.log(T)

    log_alpha = torch.empty((T_len, N), dtype=pi.dtype, device=pi.device)
    # just need to mask so that all locations where E = 0 get ll -inf.
    idx = E.argmax(dim=1) # (N, ) idx[i] gives observation mapped to for state i
    # alpha[0] = log_pi + torch.logsumexp(log_E + observation_lls[0][None, :], dim=1)
    log_alpha[0] = torch.log(pi) + observation_lls[0][idx]

    for t in range(1, T_len):
        log_T_a = log_T[a[t - 1]]
        log_alpha[t] = torch.logsumexp(log_alpha[t - 1][:, None] + log_T_a, dim=0) + observation_lls[t][idx] #TODO: understand indexing here, if you aren't gonna use the E matrix directly then it makes sense to just pass n_clones into this function instead of E

    return log_alpha[-1].logsumexp(dim=0)

def learn_encoder(encoder, input, a, T, E, pi, n_iters=100):
    """
    Trains the encoder
    """
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    for _ in range(n_iters):
        optimizer.zero_grad()
        loss = encoder_loss(encoder, input, a, T, E, pi)
        loss.backward()
        optimizer.step()

    return loss

