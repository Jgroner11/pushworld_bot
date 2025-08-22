import torch
from encoders import SmallCNN

def encoder_loss(encoder, input, a, T, E, pi, observation_probs):
    """
    input: (T, C, H, W) sequence of input images
    encoder: nn.Module that converts image to logits

    """
    logits = get_logits(encoder, input)  # (T, O)
    observation_probs = torch.exp(logits)
    probability_of_observing = forward_algorithm_with_action_dependent_transitions(T, E, pi, observation_probs, a)
    log_likelihood = torch.log(probability_of_observing)
    return -log_likelihood


def get_logits(encoder, input):
    return encoder(input)


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


def log_forward(log_T, E, log_pi, observation_lls, a):
    """
    T: (T_len, N, N) action-dependent transition matrix
    N: number of states
    O: number of observations
    E: emission matrix (N, O) E[s, o] gives 0/1 prob that state s emits observation o
    pi: (N, ) initial distribution of states
    observation_probs: (T_len, O) observation_probs[t, o] gives likelihood of being in observation o at time t
    a: action sequence
    """


    N = log_pi.shape[0]
    T_len = len(observation_lls)

    alpha = torch.empty((T_len, N), dtype=log_pi.dtype, device=log_pi.device)
    # just need to mask so that all locations where E = 0 get ll -inf.
    # alpha[0] = log_pi + torch.logsumexp(log_E + observation_lls[0][None, :], dim=1)
    alpha[0] = log_pi + observation_lls[0]
    observation_lls[0][E[E==1]]



    for t in range(1, T_len):
        log_T_a = log_T[a[t - 1]]
        alpha[t] = torch.logsumexp(alpha[t - 1][:, None] + log_T_a, dim=0) + torch.logsumexp(log_E + observation_lls[t][None, :], dim=1)

    return alpha[-1].logsumexp(dim=0)

def learn_encoder(encoder, input, a, T, E, pi, observation_probs, steps=100):
    """
    Trains the encoder
    """
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    for step in range(steps):
        optimizer.zero_grad()
        loss = encoder_loss(encoder, input, a, T, E, pi, observation_probs)
        loss.backward()
        optimizer.step()

    return loss


if __name__ == "__main__":
    encoder = SmallCNN()
