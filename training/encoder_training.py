import torch
import torch.nn.functional as F

def encoder_loss(encoder, input, a, T, E, pi, *, stabilize=True, epsilon=1e-6):
    """
    input: (T, H, W, C) sequence of input images
    encoder: nn.Module that converts image to logits
    """
    logits = encoder(input)  # (T, O)
    if torch.any(torch.isnan(logits)):
        print("[encoder_loss] Detected NaNs in encoder logits before softmax")
    observation_lls = torch.log_softmax(logits, dim=1)
    if torch.any(~torch.isfinite(observation_lls)):
        print("[encoder_loss] Non-finite observation log-likelihoods detected",
              observation_lls)

    log_likelihood, _ = log_forward(T, E, pi, observation_lls, a, stabilize=stabilize, epsilon=epsilon)
    print(f'log_likelihood={log_likelihood}')
    if not torch.isfinite(log_likelihood):
        print("[encoder_loss] log_likelihood became non-finite; skipping gradients may be necessary")
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


def log_forward(T, E, pi, observation_lls, a, stabilize=False, epsilon=1e-6):
    """
    T: (A, N, N) action-dependent transition matrix P(s'|s,a)
    N: number of states
    O: number of observations
    E: emission matrix (N, O) E[s, o] gives 0/1 prob that state s emits observation o.
    pi: (N, ) initial distribution of states
    observation_lls: (T_len, O) observation_lls[t, o] gives likelihood of being in observation o at time t
    a: action sequence
    """

    N = pi.shape[0]
    T_len = len(observation_lls)
    if stabilize:
        eps = T.new_tensor(epsilon)
        if torch.any(T <= 0):
            zero_mask = (T <= 0)
            print(f"[log_forward] Found {zero_mask.sum().item()} zero/negative transitions; adding epsilon")
        T = torch.clamp(T, min=eps)
        pi = torch.clamp(pi, min=eps)
    log_T = torch.log(T)
    log_pi = torch.log(pi)

    # (T, N) log_alpha[t, s] is logp of being in state s at time t and having observed up to time t
    log_alpha = torch.empty((T_len, N), dtype=pi.dtype, device=pi.device)
    # just need to mask so that all locations where E = 0 get ll -inf.
    idx = E.argmax(dim=1) # (N, ) idx[i] gives observation mapped to for state i

    # observation_lls[0][idx] gives (N, ) vector of likelihood of being in state i at time 0
    log_alpha[0] = log_pi + observation_lls[0][idx]

    # if torch.any(~torch.isfinite(log_alpha[0])):
        # print("[log_forward] Non-finite initial alpha", log_alpha[0])

    for t in range(1, T_len):
        log_T_a = log_T[a[t - 1]]
        candidates = log_alpha[t - 1][:, None] + log_T_a  # (N, N)
        # if torch.all(torch.isinf(candidates)):
            # print(f"[log_forward] All transitions impossible at step {t}; candidates are -inf")
        log_alpha[t] = torch.logsumexp(candidates, dim=0) + observation_lls[t][idx]
        # if torch.any(~torch.isfinite(log_alpha[t])):
            # print(f"[log_forward] Non-finite alpha at step {t}", log_alpha[t])

    return log_alpha[-1].logsumexp(dim=0), log_alpha

def log_backward(T, E, pi, observation_lls, a, stabilize=False, epsilon=1e-6):
    """
    T: (A, N, N) action-dependent transition matrix P(s'|s,a)
    N: number of states
    O: number of observations
    E: emission matrix (N, O) E[s, o] gives 0/1 prob that state s emits observation o.
    pi: (N, ) initial distribution of states
    observation_lls: (T_len, O) observation_lls[t, o] gives likelihood of being in observation o at time t
    a: action sequence
    """

    N = pi.shape[0]
    T_len = len(observation_lls)
    if stabilize:
        eps = T.new_tensor(epsilon)
        # if torch.any(T <= 0):
            # zero_mask = (T <= 0)
            # print(f"[log_forward] Found {zero_mask.sum().item()} zero/negative transitions; adding epsilon")
        T = torch.clamp(T, min=eps)
        pi = torch.clamp(pi, min=eps)
    log_T = torch.log(T)
    log_pi = torch.log(pi)

    # (T, N) log_alpha[t, s] is logp of being in state s at time t and having observed up to time t
    log_beta = torch.empty((T_len, N), dtype=pi.dtype, device=pi.device)
    # just need to mask so that all locations where E = 0 get ll -inf.
    idx = E.argmax(dim=1) # (N, ) idx[i] gives observation mapped to for state i

    # observation_lls[0][idx] gives (N, ) vector of likelihood of being in state i at time 0
    log_beta[-1] = 0.0

    for t in range(T_len - 2, -1, -1):
        log_T_a = log_T[a[t]]
        # observation_lls[0][idx] gives (N, ) vector of likelihood of being in state i at time 0
        candidates = log_beta[t+1][None, :] + observation_lls[t][idx][None, :] + log_T_a  # (N, N)
        log_beta[t] = torch.logsumexp(candidates, dim=1)

    return log_beta

def log_fw_bw(T, E, pi, observation_lls, a, stabilize=False, epsilon=1e-6):
    ll, log_alpha = log_forward(T, E, pi, observation_lls, a, stabilize=stabilize, epsilon=epsilon)
    log_beta = log_backward(T, E, pi, observation_lls, a, stabilize=stabilize, epsilon=epsilon)
    log_gamma = log_alpha + log_beta - ll
    return log_gamma

def gamma_to_obs_soft_targets(log_gamma, E):
    """
    log_gamma: (T, N)  log posterior over states
    E:        (N, O)  deterministic 0/1 rows or general rows summing to 1
    Returns:  p_soft: (T, O) posterior over observations
    """
    gamma = log_gamma.exp()          # (T, N)
    p_soft = gamma @ E.to(torch.float64)              # (T, O)
    p_soft = p_soft / p_soft.sum(dim=1, keepdim=True).clamp_min(1e-40)
    return p_soft

def cross_entropy_soft_targets(logits, p_soft):
    # logits: (T, O); p_soft: (T, O), detached
    log_q = F.log_softmax(logits, dim=1)
    return -(p_soft.detach() * log_q).sum(dim=1).mean()


def learn_encoder(encoder, input, a, T, E, pi, n_iters=3, n_inner_iters=10):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=3e-4)
    for it in range(n_iters):
        print(f'Iteration {it}:')
        # 1) compute gamma with current encoder, but STOP-GRAD through DP
        with torch.no_grad():
            logits_cur = encoder(input)                          # (T, O)
            observation_lls = F.log_softmax(logits_cur, dim=1)   # (T, O)
            print(torch.argmax(observation_lls, dim=1))
            log_gamma = log_fw_bw(T, E, pi, observation_lls, a)  # (T, N)
            p_soft = gamma_to_obs_soft_targets(log_gamma, E)     # (T, O)

        # 2) refine encoder against detached soft targets
        for inner in range(n_inner_iters):
            optimizer.zero_grad()
            logits = encoder(input)                              # reuse model
            loss = cross_entropy_soft_targets(logits, p_soft)
            loss.backward()
            optimizer.step()
        print(f'loss={loss.item()}')
    return loss


def learn_encoder_old(encoder, input, a, T, E, pi, n_iters=3):
    """
    Trains the encoder
    """


    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-5)
    for iter in range(n_iters):
        print(f'Iteration {iter}:')
        optimizer.zero_grad()
        loss = encoder_loss(encoder, input, a, T, E, pi)
        loss.backward()
        optimizer.step()

    return loss


