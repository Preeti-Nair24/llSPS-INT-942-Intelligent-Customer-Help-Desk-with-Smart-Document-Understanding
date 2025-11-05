# PPO vs DDPG for Polywell Control - Technical Comparison

## Executive Summary

This document compares two reinforcement learning algorithms for controlling polywell fusion reactor coil currents: **DDPG (Deep Deterministic Policy Gradient)** and **PPO (Proximal Policy Optimization)**.

---

## Algorithm Overview

### DDPG (Deep Deterministic Policy Gradient)

**Type:** Off-policy, Actor-Critic, Deterministic Policy
**Developed by:** DeepMind, 2016
**Best for:** Continuous control tasks requiring fine-grained deterministic actions

**Key Features:**
- Deterministic policy: μ(s) → a
- Uses experience replay buffer
- Requires external exploration noise
- Sample efficient (reuses old data)

### PPO (Proximal Policy Optimization)

**Type:** On-policy, Actor-Critic, Stochastic Policy
**Developed by:** OpenAI, 2017
**Best for:** General-purpose continuous control with stability priority

**Key Features:**
- Stochastic policy: π(a|s) ~ N(μ(s), σ(s))
- Uses fresh trajectories each iteration
- Built-in exploration (stochastic sampling)
- Very stable training

---

## Mathematical Comparison

### Policy Representation

**DDPG:**
```
Deterministic: a = μ_θ(s)

Action is a fixed function of state
Exploration via noise: a_executed = μ_θ(s) + N_t
```

**PPO:**
```
Stochastic: a ~ π_θ(·|s) = N(μ_θ(s), σ_θ(s))

Action sampled from distribution
Natural exploration via sampling
```

### Learning Objective

**DDPG:**
```
Maximize Q-value:
J(θ) = E_s[Q(s, μ_θ(s))]

Update rule:
∇_θ J = E_s[∇_θ μ(s) · ∇_a Q(s,a)|_{a=μ(s)}]
```

**PPO:**
```
Clipped surrogate objective:
L^CLIP(θ) = E_t[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

Where:
    r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    ε = 0.2 (typical clipping range)
    A_t = Advantage function
```

### Critic Function

**DDPG:**
```
Q-function (state-action value):
Q(s, a) = E[Σ γ^t r_t | s_0=s, a_0=a]

Takes both state AND action as input
```

**PPO:**
```
Value function (state value):
V(s) = E[Σ γ^t r_t | s_0=s]

Takes only state as input (simpler)
```

---

## Implementation Comparison

### Network Architecture

**DDPG:**
```
ACTOR:
State (9) → [128] → [64] → [6] → tanh → ×500 → Action

CRITIC:
State (9) → [128] → [64] ─┐
                           ├→ [32] → [1] Q-value
Action (6) → [64] ─────────┘

Total: ~35,000 parameters
```

**PPO:**
```
ACTOR (Stochastic):
State (9) → [256] → [128] ─┬→ [64] → [6] → tanh → ×500 → Mean μ
                           └→ [64] → [6] → exp → Std σ

CRITIC:
State (9) → [256] → [128] → [64] → [1] → Value V(s)

Total: ~45,000 parameters
```

### Training Loop

**DDPG:**
```python
for step in all_steps:
    # Interact with environment
    a = actor(s) + noise
    s', r = env.step(a)

    # Store in replay buffer
    buffer.add(s, a, r, s')

    # Learn from random batch
    batch = buffer.sample()
    update_critic(batch)
    update_actor(batch)

    # Slow target update
    target_networks.soft_update()
```

**PPO:**
```python
for iteration in iterations:
    # Collect trajectories
    trajectories = []
    for _ in range(horizon):
        a ~ actor(s)  # Sample from distribution
        s', r = env.step(a)
        trajectories.append((s,a,r,s'))

    # Compute advantages
    advantages = compute_GAE(trajectories)

    # Multiple optimization epochs
    for epoch in range(K):
        for batch in trajectories:
            update_actor(batch, advantages)  # With clipping
            update_critic(batch)
```

### Hyperparameters

| Parameter | DDPG | PPO |
|-----------|------|-----|
| Actor Learning Rate | 1×10⁻⁴ | 3×10⁻⁴ |
| Critic Learning Rate | 1×10⁻³ | 1×10⁻³ |
| Batch Size | 128 | 128 |
| Discount γ | 0.99 | 0.99 |
| **Replay Buffer** | 1,000,000 | Not used |
| **Target Update τ** | 0.001 | Not used |
| **Exploration Noise** | OU Process | Not needed |
| **Experience Horizon** | 1 step | 2000 steps |
| **Optimization Epochs** | 1 | 10 |
| **Clip Parameter ε** | N/A | 0.2 |
| **GAE Lambda λ** | N/A | 0.95 |
| **Entropy Weight** | N/A | 0.01 |

---

## Performance Comparison

### Convergence Speed

**DDPG:**
- Sample efficient: ~100,000 steps to converge
- Fast initial learning
- Can be unstable if hyperparameters not tuned

**PPO:**
- Less sample efficient: ~200,000 steps to converge
- Slower but steadier learning
- Very stable, robust to hyperparameters

### Training Stability

**DDPG:**
```
Reward vs Episode
    20│                     ╱───
       │                   ╱
    15│               ╱──╱
       │          ╱──╱
    10│      ╱──╱
       │  ╱──╱
     5│╱──╱
       │╱              (Can have dips/instability)
     0├────────────────────────────
       0   100   200   300   400   500
```

**PPO:**
```
Reward vs Episode
    20│                   ╱────
       │                 ╱
    15│               ╱─
       │             ╱
    10│          ╱──
       │       ╱──
     5│    ╱──
       │╱───           (Smoother, monotonic)
     0├────────────────────────────
       0   100   200   300   400   500
```

### Exploration Strategy

**DDPG:**
```matlab
% Requires explicit noise addition
action_clean = actor(state);
noise = OUNoise();  % Ornstein-Uhlenbeck process
action_executed = action_clean + noise;

Pros: Can control exploration explicitly
Cons: Need to tune noise parameters
```

**PPO:**
```matlab
% Natural exploration via sampling
mean, std = actor(state);
action = mean + std .* randn();  % Sample from Gaussian

Pros: Automatic exploration, no tuning
Cons: Less control over exploration
```

---

## Polywell-Specific Considerations

### For Deterministic, Fine-Grained Control (DDPG)

**Use when:**
- Need precise, repeatable control actions
- Have good simulator (for efficient replay)
- Can tolerate some training instability
- Sample efficiency is critical (limited data)

**Example scenario:**
- Final optimization of coil currents
- Known operating regime
- Real-time control requiring deterministic actions

### For Robust, Safe Learning (PPO)

**Use when:**
- Training stability is priority
- Environment is noisy/stochastic
- Safety constraints are critical
- Exploration is important

**Example scenario:**
- Initial learning phase (exploring parameter space)
- Uncertain plasma dynamics
- Need to avoid dangerous states
- Learning from real hardware (can't afford crashes)

---

## Computational Requirements

| Metric | DDPG | PPO |
|--------|------|-----|
| **Memory** | High (replay buffer) | Low (no buffer) |
| **CPU Time per Step** | Low | Medium |
| **GPU Utilization** | Medium | High (batch updates) |
| **Total Training Time** | 2-4 hours | 3-6 hours |
| **Inference Speed** | Very fast (deterministic) | Fast (one sample) |

---

## Recommendation for Polywell Control

### Phase 1: Initial Learning → **Use PPO**

**Reasons:**
- More stable during exploration
- Better for discovering safe operating regions
- Natural exploration finds diverse solutions
- Easier to implement and debug

### Phase 2: Fine-Tuning → **Use DDPG**

**Reasons:**
- Sample efficient (less expensive simulation time)
- Deterministic policy for deployment
- Fine-grained control adjustments
- Can initialize from PPO policy

### Best of Both Worlds: Hybrid Approach

```
1. Train PPO agent (500 episodes)
   → Discovers robust control strategy
   → Explores safely

2. Transfer to DDPG (initialize actor with PPO mean)
   → Fine-tune policy
   → Achieve deterministic control

3. Deploy DDPG agent
   → Fast inference
   → Repeatable actions
```

---

## Code Usage Examples

### Training DDPG:
```matlab
cd('POLYWELL-RL-CONTROL')
trainPolywellRLAgent  % Uses DDPG
```

### Training PPO:
```matlab
cd('POLYWELL-RL-CONTROL')
trainPolywellRLAgent_PPO  % Uses PPO
```

### Comparing Both:
```matlab
% Train both
trainPolywellRLAgent       % DDPG
trainPolywellRLAgent_PPO   % PPO

% Load and compare
load('trainedPolywellAgent.mat', 'trainingStats');
ddpg_rewards = trainingStats.EpisodeReward;

load('trainedPolywellAgent_PPO.mat', 'trainingStats');
ppo_rewards = trainingStats.EpisodeReward;

% Plot comparison
figure;
plot(smooth(ddpg_rewards), 'b', 'LineWidth', 2); hold on;
plot(smooth(ppo_rewards), 'r', 'LineWidth', 2);
legend('DDPG', 'PPO');
xlabel('Episode'); ylabel('Reward');
title('DDPG vs PPO Training Comparison');
```

---

## Conclusion

**DDPG:**
- ✅ Sample efficient
- ✅ Deterministic control
- ✅ Fast fine-tuning
- ❌ Requires careful tuning
- ❌ Can be unstable

**PPO:**
- ✅ Very stable
- ✅ Easy to implement
- ✅ Natural exploration
- ❌ Less sample efficient
- ❌ Stochastic policy

**For Polywell Control:**
Start with PPO for robust learning, then optionally fine-tune with DDPG for deployment.

---

## References

1. Lillicrap et al., "Continuous control with deep reinforcement learning" (DDPG), ICLR 2016
2. Schulman et al., "Proximal Policy Optimization Algorithms" (PPO), arXiv 2017
3. Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (GAE), ICLR 2016
4. OpenAI Spinning Up Documentation: https://spinningup.openai.com/
