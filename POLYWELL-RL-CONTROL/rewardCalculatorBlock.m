function reward = rewardCalculatorBlock(beta, tau_c, uniformity, coilCurrents)
%REWARDCALCULATORBLOCK Computes reward signal for RL agent
%
% Multi-objective reward function balancing:
% - Plasma beta targeting (primary goal)
% - Confinement time maximization
% - Field uniformity
% - Power efficiency
%
% Inputs:
%   beta - Plasma beta
%   tau_c - Confinement time (s)
%   uniformity - Field uniformity (0-1)
%   coilCurrents - [6x1] Coil currents (A)
%
% Output:
%   reward - Scalar reward value

%% Load Parameters
persistent params_loaded params

if isempty(params_loaded)
    data = load('polywellFusionParams.mat');
    params = data.params;
    params_loaded = true;
end

%% Target Values
beta_target = 0.40;              % Target beta
tau_target = params.confinement.tau_target;  % Target confinement time
I_max = params.magnetic.I_max;   % Maximum current

%% Reward Components

% 1. Beta Reward (primary) - Encourage beta close to target
beta_error = abs(beta - beta_target);
beta_reward = -beta_error;  % Negative error (maximize when error=0)

% Bonus for being within ±5% of target
if beta_error < 0.02
    beta_reward = beta_reward + 1.0;  % Bonus
end

% 2. Confinement Time Reward
% Normalize by target, encourage longer
tau_ratio = tau_c / tau_target;
confinement_reward = min(tau_ratio, 2.0);  % Cap at 2x target

% 3. Uniformity Reward
% Directly use uniformity (0-1 scale)
uniformity_reward = uniformity;

% 4. Power Efficiency Penalty
% Penalize excessive current usage (I² losses)
power_normalized = sum(coilCurrents.^2) / (6 * I_max^2);
efficiency_penalty = -power_normalized;

% 5. Safety Penalties

% Penalty for beta too high (>0.6 is dangerous - MHD instabilities)
if beta > 0.6
    safety_penalty = -10 * (beta - 0.6);
else
    safety_penalty = 0;
end

% Penalty for currents near maximum
max_current = max(abs(coilCurrents));
if max_current > 0.95 * I_max
    current_penalty = -5 * (max_current / I_max - 0.95);
else
    current_penalty = 0;
end

%% Weighted Sum
weights = params.control.rewardWeights;  % [10, 5, 2, 0.1]

reward = weights(1) * beta_reward + ...
         weights(2) * confinement_reward + ...
         weights(3) * uniformity_reward + ...
         weights(4) * efficiency_penalty + ...
         safety_penalty + ...
         current_penalty;

%% Apply Bounds
% Prevent extreme reward values
reward = max(-20, min(reward, 25));

end
