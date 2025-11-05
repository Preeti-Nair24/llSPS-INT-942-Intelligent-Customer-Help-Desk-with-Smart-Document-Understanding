%% Run Polywell Fusion Simulation with RL Control - 500 Episodes
% This script runs the complete simulation for 500 episodes and generates
% comprehensive transient plots for confinement time and other parameters.
%
% Features:
% - Trains PPO agent or loads existing
% - Runs 500 episodes with physics-based plasma dynamics
% - Logs all transient data (beta, tau, uniformity, currents, power)
% - Generates publication-quality plots
% - Saves results for analysis
%
% Prerequisites:
% - setupFusionParameters.m (generates params)
% - RL Toolbox installed

clear all; close all; clc;

fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  POLYWELL FUSION RL CONTROL - 500 EPISODE SIMULATION         ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

%% Step 1: Load Parameters
fprintf('[1/7] Loading fusion physics parameters...\n');

if ~exist('polywellFusionParams.mat', 'file')
    fprintf('      Running setupFusionParameters.m...\n');
    setupFusionParameters;
end

load('polywellFusionParams.mat', 'params');
fprintf('      ✓ Parameters loaded\n\n');

%% Step 2: Create RL Environment (Custom without Simulink)
fprintf('[2/7] Creating RL environment...\n');

% Since we're running MATLAB-only (not Simulink in loop), use custom env
env = PolywellRLEnvironment(struct(), params.geometry.coilPositions);
env.MaxSteps = 100;  % 100 steps per episode (10 seconds at 0.1s timestep)

fprintf('      ✓ Environment created\n');
fprintf('        Episodes: 500\n');
fprintf('        Steps per episode: %d\n', env.MaxSteps);
fprintf('        Total timesteps: %d\n\n', 500 * env.MaxSteps);

%% Step 3: Create or Load RL Agent
fprintf('[3/7] Setting up PPO agent...\n');

agentFile = 'polywellAgent_500ep.mat';

if exist(agentFile, 'file')
    fprintf('      Found existing agent, loading...\n');
    load(agentFile, 'agent');
    fprintf('      ✓ Agent loaded from %s\n\n', agentFile);
else
    fprintf('      Creating new PPO agent...\n');

    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);

    % Compact network for reasonable training time
    actorNet = [
        featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(128, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(64, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'fc3')
        tanhLayer('Name', 'tanh')
        scalingLayer('Name', 'scale', 'Scale', actInfo.UpperLimit)
    ];

    criticNet = [
        featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(128, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(64, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(1, 'Name', 'value')
    ];

    actor = rlStochasticActorRepresentation(actorNet, obsInfo, actInfo, 'ObservationInputNames', 'state');
    critic = rlValueRepresentation(criticNet, obsInfo, 'ObservationInputNames', 'state');

    agentOpts = rlPPOAgentOptions('SampleTime', params.control.dt, ...
        'ExperienceHorizon', 1000, 'MiniBatchSize', 128, 'NumEpoch', 5);
    agentOpts.ActorOptimizerOptions.LearnRate = 3e-4;
    agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;

    agent = rlPPOAgent(actor, critic, agentOpts);

    fprintf('      ✓ PPO agent created\n\n');
end

%% Step 4: Initialize Data Storage
fprintf('[4/7] Initializing data storage...\n');

numEpisodes = 500;
maxSteps = env.MaxSteps;

% Pre-allocate arrays for efficiency
data.beta = zeros(numEpisodes, maxSteps);
data.tau_c = zeros(numEpisodes, maxSteps);
data.uniformity = zeros(numEpisodes, maxSteps);
data.fusionPower = zeros(numEpisodes, maxSteps);
data.B_field = zeros(numEpisodes, maxSteps);
data.coilCurrents = zeros(numEpisodes, maxSteps, 6);
data.rewards = zeros(numEpisodes, maxSteps);

% Episode summaries
summary.episodeReward = zeros(numEpisodes, 1);
summary.avgBeta = zeros(numEpisodes, 1);
summary.avgTau = zeros(numEpisodes, 1);
summary.avgUniformity = zeros(numEpisodes, 1);
summary.avgFusionPower = zeros(numEpisodes, 1);

fprintf('      ✓ Data arrays allocated\n');
fprintf('        Memory: ~%.1f MB\n\n', (numel(data.beta)*8 + numel(data.coilCurrents)*8)/1e6);

%% Step 5: Run 500 Episodes with Data Collection
fprintf('[5/7] Running 500 episodes...\n');
fprintf('      This will take approximately 10-30 minutes depending on hardware.\n');
fprintf('      Progress will be displayed every 25 episodes.\n\n');

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  Episode | Reward  | Avg Beta | Avg Tau(ms) | Status          ║\n');
fprintf('╠═══════════════════════════════════════════════════════════════╣\n');

tic;

for episode = 1:numEpisodes
    % Reset environment
    obs = reset(env);
    episodeReward = 0;

    for step = 1:maxSteps
        % Get action from agent
        action = getAction(agent, obs);

        % Execute action
        [nextObs, reward, isDone] = env.step(action);
        episodeReward = episodeReward + reward;

        % Calculate magnetic field from currents
        B_field = magneticFieldBlock(env.CoilCurrents);

        % Calculate plasma physics
        [beta, tau_c, uniformity, fusionPower] = ...
            plasmaPhysicsBlock(env.CoilCurrents, B_field);

        % Store data
        data.beta(episode, step) = beta;
        data.tau_c(episode, step) = tau_c;
        data.uniformity(episode, step) = uniformity;
        data.fusionPower(episode, step) = fusionPower;
        data.B_field(episode, step) = B_field;
        data.coilCurrents(episode, step, :) = env.CoilCurrents;
        data.rewards(episode, step) = reward;

        obs = nextObs;

        if isDone
            break;
        end
    end

    % Store episode summaries
    summary.episodeReward(episode) = episodeReward;
    summary.avgBeta(episode) = mean(data.beta(episode, 1:step));
    summary.avgTau(episode) = mean(data.tau_c(episode, 1:step));
    summary.avgUniformity(episode) = mean(data.uniformity(episode, 1:step));
    summary.avgFusionPower(episode) = mean(data.fusionPower(episode, 1:step));

    % Print progress every 25 episodes
    if mod(episode, 25) == 0 || episode == 1
        status = '';
        if episode < 100
            status = 'Exploring';
        elseif summary.avgBeta(episode) < 0.35
            status = 'Learning';
        elseif summary.avgBeta(episode) < 0.39
            status = 'Improving';
        else
            status = 'Optimized!';
        end

        fprintf('║  %6d | %7.2f | %8.3f | %11.2f | %-15s ║\n', ...
            episode, episodeReward, summary.avgBeta(episode), ...
            summary.avgTau(episode)*1000, status);
    end
end

elapsedTime = toc;

fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');
fprintf('      ✓ 500 episodes completed in %.1f minutes\n\n', elapsedTime/60);

%% Step 6: Save Results
fprintf('[6/7] Saving results...\n');

% Save agent
save(agentFile, 'agent');
fprintf('      ✓ Agent saved: %s\n', agentFile);

% Save all data
save('polywellSimulation_500ep_data.mat', 'data', 'summary', 'params', '-v7.3');
fprintf('      ✓ Simulation data saved: polywellSimulation_500ep_data.mat\n');
fprintf('        File size: %.1f MB\n\n', dir('polywellSimulation_500ep_data.mat').bytes/1e6);

%% Step 7: Generate Transient Plots
fprintf('[7/7] Generating transient analysis plots...\n\n');

% Call plotting function
plotPolywellTransients(data, summary, params);

fprintf('      ✓ All plots generated and saved\n\n');

%% Final Summary
fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║  SIMULATION COMPLETE - SUMMARY                                ║\n');
fprintf('╠═══════════════════════════════════════════════════════════════╣\n');
fprintf('║                                                               ║\n');
fprintf('║  Performance Metrics:                                         ║\n');
fprintf('║  ├─ Initial avg beta:     %.3f                             ║\n', summary.avgBeta(1));
fprintf('║  ├─ Final avg beta:       %.3f                             ║\n', summary.avgBeta(end));
fprintf('║  ├─ Best beta achieved:   %.3f                             ║\n', max(summary.avgBeta));
fprintf('║  └─ Improvement:          %.1fx                              ║\n', summary.avgBeta(end)/summary.avgBeta(1));
fprintf('║                                                               ║\n');
fprintf('║  Confinement Time:                                            ║\n');
fprintf('║  ├─ Initial avg:          %.3f ms                          ║\n', summary.avgTau(1)*1000);
fprintf('║  ├─ Final avg:            %.3f ms                          ║\n', summary.avgTau(end)*1000);
fprintf('║  ├─ Target:               %.3f ms                          ║\n', params.confinement.tau_target*1000);
fprintf('║  └─ Achievement:          %.1f%%                             ║\n', summary.avgTau(end)/params.confinement.tau_target*100);
fprintf('║                                                               ║\n');
fprintf('║  Fusion Power:                                                ║\n');
fprintf('║  ├─ Average:              %.2e W                          ║\n', mean(summary.avgFusionPower));
fprintf('║  └─ Peak:                 %.2e W                          ║\n', max(summary.avgFusionPower));
fprintf('║                                                               ║\n');
fprintf('║  Files Created:                                               ║\n');
fprintf('║  ├─ polywellAgent_500ep.mat                                   ║\n');
fprintf('║  ├─ polywellSimulation_500ep_data.mat                         ║\n');
fprintf('║  ├─ ConfinementTime_Transient.png                             ║\n');
fprintf('║  ├─ PlasmaParameters_Evolution.png                            ║\n');
fprintf('║  └─ FusionPerformance.png                                     ║\n');
fprintf('║                                                               ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

fprintf('Simulation complete! Check the generated plots.\n');
fprintf('To re-plot: plotPolywellTransients(data, summary, params)\n\n');
