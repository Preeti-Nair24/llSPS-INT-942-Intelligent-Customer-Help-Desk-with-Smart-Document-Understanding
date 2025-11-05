%% Train Polywell RL Agent using PPO
% This script trains a PPO (Proximal Policy Optimization) agent to control
% the coil currents in a polywell fusion reactor for optimal plasma confinement.
%
% PPO Advantages over DDPG:
% - More stable training
% - Better exploration (stochastic policy)
% - Simpler implementation
% - Better for noisy environments
%
% Requirements:
% - MATLAB Reinforcement Learning Toolbox
% - ANSYS Maxwell simulation data (or use generateExampleData.m)
%
% Usage:
%   1. Export magnetic field data from ANSYS Maxwell
%   2. Update the dataFilePath variable below
%   3. Run this script to train the PPO agent

clear all; close all; clc;

%% Configuration
fprintf('=== Polywell PPO Controller Training ===\n\n');

% Path to ANSYS Maxwell data file
dataFilePath = 'maxwell_polywell_data.csv';

% Training parameters
numIterations = 1000;        % Number of training iterations
experienceHorizon = 2000;    % Steps to collect per iteration
miniBatchSize = 128;         % Mini-batch size for updates
numEpochsPerUpdate = 10;     % Epochs per PPO update

%% Import ANSYS Maxwell Data
fprintf('Step 1: Importing ANSYS Maxwell data...\n');

if ~exist(dataFilePath, 'file')
    warning('ANSYS Maxwell data file not found. Using dummy data for demonstration.');
    [magneticFieldData, coilPositions] = createDummyData();
else
    [magneticFieldData, coilPositions] = importMaxwellData(dataFilePath);
end

fprintf('Data imported successfully.\n\n');

%% Create RL Environment
fprintf('Step 2: Creating RL environment...\n');

env = PolywellRLEnvironment(magneticFieldData, coilPositions);

% Validate environment
validateEnvironment(env);

fprintf('Environment created and validated.\n\n');

%% Create PPO Agent
fprintf('Step 3: Creating PPO agent...\n');

% Get observation and action specifications
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Number of observations and actions
numObs = obsInfo.Dimension(1);
numAct = actInfo.Dimension(1);

%% Create Actor Network (Stochastic Policy)
% PPO uses a stochastic policy that outputs mean and standard deviation
% for a Gaussian distribution over actions

% Shared feature extraction layers
commonPath = [
    featureInputLayer(numObs, 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(256, 'Name', 'shared_fc1')
    reluLayer('Name', 'shared_relu1')
    fullyConnectedLayer(128, 'Name', 'shared_fc2')
    reluLayer('Name', 'shared_relu2')
];

% Policy path (outputs mean of action distribution)
meanPath = [
    fullyConnectedLayer(64, 'Name', 'mean_fc1')
    reluLayer('Name', 'mean_relu1')
    fullyConnectedLayer(numAct, 'Name', 'mean_fc2')
    tanhLayer('Name', 'mean_tanh')
    scalingLayer('Name', 'mean_scale', 'Scale', actInfo.UpperLimit)
];

% Standard deviation path (outputs log std for numerical stability)
stdPath = [
    fullyConnectedLayer(64, 'Name', 'std_fc1')
    reluLayer('Name', 'std_relu1')
    fullyConnectedLayer(numAct, 'Name', 'std_fc2')
];

% Combine paths
actorNetwork = layerGraph(commonPath);
actorNetwork = addLayers(actorNetwork, meanPath);
actorNetwork = addLayers(actorNetwork, stdPath);
actorNetwork = connectLayers(actorNetwork, 'shared_relu2', 'mean_fc1');
actorNetwork = connectLayers(actorNetwork, 'shared_relu2', 'std_fc1');

% Create actor with Gaussian distribution
actor = rlStochasticActorRepresentation(actorNetwork, obsInfo, actInfo, ...
    'ObservationInputNames', 'state', ...
    'ActionMeanOutputNames', 'mean_scale', ...
    'ActionStandardDeviationOutputNames', 'std_fc2');

%% Create Critic Network (Value Function)
% Critic estimates V(s) - the expected return from state s

criticNetwork = [
    featureInputLayer(numObs, 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(256, 'Name', 'critic_fc1')
    reluLayer('Name', 'critic_relu1')
    fullyConnectedLayer(128, 'Name', 'critic_fc2')
    reluLayer('Name', 'critic_relu2')
    fullyConnectedLayer(64, 'Name', 'critic_fc3')
    reluLayer('Name', 'critic_relu3')
    fullyConnectedLayer(1, 'Name', 'critic_output')
];

critic = rlValueRepresentation(criticNetwork, obsInfo, ...
    'ObservationInputNames', 'state');

%% Create PPO Agent Options
agentOptions = rlPPOAgentOptions(...
    'SampleTime', env.TimeStep, ...
    'ExperienceHorizon', experienceHorizon, ...
    'MiniBatchSize', miniBatchSize, ...
    'NumEpoch', numEpochsPerUpdate, ...
    'DiscountFactor', 0.99, ...
    'GAEFactor', 0.95, ...              % Lambda for GAE
    'ClipFactor', 0.2, ...              % Epsilon for clipping
    'EntropyLossWeight', 0.01);         % Entropy bonus weight

% Learning rates
agentOptions.ActorOptimizerOptions.LearnRate = 3e-4;
agentOptions.ActorOptimizerOptions.GradientThreshold = 1;

agentOptions.CriticOptimizerOptions.LearnRate = 1e-3;
agentOptions.CriticOptimizerOptions.GradientThreshold = 1;

% Create PPO agent
agent = rlPPOAgent(actor, critic, agentOptions);

fprintf('PPO agent created.\n\n');

%% Configure Training Options
fprintf('Step 4: Configuring training...\n');

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', numIterations, ...
    'MaxStepsPerEpisode', env.MaxSteps, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 15, ...
    'ScoreAveragingWindowLength', 20, ...
    'SaveAgentCriteria', 'EpisodeReward', ...
    'SaveAgentValue', 10);

fprintf('Training configured.\n\n');

%% Train the Agent
fprintf('Step 5: Training PPO agent...\n');
fprintf('PPO typically needs more episodes than DDPG but is more stable.\n\n');

% Train
trainingStats = train(agent, env, trainOpts);

fprintf('\nTraining completed!\n\n');

%% Save Trained Agent
fprintf('Step 6: Saving trained agent...\n');

save('trainedPolywellAgent_PPO.mat', 'agent', 'trainingStats', ...
     'magneticFieldData', 'coilPositions');

fprintf('PPO agent saved to: trainedPolywellAgent_PPO.mat\n\n');

%% Test Trained Agent
fprintf('Step 7: Testing trained PPO agent...\n');

% Run simulation with trained agent
simOptions = rlSimulationOptions('MaxSteps', env.MaxSteps);
experience = sim(env, agent, simOptions);

% Display results
fprintf('Test episode reward: %.2f\n', sum(experience.Reward.Data));

% Extract observations over time
obs_data = experience.Observation.obs1.Data;
fprintf('Average plasma beta: %.3f\n', mean(obs_data(7, :)));
fprintf('Average confinement time: %.4f s\n', mean(obs_data(8, :)) / 100);
fprintf('Average field uniformity: %.3f\n', mean(obs_data(9, :)));

% Calculate average coil currents
avg_currents = mean(obs_data(1:6, :) * env.MaxCurrent, 2);
fprintf('Average coil currents:\n');
for i = 1:6
    fprintf('  Coil %d: %.1f A\n', i, avg_currents(i));
end

fprintf('\n=== Training Complete! ===\n');
fprintf('Use visualizePolywellControl_PPO.m to visualize the trained agent.\n');

%% Helper Function: Create Dummy Data
function [magneticFieldData, coilPositions] = createDummyData()
    % Create dummy magnetic field data for demonstration

    % Create grid of points
    [X, Y, Z] = meshgrid(-0.5:0.05:0.5, -0.5:0.05:0.5, -0.5:0.05:0.5);
    positions = [X(:), Y(:), Z(:)];

    % Calculate dummy magnetic field
    r = sqrt(sum(positions.^2, 2)) + 0.01;
    Bmag = 0.1 ./ r.^2;

    % Random field directions
    Bfield = randn(size(positions)) .* Bmag;

    magneticFieldData.positions = positions;
    magneticFieldData.Bfield = Bfield;
    magneticFieldData.Bmag = Bmag;
    magneticFieldData.coilCurrents = ones(6, 1) * 1000;
    magneticFieldData.interpolant = scatteredInterpolant(positions, Bmag, 'natural', 'none');

    % Define coil positions
    L = 0.5;
    coilPositions = [
        L,  0,  0;
       -L,  0,  0;
        0,  L,  0;
        0, -L,  0;
        0,  0,  L;
        0,  0, -L
    ];
end

%% Training Tips for PPO
% 1. PPO is more stable than DDPG - easier to tune hyperparameters
% 2. If training is slow, increase experienceHorizon
% 3. If policy changes too fast, decrease ClipFactor
% 4. If exploration is insufficient, increase EntropyLossWeight
% 5. PPO naturally explores due to stochastic policy (no need for external noise)
% 6. Monitor the "Clip Fraction" - if too high (>0.5), policy is changing too fast
