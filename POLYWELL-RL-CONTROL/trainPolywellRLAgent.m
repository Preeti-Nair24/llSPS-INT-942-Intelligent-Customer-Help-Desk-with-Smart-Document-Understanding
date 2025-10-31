%% Train Polywell RL Agent
% This script trains a reinforcement learning agent to control
% the coil currents in a polywell fusion reactor for optimal
% plasma confinement.
%
% Requirements:
% - MATLAB Reinforcement Learning Toolbox
% - ANSYS Maxwell simulation data
%
% Usage:
%   1. Export magnetic field data from ANSYS Maxwell
%   2. Update the dataFilePath variable below
%   3. Run this script to train the RL agent

clear all; close all; clc;

%% Configuration
fprintf('=== Polywell RL Controller Training ===\n\n');

% Path to ANSYS Maxwell data file
dataFilePath = 'maxwell_polywell_data.csv';  % Update this path!

% Training parameters
numEpisodes = 500;
maxStepsPerEpisode = 200;
saveAgentEvery = 50;  % Save checkpoint every N episodes

%% Import ANSYS Maxwell Data
fprintf('Step 1: Importing ANSYS Maxwell data...\n');

% Check if data file exists
if ~exist(dataFilePath, 'file')
    warning('ANSYS Maxwell data file not found. Using dummy data for demonstration.');
    % Create dummy data structure
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

%% Create RL Agent (DDPG - Deep Deterministic Policy Gradient)
fprintf('Step 3: Creating DDPG agent...\n');

% Get observation and action specifications
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Create critic network
statePath = [
    featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
];

actionPath = [
    featureInputLayer(actInfo.Dimension(1), 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(64, 'Name', 'fc3')
];

commonPath = [
    additionLayer(2, 'Name', 'add')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(32, 'Name', 'fc4')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(1, 'Name', 'output')
];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
criticNetwork = connectLayers(criticNetwork, 'relu2', 'add/in1');
criticNetwork = connectLayers(criticNetwork, 'fc3', 'add/in2');

% Create actor network
actorNetwork = [
    featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'fc3')
    tanhLayer('Name', 'tanh')
    scalingLayer('Name', 'scaling', 'Scale', actInfo.UpperLimit)
];

% Create critic and actor
critic = rlQValueFunction(criticNetwork, obsInfo, actInfo, ...
    'ObservationInputNames', 'state', 'ActionInputNames', 'action');

actor = rlContinuousDeterministicActor(actorNetwork, obsInfo, actInfo);

% Create DDPG agent
agentOptions = rlDDPGAgentOptions(...
    'SampleTime', env.TimeStep, ...
    'TargetSmoothFactor', 1e-3, ...
    'ExperienceBufferLength', 1e6, ...
    'MiniBatchSize', 128, ...
    'NumStepsToLookAhead', 1);

agentOptions.NoiseOptions.Variance = 0.1;
agentOptions.NoiseOptions.VarianceDecayRate = 1e-5;

agent = rlDDPGAgent(actor, critic, agentOptions);

fprintf('DDPG agent created.\n\n');

%% Configure Training Options
fprintf('Step 4: Configuring training...\n');

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', numEpisodes, ...
    'MaxStepsPerEpisode', maxStepsPerEpisode, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 15, ...
    'ScoreAveragingWindowLength', 20, ...
    'SaveAgentCriteria', 'EpisodeReward', ...
    'SaveAgentValue', 10);

fprintf('Training configured.\n\n');

%% Train the Agent
fprintf('Step 5: Training agent...\n');
fprintf('Training may take several hours depending on your hardware.\n\n');

% Train
trainingStats = train(agent, env, trainOpts);

fprintf('\nTraining completed!\n\n');

%% Save Trained Agent
fprintf('Step 6: Saving trained agent...\n');

save('trainedPolywellAgent.mat', 'agent', 'trainingStats', 'magneticFieldData', 'coilPositions');

fprintf('Agent saved to: trainedPolywellAgent.mat\n\n');

%% Test Trained Agent
fprintf('Step 7: Testing trained agent...\n');

% Run simulation with trained agent
simOptions = rlSimulationOptions('MaxSteps', maxStepsPerEpisode);
experience = sim(env, agent, simOptions);

% Display results
fprintf('Test episode reward: %.2f\n', sum(experience.Reward.Data));
fprintf('Average plasma beta: %.3f\n', mean(experience.Observation.obs1.Data(7, :)));
fprintf('Average confinement time: %.4f s\n', mean(experience.Observation.obs1.Data(8, :)) / 100);

fprintf('\n=== Training Complete! ===\n');
fprintf('Use visualizePolywellControl.m to visualize the trained agent.\n');

%% Helper Function: Create Dummy Data
function [magneticFieldData, coilPositions] = createDummyData()
    % Create dummy magnetic field data for demonstration

    % Create grid of points
    [X, Y, Z] = meshgrid(-0.5:0.05:0.5, -0.5:0.05:0.5, -0.5:0.05:0.5);
    positions = [X(:), Y(:), Z(:)];

    % Calculate dummy magnetic field (simple dipole-like field)
    r = sqrt(sum(positions.^2, 2)) + 0.01;
    Bmag = 0.1 ./ r.^2;  % Inverse square law

    % Random field directions (for demonstration)
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
