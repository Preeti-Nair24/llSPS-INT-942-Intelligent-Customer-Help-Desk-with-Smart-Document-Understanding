%% ONE-CLICK DEMO: RL Controlling Polywell Magnetic Field
%
% This is the SIMPLEST way to see RL in action!
% Just press "Run" in MATLAB and watch!
%
% What happens:
% 1. Creates magnetic field data (30 seconds)
% 2. Trains RL agent (3-5 minutes)
% 3. Shows live control visualization
%
% Total time: ~5-8 minutes
%
% ============================================
%      JUST PRESS RUN AND WATCH! 🚀
% ============================================

clear all; close all; clc;

%% Welcome Message
fprintf('\n\n');
fprintf('    ╔═══════════════════════════════════════════════════════╗\n');
fprintf('    ║                                                       ║\n');
fprintf('    ║       POLYWELL RL CONTROL - ONE-CLICK DEMO          ║\n');
fprintf('    ║                                                       ║\n');
fprintf('    ║   Watch AI learn to control fusion reactor!          ║\n');
fprintf('    ║                                                       ║\n');
fprintf('    ╚═══════════════════════════════════════════════════════╝\n');
fprintf('\n');
fprintf('    This demo will:\n');
fprintf('    ✓ Generate magnetic field data\n');
fprintf('    ✓ Train RL agent (fast mode: ~3 minutes)\n');
fprintf('    ✓ Show live control visualization\n\n');
fprintf('    Sit back and watch the magic happen! ✨\n\n');

pause(2);

%% PHASE 1: Setup
fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('PHASE 1: SETUP\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('[1/3] Generating magnetic field data...\n');
tic;

% Quick data generation
[X, Y, Z] = meshgrid(-0.5:0.1:0.5, -0.5:0.1:0.5, -0.5:0.1:0.5);
positions = [X(:), Y(:), Z(:)];
r = sqrt(sum(positions.^2, 2)) + 0.01;
Bmag = 0.1 ./ r.^2;
Bfield = randn(size(positions)) .* Bmag;

magneticFieldData.positions = positions;
magneticFieldData.Bfield = Bfield;
magneticFieldData.Bmag = Bmag;
magneticFieldData.coilCurrents = ones(6, 1) * 1000;
magneticFieldData.interpolant = scatteredInterpolant(positions, Bmag, 'natural', 'none');

L = 0.5;
coilPositions = [L,0,0; -L,0,0; 0,L,0; 0,-L,0; 0,0,L; 0,0,-L];

fprintf('      ✓ Created %d field data points (%.1f seconds)\n\n', size(positions,1), toc);

fprintf('[2/3] Creating RL environment...\n');
tic;

env = PolywellRLEnvironment(magneticFieldData, coilPositions);
env.MaxSteps = 100;  % Shorter for faster demo

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

fprintf('      ✓ Environment ready (%.1f seconds)\n', toc);
fprintf('        • State space: %d dimensions\n', obsInfo.Dimension(1));
fprintf('        • Action space: %d dimensions\n', actInfo.Dimension(1));
fprintf('        • Max steps: %d per episode\n\n', env.MaxSteps);

fprintf('[3/3] Creating PPO agent...\n');
tic;

% Compact network for speed
actorNet = [
    featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(32, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'fc3')
    tanhLayer('Name', 'tanh')
    scalingLayer('Name', 'scale', 'Scale', actInfo.UpperLimit)
];

criticNet = [
    featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(32, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'value')
];

actor = rlStochasticActorRepresentation(actorNet, obsInfo, actInfo, 'ObservationInputNames', 'state');
critic = rlValueRepresentation(criticNet, obsInfo, 'ObservationInputNames', 'state');

agentOpts = rlPPOAgentOptions('SampleTime', 0.1, 'ExperienceHorizon', 400, ...
    'MiniBatchSize', 64, 'NumEpoch', 3, 'DiscountFactor', 0.99);
agentOpts.ActorOptimizerOptions.LearnRate = 5e-4;
agentOpts.CriticOptimizerOptions.LearnRate = 1e-3;

agent = rlPPOAgent(actor, critic, agentOpts);

fprintf('      ✓ PPO agent created (%.1f seconds)\n\n', toc);

fprintf('Setup complete! Ready to train.\n\n');
pause(1);

%% PHASE 2: Training
fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('PHASE 2: TRAINING (This will take ~3-5 minutes)\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('Training agent for 40 episodes...\n');
fprintf('Watch the "Training Progress" window that will pop up!\n');
fprintf('You''ll see the reward curve climbing as the agent learns.\n\n');

pause(2);

% Training options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 40, ...
    'MaxStepsPerEpisode', env.MaxSteps, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 8, ...
    'ScoreAveragingWindowLength', 10);

% Train!
fprintf('🎓 TRAINING STARTED - Watch the plot window!\n\n');
trainingInfo = train(agent, env, trainOpts);

fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('TRAINING COMPLETE! ✓\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

% Show results
fprintf('📊 Training Results:\n');
if ~isempty(trainingInfo.EpisodeReward)
    fprintf('   • Episodes completed: %d\n', length(trainingInfo.EpisodeReward));
    fprintf('   • Initial reward: %.2f\n', trainingInfo.EpisodeReward(1));
    fprintf('   • Final reward: %.2f\n', trainingInfo.EpisodeReward(end));
    fprintf('   • Best reward: %.2f\n', max(trainingInfo.EpisodeReward));
    fprintf('   • Improvement: %.1fx better!\n\n', trainingInfo.EpisodeReward(end)/trainingInfo.EpisodeReward(1));
end

% Save agent
save('demoAgent_trained.mat', 'agent', 'trainingInfo', 'magneticFieldData', 'coilPositions');
fprintf('💾 Trained agent saved: demoAgent_trained.mat\n\n');

pause(2);

%% PHASE 3: Live Control Demonstration
fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('PHASE 3: LIVE CONTROL DEMONSTRATION\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('Now watch the trained agent control the polywell in real-time!\n');
fprintf('A new window will show:\n');
fprintf('  • 3D coil configuration with colors showing current\n');
fprintf('  • Plasma sphere (size = confinement quality)\n');
fprintf('  • Live performance metrics\n\n');

pause(2);

% Create visualization
fig = figure('Name', 'Live Magnetic Field Control', ...
             'Position', [50, 50, 1200, 700], 'Color', 'w');

% 3D View
subplot(2,2,[1,3]);
hold on; grid on; axis equal;
title('Polywell Magnetic Coil Control (Live)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
view(45, 30);
xlim([-0.7, 0.7]); ylim([-0.7, 0.7]); zlim([-0.7, 0.7]);

% Draw coils
coilHandles = gobjects(6, 1);
coilRadius = 0.15;
for i = 1:6
    normal = -coilPositions(i, :) / norm(coilPositions(i, :));
    theta = linspace(0, 2*pi, 50);
    if abs(normal(1)) > 0.9
        cy = coilRadius * cos(theta); cz = coilRadius * sin(theta);
        cx = ones(size(cy)) * coilPositions(i, 1);
    elseif abs(normal(2)) > 0.9
        cx = coilRadius * cos(theta); cz = coilRadius * sin(theta);
        cy = ones(size(cx)) * coilPositions(i, 2);
    else
        cx = coilRadius * cos(theta); cy = coilRadius * sin(theta);
        cz = ones(size(cx)) * coilPositions(i, 3);
    end
    coilHandles(i) = plot3(cx, cy, cz, 'LineWidth', 8);
    text(coilPositions(i,1)*1.2, coilPositions(i,2)*1.2, coilPositions(i,3)*1.2, ...
        sprintf('C%d', i), 'FontSize', 11, 'FontWeight', 'bold');
end

% Plasma sphere
[xs, ys, zs] = sphere(40);
plasmaHandle = surf(xs*0.2, ys*0.2, zs*0.2, 'FaceColor', 'r', 'FaceAlpha', 0.4, 'EdgeColor', 'none');
camlight; lighting gouraud;

% Coil currents
subplot(2,2,2);
currBars = bar(1:6, zeros(6,1), 'FaceColor', [0.2,0.6,0.8]);
title('Coil Currents', 'FontWeight', 'bold');
xlabel('Coil #'); ylabel('Current (A)');
ylim([0, 5000]); grid on;

% Performance metrics
subplot(2,2,4);
betaLine = animatedline('Color', 'b', 'LineWidth', 2.5, 'DisplayName', 'Beta');
confLine = animatedline('Color', 'r', 'LineWidth', 2.5, 'DisplayName', 'Confinement×100');
uniformLine = animatedline('Color', 'g', 'LineWidth', 2.5, 'DisplayName', 'Uniformity');
hold on;
yline(0.4, '--k', 'Target', 'LineWidth', 1.5);
title('Plasma Parameters', 'FontWeight', 'bold');
xlabel('Step'); ylabel('Value');
legend('Location', 'southeast');
ylim([0, 1]); grid on;

fprintf('🎬 Starting live demonstration...\n\n');

% Run demonstration
obs = reset(env);
totalSteps = 150;

fprintf('Step | Beta  | Conf(ms) | Uniform | Currents\n');
fprintf('-----|-------|----------|---------|---------------------------\n');

for step = 1:totalSteps
    % Get action and execute
    action = getAction(agent, obs);
    [obs, ~, isDone] = env.step(action);

    % Update visualizations
    if mod(step, 3) == 0
        % Update coil colors and bars
        currBars.YData = env.CoilCurrents;
        for i = 1:6
            intensity = env.CoilCurrents(i) / env.MaxCurrent;
            if intensity < 0.5
                coilHandles(i).Color = [0, intensity*2, 1];
            else
                coilHandles(i).Color = [(intensity-0.5)*2, 1, 0];
            end
        end

        % Update plasma
        scale = 0.15 * (1 + 1.5*env.PlasmaBeta);
        set(plasmaHandle, 'XData', xs*scale, 'YData', ys*scale, 'ZData', zs*scale);
        set(plasmaHandle, 'FaceAlpha', 0.2 + 0.6*env.PlasmaBeta);

        % Update plots
        addpoints(betaLine, step, env.PlasmaBeta);
        addpoints(confLine, step, env.ConfinementTime*100);
        addpoints(uniformLine, step, env.FieldUniformity);

        drawnow limitrate;
    end

    % Print every 15 steps
    if mod(step, 15) == 0
        fprintf('%4d | %.3f | %8.2f | %7.3f | [', step, env.PlasmaBeta, ...
            env.ConfinementTime*1000, env.FieldUniformity);
        fprintf('%.0f ', env.CoilCurrents(1:3));
        fprintf('...]\n');
    end

    pause(0.04);

    if isDone, break; end
end

fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('DEMONSTRATION COMPLETE! ✓\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

%% Final Summary
fprintf('🎉 DEMO FINISHED SUCCESSFULLY!\n\n');
fprintf('What you just saw:\n');
fprintf('  1. Agent trained from scratch using PPO algorithm\n');
fprintf('  2. Learned to control 6 magnetic coils\n');
fprintf('  3. Optimized plasma beta, confinement, and uniformity\n');
fprintf('  4. All through trial-and-error learning!\n\n');

fprintf('📈 Final Performance:\n');
fprintf('   • Plasma Beta: %.3f (target: 0.40)\n', env.PlasmaBeta);
fprintf('   • Confinement: %.4f seconds\n', env.ConfinementTime);
fprintf('   • Uniformity: %.3f\n\n', env.FieldUniformity);

fprintf('📁 Files created:\n');
fprintf('   • demoAgent_trained.mat (trained agent)\n\n');

fprintf('🚀 Next Steps:\n');
fprintf('   • For longer training: trainPolywellRLAgent_PPO.m\n');
fprintf('   • For detailed viz: liveSimulationMagneticControl.m\n');
fprintf('   • Study the code to understand how it works!\n\n');

fprintf('Keep the figure window open to examine the final state.\n');
fprintf('Thank you for watching! 🎓\n\n');
