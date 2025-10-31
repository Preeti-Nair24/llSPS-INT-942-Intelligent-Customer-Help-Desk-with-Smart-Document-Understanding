%% Visualize Polywell RL Control
% This script visualizes the trained RL agent controlling the coil
% currents in a polywell fusion reactor in real-time.
%
% Features:
% - Real-time 3D visualization of coil currents
% - Plasma confinement metrics
% - Magnetic field visualization
% - Performance plots
%
% Requirements:
% - Trained RL agent (trainedPolywellAgent.mat)
%
% Usage:
%   Run this script after training the agent with trainPolywellRLAgent.m

clear all; close all; clc;

%% Load Trained Agent
fprintf('=== Polywell RL Control Visualization ===\n\n');
fprintf('Loading trained agent...\n');

if ~exist('trainedPolywellAgent.mat', 'file')
    error('Trained agent not found. Please run trainPolywellRLAgent.m first.');
end

load('trainedPolywellAgent.mat', 'agent', 'magneticFieldData', 'coilPositions');
fprintf('Agent loaded successfully.\n\n');

%% Create Environment
env = PolywellRLEnvironment(magneticFieldData, coilPositions);

%% Setup Visualization
fprintf('Setting up visualization...\n');

% Create figure with subplots
fig = figure('Name', 'Polywell RL Control Visualization', ...
             'Position', [100, 100, 1400, 800], ...
             'Color', 'w');

% Subplot 1: 3D Coil Configuration and Magnetic Field
ax1 = subplot(2, 3, [1, 4]);
title('Polywell Coil Configuration', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
grid on; axis equal; hold on;
view(45, 30);
xlim([-0.8, 0.8]); ylim([-0.8, 0.8]); zlim([-0.8, 0.8]);

% Draw coil representations
coilHandles = gobjects(6, 1);
for i = 1:6
    coilHandles(i) = plot3(coilPositions(i, 1), coilPositions(i, 2), coilPositions(i, 3), ...
                           'o', 'MarkerSize', 30, 'LineWidth', 3, 'MarkerFaceColor', 'b');
end

% Add coil labels
for i = 1:6
    text(coilPositions(i, 1)*1.15, coilPositions(i, 2)*1.15, coilPositions(i, 3)*1.15, ...
         sprintf('Coil %d', i), 'FontSize', 10, 'FontWeight', 'bold');
end

% Add plasma region (sphere at center)
[xs, ys, zs] = sphere(30);
plasmaHandle = surf(xs*0.2, ys*0.2, zs*0.2, 'FaceColor', 'r', 'FaceAlpha', 0.3, ...
                    'EdgeColor', 'none');

% Subplot 2: Coil Currents
ax2 = subplot(2, 3, 2);
currentBars = bar(1:6, zeros(6, 1), 'FaceColor', [0.2, 0.6, 0.8]);
title('Coil Currents', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Coil Number');
ylabel('Current (A)');
ylim([0, env.MaxCurrent]);
grid on;

% Subplot 3: Plasma Parameters
ax3 = subplot(2, 3, 3);
hold on;
betaLine = animatedline('Color', 'b', 'LineWidth', 2, 'DisplayName', 'Plasma Beta');
confinementLine = animatedline('Color', 'r', 'LineWidth', 2, 'DisplayName', 'Confinement (×100)');
uniformityLine = animatedline('Color', 'g', 'LineWidth', 2, 'DisplayName', 'Field Uniformity');
title('Plasma Parameters', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Time Step');
ylabel('Value');
legend('Location', 'best');
ylim([0, 1]);
grid on;

% Subplot 4: Reward
ax4 = subplot(2, 3, 5);
rewardLine = animatedline('Color', 'm', 'LineWidth', 2);
title('Cumulative Reward', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Time Step');
ylabel('Reward');
grid on;

% Subplot 5: Power Consumption
ax5 = subplot(2, 3, 6);
powerLine = animatedline('Color', [0.8, 0.4, 0], 'LineWidth', 2);
title('Total Power Consumption', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Time Step');
ylabel('Power (MW)');
grid on;

% Add text display for current metrics
metricsText = annotation('textbox', [0.72, 0.15, 0.25, 0.15], ...
    'String', 'Initializing...', 'FontSize', 10, ...
    'BackgroundColor', 'w', 'EdgeColor', 'k', 'LineWidth', 1);

fprintf('Visualization setup complete.\n\n');

%% Run Simulation with Visualization
fprintf('Running simulation...\n');
fprintf('Press Ctrl+C to stop.\n\n');

% Reset environment
obs = reset(env);
cumulativeReward = 0;
step = 0;

% Simulation loop
while true
    step = step + 1;

    % Get action from agent
    action = getAction(agent, obs);

    % Take step in environment
    [nextObs, reward, isDone, ~] = step(env, action);
    cumulativeReward = cumulativeReward + reward;

    % Update coil current visualization
    currentBars.YData = env.CoilCurrents;

    % Update coil colors based on current magnitude
    for i = 1:6
        intensity = env.CoilCurrents(i) / env.MaxCurrent;
        coilHandles(i).MarkerFaceColor = [intensity, 0, 1-intensity];
    end

    % Update plasma sphere color/size based on beta
    plasmaScale = 0.2 * (1 + env.PlasmaBeta);
    set(plasmaHandle, 'XData', xs*plasmaScale, 'YData', ys*plasmaScale, 'ZData', zs*plasmaScale);
    plasmaAlpha = 0.2 + 0.5 * env.PlasmaBeta;
    set(plasmaHandle, 'FaceAlpha', plasmaAlpha);

    % Update plasma parameters plot
    addpoints(betaLine, step, env.PlasmaBeta);
    addpoints(confinementLine, step, env.ConfinementTime * 100);
    addpoints(uniformityLine, step, env.FieldUniformity);

    % Update reward plot
    addpoints(rewardLine, step, cumulativeReward);

    % Update power consumption
    totalPower = sum(env.CoilCurrents.^2) * 1e-6;  % Convert to MW (assuming resistance)
    addpoints(powerLine, step, totalPower);

    % Update metrics text
    metricsStr = sprintf(['Current Step: %d\n' ...
                         'Plasma Beta: %.3f\n' ...
                         'Confinement Time: %.4f s\n' ...
                         'Field Uniformity: %.3f\n' ...
                         'Cumulative Reward: %.2f\n' ...
                         'Power: %.2f MW'], ...
                         step, env.PlasmaBeta, env.ConfinementTime, ...
                         env.FieldUniformity, cumulativeReward, totalPower);
    set(metricsText, 'String', metricsStr);

    % Update observation
    obs = nextObs;

    % Refresh display
    drawnow;

    % Pause for animation
    pause(0.05);

    % Check if episode is done
    if isDone
        fprintf('\nEpisode completed at step %d\n', step);
        fprintf('Final cumulative reward: %.2f\n', cumulativeReward);
        fprintf('Final plasma beta: %.3f\n', env.PlasmaBeta);
        fprintf('Final confinement time: %.4f s\n\n', env.ConfinementTime);

        % Ask to continue
        choice = questdlg('Episode finished. Run another episode?', ...
                         'Continue', 'Yes', 'No', 'Yes');

        if strcmp(choice, 'Yes')
            % Reset for new episode
            obs = reset(env);
            cumulativeReward = 0;
            step = 0;

            % Clear animated lines
            clearpoints(betaLine);
            clearpoints(confinementLine);
            clearpoints(uniformityLine);
            clearpoints(rewardLine);
            clearpoints(powerLine);

            fprintf('Starting new episode...\n\n');
        else
            break;
        end
    end
end

fprintf('Visualization ended.\n');

%% Generate Summary Report
fprintf('\n=== Session Summary ===\n');
fprintf('Total steps simulated: %d\n', step);
fprintf('Final reward: %.2f\n', cumulativeReward);
fprintf('Average coil current: %.2f A\n', mean(env.CoilCurrents));
fprintf('Max coil current: %.2f A\n', max(env.CoilCurrents));
fprintf('Min coil current: %.2f A\n', min(env.CoilCurrents));
fprintf('\nVisualization complete!\n');
