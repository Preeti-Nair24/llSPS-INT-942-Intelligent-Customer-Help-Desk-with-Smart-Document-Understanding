%% Live Simulation: RL Agent Controlling Magnetic Field via Coil Currents
% This script provides real-time visualization of the RL agent controlling
% the polywell magnetic field by dynamically adjusting coil currents.
%
% Visualizations:
% - 3D magnetic field lines (real-time updates)
% - Coil current indicators with color-coded intensity
% - Magnetic field strength heatmap
% - Plasma confinement zone
% - Live performance metrics
%
% Usage:
%   Run this script after training an agent (PPO or DDPG)
%   Or run with untrained agent to see learning in real-time

clear all; close all; clc;

%% Configuration
fprintf('=== Live Magnetic Field Control Simulation ===\n\n');

% Choose mode
USE_TRAINED_AGENT = true;  % Set to false to see random/learning behavior

%% Load Agent (if available)
if USE_TRAINED_AGENT
    fprintf('Loading trained agent...\n');
    if exist('trainedPolywellAgent_PPO.mat', 'file')
        load('trainedPolywellAgent_PPO.mat', 'agent', 'magneticFieldData', 'coilPositions');
        fprintf('PPO agent loaded.\n');
    elseif exist('trainedPolywellAgent.mat', 'file')
        load('trainedPolywellAgent.mat', 'agent', 'magneticFieldData', 'coilPositions');
        fprintf('DDPG agent loaded.\n');
    else
        fprintf('No trained agent found. Generating dummy data...\n');
        [magneticFieldData, coilPositions] = createDummyData();
        agent = [];
        USE_TRAINED_AGENT = false;
    end
else
    fprintf('Running without trained agent (random actions)...\n');
    [magneticFieldData, coilPositions] = createDummyData();
    agent = [];
end

%% Create Environment
fprintf('Creating environment...\n');
env = PolywellRLEnvironment(magneticFieldData, coilPositions);

%% Setup Enhanced Visualization
fprintf('Setting up live visualization...\n\n');

% Create main figure
fig = figure('Name', 'Live Magnetic Field Control by RL Agent', ...
             'Position', [50, 50, 1600, 900], ...
             'Color', 'w', ...
             'KeyPressFcn', @(~,~) fprintf('Press Ctrl+C to stop\n'));

%% Subplot 1: 3D Magnetic Field with Field Lines (MAIN VIEW)
ax1 = subplot(2, 3, [1, 4]);
title('3D Magnetic Field Control (Live)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
grid on; axis equal; hold on;
view(45, 30);
xlim([-0.7, 0.7]); ylim([-0.7, 0.7]); zlim([-0.7, 0.7]);
set(ax1, 'Color', [0.95 0.95 0.95]);

% Draw coil representations (large 3D cylinders)
coilHandles = gobjects(6, 1);
coilTextHandles = gobjects(6, 1);
coilRadius = 0.15;

for i = 1:6
    % Determine coil orientation
    normal = -coilPositions(i, :) / norm(coilPositions(i, :));

    % Draw coil as thick circle
    theta = linspace(0, 2*pi, 50);
    if abs(normal(1)) > 0.9  % X-axis coils
        cy = coilRadius * cos(theta);
        cz = coilRadius * sin(theta);
        cx = ones(size(cy)) * coilPositions(i, 1);
    elseif abs(normal(2)) > 0.9  % Y-axis coils
        cx = coilRadius * cos(theta);
        cz = coilRadius * sin(theta);
        cy = ones(size(cx)) * coilPositions(i, 2);
    else  % Z-axis coils
        cx = coilRadius * cos(theta);
        cy = coilRadius * sin(theta);
        cz = ones(size(cx)) * coilPositions(i, 3);
    end

    coilHandles(i) = plot3(cx, cy, cz, 'LineWidth', 8, 'Color', 'b');

    % Label
    labelPos = coilPositions(i, :) * 1.3;
    coilTextHandles(i) = text(labelPos(1), labelPos(2), labelPos(3), ...
        sprintf('Coil %d\n%.0fA', i, env.CoilCurrents(i)), ...
        'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end

% Plasma sphere (will update size/color based on beta)
[xs, ys, zs] = sphere(50);
plasmaHandle = surf(xs*0.2, ys*0.2, zs*0.2, ...
    'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');

% Magnetic field lines (streamlines)
fieldLineHandles = cell(12, 1);
numFieldLines = 12;

% Starting points for field lines (arranged around center)
phi = linspace(0, 2*pi, numFieldLines);
startPoints = 0.15 * [cos(phi)', sin(phi)', zeros(numFieldLines, 1)];

for i = 1:numFieldLines
    fieldLineHandles{i} = plot3(0, 0, 0, 'Color', [0.3, 0.3, 0.8], 'LineWidth', 1.5);
end

% Add lighting for better 3D effect
camlight('headlight');
lighting gouraud;
material shiny;

%% Subplot 2: Coil Current Bar Chart
ax2 = subplot(2, 3, 2);
currentBars = bar(1:6, zeros(6, 1), 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
title('Coil Currents (Live Control)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Coil Number');
ylabel('Current (A)');
ylim([0, env.MaxCurrent]);
grid on;
% Add target line
hold on;
yline(2500, '--r', 'Nominal', 'LineWidth', 2);

%% Subplot 3: Magnetic Field Strength Heatmap (Z=0 plane)
ax3 = subplot(2, 3, 3);
% Create grid for field strength
[Xgrid, Ygrid] = meshgrid(-0.5:0.05:0.5, -0.5:0.05:0.5);
Zgrid = zeros(size(Xgrid));
fieldStrengthData = zeros(size(Xgrid));
fieldHeatmap = imagesc([-0.5, 0.5], [-0.5, 0.5], fieldStrengthData);
colorbar;
colormap(ax3, 'hot');
title('Magnetic Field Strength (Z=0)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('X (m)'); ylabel('Y (m)');
axis equal tight;
caxis([0, 0.5]);  % Will be updated

%% Subplot 4: Plasma Parameters Over Time
ax4 = subplot(2, 3, 5);
hold on;
betaLine = animatedline('Color', 'b', 'LineWidth', 2.5, 'DisplayName', 'Plasma Beta');
confinementLine = animatedline('Color', 'r', 'LineWidth', 2.5, 'DisplayName', 'Confinement (×100)');
uniformityLine = animatedline('Color', 'g', 'LineWidth', 2.5, 'DisplayName', 'Field Uniformity');
yline(0.4, '--b', 'β Target', 'LineWidth', 1.5);
title('Plasma Performance Metrics', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Time Step');
ylabel('Value');
legend('Location', 'best');
ylim([0, 1]);
grid on;

%% Subplot 5: Control Actions (Current Changes)
ax5 = subplot(2, 3, 6);
actionBars = bar(1:6, zeros(6, 1), 'FaceColor', [0.8, 0.4, 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5);
title('RL Agent Actions (ΔI)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Coil Number');
ylabel('Current Change (A)');
ylim([-env.MaxCurrentChange, env.MaxCurrentChange]);
grid on;
hold on;
yline(0, 'k', 'LineWidth', 1.5);

%% Status Display
statusText = annotation('textbox', [0.02, 0.88, 0.15, 0.10], ...
    'String', 'Initializing...', 'FontSize', 11, ...
    'BackgroundColor', [1, 1, 0.8], 'EdgeColor', 'k', 'LineWidth', 2, ...
    'FontWeight', 'bold');

metricsText = annotation('textbox', [0.68, 0.02, 0.30, 0.28], ...
    'String', 'Starting simulation...', 'FontSize', 10, ...
    'BackgroundColor', 'w', 'EdgeColor', 'k', 'LineWidth', 1);

fprintf('Visualization ready. Starting simulation...\n\n');

%% Main Simulation Loop
obs = reset(env);
cumulativeReward = 0;
step = 0;
episode = 1;

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  LIVE SIMULATION RUNNING - Press Ctrl+C to stop           ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

while true
    step = step + 1;

    % Get action from agent or random
    if ~isempty(agent)
        action = getAction(agent, obs);
    else
        % Random exploration (if no agent)
        action = (rand(6, 1) - 0.5) * 2 * env.MaxCurrentChange;
    end

    % Take step in environment
    [nextObs, reward, isDone, ~] = step(env, action);
    cumulativeReward = cumulativeReward + reward;

    %% UPDATE VISUALIZATIONS

    % 1. Update coil currents and colors
    currentBars.YData = env.CoilCurrents;
    for i = 1:6
        intensity = env.CoilCurrents(i) / env.MaxCurrent;
        % Color: Blue (low) -> Cyan -> Green -> Yellow -> Red (high)
        if intensity < 0.5
            color = [0, 0.5*intensity*2, 1-intensity];
        else
            color = [2*(intensity-0.5), 1-0.5*(intensity-0.5), 0];
        end
        coilHandles(i).Color = color;
        coilTextHandles(i).String = sprintf('Coil %d\n%.0fA', i, env.CoilCurrents(i));
    end

    % 2. Update plasma sphere (size and transparency based on beta)
    plasmaScale = 0.15 * (1 + 1.5*env.PlasmaBeta);
    set(plasmaHandle, 'XData', xs*plasmaScale, 'YData', ys*plasmaScale, 'ZData', zs*plasmaScale);
    plasmaAlpha = 0.2 + 0.6 * env.PlasmaBeta;
    plasmaColor = [1, 0.3, 0.3] * (0.5 + 0.5*env.PlasmaBeta);  % Brighter = better beta
    set(plasmaHandle, 'FaceAlpha', plasmaAlpha, 'FaceColor', plasmaColor);

    % 3. Calculate and visualize magnetic field lines
    if mod(step, 5) == 0  % Update field lines every 5 steps (expensive)
        % Calculate field strength based on current coil currents
        for i = 1:numFieldLines
            % Trace field line from starting point
            currentPoint = startPoints(i, :);
            linePoints = currentPoint;

            for j = 1:30  % 30 steps along field line
                % Calculate magnetic field at current point
                Bfield = calculateMagneticField(currentPoint, coilPositions, env.CoilCurrents);

                if norm(Bfield) < 1e-6
                    break;
                end

                % Move along field direction
                stepSize = 0.03;
                currentPoint = currentPoint + stepSize * Bfield / norm(Bfield);

                % Check if out of bounds
                if norm(currentPoint) > 0.6
                    break;
                end

                linePoints = [linePoints; currentPoint];
            end

            % Update field line
            set(fieldLineHandles{i}, 'XData', linePoints(:,1), ...
                'YData', linePoints(:,2), 'ZData', linePoints(:,3));
        end
    end

    % 4. Update magnetic field strength heatmap
    if mod(step, 10) == 0  % Update every 10 steps
        for ix = 1:size(Xgrid, 1)
            for iy = 1:size(Xgrid, 2)
                point = [Xgrid(ix, iy), Ygrid(ix, iy), 0];
                Bfield = calculateMagneticField(point, coilPositions, env.CoilCurrents);
                fieldStrengthData(ix, iy) = norm(Bfield);
            end
        end
        set(fieldHeatmap, 'CData', fieldStrengthData);
        caxis(ax3, [0, max(fieldStrengthData(:))*1.1]);
    end

    % 5. Update action bars
    actionBars.YData = action;

    % 6. Update plasma parameters plot
    addpoints(betaLine, step, env.PlasmaBeta);
    addpoints(confinementLine, step, env.ConfinementTime * 100);
    addpoints(uniformityLine, step, env.FieldUniformity);

    % 7. Update status text
    if ~isempty(agent)
        agentType = 'RL Agent (Trained)';
    else
        agentType = 'Random Actions (No Agent)';
    end

    statusStr = sprintf(['Episode: %d | Step: %d\n' ...
                        'Mode: %s\n' ...
                        'Status: %s'], ...
                        episode, step, agentType, ...
                        isDone ? 'Episode Ended' : 'Running');
    set(statusText, 'String', statusStr);

    % 8. Update metrics text
    avgCurrent = mean(env.CoilCurrents);
    maxCurrent = max(env.CoilCurrents);
    minCurrent = min(env.CoilCurrents);
    currentImbalance = std(env.CoilCurrents) / mean(env.CoilCurrents);

    metricsStr = sprintf(['╔══════ LIVE METRICS ══════╗\n' ...
                         '║ Plasma Beta:      %.3f  ║\n' ...
                         '║ Target Beta:      %.3f  ║\n' ...
                         '║ Confinement:   %.4f s ║\n' ...
                         '║ Uniformity:       %.3f  ║\n' ...
                         '║                          ║\n' ...
                         '║ Avg Current:   %.0f A   ║\n' ...
                         '║ Max Current:   %.0f A   ║\n' ...
                         '║ Min Current:   %.0f A   ║\n' ...
                         '║ Imbalance:        %.1f%%  ║\n' ...
                         '║                          ║\n' ...
                         '║ Reward:          %.2f   ║\n' ...
                         '║ Cumulative:      %.2f   ║\n' ...
                         '╚══════════════════════════╝'], ...
                         env.PlasmaBeta, env.TargetBeta, ...
                         env.ConfinementTime, env.FieldUniformity, ...
                         avgCurrent, maxCurrent, minCurrent, ...
                         currentImbalance*100, reward, cumulativeReward);
    set(metricsText, 'String', metricsStr);

    % Update observation
    obs = nextObs;

    % Refresh display
    drawnow limitrate;

    % Pause for animation (adjustable speed)
    pause(0.05);

    % Print periodic updates
    if mod(step, 20) == 0
        fprintf('Step %3d | β=%.3f | τ=%.4fs | U=%.3f | Reward=%.2f\n', ...
            step, env.PlasmaBeta, env.ConfinementTime, env.FieldUniformity, cumulativeReward);
    end

    % Check if episode is done
    if isDone
        fprintf('\n┌─────────────────────────────────────────────┐\n');
        fprintf('│ EPISODE %d COMPLETED                        │\n', episode);
        fprintf('│ Steps: %d                                  │\n', step);
        fprintf('│ Final Beta: %.3f                           │\n', env.PlasmaBeta);
        fprintf('│ Final Confinement: %.4f s                  │\n', env.ConfinementTime);
        fprintf('│ Total Reward: %.2f                         │\n', cumulativeReward);
        fprintf('└─────────────────────────────────────────────┘\n\n');

        % Ask to continue
        choice = questdlg('Episode finished. Start new episode?', ...
                         'Continue', 'Yes', 'No', 'Yes');

        if strcmp(choice, 'Yes')
            % Reset for new episode
            obs = reset(env);
            cumulativeReward = 0;
            step = 0;
            episode = episode + 1;

            % Clear animated lines
            clearpoints(betaLine);
            clearpoints(confinementLine);
            clearpoints(uniformityLine);

            fprintf('Starting Episode %d...\n\n', episode);
        else
            fprintf('\nSimulation ended by user.\n');
            break;
        end
    end
end

fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  SIMULATION COMPLETE                                       ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');

%% Helper Functions

function Bfield = calculateMagneticField(point, coilPositions, coilCurrents)
    % Calculate magnetic field at a point due to all coils
    % Simplified dipole model

    mu0 = 4*pi*1e-7;  % Permeability of free space
    coilRadius = 0.15;
    Bfield = [0; 0; 0];

    for i = 1:size(coilPositions, 1)
        r = point - coilPositions(i, :);
        rMag = norm(r);

        if rMag < 0.01
            continue;  % Skip if too close to coil
        end

        % Coil normal (pointing inward)
        normal = -coilPositions(i, :)' / norm(coilPositions(i, :));

        % Magnetic dipole moment
        m = coilCurrents(i) * pi * coilRadius^2 * normal;

        % Dipole field
        rHat = r' / rMag;
        mDotR = dot(m, rHat);

        B = (mu0 / (4*pi)) * (1/rMag^3) * (3 * mDotR * rHat - m);

        Bfield = Bfield + B;
    end
end

function [magneticFieldData, coilPositions] = createDummyData()
    % Create dummy magnetic field data
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
end
