function plotPolywellTransients(data, summary, params)
%PLOTPOLYWELL

TRANSIENTS Generate comprehensive transient plots for 500 episode simulation
%
% Creates publication-quality plots showing:
% - Confinement time evolution over 500 episodes
% - Plasma beta progression
% - Magnetic field uniformity
% - Fusion power output
% - Control performance metrics
%
% Inputs:
%   data - Struct with full time-series data
%   summary - Struct with episode summaries
%   params - Physics parameters
%
% Outputs:
%   Generates and saves PNG figures

fprintf('Generating transient analysis plots...\n');

numEpisodes = size(data.beta, 1);
episodes = 1:numEpisodes;

%% Figure 1: Confinement Time Transient (Main Result)
fig1 = figure('Position', [100, 100, 1200, 800], 'Color', 'w');

% Subplot 1: Confinement Time vs Episode
subplot(3, 2, 1);
plot(episodes, summary.avgTau*1000, 'b-', 'LineWidth', 2);
hold on;
plot(episodes, movmean(summary.avgTau*1000, 20), 'r-', 'LineWidth', 3);
yline(params.confinement.tau_target*1000, '--k', 'Target', 'LineWidth', 2);
xlabel('Episode', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Confinement Time (ms)', 'FontSize', 12, 'FontWeight', 'bold');
title('Energy Confinement Time Evolution', 'FontSize', 14, 'FontWeight', 'bold');
legend('Per Episode', '20-Episode Moving Avg', 'Target', 'Location', 'southeast');
grid on;
xlim([1, numEpisodes]);

% Subplot 2: Confinement Time Distribution
subplot(3, 2, 2);
histogram(summary.avgTau*1000, 30, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'k');
hold on;
xline(params.confinement.tau_target*1000, '--r', 'Target', 'LineWidth', 3);
xline(mean(summary.avgTau)*1000, '--g', 'Mean', 'LineWidth', 2);
xlabel('Confinement Time (ms)', 'FontSize', 12);
ylabel('Frequency', 'FontSize', 12);
title('Confinement Time Distribution (500 Episodes)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% Subplot 3: Learning Curve - First 100 Episodes
subplot(3, 2, 3);
plot(episodes(1:100), summary.avgTau(1:100)*1000, 'b-', 'LineWidth', 2);
hold on;
plot(episodes(1:100), movmean(summary.avgTau(1:100)*1000, 5), 'r-', 'LineWidth', 3);
xlabel('Episode', 'FontSize', 12);
ylabel('Confinement Time (ms)', 'FontSize', 12);
title('Learning Phase (Episodes 1-100)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Actual', '5-Ep Moving Avg', 'Location', 'southeast');
grid on;

% Subplot 4: Steady State - Last 100 Episodes
subplot(3, 2, 4);
plot(episodes(401:500), summary.avgTau(401:500)*1000, 'b-', 'LineWidth', 2);
hold on;
plot(episodes(401:500), movmean(summary.avgTau(401:500)*1000, 5), 'r-', 'LineWidth', 3);
yline(params.confinement.tau_target*1000, '--k', 'Target', 'LineWidth', 2);
xlabel('Episode', 'FontSize', 12);
ylabel('Confinement Time (ms)', 'FontSize', 12);
title('Optimized Control (Episodes 401-500)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Actual', '5-Ep Moving Avg', 'Target', 'Location', 'southeast');
grid on;

% Subplot 5: Improvement Rate
subplot(3, 2, 5);
windowSize = 50;
improvement = zeros(numEpisodes - windowSize, 1);
for i = 1:(numEpisodes - windowSize)
    improvement(i) = (mean(summary.avgTau(i+windowSize:i+windowSize)) - ...
                      mean(summary.avgTau(i:i+windowSize))) / ...
                      mean(summary.avgTau(i:i+windowSize)) * 100;
end
plot(episodes(1:end-windowSize), improvement, 'g-', 'LineWidth', 2);
xlabel('Episode', 'FontSize', 12);
ylabel('Improvement Rate (%)', 'FontSize', 12);
title(sprintf('Rolling %d-Episode Improvement', windowSize), 'FontSize', 14, 'FontWeight', 'bold');
grid on;
yline(0, '--k', 'LineWidth', 1.5);

% Subplot 6: Cumulative Average
subplot(3, 2, 6);
cumAvg = cumsum(summary.avgTau) ./ episodes';
plot(episodes, cumAvg*1000, 'b-', 'LineWidth', 2.5);
hold on;
yline(params.confinement.tau_target*1000, '--r', 'Target', 'LineWidth', 2);
xlabel('Episode', 'FontSize', 12);
ylabel('Cumulative Average τ (ms)', 'FontSize', 12);
title('Cumulative Average Confinement Time', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
legend('Cumulative Avg', 'Target', 'Location', 'southeast');

% Save figure
saveas(fig1, 'ConfinementTime_Transient.png');
fprintf('  ✓ Saved: ConfinementTime_Transient.png\n');

%% Figure 2: Plasma Parameters Evolution
fig2 = figure('Position', [150, 150, 1400, 900], 'Color', 'w');

% Subplot 1: Plasma Beta
subplot(3, 3, 1);
plot(episodes, summary.avgBeta, 'b-', 'LineWidth', 2);
hold on;
plot(episodes, movmean(summary.avgBeta, 20), 'r-', 'LineWidth', 3);
yline(0.4, '--k', 'Target', 'LineWidth', 2);
xlabel('Episode'); ylabel('Plasma Beta');
title('Plasma Beta Evolution', 'FontWeight', 'bold');
legend('Actual', '20-Ep Avg', 'Target', 'Location', 'southeast');
grid on;

% Subplot 2: Uniformity
subplot(3, 3, 2);
plot(episodes, summary.avgUniformity, 'g-', 'LineWidth', 2);
hold on;
plot(episodes, movmean(summary.avgUniformity, 20), 'k-', 'LineWidth', 3);
xlabel('Episode'); ylabel('Field Uniformity');
title('Magnetic Field Uniformity', 'FontWeight', 'bold');
legend('Actual', '20-Ep Avg', 'Location', 'southeast');
grid on;
ylim([0, 1]);

% Subplot 3: Fusion Power
subplot(3, 3, 3);
semilogy(episodes, summary.avgFusionPower*1e6, 'r-', 'LineWidth', 2);
hold on;
semilogy(episodes, movmean(summary.avgFusionPower*1e6, 20), 'k-', 'LineWidth', 3);
xlabel('Episode'); ylabel('Fusion Power (μW)');
title('Fusion Power Output', 'FontWeight', 'bold');
legend('Actual', '20-Ep Avg', 'Location', 'best');
grid on;

% Subplot 4: Beta vs Tau (Correlation)
subplot(3, 3, 4);
scatter(summary.avgBeta, summary.avgTau*1000, 30, episodes, 'filled');
colorbar('Label', 'Episode Number');
xlabel('Average Beta'); ylabel('Average τ (ms)');
title('Beta-Confinement Correlation', 'FontWeight', 'bold');
grid on;
colormap(gca, 'jet');

% Subplot 5: Uniformity vs Tau
subplot(3, 3, 5);
scatter(summary.avgUniformity, summary.avgTau*1000, 30, episodes, 'filled');
colorbar('Label', 'Episode Number');
xlabel('Average Uniformity'); ylabel('Average τ (ms)');
title('Uniformity-Confinement Correlation', 'FontWeight', 'bold');
grid on;
colormap(gca, 'jet');

% Subplot 6: Lawson Criterion Progress
subplot(3, 3, 6);
nTau = params.plasma.n_avg * summary.avgTau;
semilogy(episodes, nTau, 'b-', 'LineWidth', 2);
hold on;
semilogy(episodes, movmean(nTau, 20), 'r-', 'LineWidth', 3);
yline(params.confinement.lawson_DD, '--k', 'D-D Criterion', 'LineWidth', 2);
xlabel('Episode'); ylabel('n×τ (s/m³)');
title('Lawson Criterion Progress', 'FontWeight', 'bold');
legend('Actual', '20-Ep Avg', 'D-D Target', 'Location', 'southeast');
grid on;

% Subplot 7: Episode Reward
subplot(3, 3, 7);
plot(episodes, summary.episodeReward, 'b-', 'LineWidth', 1.5);
hold on;
plot(episodes, movmean(summary.episodeReward, 20), 'r-', 'LineWidth', 3);
xlabel('Episode'); ylabel('Total Reward');
title('RL Agent Reward Evolution', 'FontWeight', 'bold');
legend('Episode Reward', '20-Ep Avg', 'Location', 'southeast');
grid on;

% Subplot 8: Average Coil Current
subplot(3, 3, 8);
avgCurrent = squeeze(mean(mean(data.coilCurrents, 2), 3));
plot(episodes, avgCurrent, 'b-', 'LineWidth', 2);
hold on;
plot(episodes, movmean(avgCurrent, 20), 'r-', 'LineWidth', 3);
xlabel('Episode'); ylabel('Average Current (A)');
title('Mean Coil Current Evolution', 'FontWeight', 'bold');
legend('Actual', '20-Ep Avg', 'Location', 'best');
grid on;

% Subplot 9: Magnetic Field Strength
subplot(3, 3, 9);
avgB = mean(data.B_field, 2);
plot(episodes, avgB, 'b-', 'LineWidth', 2);
hold on;
plot(episodes, movmean(avgB, 20), 'r-', 'LineWidth', 3);
yline(params.magnetic.B0_design, '--k', 'Design', 'LineWidth', 2);
xlabel('Episode'); ylabel('Magnetic Field (T)');
title('Average Magnetic Field', 'FontWeight', 'bold');
legend('Actual', '20-Ep Avg', 'Design', 'Location', 'best');
grid on;

saveas(fig2, 'PlasmaParameters_Evolution.png');
fprintf('  ✓ Saved: PlasmaParameters_Evolution.png\n');

%% Figure 3: Detailed Time Series (Sample Episodes)
fig3 = figure('Position', [200, 200, 1400, 800], 'Color', 'w');

% Select representative episodes
ep_early = 10;
ep_mid = 250;
ep_late = 490;

time_axis = (1:size(data.beta, 2)) * params.control.dt;

% Confinement time within episodes
subplot(2, 3, 1);
plot(time_axis, data.tau_c(ep_early, :)*1000, 'b-', 'LineWidth', 2); hold on;
plot(time_axis, data.tau_c(ep_mid, :)*1000, 'g-', 'LineWidth', 2);
plot(time_axis, data.tau_c(ep_late, :)*1000, 'r-', 'LineWidth', 2);
yline(params.confinement.tau_target*1000, '--k', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('τ (ms)');
title('Confinement Time Within Episodes', 'FontWeight', 'bold');
legend(sprintf('Ep %d', ep_early), sprintf('Ep %d', ep_mid), ...
       sprintf('Ep %d', ep_late), 'Target', 'Location', 'best');
grid on;

% Beta within episodes
subplot(2, 3, 2);
plot(time_axis, data.beta(ep_early, :), 'b-', 'LineWidth', 2); hold on;
plot(time_axis, data.beta(ep_mid, :), 'g-', 'LineWidth', 2);
plot(time_axis, data.beta(ep_late, :), 'r-', 'LineWidth', 2);
yline(0.4, '--k', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Beta');
title('Plasma Beta Within Episodes', 'FontWeight', 'bold');
legend(sprintf('Ep %d', ep_early), sprintf('Ep %d', ep_mid), ...
       sprintf('Ep %d', ep_late), 'Target', 'Location', 'best');
grid on;

% Uniformity within episodes
subplot(2, 3, 3);
plot(time_axis, data.uniformity(ep_early, :), 'b-', 'LineWidth', 2); hold on;
plot(time_axis, data.uniformity(ep_mid, :), 'g-', 'LineWidth', 2);
plot(time_axis, data.uniformity(ep_late, :), 'r-', 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Uniformity');
title('Field Uniformity Within Episodes', 'FontWeight', 'bold');
legend(sprintf('Ep %d', ep_early), sprintf('Ep %d', ep_mid), ...
       sprintf('Ep %d', ep_late), 'Location', 'best');
grid on;

% Coil currents - Episode 10
subplot(2, 3, 4);
for i = 1:6
    plot(time_axis, squeeze(data.coilCurrents(ep_early, :, i)), 'LineWidth', 1.5);
    hold on;
end
xlabel('Time (s)'); ylabel('Current (A)');
title(sprintf('Coil Currents - Episode %d (Early)', ep_early), 'FontWeight', 'bold');
legend('Coil 1', 'Coil 2', 'Coil 3', 'Coil 4', 'Coil 5', 'Coil 6', 'Location', 'best');
grid on;

% Coil currents - Episode 250
subplot(2, 3, 5);
for i = 1:6
    plot(time_axis, squeeze(data.coilCurrents(ep_mid, :, i)), 'LineWidth', 1.5);
    hold on;
end
xlabel('Time (s)'); ylabel('Current (A)');
title(sprintf('Coil Currents - Episode %d (Mid)', ep_mid), 'FontWeight', 'bold');
legend('Coil 1', 'Coil 2', 'Coil 3', 'Coil 4', 'Coil 5', 'Coil 6', 'Location', 'best');
grid on;

% Coil currents - Episode 490
subplot(2, 3, 6);
for i = 1:6
    plot(time_axis, squeeze(data.coilCurrents(ep_late, :, i)), 'LineWidth', 1.5);
    hold on;
end
xlabel('Time (s)'); ylabel('Current (A)');
title(sprintf('Coil Currents - Episode %d (Late)', ep_late), 'FontWeight', 'bold');
legend('Coil 1', 'Coil 2', 'Coil 3', 'Coil 4', 'Coil 5', 'Coil 6', 'Location', 'best');
grid on;

saveas(fig3, 'EpisodeTimeSeries.png');
fprintf('  ✓ Saved: EpisodeTimeSeries.png\n');

%% Figure 4: Performance Summary
fig4 = figure('Position', [250, 250, 1200, 600], 'Color', 'w');

% Performance metrics over training phases
phases = [1, 100, 200, 300, 400, 500];
phase_labels = {'Start', 'Ep 100', 'Ep 200', 'Ep 300', 'Ep 400', 'Final'};

metrics = zeros(6, 4);
for i = 1:6
    ep = phases(i);
    metrics(i, 1) = summary.avgBeta(ep);
    metrics(i, 2) = summary.avgTau(ep) * 1000;  % ms
    metrics(i, 3) = summary.avgUniformity(ep);
    metrics(i, 4) = summary.episodeReward(ep);
end

subplot(1, 2, 1);
bar(metrics(:, 1:3));
set(gca, 'XTickLabel', phase_labels);
ylabel('Value');
title('Performance Metrics Across Training', 'FontSize', 14, 'FontWeight', 'bold');
legend('Beta', 'τ (ms)', 'Uniformity', 'Location', 'northwest');
grid on;

subplot(1, 2, 2);
bar(metrics(:, 4));
set(gca, 'XTickLabel', phase_labels);
ylabel('Reward');
title('Episode Reward Progression', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

saveas(fig4, 'FusionPerformance.png');
fprintf('  ✓ Saved: FusionPerformance.png\n');

fprintf('\nAll transient plots generated successfully!\n');

end
