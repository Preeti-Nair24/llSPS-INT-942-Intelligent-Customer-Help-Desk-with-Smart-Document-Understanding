%% Generate Example ANSYS Maxwell Data
% This script generates synthetic magnetic field data that mimics
% ANSYS Maxwell output for testing purposes.
%
% Use this to test the RL system before you have real ANSYS Maxwell data.
%
% Usage:
%   generateExampleData
%   % This will create 'maxwell_polywell_data.csv'

clear all; close all; clc;

fprintf('=== Generating Example ANSYS Maxwell Data ===\n\n');

%% Define Polywell Geometry
fprintf('Step 1: Defining polywell geometry...\n');

% Coil parameters
L = 0.5;  % Half side length (meters)
coilRadius = 0.15;  % Coil radius (meters)
coilCurrent = 1000;  % Current per coil (Amperes)

% Coil positions (cubic arrangement)
coilPositions = [
    L,  0,  0;   % +X face
   -L,  0,  0;   % -X face
    0,  L,  0;   % +Y face
    0, -L,  0;   % -Y face
    0,  0,  L;   % +Z face
    0,  0, -L    % -Z face
];

% Coil normal directions (pointing inward)
coilNormals = -coilPositions ./ vecnorm(coilPositions, 2, 2);

fprintf('Number of coils: %d\n', size(coilPositions, 1));

%% Create Sampling Grid
fprintf('Step 2: Creating sampling grid...\n');

% Grid resolution
resolution = 0.05;  % meters
gridRange = -0.6:resolution:0.6;

% Create 3D grid
[X, Y, Z] = meshgrid(gridRange, gridRange, gridRange);
positions = [X(:), Y(:), Z(:)];

fprintf('Grid points: %d\n', size(positions, 1));

%% Calculate Magnetic Field
fprintf('Step 3: Calculating magnetic field from coils...\n');

% Initialize field components
Bx = zeros(size(positions, 1), 1);
By = zeros(size(positions, 1), 1);
Bz = zeros(size(positions, 1), 1);

% Permeability of free space
mu0 = 4*pi*1e-7;  % T·m/A

% Calculate field contribution from each coil
for coilIdx = 1:size(coilPositions, 1)
    fprintf('  Processing coil %d/%d...\n', coilIdx, size(coilPositions, 1));

    coilPos = coilPositions(coilIdx, :);
    coilNormal = coilNormals(coilIdx, :);

    % For each point in space
    for i = 1:size(positions, 1)
        r = positions(i, :) - coilPos;  % Vector from coil to point
        rMag = norm(r);

        if rMag < 1e-6  % Avoid singularity at coil center
            continue;
        end

        % Simplified magnetic field calculation for circular current loop
        % Using magnetic dipole approximation for far field
        if rMag > coilRadius * 2
            % Magnetic dipole moment
            m = coilCurrent * pi * coilRadius^2 * coilNormal;

            % Dipole field components
            rHat = r / rMag;
            mDotR = dot(m, rHat);

            % B = (mu0/4pi) * (1/r^3) * (3*(m·r̂)r̂ - m)
            Bfield = (mu0 / (4*pi)) * (1/rMag^3) * ...
                     (3 * mDotR * rHat - m);

            Bx(i) = Bx(i) + Bfield(1);
            By(i) = By(i) + Bfield(2);
            Bz(i) = Bz(i) + Bfield(3);
        else
            % Near field: use simplified approximation
            % On-axis field for circular loop
            z_axis = dot(r, coilNormal);
            rho = norm(r - z_axis * coilNormal);

            % Axial component
            B_axial = (mu0 * coilCurrent * coilRadius^2) / ...
                      (2 * (coilRadius^2 + rMag^2)^(3/2));

            % Add to total field
            Bfield = B_axial * coilNormal;
            Bx(i) = Bx(i) + Bfield(1);
            By(i) = By(i) + Bfield(2);
            Bz(i) = Bz(i) + Bfield(3);
        end
    end
end

% Calculate magnitude
Bmag = sqrt(Bx.^2 + By.^2 + Bz.^2);

fprintf('Field calculation complete.\n');
fprintf('Field range: %.2e to %.2e Tesla\n', min(Bmag), max(Bmag));

%% Create Data Table
fprintf('\nStep 4: Creating data table...\n');

dataTable = table(positions(:,1), positions(:,2), positions(:,3), ...
                  Bx, By, Bz, Bmag, ...
                  'VariableNames', {'X', 'Y', 'Z', 'Bx', 'By', 'Bz', 'Bmag'});

fprintf('Data table created with %d rows.\n', height(dataTable));

%% Export to CSV
fprintf('\nStep 5: Exporting to CSV...\n');

outputFile = 'maxwell_polywell_data.csv';
writetable(dataTable, outputFile);

fprintf('Data exported to: %s\n', outputFile);
fprintf('File size: %.2f MB\n', dir(outputFile).bytes / 1e6);

%% Visualize the Field
fprintf('\nStep 6: Creating visualization...\n');

figure('Name', 'Generated Magnetic Field', 'Position', [100, 100, 1200, 500]);

% Subplot 1: Field magnitude slice
subplot(1, 3, 1);
sliceZ = abs(positions(:,3)) < resolution/2;  % z ≈ 0 plane
scatter(positions(sliceZ, 1), positions(sliceZ, 2), 20, Bmag(sliceZ), 'filled');
colorbar;
title('Magnetic Field Magnitude (Z=0 plane)');
xlabel('X (m)'); ylabel('Y (m)');
axis equal tight;
colormap jet;

% Subplot 2: Field lines (quiver plot)
subplot(1, 3, 2);
sliceZ2 = abs(positions(:,3)) < resolution/2;
quiver(positions(sliceZ2, 1), positions(sliceZ2, 2), ...
       Bx(sliceZ2), By(sliceZ2), 2);
hold on;
plot(coilPositions([3,4], 1), coilPositions([3,4], 2), ...
     'ro', 'MarkerSize', 15, 'LineWidth', 3);
title('Magnetic Field Lines (Z=0 plane)');
xlabel('X (m)'); ylabel('Y (m)');
axis equal tight;
grid on;

% Subplot 3: 3D coil configuration
subplot(1, 3, 3);
hold on;
for i = 1:size(coilPositions, 1)
    plot3(coilPositions(i,1), coilPositions(i,2), coilPositions(i,3), ...
          'o', 'MarkerSize', 20, 'LineWidth', 3, 'MarkerFaceColor', 'b');
    % Draw coil orientation
    quiver3(coilPositions(i,1), coilPositions(i,2), coilPositions(i,3), ...
            coilNormals(i,1)*0.2, coilNormals(i,2)*0.2, coilNormals(i,3)*0.2, ...
            'LineWidth', 2, 'Color', 'r', 'MaxHeadSize', 1);
end
title('Polywell Coil Configuration');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
grid on; axis equal;
view(45, 30);

fprintf('Visualization complete.\n');

%% Statistics
fprintf('\n=== Data Statistics ===\n');
fprintf('Number of points: %d\n', height(dataTable));
fprintf('Spatial extent:\n');
fprintf('  X: [%.2f, %.2f] m\n', min(positions(:,1)), max(positions(:,1)));
fprintf('  Y: [%.2f, %.2f] m\n', min(positions(:,2)), max(positions(:,2)));
fprintf('  Z: [%.2f, %.2f] m\n', min(positions(:,3)), max(positions(:,3)));
fprintf('\nMagnetic field:\n');
fprintf('  Min: %.2e Tesla\n', min(Bmag));
fprintf('  Max: %.2e Tesla\n', max(Bmag));
fprintf('  Mean: %.2e Tesla\n', mean(Bmag));
fprintf('  Std: %.2e Tesla\n', std(Bmag));

fprintf('\n=== Generation Complete! ===\n');
fprintf('You can now use this data with trainPolywellRLAgent.m\n');
