function B_field = magneticFieldBlock(coilCurrents)
%MAGNETICFIELDBLOCK Calculates magnetic field strength from coil currents
%
% This function computes the magnetic field at the center of the polywell
% reactor based on the currents in the 6 magnetic coils.
%
% Physics:
% - Each coil acts as a magnetic dipole
% - Fields superpose (linear combination)
% - Uses Biot-Savart law approximation
%
% Input:
%   coilCurrents - [6x1] vector of coil currents (Amperes)
%
% Output:
%   B_field - Magnetic field magnitude at center (Tesla)

%% Physical Constants
mu0 = 4*pi*1e-7;  % Permeability of free space (H/m)

%% Load Geometry (persistent)
persistent params_loaded params

if isempty(params_loaded)
    data = load('polywellFusionParams.mat');
    params = data.params;
    params_loaded = true;
end

%% Coil Geometry
R_coil = params.geometry.coilRadius;      % Coil radius (m)
positions = params.geometry.coilPositions; % [6x3] coil centers

%% Calculate Field at Center (0,0,0)
% For a circular current loop, on-axis field:
% B = (μ₀ * I * R²) / (2 * (R² + z²)^(3/2))
%
% At center (z = distance from coil to center):
center = [0, 0, 0];
B_total = [0, 0, 0];

for i = 1:6
    % Distance from coil to center
    r_vec = center - positions(i, :);
    z = norm(r_vec);

    % Coil normal direction (pointing inward)
    normal = -positions(i, :) / norm(positions(i, :));

    % On-axis magnetic field magnitude
    B_mag = (mu0 * coilCurrents(i) * R_coil^2) / ...
            (2 * (R_coil^2 + z^2)^(3/2));

    % Field vector (along normal direction)
    B_vec = B_mag * normal;

    % Superposition
    B_total = B_total + B_vec;
end

%% Calculate Total Field Magnitude
B_field = norm(B_total);

%% Apply Physical Limits
B_max = params.magnetic.B_max;  % Maximum possible field
B_min = params.magnetic.B_min;  % Residual field

B_field = max(B_min, min(B_field, B_max));

%% Nonlinear Effects (Saturation)
% At very high currents, iron core saturation reduces field efficiency
I_total = sum(abs(coilCurrents));
I_saturation = 25000;  % 25 kA total (conservative)

if I_total > I_saturation
    saturation_factor = I_saturation / I_total;
    B_field = B_field * (0.7 + 0.3*saturation_factor);  % Reduced efficiency
end

end
