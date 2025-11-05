function [beta, tau_c, uniformity, fusionPower] = plasmaPhysicsBlock(coilCurrents, B_field)
%PLASMAPHYSICSBLOCK Calculates plasma parameters from magnetic field
%
% This function implements realistic plasma physics for a polywell fusion reactor
% Including:
% - Plasma beta calculation
% - Confinement time (energy confinement)
% - Fusion reaction rates
% - Magnetic field uniformity effects
%
% Inputs:
%   coilCurrents - [6x1] Current in each coil (Amperes)
%   B_field - Magnetic field strength (Tesla)
%
% Outputs:
%   beta - Plasma beta (pressure ratio)
%   tau_c - Energy confinement time (seconds)
%   uniformity - Field uniformity measure (0-1)
%   fusionPower - Fusion power output (Watts)

%% Physical Constants
kb = 1.381e-23;           % Boltzmann constant
e = 1.602e-19;            % Elementary charge
mu0 = 4*pi*1e-7;          % Permeability

%% Load Parameters (persistent for speed)
persistent params_loaded params

if isempty(params_loaded)
    % Load once and keep in memory
    data = load('polywellFusionParams.mat');
    params = data.params;
    params_loaded = true;
end

%% Calculate Field Uniformity
% Measure of how balanced the coil currents are
I_mean = mean(coilCurrents);
I_std = std(coilCurrents);

if I_mean > 1e-6
    uniformity = exp(-I_std / I_mean);  % 0 (unbalanced) to 1 (perfect balance)
else
    uniformity = 0;
end

%% Calculate Magnetic Pressure and Plasma Pressure
% Magnetic pressure
P_mag = B_field^2 / (2 * mu0);  % Pascal

% Plasma pressure (from kinetic pressure)
% P = n*k*T, but density varies with confinement
n_eff = params.plasma.n_avg * (0.5 + 0.5*uniformity);  % Density depends on field quality
P_plasma = n_eff * kb * params.plasma.T_K;

%% Calculate Beta
beta = P_plasma / (P_mag + 1e-10);  % Avoid division by zero

% Physical limits
beta = max(0, min(beta, 0.8));  % Beta can't exceed ~0.8 (ideal MHD limit)

%% Calculate Confinement Time
% Based on empirical scaling laws for IEC devices
% τ_E ∝ β × uniformity × (B/B₀)^α

B_normalized = B_field / params.magnetic.B0_design;

% Bohm scaling (pessimistic): τ ∝ B
% Gyro-Bohm scaling (optimistic): τ ∝ B²
% Use intermediate: τ ∝ B^1.5
tau_base = params.confinement.tau_target;
tau_c = tau_base * beta * uniformity * B_normalized^1.5;

% Add anomalous transport losses (degrade confinement)
if uniformity < 0.7
    % Poor uniformity → increased transport
    tau_c = tau_c * (0.5 + 0.5*uniformity);
end

% Physical limits
tau_c = max(params.confinement.tau_min, min(tau_c, params.confinement.tau_max));

%% Calculate Fusion Reaction Rate
% D-D fusion cross-section (temperature dependent)
T_keV = params.plasma.T_keV;

% Bosch-Hale parameterization for D-D (simplified)
if T_keV > 0
    sigmav_DD = 2.33e-20 * T_keV^(-2/3) * exp(-18.76 * T_keV^(-1/3));  % m³/s
else
    sigmav_DD = 0;
end

% Reaction rate: R = 0.5 * n² * <σv> * V
% Factor 0.5 because identical particles
n_squared = n_eff^2;
V_eff = params.geometry.volume * beta * uniformity;  % Effective volume with good confinement

reactionRate = 0.5 * n_squared * sigmav_DD * V_eff;  % reactions/second

%% Calculate Fusion Power
% D + D → He3 + n + 3.27 MeV (50%)
% D + D → T + p + 4.03 MeV (50%)
% Average: 3.65 MeV per reaction

E_fusion_avg = 3.65e6 * e;  % Joules per reaction
fusionPower = reactionRate * E_fusion_avg;  % Watts

% Realistic limit (current polywell experiments: μW to mW range)
fusionPower = min(fusionPower, 1e-3);  % Cap at 1 mW for now

%% Add Physics-Based Noise
% Real plasma has fluctuations
beta = beta * (1 + 0.05*randn());  % 5% noise
tau_c = tau_c * (1 + 0.03*randn());  % 3% noise

% Ensure non-negative
beta = max(0, beta);
tau_c = max(0, tau_c);
uniformity = max(0, min(1, uniformity));
fusionPower = max(0, fusionPower);

end
