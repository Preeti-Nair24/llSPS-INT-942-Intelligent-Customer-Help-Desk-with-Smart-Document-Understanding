%% Polywell Fusion Reactor - Physical Parameters Setup
% This script defines realistic fusion parameters for the polywell simulation
% Based on actual IEC/Polywell fusion research and plasma physics
%
% References:
% - Bussard, R.W. "Inertial Electrostatic Confinement" (1991)
% - Lawson Criterion for fusion
% - NRL Plasma Formulary

clear all;

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  POLYWELL FUSION REACTOR - PHYSICS PARAMETERS             ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

%% Universal Constants
constants.e = 1.602e-19;          % Elementary charge (C)
constants.me = 9.109e-31;         % Electron mass (kg)
constants.mp = 1.673e-27;         % Proton mass (kg)
constants.md = 3.344e-27;         % Deuterium mass (kg)
constants.mt = 5.008e-27;         % Tritium mass (kg)
constants.eps0 = 8.854e-12;       % Permittivity of free space (F/m)
constants.mu0 = 4*pi*1e-7;        % Permeability of free space (H/m)
constants.kb = 1.381e-23;         % Boltzmann constant (J/K)
constants.c = 2.998e8;            % Speed of light (m/s)

fprintf('✓ Universal constants loaded\n');

%% Polywell Geometry
geometry.coilRadius = 0.15;           % Coil radius (m) - 15 cm
geometry.reactorRadius = 0.50;        % Reactor chamber radius (m) - 50 cm
geometry.coilPositions = [            % 6 coils in cubic arrangement
    0.5,  0,  0;   % +X
   -0.5,  0,  0;   % -X
    0,  0.5,  0;   % +Y
    0, -0.5,  0;   % -Y
    0,  0,  0.5;   % +Z
    0,  0, -0.5    % -Z
];
geometry.numCoils = 6;
geometry.volume = (4/3) * pi * geometry.reactorRadius^3;  % Reactor volume (m³)

fprintf('✓ Polywell geometry defined\n');
fprintf('  • Reactor radius: %.2f m\n', geometry.reactorRadius);
fprintf('  • Volume: %.4f m³\n', geometry.volume);

%% Plasma Parameters (Realistic for Polywell)
plasma.fuelType = 'D-D';              % Deuterium-Deuterium fusion

% Operating conditions
plasma.T_keV = 30;                    % Plasma temperature (keV) - typical for D-D
plasma.T_K = plasma.T_keV * 1.1605e7; % Temperature in Kelvin
plasma.n0 = 1e19;                     % Central plasma density (m⁻³)
plasma.n_avg = 5e18;                  % Average density (m⁻³)

% Derived quantities
plasma.vth_e = sqrt(2*constants.kb*plasma.T_K / constants.me);  % Electron thermal velocity
plasma.vth_i = sqrt(2*constants.kb*plasma.T_K / constants.md);  % Ion thermal velocity

fprintf('✓ Plasma parameters set\n');
fprintf('  • Fuel: %s\n', plasma.fuelType);
fprintf('  • Temperature: %.1f keV (%.2e K)\n', plasma.T_keV, plasma.T_K);
fprintf('  • Density: %.2e m⁻³\n', plasma.n_avg);

%% Magnetic Field Parameters
magnetic.B0_design = 0.3;             % Design field strength (Tesla)
magnetic.B_min = 0.05;                % Minimum field (T)
magnetic.B_max = 0.8;                 % Maximum field (T)
magnetic.I_nominal = 2500;            % Nominal coil current (A)
magnetic.I_max = 5000;                % Maximum coil current (A)
magnetic.L_coil = 1e-3;               % Coil inductance (H)
magnetic.R_coil = 0.1;                % Coil resistance (Ohm)

% Magnetic pressure and beta
magnetic.P_mag = magnetic.B0_design^2 / (2*constants.mu0);  % Magnetic pressure (Pa)
plasma.P_plasma = plasma.n_avg * constants.kb * plasma.T_K;  % Plasma pressure (Pa)
plasma.beta = plasma.P_plasma / magnetic.P_mag;              % Plasma beta

fprintf('✓ Magnetic field parameters\n');
fprintf('  • Design field: %.2f T\n', magnetic.B0_design);
fprintf('  • Plasma beta: %.4f\n', plasma.beta);

%% Confinement Parameters
confinement.tau_target = 0.010;       % Target confinement time (10 ms)
confinement.tau_min = 0.001;          % Minimum (1 ms)
confinement.tau_max = 0.100;          % Maximum (100 ms)

% Lawson criterion (for D-D fusion at 30 keV: n*τ > 1e21 s/m³)
confinement.lawson_DT = 1e20;         % D-T Lawson criterion (s/m³)
confinement.lawson_DD = 1e21;         % D-D Lawson criterion (s/m³)
confinement.nTau_actual = plasma.n_avg * confinement.tau_target;

fprintf('✓ Confinement parameters\n');
fprintf('  • Target τ: %.3f ms\n', confinement.tau_target*1000);
fprintf('  • n×τ: %.2e s/m³ (need %.2e for D-D)\n', ...
    confinement.nTau_actual, confinement.lawson_DD);

%% Fusion Reaction Rates (D-D Fusion)
% D + D → He³ + n + 3.27 MeV (50%)
% D + D → T + p + 4.03 MeV (50%)

% Fusion cross-section data (simplified parameterization)
% σv(T) for D-D reaction (Bosch-Hale formula simplified)
fusion.calcReactionRate = @(T_keV) 2.33e-20 * T_keV^(-2/3) * exp(-18.76 * T_keV^(-1/3));

fusion.sigmav_DD = fusion.calcReactionRate(plasma.T_keV);  % <σv> in m³/s
fusion.P_fusion = 0.25 * plasma.n_avg^2 * fusion.sigmav_DD * ...
                  3.65e-13 * geometry.volume;  % Fusion power (W), avg 3.65 MeV

fusion.reactionRate = 0.5 * plasma.n_avg^2 * fusion.sigmav_DD * geometry.volume;  % reactions/s

fprintf('✓ Fusion reaction parameters\n');
fprintf('  • <σv>: %.2e m³/s\n', fusion.sigmav_DD);
fprintf('  • Reaction rate: %.2e reactions/s\n', fusion.reactionRate);
fprintf('  • Fusion power: %.2e W\n', fusion.P_fusion);

%% Energy Balance
% Power losses
energy.P_bremsstrahlung = 5.35e-37 * plasma.n_avg^2 * sqrt(plasma.T_K) * ...
                          geometry.volume;  % Bremsstrahlung radiation (W)
energy.P_synchrotron = 6.2e-17 * plasma.n_avg * plasma.T_keV^2.5 * ...
                       magnetic.B0_design^2 * geometry.volume;  % Synchrotron (W)
energy.P_conduction = (3*plasma.n_avg*constants.kb*plasma.T_K*geometry.volume) / ...
                      confinement.tau_target;  % Conduction losses (W)

energy.P_total_loss = energy.P_bremsstrahlung + energy.P_synchrotron + energy.P_conduction;
energy.Q_factor = fusion.P_fusion / max(energy.P_total_loss, 1);  % Fusion gain

fprintf('✓ Energy balance\n');
fprintf('  • Bremsstrahlung: %.2e W\n', energy.P_bremsstrahlung);
fprintf('  • Synchrotron: %.2e W\n', energy.P_synchrotron);
fprintf('  • Conduction: %.2e W\n', energy.P_conduction);
fprintf('  • Q factor: %.4f\n', energy.Q_factor);

%% Plasma Frequencies and Scales
plasma.omega_pe = sqrt(plasma.n_avg * constants.e^2 / (constants.eps0 * constants.me));  % Electron plasma freq
plasma.omega_ce = constants.e * magnetic.B0_design / constants.me;  % Electron cyclotron freq
plasma.omega_ci = constants.e * magnetic.B0_design / constants.md;  % Ion cyclotron freq

plasma.lambda_D = sqrt(constants.eps0 * constants.kb * plasma.T_K / ...
                       (plasma.n_avg * constants.e^2));  % Debye length
plasma.rho_i = constants.md * plasma.vth_i / (constants.e * magnetic.B0_design);  % Ion Larmor radius

fprintf('✓ Plasma characteristic scales\n');
fprintf('  • Debye length: %.2e m\n', plasma.lambda_D);
fprintf('  • Ion Larmor radius: %.4f m\n', plasma.rho_i);
fprintf('  • Electron plasma freq: %.2e rad/s\n', plasma.omega_pe);

%% RL Control Parameters
control.numCoils = 6;
control.stateSize = 9;                % [6 currents, beta, tau, uniformity]
control.actionSize = 6;               % Current changes for 6 coils
control.actionMax = 500;              % Max current change per step (A)
control.dt = 0.1;                     % Control time step (100 ms)
control.rewardWeights = [10, 5, 2, 0.1];  % [beta, confinement, uniformity, efficiency]

fprintf('✓ RL control parameters\n');
fprintf('  • Control timestep: %.1f ms\n', control.dt*1000);

%% Physics Model Parameters (for Simulink)
model.tau_coefficient = 0.01;         % τ = coeff × β × uniformity
model.beta_coefficient = 0.5;         % β = coeff × field_strength × uniformity
model.uniformity_decay = 0.95;        % Exponential uniformity calculation

model.transportCoeff = 1e-3;          % Anomalous transport coefficient
model.diffusionTime = geometry.reactorRadius^2 / ...
                      (model.transportCoeff * plasma.vth_i);  % Diffusion timescale

fprintf('✓ Physics model coefficients\n');

%% Save all parameters
fprintf('\nSaving parameters to workspace...\n');

% Create single structure with all parameters
params.constants = constants;
params.geometry = geometry;
params.plasma = plasma;
params.magnetic = magnetic;
params.confinement = confinement;
params.fusion = fusion;
params.energy = energy;
params.control = control;
params.model = model;

% Save to file
save('polywellFusionParams.mat', 'params');

fprintf('✓ Parameters saved to: polywellFusionParams.mat\n\n');

%% Display Summary Table
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  PARAMETER SUMMARY                                         ║\n');
fprintf('╠════════════════════════════════════════════════════════════╣\n');
fprintf('║ Temperature:        %6.1f keV                           ║\n', plasma.T_keV);
fprintf('║ Density:            %6.2e m⁻³                         ║\n', plasma.n_avg);
fprintf('║ Magnetic Field:     %6.2f T                             ║\n', magnetic.B0_design);
fprintf('║ Beta:               %6.4f                               ║\n', plasma.beta);
fprintf('║ Confinement Time:   %6.2f ms                            ║\n', confinement.tau_target*1000);
fprintf('║ n×τ:                %6.2e s/m³                         ║\n', confinement.nTau_actual);
fprintf('║ Fusion Power:       %6.2e W                           ║\n', fusion.P_fusion);
fprintf('║ Q Factor:           %6.4f                               ║\n', energy.Q_factor);
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

fprintf('Parameters ready for Simulink model!\n');
fprintf('Next: Run buildPolywellSimulinkModel.m to create Simulink model\n\n');
