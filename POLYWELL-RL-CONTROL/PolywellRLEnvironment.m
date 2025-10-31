classdef PolywellRLEnvironment < rl.env.MATLABEnvironment
    % POLYWELLRLENVIRONMENT Custom RL Environment for Polywell Coil Current Control
    %
    % This environment simulates the control of 6 magnetic coil currents
    % in a polywell fusion reactor to optimize plasma confinement.
    %
    % State: [6 coil currents, plasma beta, confinement time, field uniformity]
    % Action: Change in coil currents (continuous)
    % Reward: Based on plasma confinement quality and energy efficiency

    properties
        % Magnetic field data from ANSYS Maxwell
        MagneticFieldData
        CoilPositions

        % Current state
        CoilCurrents        % [6 x 1] Current in each coil (Amperes)
        PlasmaBeta          % Plasma pressure / Magnetic pressure
        ConfinementTime     % Plasma confinement time (seconds)
        FieldUniformity     % Measure of field uniformity (0-1)

        % Physical constraints
        MaxCurrent = 5000   % Maximum coil current (A)
        MinCurrent = 0      % Minimum coil current (A)
        MaxCurrentChange = 500  % Maximum current change per step (A)

        % Simulation parameters
        TimeStep = 0.1      % Time step (seconds)
        MaxSteps = 200      % Maximum steps per episode
        CurrentStep = 0

        % Target parameters
        TargetBeta = 0.4    % Target plasma beta
        TargetConfinement = 0.01  % Target confinement time (s)

        % Visualization handles
        FigureHandle
        AxisHandles
    end

    methods
        function this = PolywellRLEnvironment(magneticFieldData, coilPositions)
            % Initialize observation space
            % State: [6 currents, beta, confinement, uniformity] = 9 dimensions
            ObservationInfo = rlNumericSpec([9 1]);
            ObservationInfo.Name = 'Polywell State';
            ObservationInfo.Description = 'Coil Currents, Beta, Confinement, Uniformity';

            % Initialize action space
            % Action: Change in current for each of 6 coils
            ActionInfo = rlNumericSpec([6 1], ...
                'LowerLimit', -this.MaxCurrentChange * ones(6, 1), ...
                'UpperLimit', this.MaxCurrentChange * ones(6, 1));
            ActionInfo.Name = 'Coil Current Changes';

            % Call superclass constructor
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);

            % Store magnetic field data
            this.MagneticFieldData = magneticFieldData;
            this.CoilPositions = coilPositions;
        end

        function [Observation, Reward, IsDone, LoggedSignals] = step(this, Action)
            % STEP Execute one time step
            LoggedSignals = [];

            % Update coil currents based on action
            this.CoilCurrents = this.CoilCurrents + Action;

            % Enforce current limits
            this.CoilCurrents = max(min(this.CoilCurrents, this.MaxCurrent), this.MinCurrent);

            % Calculate plasma parameters based on new currents
            this = updatePlasmaParameters(this);

            % Calculate reward
            Reward = calculateReward(this);

            % Check termination conditions
            this.CurrentStep = this.CurrentStep + 1;
            IsDone = this.CurrentStep >= this.MaxSteps || ...
                     this.PlasmaBeta < 0.1 || ...  % Plasma lost
                     max(this.CoilCurrents) > this.MaxCurrent * 0.95;  % Near limit

            % Update observation
            Observation = getObservation(this);
        end

        function InitialObservation = reset(this)
            % RESET Reset environment to initial state

            % Initialize coil currents (with some randomness)
            this.CoilCurrents = 1000 * ones(6, 1) + randn(6, 1) * 100;
            this.CoilCurrents = max(min(this.CoilCurrents, this.MaxCurrent), this.MinCurrent);

            % Initialize plasma parameters
            this.PlasmaBeta = 0.2;
            this.ConfinementTime = 0.005;
            this.FieldUniformity = 0.5;
            this.CurrentStep = 0;

            % Update plasma parameters
            this = updatePlasmaParameters(this);

            % Return initial observation
            InitialObservation = getObservation(this);
        end
    end

    methods (Access = private)
        function Observation = getObservation(this)
            % GET_OBSERVATION Return current state
            Observation = [
                this.CoilCurrents / this.MaxCurrent;  % Normalized currents
                this.PlasmaBeta;
                this.ConfinementTime * 100;  % Scale for better learning
                this.FieldUniformity
            ];
        end

        function this = updatePlasmaParameters(this)
            % UPDATE_PLASMA_PARAMETERS Calculate plasma parameters from coil currents

            % Calculate average magnetic field strength
            avgField = mean(this.CoilCurrents) / 1000;  % Normalized

            % Calculate field uniformity (how balanced the coils are)
            currentStd = std(this.CoilCurrents);
            currentMean = mean(this.CoilCurrents);
            this.FieldUniformity = exp(-currentStd / (currentMean + 1e-6));

            % Model plasma beta (simplified physics model)
            % Beta increases with field strength and uniformity
            fieldStrength = avgField / 5;  % Normalize to max current
            this.PlasmaBeta = 0.5 * fieldStrength * this.FieldUniformity + ...
                             0.1 * randn();  % Add noise
            this.PlasmaBeta = max(0, min(1, this.PlasmaBeta));

            % Model confinement time (simplified)
            % Better confinement with higher beta and uniformity
            this.ConfinementTime = 0.01 * this.PlasmaBeta * ...
                                  this.FieldUniformity * (1 + 0.1 * randn());
            this.ConfinementTime = max(0, this.ConfinementTime);
        end

        function Reward = calculateReward(this)
            % CALCULATE_REWARD Calculate reward based on performance

            % Reward components
            % 1. Plasma beta close to target
            betaReward = -abs(this.PlasmaBeta - this.TargetBeta);

            % 2. Confinement time (higher is better)
            confinementReward = (this.ConfinementTime / this.TargetConfinement);

            % 3. Field uniformity (higher is better)
            uniformityReward = this.FieldUniformity;

            % 4. Energy efficiency (penalize high currents)
            powerConsumption = sum(this.CoilCurrents.^2) / (6 * this.MaxCurrent^2);
            efficiencyPenalty = -0.1 * powerConsumption;

            % Total reward (weighted sum)
            Reward = 10 * betaReward + ...
                     5 * confinementReward + ...
                     2 * uniformityReward + ...
                     efficiencyPenalty;
        end
    end
end
