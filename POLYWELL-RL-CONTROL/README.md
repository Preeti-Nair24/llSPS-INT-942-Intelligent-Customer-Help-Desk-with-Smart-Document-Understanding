# Polywell Magnetic Confinement RL Control System

This system integrates ANSYS Maxwell magnetic field simulations with MATLAB Reinforcement Learning to optimize plasma confinement in a polywell fusion reactor by controlling the currents in 6 magnetic coils.

## Overview

The polywell fusion reactor uses 6 magnetic coils arranged in a cubic configuration to create a magnetic field that confines plasma. This RL-based control system learns to optimize the coil currents to maximize:
- Plasma beta (ratio of plasma pressure to magnetic pressure)
- Plasma confinement time
- Magnetic field uniformity
- Energy efficiency

## System Architecture

```
ANSYS Maxwell Simulation → MATLAB Data Import → RL Environment → Trained Agent → Real-time Control Visualization
```

## Files

1. **importMaxwellData.m** - Imports magnetic field data from ANSYS Maxwell
2. **PolywellRLEnvironment.m** - Custom RL environment for coil current control
3. **trainPolywellRLAgent.m** - Trains the DDPG agent
4. **visualizePolywellControl.m** - Real-time visualization of RL control

## Requirements

### Software
- MATLAB R2020b or later
- MATLAB Reinforcement Learning Toolbox
- MATLAB Deep Learning Toolbox
- ANSYS Maxwell (for generating simulation data)

### Hardware
- Recommended: GPU for faster training
- Minimum RAM: 8 GB
- Recommended RAM: 16 GB or more

## Quick Start Guide

### Step 1: Export Data from ANSYS Maxwell

In ANSYS Maxwell, export your magnetic field simulation data:

1. Complete your polywell geometry simulation in Maxwell
2. Go to **Maxwell 3D → Fields → Fields Calculator**
3. Export magnetic field data (B-field) to CSV format with the following columns:
   - X, Y, Z (position coordinates in meters)
   - Bx, By, Bz (magnetic field components in Tesla)
   - Bmag (magnetic field magnitude in Tesla)

4. Save the file as `maxwell_polywell_data.csv` in the POLYWELL-RL-CONTROL directory

**Example CSV format:**
```csv
X,Y,Z,Bx,By,Bz,Bmag
-0.5,-0.5,-0.5,0.001,0.002,0.003,0.0037
-0.5,-0.5,-0.4,0.0015,0.0022,0.0031,0.0041
...
```

### Step 2: Train the RL Agent

```matlab
% Open MATLAB and navigate to the POLYWELL-RL-CONTROL directory
cd('path/to/POLYWELL-RL-CONTROL')

% Run the training script
trainPolywellRLAgent

% This will:
% - Import your ANSYS Maxwell data
% - Create the RL environment
% - Train a DDPG agent (may take several hours)
% - Save the trained agent to 'trainedPolywellAgent.mat'
```

**Training Parameters (configurable in script):**
- Episodes: 500
- Max steps per episode: 200
- Algorithm: DDPG (Deep Deterministic Policy Gradient)
- Estimated training time: 2-6 hours (depending on hardware)

### Step 3: Visualize the Trained Agent

```matlab
% Run the visualization script
visualizePolywellControl

% This will display:
% - 3D coil configuration with real-time current updates
% - Plasma confinement sphere (size/color indicates beta)
% - Real-time plots of plasma parameters
% - Cumulative reward
% - Power consumption
```

## Understanding the RL Environment

### State Space (9 dimensions)
1. Coil 1-6 currents (normalized, 0-1)
2. Plasma beta (0-1)
3. Confinement time (scaled)
4. Magnetic field uniformity (0-1)

### Action Space (6 dimensions)
- Change in current for each coil (-500A to +500A per step)

### Reward Function
```
Reward = 10 × (beta_error) +
         5 × (confinement_reward) +
         2 × (uniformity_reward) -
         0.1 × (power_penalty)
```

### Physical Constraints
- Max coil current: 5000 A
- Min coil current: 0 A
- Max current change per step: 500 A

## Customization

### Modify Target Parameters

In `PolywellRLEnvironment.m`, adjust:

```matlab
% Target parameters
TargetBeta = 0.4    % Target plasma beta (0-1)
TargetConfinement = 0.01  % Target confinement time (seconds)
```

### Modify Physical Constraints

```matlab
% Physical constraints
MaxCurrent = 5000   % Maximum coil current (Amperes)
MinCurrent = 0      % Minimum coil current (Amperes)
MaxCurrentChange = 500  % Maximum current change per step (Amperes)
```

### Modify Coil Geometry

In `importMaxwellData.m`, adjust the coil positions:

```matlab
L = 0.5;  % Half side length of cube (meters)
coilPositions = [
    L,  0,  0;   % +X face
   -L,  0,  0;   % -X face
    0,  L,  0;   % +Y face
    0, -L,  0;   % -Y face
    0,  0,  L;   % +Z face
    0,  0, -L    % -Z face
];
```

## Training Tips

1. **Start with dummy data**: The training script includes dummy data generation if ANSYS data is not available
2. **Monitor training**: Use the training progress plot to monitor convergence
3. **Adjust hyperparameters**: If training is unstable, reduce learning rate or increase mini-batch size
4. **GPU acceleration**: Enable GPU in MATLAB for faster training:
   ```matlab
   gpuDevice(1)  % Use first GPU
   ```

## Troubleshooting

### Issue: "File not found" error
**Solution**: Ensure `maxwell_polywell_data.csv` is in the POLYWELL-RL-CONTROL directory or update the path in `trainPolywellRLAgent.m`

### Issue: Training is too slow
**Solution**:
- Reduce number of episodes or max steps
- Enable GPU acceleration
- Reduce network size in training script

### Issue: Agent not learning (reward not improving)
**Solution**:
- Adjust reward function weights
- Increase exploration noise
- Increase training episodes
- Check that ANSYS data is properly formatted

### Issue: Visualization is choppy
**Solution**: Increase pause duration in visualization loop:
```matlab
pause(0.1);  % Increase from 0.05 to 0.1
```

## Advanced Usage

### Export Trained Controller for Hardware

```matlab
% Load trained agent
load('trainedPolywellAgent.mat');

% Create deployment function
function currents = getCoilCurrents(state)
    action = getAction(agent, state);
    currents = state(1:6) * MaxCurrent + action;
end
```

### Batch Testing

```matlab
% Test agent over multiple episodes
numTests = 10;
rewards = zeros(numTests, 1);

for i = 1:numTests
    simOpts = rlSimulationOptions('MaxSteps', 200);
    experience = sim(env, agent, simOpts);
    rewards(i) = sum(experience.Reward.Data);
end

fprintf('Average reward: %.2f ± %.2f\n', mean(rewards), std(rewards));
```

## References

- ANSYS Maxwell Documentation: https://www.ansys.com/products/electronics/ansys-maxwell
- MATLAB Reinforcement Learning Toolbox: https://www.mathworks.com/products/reinforcement-learning.html
- Polywell Fusion: Bussard, R. W. (1991). "Some physics considerations of magnetic inertial-electrostatic confinement"

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review MATLAB documentation for RL Toolbox
3. Ensure ANSYS Maxwell data export is correct

## License

This code is provided for research and educational purposes.
