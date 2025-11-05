# Polywell Fusion Reactor - RL Control with Simulink/MATLAB
## 500-Episode Simulation with Realistic Fusion Parameters

This simulation implements a complete physics-based model of a polywell fusion reactor controlled by a reinforcement learning agent, with transient analysis over 500 episodes.

---

## 🎯 Features

### Realistic Fusion Physics
- **D-D Fusion Reactions** with Bosch-Hale cross-sections
- **Plasma Parameters**: T=30 keV, n=5×10¹⁸ m⁻³
- **Magnetic Confinement**: B₀=0.3 T, β targeting 0.40
- **Lawson Criterion** tracking (n×τ for fusion breakeven)
- **Energy Balance**: Bremsstrahlung, synchrotron, conduction losses
- **Fusion Power Output** calculation

### RL Control System
- **PPO Algorithm** (Proximal Policy Optimization)
- **6-DOF Control**: Independent current control for each magnetic coil
- **Multi-Objective Optimization**: Beta, confinement time, uniformity, efficiency
- **Physics-Based Reward**: Incorporates safety constraints

### Comprehensive Data Analysis
- **500 Episodes** of training/operation
- **50,000 Total Timesteps** (100 steps × 500 episodes)
- **Transient Plots**: Confinement time evolution, learning curves
- **Performance Metrics**: Beta, τ, uniformity, fusion power
- **Correlation Analysis**: Parameter relationships

---

## 📁 File Structure

```
POLYWELL-RL-CONTROL/
├── setupFusionParameters.m           # Physics constants and parameters
├── runPolywellSimulation_500Episodes.m   # Main simulation runner
├── plasmaPhysicsBlock.m              # Plasma dynamics calculator
├── magneticFieldBlock.m              # B-field from coil currents
├── rewardCalculatorBlock.m           # RL reward function
├── plotPolywellTransients.m          # Transient plot generator
├── buildPolywellSimulinkModel.m      # (Optional) Simulink model builder
└── SIMULINK_SIMULATION_README.md     # This file
```

---

## 🚀 Quick Start

### Prerequisites
```matlab
% Required MATLAB Toolboxes:
% - Reinforcement Learning Toolbox
% - (Optional) Simulink for graphical modeling
```

### Run the Complete 500-Episode Simulation

```matlab
% Step 1: Setup physics parameters
setupFusionParameters

% Step 2: Run 500-episode simulation (takes 10-30 minutes)
runPolywellSimulation_500Episodes
```

**That's it!** The script will:
1. Load/create fusion physics parameters
2. Create or load RL agent
3. Run 500 episodes with full data logging
4. Generate transient analysis plots
5. Save all results

---

## 📊 Output Files

### Data Files (Created Automatically)
```
polywellFusionParams.mat              # Physics parameters
polywellAgent_500ep.mat               # Trained RL agent
polywellSimulation_500ep_data.mat     # All simulation data (~50-100 MB)
```

### Transient Plots (Auto-Generated)
```
ConfinementTime_Transient.png         # 6-panel confinement analysis
PlasmaParameters_Evolution.png        # 9-panel parameter evolution
EpisodeTimeSeries.png                 # Sample episode time series
FusionPerformance.png                 # Performance summary
```

---

## 📈 Understanding the Results

### Confinement Time Transient Plot

Shows 6 subplots:

1. **Confinement Time vs Episode**
   - Blue line: Per-episode average
   - Red line: 20-episode moving average
   - Black dashed: Target (10 ms)
   - **Expected**: Rise from ~3-5 ms → 8-10 ms

2. **Distribution**
   - Histogram of all 500 episodes
   - Shows concentration around target value

3. **Learning Phase (Ep 1-100)**
   - Initial exploration and rapid improvement
   - High variability early on

4. **Steady State (Ep 401-500)**
   - Optimized control with low variance
   - Tight clustering around target

5. **Improvement Rate**
   - Rolling window improvement percentage
   - Shows learning velocity

6. **Cumulative Average**
   - Running mean over all episodes
   - Demonstrates convergence

---

## 🔬 Physical Parameters

### Plasma Conditions
| Parameter | Value | Units | Notes |
|-----------|-------|-------|-------|
| Temperature | 30 | keV | Typical for D-D fusion |
| Density (avg) | 5×10¹⁸ | m⁻³ | IEC-achievable range |
| Magnetic Field | 0.3 | T | Design value |
| Plasma Beta | 0.4 | - | Target value |
| Confinement Time | 10 | ms | Target value |
| Reactor Volume | 0.524 | m³ | 50 cm radius sphere |

### Fusion Performance
| Metric | Value | Units |
|--------|-------|-------|
| <σv> D-D @ 30 keV | ~1.0×10⁻²² | m³/s |
| Reaction Rate | ~10⁶-10⁸ | reactions/s |
| Fusion Power | 10⁻⁶-10⁻³ | W |
| n×τ | ~5×10¹⁸ | s/m³ |
| Lawson (D-D) | 10²¹ | s/m³ |

### Magnetic System
| Parameter | Value | Units |
|-----------|-------|-------|
| Number of Coils | 6 | - |
| Coil Radius | 15 | cm |
| Nominal Current | 2500 | A |
| Maximum Current | 5000 | A |
| Field Range | 0.05 - 0.8 | T |

---

## 🧮 Physics Models Implemented

### 1. Magnetic Field Calculation
```
B = (μ₀/2) × Σᵢ (Iᵢ × R²) / (R² + zᵢ²)^(3/2)
```
- Biot-Savart law for circular current loops
- Superposition of 6 coils
- Saturation effects at high current

### 2. Plasma Beta
```
β = P_plasma / P_magnetic
P_plasma = n × k × T
P_magnetic = B² / (2μ₀)
```
- Ratio of plasma to magnetic pressure
- MHD stability limit: β < 0.8

### 3. Confinement Time (Empirical Scaling)
```
τ = τ₀ × β × U × (B/B₀)^1.5
```
Where:
- τ₀ = base confinement time
- U = field uniformity (0-1)
- B/B₀ = normalized field strength

### 4. Fusion Reaction Rate
```
R = 0.5 × n² × <σv> × V_eff
<σv> = 2.33×10⁻²⁰ × T^(-2/3) × exp(-18.76 × T^(-1/3))
```
- Bosch-Hale parameterization for D-D
- Temperature-dependent cross-section

### 5. Power Balance
```
P_fusion = R × E_avg                    (3.65 MeV/reaction)
P_brem = 5.35×10⁻³⁷ × n² × √T × V     (Bremsstrahlung)
P_sync = 6.2×10⁻¹⁷ × n × T^2.5 × B² × V (Synchrotron)
P_cond = (3nkTV) / τ                    (Conduction)
```

---

## 🎮 RL Control Strategy

### State Space (9 Dimensions)
```matlab
state = [I₁/I_max, I₂/I_max, ..., I₆/I_max,  % Normalized currents
         β,                                    % Plasma beta
         τ × 100,                              % Scaled confinement
         U]                                    % Uniformity
```

### Action Space (6 Dimensions)
```matlab
action = [ΔI₁, ΔI₂, ..., ΔI₆]  % Current changes, ±500 A/step
```

### Reward Function
```matlab
R = 10 × (β accuracy) +         % Weight: 10 (highest priority)
    5 × (τ performance) +        % Weight: 5
    2 × (uniformity) -           % Weight: 2
    0.1 × (power usage) +        % Weight: 0.1
    safety_penalties             % Prevent β>0.6, I>I_max
```

---

## 📖 Detailed Usage Examples

### Example 1: Run Full Simulation
```matlab
% Complete 500-episode run
runPolywellSimulation_500Episodes

% Check results
load('polywellSimulation_500ep_data.mat');
fprintf('Final avg beta: %.3f\n', summary.avgBeta(end));
fprintf('Final avg tau: %.3f ms\n', summary.avgTau(end)*1000);
```

### Example 2: Re-Plot Data
```matlab
% Load saved data
load('polywellSimulation_500ep_data.mat');

% Regenerate all plots
plotPolywellTransients(data, summary, params);
```

### Example 3: Analyze Specific Episodes
```matlab
load('polywellSimulation_500ep_data.mat');

% Plot episode 250
episode = 250;
time = (1:100) * params.control.dt;

figure;
subplot(2,1,1);
plot(time, data.beta(episode, :));
title(sprintf('Beta - Episode %d', episode));
xlabel('Time (s)'); ylabel('Beta');

subplot(2,1,2);
plot(time, data.tau_c(episode, :)*1000);
title(sprintf('Confinement Time - Episode %d', episode));
xlabel('Time (s)'); ylabel('τ (ms)');
```

### Example 4: Export Data to CSV
```matlab
load('polywellSimulation_500ep_data.mat');

% Create table with summary data
T = table((1:500)', summary.avgBeta, summary.avgTau*1000, ...
          summary.avgUniformity, summary.episodeReward, ...
          'VariableNames', {'Episode', 'AvgBeta', 'AvgTau_ms', ...
                           'AvgUniformity', 'EpisodeReward'});

% Write to CSV
writetable(T, 'polywellResults_500ep.csv');
fprintf('Data exported to: polywellResults_500ep.csv\n');
```

---

## 🔧 Customization

### Modify Physics Parameters

Edit `setupFusionParameters.m`:

```matlab
% Change plasma temperature
plasma.T_keV = 50;  % 50 keV instead of 30 keV

% Change target beta
plasma.beta_target = 0.5;  % Instead of 0.4

% Change reactor size
geometry.reactorRadius = 0.75;  % 75 cm instead of 50 cm
```

### Modify RL Training

Edit `runPolywellSimulation_500Episodes.m`:

```matlab
% Change number of episodes
numEpisodes = 1000;  % Instead of 500

% Change steps per episode
env.MaxSteps = 200;  % Instead of 100

% Change network size
fullyConnectedLayer(256, ...)  % Instead of 128
```

### Modify Reward Function

Edit `rewardCalculatorBlock.m`:

```matlab
% Change reward weights
weights = [15, 5, 2, 0.1];  % More emphasis on beta

% Add new reward component
fusionPowerReward = log10(fusionPower + 1e-10);
reward = reward + 0.5 * fusionPowerReward;
```

---

## 🐛 Troubleshooting

### Issue: Simulation runs slowly

**Solutions:**
1. Reduce episodes: `numEpisodes = 100;`
2. Reduce steps: `env.MaxSteps = 50;`
3. Use smaller network: `fullyConnectedLayer(64, ...)`
4. Use GPU: `executionEnvironment = 'gpu'` (if available)

### Issue: Agent not learning (reward not improving)

**Check:**
1. Reward function balance (adjust weights)
2. Learning rates (try higher: 1e-3)
3. Exploration (increase entropy weight)
4. Target values (make sure they're achievable)

### Issue: Out of memory

**Solutions:**
1. Reduce episodes
2. Save data incrementally
3. Use lower precision: `single()` instead of `double()`
4. Clear intermediate variables

### Issue: Plots not generating

**Check:**
1. Data was saved: `exist('polywellSimulation_500ep_data.mat')`
2. Function path: `which plotPolywellTransients`
3. Figure handles: Close all figures before replotting
4. Permissions: Check write permissions in directory

---

## 📚 References

### Fusion Physics
1. Lawson, J.D. (1957). "Some Criteria for a Power Producing Thermonuclear Reactor"
2. Bosch, H.S. & Hale, G.M. (1992). "Improved formulas for fusion cross-sections and thermal reactivities"
3. Bussard, R.W. (1991). "Some physics considerations of magnetic inertial-electrostatic confinement"

### RL Algorithms
1. Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms"
2. Degrave, J. et al. (2022). "Magnetic control of tokamak plasmas through deep reinforcement learning"

### Plasma Physics
- NRL Plasma Formulary (2019)
- Freidberg, J.P. "Plasma Physics and Fusion Energy"

---

## 💾 Data Structure Reference

### `data` Structure
```matlab
data.beta           [500 × 100]  % Plasma beta over time
data.tau_c          [500 × 100]  % Confinement time (s)
data.uniformity     [500 × 100]  % Field uniformity
data.fusionPower    [500 × 100]  % Fusion power (W)
data.B_field        [500 × 100]  % Magnetic field (T)
data.coilCurrents   [500 × 100 × 6]  % All coil currents (A)
data.rewards        [500 × 100]  % Reward signal
```

### `summary` Structure
```matlab
summary.episodeReward      [500 × 1]  % Total reward per episode
summary.avgBeta            [500 × 1]  % Episode-averaged beta
summary.avgTau             [500 × 1]  % Episode-averaged τ
summary.avgUniformity      [500 × 1]  % Episode-averaged uniformity
summary.avgFusionPower     [500 × 1]  % Episode-averaged power
```

---

## ✨ Key Results to Expect

### Learning Progression
| Phase | Episodes | Avg Beta | Avg τ (ms) | Characteristics |
|-------|----------|----------|------------|-----------------|
| **Initial** | 1-50 | 0.20-0.25 | 3-5 | Random exploration |
| **Learning** | 51-200 | 0.25-0.35 | 5-7 | Discovering patterns |
| **Improving** | 201-400 | 0.35-0.39 | 7-9 | Refining control |
| **Optimized** | 401-500 | 0.38-0.42 | 9-11 | Near-optimal, stable |

### Performance Improvement
- **Beta**: 2-3× improvement (0.20 → 0.40)
- **Confinement**: 2-3× improvement (3 ms → 9 ms)
- **Uniformity**: 1.5-2× improvement (0.5 → 0.9)
- **Reward**: 5-10× improvement (-5 → +15)

---

## 🎓 For Academic Use

### Citation
If using this code for research, please cite:
```
Polywell Fusion Reactor RL Control System
GitHub: [Your Repository]
Year: 2025
```

### Thesis/Report Sections

**Methods**: Describe PPO algorithm, physics models, reward function
**Results**: Include transient plots, performance tables, correlation analysis
**Discussion**: Learning progression, physical interpretations, limitations

### Suggested Experiments
1. **Temperature Variation**: Test at 20, 30, 40, 50 keV
2. **Density Scaling**: Vary from 10¹⁸ to 10¹⁹ m⁻³
3. **Reward Tuning**: Try different weight combinations
4. **Algorithm Comparison**: Test PPO vs DDPG vs SAC
5. **Network Architecture**: Compare shallow vs deep networks

---

## 🚀 Next Steps

### Enhancements
- [ ] Add D-T fusion option (higher cross-section)
- [ ] Implement bootstrap current effects
- [ ] Add turbulence models (gyro-kinetic effects)
- [ ] Multi-agent control (cooperative coils)
- [ ] Real-time constraints (actuator dynamics)

### Validation
- [ ] Compare with experimental IEC/Polywell data
- [ ] Benchmark against PID controller
- [ ] Sensitivity analysis on physics parameters
- [ ] Uncertainty quantification

---

## 📞 Support

For issues or questions:
1. Check this README thoroughly
2. Review error messages in MATLAB console
3. Verify all files are in correct directory
4. Check MATLAB/toolbox versions

---

**Happy simulating! May your plasma be well-confined! 🔥⚛️**
