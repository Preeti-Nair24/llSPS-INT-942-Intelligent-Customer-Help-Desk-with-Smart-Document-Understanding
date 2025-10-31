# Quick Start Guide - Polywell RL Control

## 3-Step Quick Start (Without ANSYS Data)

If you don't have ANSYS Maxwell data yet, you can test the system with synthetic data:

### Step 1: Generate Example Data
```matlab
cd('POLYWELL-RL-CONTROL')
generateExampleData
```
This creates `maxwell_polywell_data.csv` with synthetic magnetic field data.

### Step 2: Train the Agent
```matlab
trainPolywellRLAgent
```
This will train for 500 episodes (2-6 hours depending on hardware).

### Step 3: Visualize Results
```matlab
visualizePolywellControl
```
Watch the RL agent control the coils in real-time!

---

## With ANSYS Maxwell Data

### Step 1: Export from ANSYS Maxwell

1. Complete your polywell simulation in ANSYS Maxwell
2. Export field data to CSV with columns: X, Y, Z, Bx, By, Bz, Bmag
3. Save as `maxwell_polywell_data.csv` in POLYWELL-RL-CONTROL folder

### Step 2: Train
```matlab
trainPolywellRLAgent
```

### Step 3: Visualize
```matlab
visualizePolywellControl
```

---

## Expected Results

**Training:**
- Initial reward: around -10 to 0
- Final reward: 10-20 (after 200-500 episodes)
- Training time: 2-6 hours

**Visualization:**
- Coils change color based on current (red = high, blue = low)
- Plasma sphere grows/shrinks with beta value
- Real-time plots show convergence to optimal parameters

---

## Troubleshooting Quick Fixes

**Error: "File not found"**
```matlab
generateExampleData  % Create synthetic data
```

**Training too slow:**
```matlab
% Edit trainPolywellRLAgent.m, line 19:
numEpisodes = 100;  % Reduce from 500
```

**Visualization choppy:**
```matlab
% Edit visualizePolywellControl.m, line 162:
pause(0.2);  % Increase pause time
```

---

## What Each Visualization Shows

1. **3D Coils** - Position and current intensity (color)
2. **Plasma Sphere** - Size = beta value, transparency = confinement
3. **Coil Currents Bar Chart** - Real-time current in each coil
4. **Plasma Parameters** - Beta, confinement time, field uniformity
5. **Cumulative Reward** - How well the agent is performing
6. **Power Consumption** - Total energy usage

---

## Files You'll Create

- `maxwell_polywell_data.csv` - Magnetic field data (from ANSYS or generated)
- `trainedPolywellAgent.mat` - Trained RL agent (saved after training)

---

## Next Steps

1. Experiment with different reward weights in `PolywellRLEnvironment.m:128`
2. Adjust coil current limits in `PolywellRLEnvironment.m:28-30`
3. Try different network architectures in `trainPolywellRLAgent.m:66-104`
4. Export the trained agent for hardware deployment

---

## Questions?

See the full README.md for detailed documentation.
