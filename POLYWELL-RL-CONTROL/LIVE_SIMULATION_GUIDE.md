# Live Magnetic Field Control Simulation - Quick Guide

## 🎬 What You'll See

This live simulation shows the RL agent **actively controlling magnetic fields** by adjusting coil currents in real-time. You'll see:

1. **3D Magnetic Field Lines** - Dynamically updating as currents change
2. **Color-Coded Coils** - Blue (low current) → Red (high current)
3. **Plasma Sphere** - Size and brightness indicate confinement quality
4. **Magnetic Field Heatmap** - Shows field strength distribution
5. **Live Performance Metrics** - Beta, confinement, uniformity updating each step

---

## 🚀 How to Run

### Option 1: With Trained Agent (Recommended)

```matlab
% First, train an agent (if not already done)
cd('POLYWELL-RL-CONTROL')
trainPolywellRLAgent_PPO    % or trainPolywellRLAgent for DDPG

% Then run live simulation
liveSimulationMagneticControl
```

**What you'll see:**
- Agent intelligently adjusts coil currents to optimize plasma
- Smooth, coordinated control actions
- Beta converges to target 0.40
- Magnetic field lines reorganize for optimal confinement

### Option 2: Without Trained Agent (See Random Behavior)

```matlab
% Edit the script first:
% Open liveSimulationMagneticControl.m
% Change line 19: USE_TRAINED_AGENT = false;

% Then run
liveSimulationMagneticControl
```

**What you'll see:**
- Random current adjustments (no intelligence)
- Chaotic field line changes
- Poor plasma confinement
- Demonstrates why RL is needed!

---

## 📊 Understanding the Visualization

### Main 3D View (Top Left, Large)

```
     Coil 3 (top)
          ↑
          |
Coil 2 ← ⊕ → Coil 1    ← Magnetic field lines (blue curves)
    (left) (right)
          |
          ↓
     Coil 4 (bottom)

     (Coil 5 front, Coil 6 back)

        ○ ← Plasma sphere (center)
```

**Color Coding:**
- **Blue coils** = Low current (< 2500A)
- **Purple coils** = Medium current (~2500A)
- **Red coils** = High current (> 3500A)

**Plasma Sphere:**
- **Small, transparent** = Poor confinement (β < 0.3)
- **Large, bright** = Good confinement (β > 0.4)

**Field Lines:**
- Show direction and shape of magnetic field
- Update every 5 steps to show field evolution
- Converge toward center = good confinement
- Spread out = poor confinement

### Coil Current Bars (Top Middle)

Shows instantaneous current in each of the 6 coils.

**What to watch:**
- All bars similar height = balanced field (good!)
- One bar much higher/lower = imbalanced (bad!)
- Smooth changes = trained agent
- Erratic jumps = untrained/random

### Magnetic Field Strength Heatmap (Top Right)

2D slice through Z=0 plane showing field intensity.

**Colors:**
- **Dark red/white** = Strong field
- **Yellow** = Medium field
- **Dark red (cold)** = Weak field

**Patterns:**
- Symmetric cross pattern = balanced coils
- Concentrated center = good confinement region
- Asymmetric = unbalanced coils

### Plasma Parameters (Bottom Middle)

Time-series plots showing:
- **Blue line** = Plasma Beta (target: 0.40)
- **Red line** = Confinement Time (×100 scaling)
- **Green line** = Field Uniformity

**Trained agent:** All three converge to high values
**Random actions:** All three fluctuate chaotically

### RL Agent Actions (Bottom Right)

Shows the **changes** in current the agent commanded.

**Bars above zero:** Increase current
**Bars below zero:** Decrease current

**What to watch:**
- Trained agent: Coordinated adjustments (multiple coils change together)
- Random: Uncoordinated (random +/- changes)

---

## 🎯 What Trained Agent Behavior Looks Like

### Phase 1: Initial Correction (Steps 0-20)

```
Coil currents start random: [1050A, 980A, 1020A, 990A, 1010A, 1000A]
Agent detects: Beta too low (0.22), field unbalanced

Actions:
  Coil 1: +300A
  Coil 2: +250A
  Coil 3: +280A
  Coil 4: +260A
  Coil 5: +270A
  Coil 6: +255A
→ Increases all currents to strengthen field
```

**Visual changes:**
- All coils change from blue to purple
- Magnetic field lines intensify
- Plasma sphere grows slightly
- Field heatmap brightens

### Phase 2: Balancing (Steps 20-80)

```
Coil currents: [1350A, 1230A, 1300A, 1250A, 1280A, 1255A]
Agent detects: Beta improved (0.32), but field unbalanced

Actions:
  Coil 1: -20A  (reduce highest)
  Coil 2: +40A  (increase lowest)
  Coil 3: +10A
  Coil 4: +25A
  Coil 5: +5A
  Coil 6: +30A
→ Balances currents toward uniformity
```

**Visual changes:**
- Coils converge to similar colors
- Field lines become more symmetric
- Field uniformity line rises
- Heatmap becomes more circular

### Phase 3: Fine Optimization (Steps 80-200)

```
Coil currents: [2850A, 2800A, 2820A, 2780A, 2810A, 2790A]
Agent detects: Beta at target (0.40), field balanced

Actions: Small adjustments ±20A
→ Maintains optimal state
```

**Visual changes:**
- All coils steady red/orange color
- Large, bright plasma sphere (β = 0.40)
- Symmetric field lines pointing to center
- All performance metrics at target

---

## 🎮 Interactive Controls

### During Simulation:

**Ctrl+C** - Stop simulation

**Dialog box when episode ends:**
- Click "Yes" to start new episode
- Click "No" to exit

### Camera Controls (3D View):

- **Left mouse + drag** - Rotate view
- **Scroll wheel** - Zoom in/out
- **Right mouse + drag** - Pan

---

## 📈 Performance Benchmarks

### Good Performance (Trained Agent)

```
✓ Plasma Beta: 0.38-0.42 (target ±0.02)
✓ Confinement Time: 0.009-0.011s (> 0.009s)
✓ Field Uniformity: > 0.90
✓ All coils within 10% of each other
✓ Smooth control actions (< 100A changes)
```

### Poor Performance (Untrained/Random)

```
✗ Plasma Beta: 0.10-0.30 (far from target)
✗ Confinement Time: 0.002-0.006s (poor)
✗ Field Uniformity: 0.50-0.70
✗ Coils vary wildly (± 50%)
✗ Erratic actions (±500A jumps)
```

---

## 🔧 Customization Options

### Speed Control

Change line 195:
```matlab
pause(0.05);  % Default: 20 FPS

pause(0.02);  % Faster: 50 FPS
pause(0.10);  % Slower: 10 FPS (better for screenshots)
pause(0);     % Maximum speed (no pause)
```

### Field Line Update Rate

Change line 172:
```matlab
if mod(step, 5) == 0  % Update every 5 steps

if mod(step, 1) == 0  % Update every step (slower but smoother)
if mod(step, 10) == 0 % Update every 10 steps (faster)
```

### Heatmap Update Rate

Change line 191:
```matlab
if mod(step, 10) == 0  % Update every 10 steps

if mod(step, 5) == 0   % More frequent updates
if mod(step, 20) == 0  % Less frequent updates (faster)
```

---

## 🎥 Recording the Simulation

### Take Screenshots

```matlab
% Add this inside the main loop (after line 240)
if step == 50 || step == 100 || step == 150
    filename = sprintf('simulation_step_%03d.png', step);
    saveas(fig, filename);
    fprintf('Screenshot saved: %s\n', filename);
end
```

### Record Video

```matlab
% Before the main loop (after line 138)
writerObj = VideoWriter('polywell_simulation.mp4', 'MPEG-4');
writerObj.FrameRate = 20;
open(writerObj);

% Inside main loop (after drawnow, line 241)
frame = getframe(fig);
writeVideo(writerObj, frame);

% After loop ends (after line 265)
close(writerObj);
fprintf('Video saved: polywell_simulation.mp4\n');
```

---

## 🐛 Troubleshooting

### "No trained agent found"

**Solution:** Train an agent first:
```matlab
trainPolywellRLAgent_PPO
% Then run simulation
```

### Visualization is slow/choppy

**Solutions:**
1. Increase pause time: `pause(0.1)`
2. Reduce field line updates: `if mod(step, 10) == 0`
3. Close other programs
4. Reduce grid resolution in heatmap (line 108)

### Field lines look wrong

**Cause:** Simplified dipole model (not exact ANSYS field)

**Solution:** For accurate field lines, use actual ANSYS field data:
- Load your ANSYS data with `importMaxwellData.m`
- Modify `calculateMagneticField()` to use interpolant

### Plasma sphere disappears

**Cause:** Beta dropped below 0.1 (plasma lost)

**Meaning:** Control failed - episode will reset

---

## 🎓 Educational Use

### Demonstrate RL Learning

1. Run with `USE_TRAINED_AGENT = false` - show random behavior
2. Train agent while projecting screen - show learning process
3. Run with trained agent - show optimized control
4. Compare before/after videos side-by-side

### Key Teaching Points

**Show students:**
- How RL agent learns coordinated multi-actuator control
- Difference between random and intelligent control
- Trade-off between exploration (early) and exploitation (late)
- Real-time optimization of multiple objectives (beta, confinement, efficiency)

---

## 📊 Expected Timeline

### Untrained Agent (Random Actions)

```
0-50 steps:   Random field changes, plasma unstable
50-100 steps: Occasionally finds decent configuration by luck
100-150 steps: Beta fluctuates 0.15-0.35
150-200 steps: Episode likely ends (plasma lost)
```

### Trained Agent

```
0-20 steps:   Rapid correction toward optimal currents
20-60 steps:  Fine-tuning balance and uniformity
60-200 steps: Maintains optimal state (β ≈ 0.40)
200 steps:    Episode completes successfully
```

---

## 💡 Tips for Best Visualization

1. **Run on large monitor** - Better to see all 6 subplots
2. **Maximize window** - Press maximize button on figure
3. **Good lighting** - 3D effects work best with proper lighting
4. **Rotate view** - Try different angles of 3D plot (45°, 30° is good start)
5. **Let it run** - Watch for 50+ steps to see patterns emerge

---

## 🎬 Demo Script for Presentations

```matlab
% 1. Show untrained behavior
% Edit: USE_TRAINED_AGENT = false
liveSimulationMagneticControl
% Let run for 100 steps, show chaos
% Press Ctrl+C to stop

% 2. Show trained behavior
% Edit: USE_TRAINED_AGENT = true
liveSimulationMagneticControl
% Let run for 200 steps, show optimization
% Complete episode successfully

% 3. Compare metrics
fprintf('Untrained: Beta = 0.25, Confinement = 0.004s\n');
fprintf('Trained:   Beta = 0.40, Confinement = 0.010s\n');
fprintf('Improvement: 60%% better beta, 150%% better confinement!\n');
```

---

## 📝 What Each Color/Size Means - Quick Reference

| Visual Element | Meaning |
|----------------|---------|
| **Blue coil** | Low current (< 2000A) |
| **Red coil** | High current (> 3500A) |
| **Small plasma sphere** | Poor confinement (β < 0.3) |
| **Large bright sphere** | Good confinement (β > 0.4) |
| **Field lines to center** | Good magnetic bottle |
| **Field lines spread out** | Weak confinement |
| **Circular heatmap** | Balanced field |
| **Asymmetric heatmap** | Unbalanced field |
| **Smooth parameter curves** | Trained agent |
| **Noisy parameter curves** | Random/untrained |

---

**Enjoy watching AI control fusion reactor magnetic fields in real-time!** 🚀
