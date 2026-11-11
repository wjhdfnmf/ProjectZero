# The problem: Booster Landing

**The task:** a reusable rocket booster is falling back to Earth after a mission. Using only its main engine, it must touch down **upright and gently on a small landing pad**, which in hard mode is a drone ship rocking in the sea. A neural network is the flight computer. Every 0.05 s it reads the rocket's sensors and sets the engine's throttle and steering.

Nobody tells the network *how* to land. It only gets a score at the end of each flight, and evolution has to discover a landing strategy that works across thousands of different starting situations.

Everything below is defined in [`rocket.cuh`](rocket.cuh) (physics, sensors, commands, scoring) and [`rocket.cu`](rocket.cu) (random scenarios, training, testing).

---

## 1. The vehicle

| Property | Value | Real-world comparison |
|---|---|---|
| Dry mass | 20,000 kg | A Falcon 9 first stage is ~25 t empty |
| Propellant at start | 4,000–7,000 kg (random) | Landing burns typically keep a few tonnes |
| Max thrust | 600 kN | Thrust-to-weight 2.2 (full) to 2.9 (nearly empty) |
| Specific impulse | 300 s | Kerosene engine at sea level ≈ 282 s |
| Fuel burn at full thrust | ~204 kg/s | So 4 t of propellant lasts ~20 s at full thrust |
| Engine gimbal | ±0.15 rad (±8.6°) | Tilting the engine both steers and rotates the rocket |
| Engine to center of mass | 20 m | Gives the gimbal a long lever arm |
| Length (for rotation inertia) | 40 m | Rotational inertia = m·L²/12 |
| Air drag | F = 20 kg/m × \|v_air\| × v_air | Terminal velocity ~110 m/s; wind pushes the rocket sideways |

**Physics:** a 2D rigid body, stepped every **0.05 s** (semi-implicit Euler), for at most **80 s** (1,600 steps). The forces are engine thrust (along the rocket's axis, tilted by the gimbal), gravity (9.81 m/s²) and air drag relative to the wind. The gimbal also creates a torque that rotates the rocket. Mass drops as propellant burns, and with no propellant left the engine produces nothing.

## 2. Starting situations (random every flight)

| | Range |
|---|---|
| Altitude | 1,500–2,500 m |
| Sideways distance from pad | −600 to +600 m |
| Sideways speed | −40 to +40 m/s |
| Falling speed | 50–90 m/s |
| Tilt | ±0.25 rad (±14°), spinning at up to ±0.05 rad/s |
| Propellant | 4,000–7,000 kg |
| Steady wind | ±15 m/s (easy) / ±20 m/s (hard) |
| Gusts | sine-wave gusts of up to 8 m/s (easy) / 12 m/s (hard), period 6–31 s |
| Drone ship (hard mode only) | pad sways sideways up to ±5 m (`ship=5`, the default) with an 8–14 s period: deck speed up to 3.9 m/s, acceleration up to 3.1 m/s² |

The rocket **cannot sense the wind**. It only sees its effect, as drift.

## 3. What the network sees (inputs)

Every 0.05 s, all scaled to roughly −1…1:

| # | Input | Scale |
|---|---|---|
| 1 | Sideways position relative to the pad | ÷ 500 m |
| 2 | Altitude | ÷ 1,000 m |
| 3 | Sideways speed relative to the pad | ÷ 50 m/s |
| 4 | Vertical speed | ÷ 50 m/s |
| 5–6 | sin and cos of tilt | avoids a jump at ±180° |
| 7 | Rotation rate | rad/s |
| 8 | Propellant left | ÷ 7,000 kg |
| 9 | *(hard)* Drone ship's sideways speed | ÷ 10 m/s |
| 10 | *(hard)* Actual engine throttle | 0–1 |
| 11–22 | *(with memory, `hist=1`)* The previous step's inputs 1–10, plus the previous throttle and gimbal command | Lets the network work out accelerations, such as the wind's push, from the change between two steps |

## 4. What the network controls (outputs)

Two outputs, each −1…1 (tanh):

| Output | Easy mode | Hard mode |
|---|---|---|
| Throttle | 0–100% of 600 kN | Below −0.2 = **engine off**. Otherwise **40–100%**: a lit engine can't go lower, like real rocket engines. Changes take effect with a **0.3 s lag**; shutdown is instant. |
| Gimbal | ±8.6° | ±8.6° |

**Why the 40% minimum matters:** near the ground the rocket weighs about 21 t, so even 40% thrust (240 kN) exceeds its weight (~206 kN). **It cannot hover.** It has to begin a single braking burn at exactly the right moment so its speed reaches zero just as it touches the deck. SpaceX calls this a "hoverslam". Start too early and it stops in mid-air and has to cut the engine and fall again. Start too late and it crashes.

## 5. Success and scoring

**A landing succeeds** only if, at touchdown:
- it's within **15 m** of the pad's center (the pad is 30 m wide), measured from the ship's position at that moment
- it's descending at **3 m/s or less**
- it's moving sideways at **1.5 m/s or less** relative to the pad
- it's tilted **5° or less**
- it's rotating at **0.2 rad/s or less**

**Other endings:**
- **crashed:** touched down but missed any of the limits above
- **tipped:** tilted past 90° in flight
- **out of bounds:** more than 3 km sideways or above 4 km
- **timeout:** still flying after 80 s

**Cost** (training minimizes this, so lower is better):

| Ending | Cost |
|---|---|
| Landed | 0.02·\|x\| + impact speed + 20·\|tilt\| + 5·\|rotation\| **− 30 − 10 × (share of propellant left)** |
| Crashed | 0.02·\|x\| + impact speed + 20·\|tilt\| + 5·\|rotation\| |
| Timeout | 80 + 0.02·distance to pad + 0.2·\|vertical speed\| |
| Tipped / out of bounds | 100 + 0.02·distance to pad |

The cost is **graded**, so evolution gets a signal from the very first, terrible flights: crashing at 30 m/s scores better than crashing at 100 m/s. The bonus for propellant left rewards efficient flying, like a late, hard burn.

## 6. How a network is judged

- **Training:** every generation flies a fresh batch of 16–32 random scenarios. A network's score is its average cost over the batch.
- **Choosing the final network:** the best candidates are re-flown on **500 validation scenarios**, a fixed set never used for training, and the best one is kept.
- **Final test:** the chosen network flies **1,000 test scenarios** it has never seen, from a third, separate random seed. The **landing rate on this test set** is the headline number in the report.
- **Cross-check:** the first 200 test flights are re-flown on the CPU with a separate implementation of the network. Every reported run matched the GPU on all 200 outcomes except one, at 199/200. The GPU uses fused multiply-add and its own `tanhf`, so tiny rounding differences can grow over a 50-second flight and flip a borderline landing.

## 7. Easy vs hard mode

| | Easy | Hard |
|---|---|---|
| Landing pad | Fixed | Drone ship, sways up to ±5 m |
| Throttle | 0–100%, instant | Off or 40–100%, 0.3 s lag → can't hover |
| Wind | ±15 m/s + 8 m/s gusts | ±20 m/s + 12 m/s gusts |
| What a good network does | Free-falls, brakes hard, then descends slowly to the pad | Has to time one braking burn precisely while tracking a moving target |

Easy mode turned out to be solvable by a tiny 90-weight network (99.5–100% landings), so hard mode was added to give larger networks something to win at.

**A correction made overnight:** the first version of hard mode swayed the ship up to ±20 m (`ship=20`). That's a deck moving at up to 15.7 m/s and accelerating at up to 12 m/s², far beyond a real drone ship. 75% of that version's crashes were gentle touchdowns on the pad that were still sliding sideways relative to the deck, so it measured luck more than skill. The default is now ±5 m. The `ship=20` results are kept in the report as a lesson.

## 8. Real mode (`mode=real`, added day 2)

The overnight flagship found two weaknesses in hard mode:
- **It relit the engine 35–115 times per flight** to fake a hover. Measured on the old hard-mode models, that's an average of 60 starts per flight.
- **The score rewarded saving propellant almost as much as landing.**

Real mode is hard mode plus five changes. Each one has a setting, so their effects can be measured separately:

| Change | Setting (real-mode default) | Details |
|---|---|---|
| **Limited engine starts** | `ignitions=3` | The engine can be lit at most 3 times per flight. Once they're used up, it stays off. The network sees how many starts it has left. This forces a genuine **single-burn hoverslam**, or at most a couple of corrections. |
| **Safety-first score** | `score=1` | Landing bonus 60, up from 30; propellant bonus at most 5, down from 10; any crash costs an extra 20. A safe landing now clearly outweighs any fuel saving. |
| **Sensor noise** | `noise=1` | Gaussian noise roughly matching GNSS + IMU fusion: position ±2 m, altitude ±1 m, velocities ±0.3 m/s, tilt and rotation ±0.005, propellant ±14 kg, deck motion ±0.1 / ±0.05 m/s. The engine's own state (throttle, starts left) is known exactly. |
| **Sensor delay** | `delay=2` | Sensor readings arrive 2 steps (0.1 s) late. |
| **Deck heave** | `heave=1.5` | The deck also moves up and down, by up to ±1.5 m over a 6–10 s period. Touchdown happens when the rocket reaches the *moving* deck, and the 3 m/s limit applies to vertical speed relative to the deck. The network sees altitude and vertical speed relative to the deck, plus the deck's vertical speed. |

**Inputs in real mode (12):** the 10 hard-mode inputs, with altitude and vertical speed now relative to the deck, plus engine starts left and the deck's vertical speed.

The noise is drawn from a hash of (scenario, step, sensor), so a flight is exactly repeatable, and the CPU reference sees the same noise as the GPU.

## 9. Simplifications to be aware of

- **2D only:** no out-of-plane motion and no roll.
- **Simple aerodynamics:** drag acts on the rocket's center with one coefficient. There are no grid fins, no aerodynamic torque and no lift.
- **No landing legs or touchdown dynamics:** landing is judged at the instant the rocket reaches the deck.
- **The drone ship only sways sideways.** The deck doesn't heave or pitch.
- **Perfect sensors** in easy and hard mode: no noise and no delay, apart from the engine's lag. Real mode adds both. The wind is never observed.
- **Unlimited engine relights** in easy and hard mode. Real mode limits them to 3.
- **The deck doesn't pitch or roll.**
- **No landing legs and no touchdown dynamics.**

Real mode addressed the first two. The remaining realism gaps are deck pitch, landing legs and 3D.
