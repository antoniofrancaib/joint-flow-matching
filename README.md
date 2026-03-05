# Joint Discrete-Continuous MolFLM — Interactive Visualization

An **interactive scroll-driven website** that visually explains **Algorithm 3 (Joint Discrete-Continuous ODE Sampling)** — and potentially **Algorithm 2 (Flow Map Sampling)** — from the project [Joint Discrete-Continuous Flow Matching for Molecules](https://www.notion.so/Joint-Discrete-Continuous-Flow-Matching-for-Molecules-319e5960818f806e995ac0a40b9b8734?pvs=21). The goal is to pitch this research idea to a supervisor using an intuitive, polished, interactive web visualization rather than static slides or equations alone.

---

## Project Context

The core idea: a **single deterministic ODE** in $\mathbb{R}^{n \times (K+3)}$ simultaneously generates **discrete atom types** and **3D molecular coordinates** from Gaussian noise. We embed atom types as one-hot vectors in $\mathbb{R}^K$ and concatenate with 3D coordinates in $\mathbb{R}^3$, forming a joint state $\mathbf{y} = [\mathbf{e}\;\mathbf{x}]$. A single velocity field transports noise to data — no CTMC machinery, no factorization.

The website makes this tangible with a **dual-panel canvas** that evolves in lockstep as the user scrolls:

- **Left panel — "3D Coordinate Flow":** A shared pseudo-3D space showing all atoms as dots connected by edges, evolving from a random noise cloud into a structured molecule. The molecular graph emerges visually as atoms self-organize. Continuous slow rotation gives depth cues.
- **Right panel — "Atom Type Flow on Simplex":** A shared 2D probability simplex (equilateral triangle with vertices C, N, O) where atom dots flow from scattered positions near the centroid (softmax of Gaussian noise ≈ uniform) toward their target simplex vertices as discrete types sharpen.

Both panels are driven by the same scroll-controlled time parameter $t \in [0,1]$, making it obvious this is **one unified ODE**, not two separate processes.

---

## What We Are Visualizing

### Algorithm 3 — Joint Discrete-Continuous ODE Sampling

This is the primary algorithm being visualized:

```
Algorithm 3 — Joint Discrete-Continuous ODE Sampling

1:  input   trained joint denoiser D_θ (from Algorithm 1); number of steps N ≥ 1
2:  init    time grid {t_0, t_1, …, t_N} with t_m = t(m/N) via reparameterisation inverse
            ▷ Uniform in α-space (alpha_to_gamma LUT)
3:  Sample initial noise: y_{t_0} ~ N(0, I_{n(K+3)})
4:  for m = 0, …, N-1 do
5:      Compute velocity: b_{t_m}(y_{t_m}) = (D_θ(y_{t_m}, t_m) - y_{t_m}) / (1 - t_m)
            ▷ Evaluate denoiser to get instantaneous velocity
6:      Euler step: y_{t_{m+1}} = y_{t_m} + (t_{m+1} - t_m) · b_{t_m}(y_{t_m})
            ▷ Forward Euler integration; incurs discretisation error
7:  end for
8:  Split terminal state: y_{t_N} = [ŷ^disc ; ŷ^cont]
9:  Decode discrete atom types: ĉ^i = argmax_k ŷ^disc_{i,k}  for each i = 1, …, n
            ▷ Argmax decoding from continuous space
10: Read continuous coordinates: x̂^i = ŷ^cont_i ∈ R^3  for each i = 1, …, n
            ▷ Direct readout; no projection needed
11: return generated molecule (ĉ, x̂)
```

### Algorithm 2 — Joint Flow Map Sampling (Future Visualization)

Algorithm 2 uses a **distilled flow map** $\hat{X}_{s,t}$ instead of Euler integration. It jumps exactly along the learned trajectory with **no discretisation error** per step. At $N = 1$, it enables **one-step molecular generation** — the key differentiator of this project. A future visualization could contrast the Euler-step trajectory of Algorithm 3 with the exact flow-map jump of Algorithm 2.

```
Algorithm 2 — Joint Discrete-Continuous Flow Map Sampling

1:  input   trained flow map X̂_{s,t} (distilled from D_θ); number of steps N ≥ 1
2:  init    time grid {t_0, …, t_N} via reparameterisation inverse
3:  Sample initial noise: y_{t_0} ~ N(0, I_{n(K+3)})
4:  for m = 0, …, N-1 do
5:      y_{t_{m+1}} = X̂_{t_m, t_{m+1}}(y_{t_m})
            ▷ Exact flow map jump; no discretisation error
6:  end for
7:  Split: y_{t_N} = [ŷ^disc ; ŷ^cont]
8:  Decode: ĉ^i = argmax_k ŷ^disc_{i,k}
9:  Read:   x̂^i = ŷ^cont_i
10: return molecule (ĉ, x̂)
```

---

## Key Scientific Ideas

### 1. Joint Euclidean Embedding

Discrete atom types (C, N, O) are embedded as one-hot vectors in $\mathbb{R}^K$ and concatenated with 3D coordinates to form $\mathbf{y} \in \mathbb{R}^{n \times (K+3)}$. This enables a single ODE — no separate discrete/continuous processes.

### 2. Single Velocity Field

The ODE $\dot{\mathbf{y}}_t = b_t(\mathbf{y}_t) = \frac{D_\theta(\mathbf{y}_t, t) - \mathbf{y}_t}{1 - t}$ acts on the full joint state. At each instant, the denoiser $D_\theta$ predicts clean data and the velocity pushes the current noisy state toward it. Cross-modal correlations (atom type ↔ bond length) are captured by a shared backbone.

### 3. Discrete Component Sharpens via Softmax

The discrete part of the state lives in $\mathbb{R}^K$ (not on the simplex). Applying softmax gives a probability distribution over atom types that starts near-uniform at $t=0$ (softmax of Gaussian noise ≈ $1/K$ per class) and concentrates on a single vertex as $t \to 1$. The visualization shows this on the 2D simplex.

### 4. Argmax Decode at Terminal Time

At $t=1$, discrete types are decoded via $\hat{c}^i = \text{argmax}_k\;\hat{\mathbf{y}}^{\text{disc}}_{i,k}$ and coordinates are read out directly: $\hat{\mathbf{x}}^i = \hat{\mathbf{y}}^{\text{cont}}_i$. No post-processing or projection needed.

### 5. No CTMC, No Factorization

Unlike SemlaFlow/MultiFlow which couple separate discrete (CTMC-based) and continuous flows, this approach uses one unified velocity field. This eliminates factorization error and the entire CTMC apparatus (rate matrices, time-ordered exponentials, masking schedules).

---

## Toy Molecule Data

The visualization uses a 5-atom chain ($n=5$, $K=3$ types) loaded from `data/molecules.json`:

- **Atom types:** [C, C, N, O, O] — indices [0, 0, 1, 2, 2]
- **Structure:** A 5-atom chain
- **Bonds:** [[0,1], [1,2], [2,3], [3,4]]
- **Atom type colors:** C = `#E24A33` (warm red), N = `#348ABD` (blue), O = `#988ED5` (purple)

Trajectories are precomputed by `scripts/gen_trajectories.py` using an **oracle denoiser** $D(\mathbf{y}_t, t) = \mathbf{y}_1$ (the clean target). 60 Euler steps from $t=0$ to $t=1$. Run `scripts/generate_all.sh` to regenerate data.

---

## Architecture & Tech Stack

- **Vanilla HTML + CSS + Canvas API** — no frameworks, no build tools
- **KaTeX** for math rendering (loaded from CDN)
- **Google Fonts** — Inter (body) + JetBrains Mono (code/time indicator)
- **No other dependencies**

### File Structure

| File | Purpose |
|------|---------|
| `index.html` | Page structure: hero, static method explainer, time slider, dual-panel canvas, footer |
| `style.css` | Layout, typography, scroll-step transitions, responsive breakpoints |
| `main.js` | Animation logic: data fetch, dual-panel renderer, scroll observer, animation loop |
| `data/molecules.json` | Molecule configs (types, coords, bonds, labels) |
| `data/trajectories/*.json` | Precomputed ODE trajectories per molecule |
| `scripts/gen_molecules.py` | Generate molecule configs → `data/molecules.json` |
| `scripts/gen_trajectories.py` | Generate ODE trajectories → `data/trajectories/<name>.json` |
| `scripts/generate_all.sh` | Regenerate all data (run once to create/update JSON files) |

### CSS Layout Pattern

- **Hero** — Centered title and abstract
- **Method explainer** — Static text (~720px max-width) with inline KaTeX equations
- **Visualization** — Time slider (0→1) + dual-panel canvas (~70vh)
- **Footer** — Algorithm 3 pseudocode box

### JS Architecture

- **`DualPanelRenderer`** — Canvas with HiDPI support and primitives (dot, label, curve, curveSmooth, bary, dashedLine). Two panels: 3D coordinates (left) and simplex (right).
- **`SectionMolFlow`** — Controller with `draw(dt)`. Time $t$ comes from the **slider** (0–1), not scroll. Draws full trajectory curves always; current position moves with slider.
- **Slider** — `<input type="range">` 0–1000 maps to $t \in [0,1]$. No scroll observer.
- **Data loading** — Fetches `data/molecules.json` and `data/trajectories/<name>.json`.

### Canvas Split

- **Left half** [0, midX): 3D coordinate flow with pseudo-3D projection (rotateY + rotateX + weak perspective), continuous rotation via `performance.now()`
- **Right half** [midX, width]: 2D simplex with equilateral triangle, softmax mapping from $\mathbb{R}^3$ to barycentric coordinates
- 1px faint divider at midpoint, panel subtitles at top, shared time indicator "t = 0.XX" centered at top

---

## Color Palette

| Token | Hex | Usage |
|-------|-----|------|
| `bg` | `#ffffff` | Page background |
| `ink` | `#2e4552` | Text, edges, axes |
| `teal` | `#2f533f` | Accent, rule, time indicator |
| `simplexFill` | `#f2eee0` | Simplex triangle fill |
| `cloud` | `#90a0aa` | Noise dots, faint elements |
| `gold` | `#d4a843` | Velocity arrows (currently using cloud) |
| `typeC` | `#E24A33` | Carbon — warm red |
| `typeN` | `#348ABD` | Nitrogen — blue |
| `typeO` | `#988ED5` | Oxygen — purple |

---

## Design Inspiration

The visual style is modelled after the **Discrete Flow Maps project page** (Lee et al., 2026) — a scroll-driven two-column layout with sticky canvas animations, clean white background, Inter font, muted color palette, and smooth eased transitions. The architecture follows the same class structure: a Renderer wrapping the canvas, an Anim class for step-based easing, section controller classes, IntersectionObserver scroll detection, and a requestAnimationFrame loop.

---

## Current State & Known Issues

The initial version (v0) has the basic dual-panel visualization working. Areas for improvement:

- **Velocity arrows** currently use `cloud` color — could switch to `gold` (`#d4a843`) for more visual pop
- **Trail coloring** could use a blue→red time gradient instead of a flat lerp to the type color
- **3D rotation** is time-driven (always rotates) — consider also adding a slight tilt parameter
- **Edge appearance** is purely opacity-based — could also transition line width and color from faint gray to solid ink
- **Simplex dot scatter at t=0** depends on the actual Gaussian noise — check if the centroid clustering is visually clear enough
- **Step 3 glow/pulse** could be more dramatic (breathing radial glow behind the molecule)
- **Responsive layout** collapses to single column below 900px — canvas moves on top

---

## Future Directions

### Algorithm 2 Visualization

Add a second scroll section below the current one that visualizes **flow map sampling**. Show the contrast: Algorithm 3's N Euler steps vs Algorithm 2's exact flow-map jump. At $N=1$, Algorithm 3 gives a single bad Euler step, while Algorithm 2 applies the full distilled flow map for high-quality one-step generation.

### Training Visualization (Algorithm 1)

A possible third section could show how the denoiser is trained: sample a random time $t$, construct the interpolant $\mathbf{y}_t = (1-t)\mathbf{y}_0 + t\mathbf{y}_1$, and train with cross-entropy (discrete) + MSE (continuous).

### Distillation Visualization (Algorithm 4)

Visualize the semigroup condition: how the flow map $\hat{X}_{s,t}$ is distilled by requiring that jumping $s \to t$ equals jumping $s \to u$ then $u \to t$.

### Interactive Controls

- Manual time slider instead of (or in addition to) scroll
- Toggle between Algorithm 2 and Algorithm 3
- Adjustable NFE count to show quality degradation at low step counts

---

## How to Run

The site works out of the box: `data/molecules.json` and `data/trajectories/` are committed. To regenerate data:

```bash
./scripts/generate_all.sh
```

Then serve with any static file server:

```bash
# Python
python -m http.server 8000

# Node
npx serve .

# Or open index.html directly (fetch may fail with file:// protocol)
```

No build step required. All dependencies are loaded from CDN.
