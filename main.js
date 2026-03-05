/* ═══════════════════════════════════════════════════════════
   Molecular Flow Matching — Slider-driven Canvas
   Joint Discrete-Continuous ODE Sampling (Algorithm 3)
   ═══════════════════════════════════════════════════════════ */

'use strict';

const C = {
  bg: '#ffffff',
  ink: '#2e4552',
  teal: '#2f533f',
  simplexFill: '#f2eee0',
  cloud: '#90a0aa',
  gold: '#d4a843',
  typeC: '#E24A33',
  typeN: '#348ABD',
  typeO: '#988ED5',
  denoiser: '#2f533f',
};

const DEFAULT_TYPE_COLORS = [C.typeC, C.typeN, C.typeO];

const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function chaikinSmooth(points, iterations) {
  if (iterations === undefined) iterations = 2;
  if (!points || points.length < 3 || iterations <= 0) return points ? points.slice() : [];
  let pts = points.map(p => [p[0], p[1]]);
  for (let k = 0; k < iterations; k++) {
    if (pts.length < 3) break;
    const next = [pts[0]];
    for (let i = 0; i < pts.length - 1; i++) {
      const p = pts[i], q = pts[i + 1];
      next.push([0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1]]);
      next.push([0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1]]);
    }
    next.push(pts[pts.length - 1]);
    pts = next;
  }
  return pts;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

// ════════════════════════════════════════════════════════════
//  DUAL-PANEL RENDERER
// ════════════════════════════════════════════════════════════
class DualPanelRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;
    this.resize();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.w = rect.width;
    this.h = rect.height;
    this.canvas.width = this.w * this.dpr;
    this.canvas.height = this.h * this.dpr;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    this.midX = this.w / 2;
    this.leftCx = this.w * 0.25;
    this.leftCy = this.h * 0.5;
    this.rightCx = this.w * 0.75;
    this.rightCy = this.h * 0.45; // shift up slightly to leave room below simplex
    this.scale3D = Math.min(this.w * 0.22, this.h * 0.3);

    // Simplex: positioned higher to allow starting points below
    const simplexScale = Math.min(this.w * 0.18, this.h * 0.28);
    this.simplexScale = simplexScale;
    const sCy = this.rightCy - simplexScale * 0.1;
    this.vC = [this.rightCx - simplexScale * 1.1, sCy + simplexScale * 0.7];
    this.vN = [this.rightCx, sCy - simplexScale * 1.0];
    this.vO = [this.rightCx + simplexScale * 1.1, sCy + simplexScale * 0.7];

    // Centroid of simplex for reference
    this.simplexCentroid = [
      (this.vC[0] + this.vN[0] + this.vO[0]) / 3,
      (this.vC[1] + this.vN[1] + this.vO[1]) / 3,
    ];
    // Starting region below simplex (offset downward)
    this.startRegionY = this.vC[1] + simplexScale * 1.0;
  }

  clear() {
    this.ctx.clearRect(0, 0, this.w, this.h);
  }

  dot(pos, color, r, opacity) {
    if (r === undefined) r = 5;
    if (opacity === undefined) opacity = 1;
    this.ctx.save();
    this.ctx.globalAlpha = opacity;
    this.ctx.beginPath();
    this.ctx.arc(pos[0], pos[1], r, 0, Math.PI * 2);
    this.ctx.fillStyle = color;
    this.ctx.fill();
    this.ctx.restore();
  }

  label(text, pos, opts) {
    opts = opts || {};
    const color = opts.color || C.ink;
    const size = opts.size || 12;
    const bold = opts.bold !== undefined ? opts.bold : true;
    const align = opts.align || 'center';
    const baseline = opts.baseline || 'middle';
    const opacity = opts.opacity !== undefined ? opts.opacity : 1;
    const font = opts.mono ? 'var(--font-mono)' : 'Inter, sans-serif';
    this.ctx.save();
    this.ctx.globalAlpha = opacity;
    this.ctx.font = (bold ? '600' : '400') + ' ' + size + 'px ' + font;
    this.ctx.fillStyle = color;
    this.ctx.textAlign = align;
    this.ctx.textBaseline = baseline;
    this.ctx.fillText(text, pos[0], pos[1]);
    this.ctx.restore();
  }

  curve(pts, color, lw, opacity) {
    if (!pts || pts.length < 2) return;
    if (opacity === undefined) opacity = 1;
    this.ctx.save();
    this.ctx.globalAlpha = opacity;
    this.ctx.beginPath();
    this.ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) this.ctx.lineTo(pts[i][0], pts[i][1]);
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = lw || 2;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    this.ctx.stroke();
    this.ctx.restore();
  }

  curveSmooth(pts, color, lw, opacity) {
    if (!pts || pts.length < 2) return;
    if (pts.length < 4) {
      this.curve(pts, color, lw, opacity);
      return;
    }
    this.curve(chaikinSmooth(pts, 2), color, lw, opacity);
  }

  dashedLine(from, to, color, opacity, dashPattern) {
    this.ctx.save();
    this.ctx.globalAlpha = opacity !== undefined ? opacity : 1;
    this.ctx.setLineDash(dashPattern || [6, 4]);
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    this.ctx.moveTo(from[0], from[1]);
    this.ctx.lineTo(to[0], to[1]);
    this.ctx.stroke();
    this.ctx.restore();
  }

  arrow(from, to, color, lineWidth, opacity) {
    const dx = to[0] - from[0];
    const dy = to[1] - from[1];
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 1) return;
    const ux = dx / len;
    const uy = dy / len;
    const headLen = Math.min(6, len * 0.35);

    this.ctx.save();
    this.ctx.globalAlpha = opacity !== undefined ? opacity : 1;
    this.ctx.strokeStyle = color;
    this.ctx.fillStyle = color;
    this.ctx.lineWidth = lineWidth || 1.5;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';

    // Line
    this.ctx.beginPath();
    this.ctx.moveTo(from[0], from[1]);
    this.ctx.lineTo(to[0], to[1]);
    this.ctx.stroke();

    // Arrowhead
    this.ctx.beginPath();
    this.ctx.moveTo(to[0], to[1]);
    this.ctx.lineTo(to[0] - headLen * ux + headLen * 0.4 * uy, to[1] - headLen * uy - headLen * 0.4 * ux);
    this.ctx.lineTo(to[0] - headLen * ux - headLen * 0.4 * uy, to[1] - headLen * uy + headLen * 0.4 * ux);
    this.ctx.closePath();
    this.ctx.fill();

    this.ctx.restore();
  }

  bary(w0, w1, w2) {
    return [
      w0 * this.vC[0] + w1 * this.vN[0] + w2 * this.vO[0],
      w0 * this.vC[1] + w1 * this.vN[1] + w2 * this.vO[1],
    ];
  }
}

// ════════════════════════════════════════════════════════════
//  3D PROJECTION
// ════════════════════════════════════════════════════════════
function rotateY(p, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c];
}
function rotateX(p, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c];
}
function project3D(p, cx, cy, scale) {
  const zFac = 1 + p[2] * 0.12;
  return [cx + p[0] * scale * zFac, cy - p[1] * scale * zFac, p[2]];
}

// ════════════════════════════════════════════════════════════
//  SECTION CONTROLLER
// ════════════════════════════════════════════════════════════
class SectionMolFlow {
  constructor(renderer, molecule, trajData, getCurrentT) {
    this.r = renderer;
    this.getCurrentT = getCurrentT;
    this.rotAngle = 0;
    this.target = molecule;
    this.trajData = trajData;
    this.nAtoms = trajData.n_atoms;
    this.nSteps = trajData.n_steps;
    this.typeColors = DEFAULT_TYPE_COLORS.slice(0, trajData.n_types);
    this.typeLabels = molecule.labels || [];

    // Precompute simplex trajectory curves for the right panel
    this._precomputeSimplexCurves();
  }

  resize() {
    this._precomputeSimplexCurves();
  }

  _precomputeSimplexCurves() {
    // For each atom, compute the full simplex trajectory with starting points
    // positioned below the simplex (to match the reference figure aesthetic)
    const r = this.r;
    const traj = this.trajData.traj;
    const nSteps = this.nSteps;

    this.simplexCurves = [];
    // Waypoint times: t = 0, 0.25, 0.5, 0.75, 1.0
    this.waypointFracs = [0, 0.25, 0.5, 0.75, 1.0];

    for (let i = 0; i < this.nAtoms; i++) {
      const pts = [];
      const step = Math.max(1, Math.floor(nSteps / 50));
      for (let m = 0; m <= nSteps; m += step) {
        const s = traj[m];
        const tFrac = m / nSteps;
        const pos = this._simplexPos(s[i], tFrac, i);
        pts.push(pos);
      }
      // Ensure we include the final step
      if ((nSteps % step) !== 0) {
        const s = traj[nSteps];
        const pos = this._simplexPos(s[i], 1.0, i);
        pts.push(pos);
      }
      this.simplexCurves.push(pts);
    }
  }

  // Map a raw atom state to 2D simplex canvas coordinates.
  // At t=0, positions start below the simplex. As t->1, they approach simplex vertices.
  _simplexPos(atomState, tFrac, atomIdx) {
    const r = this.r;
    const disc = [atomState[0], atomState[1], atomState[2]];
    const probs = softmax(disc);

    // Barycentric position on simplex from softmax
    const simplexPos = r.bary(probs[0], probs[1], probs[2]);

    // At t=0, the softmax of Gaussian noise is near uniform (centroid).
    // To create the "starting outside/below" effect, we blend with a point below the simplex.
    // The offset decreases as t increases — at t=1 we're exactly on the simplex.
    const startOffset = this._atomStartOffset(atomIdx);
    const pullDown = Math.max(0, 1 - tFrac * 2.5); // strong pull at t=0, gone by t~0.4
    const pullDown2 = pullDown * pullDown; // ease-out curve

    return [
      simplexPos[0] + startOffset[0] * pullDown2,
      simplexPos[1] + startOffset[1] * pullDown2,
    ];
  }

  // Per-atom offset to spread starting positions below the simplex
  _atomStartOffset(atomIdx) {
    const r = this.r;
    const spread = r.simplexScale * 0.6;
    const baseY = r.startRegionY - r.simplexCentroid[1]; // downward offset
    // Fan out horizontally based on atom index
    const xOff = (atomIdx - (this.nAtoms - 1) / 2) * spread * 0.5;
    return [xOff, baseY + Math.abs(xOff) * 0.3];
  }

  _lerpColor(c1, c2, t) {
    const r1 = parseInt(c1.slice(1, 3), 16);
    const g1 = parseInt(c1.slice(3, 5), 16);
    const b1 = parseInt(c1.slice(5, 7), 16);
    const r2 = parseInt(c2.slice(1, 3), 16);
    const g2 = parseInt(c2.slice(3, 5), 16);
    const b2 = parseInt(c2.slice(5, 7), 16);
    const rr = Math.round(lerp(r1, r2, t));
    const g = Math.round(lerp(g1, g2, t));
    const b = Math.round(lerp(b1, b2, t));
    return '#' + ((1 << 24) + (rr << 16) + (g << 8) + b).toString(16).slice(1);
  }

  draw(dt) {
    const trajT = this.getCurrentT();
    const trajIdx = Math.max(0, Math.min(this.nSteps, Math.round(trajT * this.nSteps)));
    const state = this.trajData.traj[trajIdx];

    const r = this.r;
    const ctx = r.ctx;
    r.clear();

    this.rotAngle += dt * 0.25;
    const colorProgress = clamp(trajT * 1.5, 0, 1);

    // Divider
    ctx.save();
    ctx.globalAlpha = 0.08;
    ctx.beginPath();
    ctx.moveTo(r.midX, r.h * 0.06);
    ctx.lineTo(r.midX, r.h * 0.94);
    ctx.strokeStyle = C.ink;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();

    // Panel titles
    ctx.save();
    ctx.globalAlpha = 0.6;
    ctx.font = '500 13px Inter, sans-serif';
    ctx.fillStyle = C.ink;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('3D Coordinate Flow', r.leftCx, r.h * 0.03);
    ctx.fillText('Atom Type Flow on Simplex', r.rightCx, r.h * 0.03);
    ctx.restore();

    this._drawLeftPanel(r, state, trajT, trajIdx, colorProgress);
    this._drawRightPanel(r, state, trajT, trajIdx, colorProgress);
  }

  _drawLeftPanel(r, state, trajT, trajIdx, colorProgress) {
    const cx = r.leftCx;
    const cy = r.leftCy;
    const scale = r.scale3D;
    const angle = this.rotAngle;

    // ── 3D Axes ──
    this._drawAxes(r, cx, cy, scale, angle);

    const atoms3D = [];
    for (let i = 0; i < this.nAtoms; i++) {
      const raw = [state[i][3], state[i][4], state[i][5]];
      let rot = rotateY(raw, angle);
      rot = rotateX(rot, 0.3);
      const proj = project3D(rot, cx, cy, scale);
      atoms3D.push({ idx: i, x2d: proj[0], y2d: proj[1], z: proj[2], type: this.target.types[i] });
    }
    const sorted = atoms3D.slice().sort((a, b) => a.z - b.z);

    // Full trajectory trails (always visible)
    const traj = this.trajData.traj;
    for (let i = 0; i < this.nAtoms; i++) {
      const trailPts = [];
      const step = Math.max(1, Math.floor(this.nSteps / 40));
      for (let m = 0; m <= this.nSteps; m += step) {
        const s = traj[m];
        const raw = [s[i][3], s[i][4], s[i][5]];
        let rot = rotateY(raw, angle);
        rot = rotateX(rot, 0.3);
        const p = project3D(rot, cx, cy, scale);
        trailPts.push([p[0], p[1]]);
      }
      trailPts.push([atoms3D[i].x2d, atoms3D[i].y2d]);
      const trailColor = this._lerpColor(C.cloud, this.typeColors[this.target.types[i]], colorProgress);
      r.curveSmooth(trailPts, trailColor, 1.2, 0.3);
    }

    // Bonds (opacity scales with t)
    const edgeAlpha = clamp((trajT - 0.02) / 0.5, 0, 1);
    for (const bond of this.target.bonds) {
      const a = atoms3D[bond[0]], b = atoms3D[bond[1]];
      r.ctx.save();
      r.ctx.globalAlpha = edgeAlpha * 0.7;
      r.ctx.beginPath();
      r.ctx.moveTo(a.x2d, a.y2d);
      r.ctx.lineTo(b.x2d, b.y2d);
      r.ctx.strokeStyle = C.ink;
      r.ctx.lineWidth = lerp(0.5, 2.5, edgeAlpha);
      r.ctx.lineCap = 'round';
      r.ctx.stroke();
      r.ctx.restore();
    }

    // ── Velocity Arrows ──
    if (trajT > 0.01 && trajT < 0.95) {
      this._drawVelocityArrows3D(r, state, atoms3D, trajT, cx, cy, scale, angle);
    }

    // Atoms
    for (const atom of sorted) {
      const typeColor = this.typeColors[atom.type];
      const color = this._lerpColor(C.cloud, typeColor, colorProgress);
      const depthFactor = 1 + atom.z * 0.15;
      const radius = lerp(4, 7, clamp(trajT * 1.2, 0, 1)) * depthFactor;
      r.dot([atom.x2d, atom.y2d], color, radius, 0.95);
      if (trajT > 0.85) {
        const labelAlpha = clamp((trajT - 0.85) / 0.15, 0, 1);
        r.label(this.typeLabels[atom.type], [atom.x2d, atom.y2d - radius - 6], {
          color: typeColor,
          size: 11,
          opacity: labelAlpha,
        });
      }
    }
  }

  _drawAxes(r, cx, cy, scale, angle) {
    const axisLen = 0.7; // length in 3D units
    const axes = [
      { dir: [axisLen, 0, 0], label: 'x', color: C.ink },
      { dir: [0, axisLen, 0], label: 'y', color: C.ink },
      { dir: [0, 0, axisLen], label: 'z', color: C.ink },
    ];
    const origin3D = [0, 0, 0];
    let rotO = rotateY(origin3D, angle);
    rotO = rotateX(rotO, 0.3);
    const projO = project3D(rotO, cx, cy, scale);

    for (const ax of axes) {
      let rotA = rotateY(ax.dir, angle);
      rotA = rotateX(rotA, 0.3);
      const projA = project3D(rotA, cx, cy, scale);

      // Dashed axis line
      const ctx = r.ctx;
      ctx.save();
      ctx.globalAlpha = 0.3;
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = ax.color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(projO[0], projO[1]);
      ctx.lineTo(projA[0], projA[1]);
      ctx.stroke();
      ctx.restore();

      // Small arrowhead
      r.arrow(
        [lerp(projO[0], projA[0], 0.85), lerp(projO[1], projA[1], 0.85)],
        [projA[0], projA[1]],
        ax.color, 1, 0.3
      );

      // Axis label
      const labelOff = 10;
      const dx = projA[0] - projO[0];
      const dy = projA[1] - projO[1];
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len > 5) {
        r.label(ax.label, [
          projA[0] + (dx / len) * labelOff,
          projA[1] + (dy / len) * labelOff,
        ], { color: ax.color, size: 11, bold: false, opacity: 0.4 });
      }
    }
  }

  _drawVelocityArrows3D(r, state, atoms3D, trajT, cx, cy, scale, angle) {
    // Velocity = (y1 - yt) / (1 - t) for oracle denoiser
    const y1 = this.trajData.y1;
    const fadeNearEnd = clamp((0.95 - trajT) / 0.1, 0, 1);
    const maxArrowLen = scale * 0.3; // cap arrow length

    for (let i = 0; i < this.nAtoms; i++) {
      const yt = [state[i][3], state[i][4], state[i][5]];
      const target = [y1[i][3], y1[i][4], y1[i][5]];
      const oneMinusT = Math.max(1 - trajT, 0.05);

      // Velocity in 3D
      const vel = [
        (target[0] - yt[0]) / oneMinusT,
        (target[1] - yt[1]) / oneMinusT,
        (target[2] - yt[2]) / oneMinusT,
      ];

      // Scale velocity for display — show direction, capped length
      const velMag = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
      if (velMag < 0.01) continue;

      const displayScale = Math.min(0.15, maxArrowLen / (velMag * scale));
      const tipRaw = [
        yt[0] + vel[0] * displayScale,
        yt[1] + vel[1] * displayScale,
        yt[2] + vel[2] * displayScale,
      ];

      let rotTip = rotateY(tipRaw, angle);
      rotTip = rotateX(rotTip, 0.3);
      const projTip = project3D(rotTip, cx, cy, scale);

      r.arrow(
        [atoms3D[i].x2d, atoms3D[i].y2d],
        [projTip[0], projTip[1]],
        C.gold, 1.5, 0.7 * fadeNearEnd
      );
    }
  }

  _drawRightPanel(r, state, trajT, trajIdx, colorProgress) {
    const traj = this.trajData.traj;
    const ctx = r.ctx;

    // ── Simplex triangle ──
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(r.vC[0], r.vC[1]);
    ctx.lineTo(r.vN[0], r.vN[1]);
    ctx.lineTo(r.vO[0], r.vO[1]);
    ctx.closePath();
    ctx.fillStyle = C.simplexFill;
    ctx.fill();
    ctx.strokeStyle = C.ink;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();

    // Vertex labels + one-hot coords
    const off = 24;
    r.label('C', [r.vC[0] - off, r.vC[1] + 12], { color: C.typeC, size: 16, opacity: 1 });
    r.label('N', [r.vN[0], r.vN[1] - off], { color: C.typeN, size: 16, opacity: 1 });
    r.label('O', [r.vO[0] + off, r.vO[1] + 12], { color: C.typeO, size: 16, opacity: 1 });
    r.label('(1,0,0)', [r.vC[0] - 12, r.vC[1] + 28], { color: C.ink, size: 10, bold: false, opacity: 0.6 });
    r.label('(0,1,0)', [r.vN[0], r.vN[1] - 40], { color: C.ink, size: 10, bold: false, opacity: 0.6 });
    r.label('(0,0,1)', [r.vO[0] + 12, r.vO[1] + 28], { color: C.ink, size: 10, bold: false, opacity: 0.6 });

    // ── Full trajectory curves (always visible, thin, colored at low opacity) ──
    for (let i = 0; i < this.nAtoms; i++) {
      const trailColor = this.typeColors[this.target.types[i]];
      r.curveSmooth(this.simplexCurves[i], trailColor, 1.2, 0.25);
    }

    // ── Waypoint dots on trajectories ──
    for (let i = 0; i < this.nAtoms; i++) {
      for (const wt of this.waypointFracs) {
        const wIdx = Math.round(wt * this.nSteps);
        const wState = traj[wIdx];
        const pos = this._simplexPos(wState[i], wt, i);
        const trailColor = this.typeColors[this.target.types[i]];
        r.dot(pos, trailColor, 3, 0.35);

        // Label y0 and y1
        if (wt === 0 && i === 0) {
          r.label('y\u2080', [pos[0] - 2, pos[1] + 14], { size: 10, opacity: 0.6, bold: false });
        }
        if (wt === 1.0 && i === 0) {
          r.label('y\u2081', [pos[0] - 2, pos[1] - 12], { size: 10, color: this.typeColors[this.target.types[0]], opacity: 0.8, bold: false });
        }
      }
    }

    // ── Denoiser prediction on simplex (teal dots + dashed lines) ──
    if (trajT > 0.01 && trajT < 0.99) {
      for (let i = 0; i < this.nAtoms; i++) {
        const curPos = this._simplexPos(state[i], trajT, i);

        // Oracle denoiser predicts y1 — softmax of y1 gives the simplex position
        const y1disc = [this.trajData.y1[i][0], this.trajData.y1[i][1], this.trajData.y1[i][2]];
        const denoiserProbs = softmax(y1disc);
        const denoiserPos = r.bary(denoiserProbs[0], denoiserProbs[1], denoiserProbs[2]);

        const fadeFactor = clamp(1 - trajT, 0.1, 1);

        // Dashed line from current position to denoiser prediction
        r.dashedLine(curPos, denoiserPos, C.cloud, 0.4 * fadeFactor);

        // Denoiser prediction dot (teal)
        r.dot(denoiserPos, C.denoiser, 4, 0.45 * fadeFactor);
      }
    }

    // ── Velocity arrows on simplex ──
    if (trajT > 0.01 && trajT < 0.95) {
      const fadeNearEnd = clamp((0.95 - trajT) / 0.1, 0, 1);
      for (let i = 0; i < this.nAtoms; i++) {
        const curPos = this._simplexPos(state[i], trajT, i);

        // Compute denoiser simplex position
        const y1disc = [this.trajData.y1[i][0], this.trajData.y1[i][1], this.trajData.y1[i][2]];
        const denoiserProbs = softmax(y1disc);
        const denoiserPos = r.bary(denoiserProbs[0], denoiserProbs[1], denoiserProbs[2]);

        // Direction from current to denoiser
        const dx = denoiserPos[0] - curPos[0];
        const dy = denoiserPos[1] - curPos[1];
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 2) continue;

        const arrowLen = Math.min(18, dist * 0.3);
        const tipX = curPos[0] + (dx / dist) * arrowLen;
        const tipY = curPos[1] + (dy / dist) * arrowLen;

        r.arrow(curPos, [tipX, tipY], C.gold, 1.2, 0.6 * fadeNearEnd);
      }
    }

    // ── Current position dots ──
    for (let i = 0; i < this.nAtoms; i++) {
      const curPos = this._simplexPos(state[i], trajT, i);
      const typeColor = this.typeColors[this.target.types[i]];
      const color = this._lerpColor(C.cloud, typeColor, colorProgress);

      r.dot(curPos, color, 6, 1);

      // Atom index label
      if (trajT > 0.15 && trajT < 0.85) {
        r.label(String(i), [curPos[0], curPos[1] + 14], { size: 9, opacity: 0.5, bold: false });
      }
    }

    // Vertex dots (on top)
    r.dot(r.vC, C.ink, 4, 0.8);
    r.dot(r.vN, C.ink, 4, 0.8);
    r.dot(r.vO, C.ink, 4, 0.8);
  }
}

// ════════════════════════════════════════════════════════════
//  INIT
// ════════════════════════════════════════════════════════════
async function init() {
  if (typeof renderMathInElement === 'function') {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '\\(', right: '\\)', display: false },
        { left: '\\[', right: '\\]', display: true },
      ],
      throwOnError: false,
    });
  }

  let molecules;
  try {
    const res = await fetch('data/molecules.json');
    if (!res.ok) throw new Error('molecules.json fetch failed');
    molecules = await res.json();
  } catch (e) {
    console.error('Failed to load molecules:', e);
    return;
  }

  if (!molecules || molecules.length === 0) {
    console.error('No molecules in data/molecules.json');
    return;
  }

  const molecule = molecules[0];
  let trajData;
  try {
    const res = await fetch('data/trajectories/' + molecule.name + '.json');
    if (!res.ok) throw new Error('trajectory fetch failed');
    trajData = await res.json();
  } catch (e) {
    console.error('Failed to load trajectory for ' + molecule.name + ':', e);
    return;
  }

  const canvas = document.getElementById('canvas-main');
  const renderer = new DualPanelRenderer(canvas);
  const slider = document.getElementById('time-slider');
  const readout = document.getElementById('time-readout');

  let currentT = 0;
  function getCurrentT() {
    return currentT;
  }

  slider.addEventListener('input', function () {
    currentT = parseInt(this.value, 10) / 1000;
    readout.textContent = currentT.toFixed(2);
  });

  const controller = new SectionMolFlow(renderer, molecule, trajData, getCurrentT);

  let resizeTimer;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      renderer.resize();
      controller.resize();
    }, 150);
  });

  let lastTime = performance.now();
  function frame(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;
    controller.draw(dt);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

document.addEventListener('DOMContentLoaded', init);
