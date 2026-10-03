/* Browser sheet for Guess the Number. Same rules as the pygame game. */

const PAPER_W = 680;
const PAPER_H = 390;
const BRUSH = 12;
const SPACING = BRUSH * 0.45;
const INPUT = 28;
const IN_DIM = INPUT * INPUT;
const DIGITS = 13;
const REJECT = 13;
const OUTPUTS = 14;
const H1 = 128;
const H2 = 64;
const CUTOFF = 0.55;
const MIN_INK = 40;
const MIN_SIDE = 36;
const OOC = "out of context";

const REPLAY_MAX = 32;
const LEARN_STEPS = 8;
const LEARN_LR = 0.05;
const LEARN_REPEAT = 4;
const ANCHOR_BATCH = 16;
const REPLAY_BATCH = 8;
const STORE_KEY = "guess-the-number-brain";
const BEST_KEY = "guess-the-number-best";

const PAPER = [244, 239, 228];
const PAPER_CSS = "#f4efe4";
const INK = "#1c1a18";
const PAPER_LUM = 0.2126 * PAPER[0] + 0.7152 * PAPER[1] + 0.0722 * PAPER[2];

const canvas = document.getElementById("sheet");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const lineEl = document.getElementById("line");
const hintEl = document.getElementById("hint");
const tallyEl = document.getElementById("tally");
const labelsEl = document.getElementById("labels");
const undoBtn = document.getElementById("undo");
const clearBtn = document.getElementById("clear");
const guessBtn = document.getElementById("guess");
const rightBtn = document.getElementById("right");
const wrongBtn = document.getElementById("wrong");

const strokes = [];
let current = [];
let drawing = false;
let mode = "draw";
let line = "Draw a number.";
let pending = null;
let rightCount = 0;
let brain = null;

function clearPixels() {
  ctx.fillStyle = PAPER_CSS;
  ctx.fillRect(0, 0, PAPER_W, PAPER_H);
}

function paintStroke(points, radius) {
  if (!points.length) return;
  const raw = points.map((point) => [Math.trunc(point[0]), Math.trunc(point[1])]);
  const pts = [raw[0]];
  for (let i = 1; i < raw.length; i++) {
    const point = raw[i];
    const last = pts[pts.length - 1];
    if (point[0] !== last[0] || point[1] !== last[1]) pts.push(point);
  }
  ctx.fillStyle = INK;
  ctx.strokeStyle = INK;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (pts.length === 1) {
    ctx.beginPath();
    ctx.arc(pts[0][0], pts[0][1], radius, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  ctx.lineWidth = Math.max(1, radius * 2 - 1);
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.stroke();
  for (const point of pts) {
    ctx.beginPath();
    ctx.arc(point[0], point[1], radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function extendStroke(stroke, x, y, spacing) {
  if (!stroke.length) {
    stroke.push([x, y]);
    return;
  }
  const last = stroke[stroke.length - 1];
  const dist = Math.hypot(x - last[0], y - last[1]);
  if (dist < spacing) return;
  const steps = Math.max(1, Math.trunc(dist / spacing));
  for (let step = 1; step <= steps; step++) {
    const t = step / steps;
    stroke.push([last[0] + (x - last[0]) * t, last[1] + (y - last[1]) * t]);
  }
}

function redraw() {
  clearPixels();
  for (const stroke of strokes) paintStroke(stroke, BRUSH);
  if (current.length) paintStroke(current, BRUSH);
}

function showLine() {
  lineEl.textContent = line;
  lineEl.className = "line";
  if (line === OOC) lineEl.classList.add("ooc");
  else if (/^\d+$/.test(line)) lineEl.classList.add("digit");
  else lineEl.classList.add("prompt");
}

function showChrome() {
  const answering = mode === "answer";
  const picking = mode === "pick";
  rightBtn.hidden = !answering;
  wrongBtn.hidden = !answering;
  labelsEl.hidden = !picking;
  if (picking) hintEl.textContent = "What is it?";
  else if (answering) hintEl.textContent = "Right if that is the number. Wrong if it is not.";
  else hintEl.textContent = "Draw on the sheet. Enter guesses.";
  tallyEl.textContent = `${rightCount} right`;
  showLine();
}

function cancelRound() {
  mode = "draw";
  pending = null;
  line = "Draw a number.";
  showChrome();
}

function finishRound() {
  mode = "draw";
  pending = null;
  strokes.length = 0;
  current = [];
  drawing = false;
  clearPixels();
  showChrome();
}

function sheetPos(event) {
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) * canvas.width) / rect.width;
  const y = ((event.clientY - rect.top) * canvas.height) / rect.height;
  return [
    Math.min(PAPER_W - 1, Math.max(0, x)),
    Math.min(PAPER_H - 1, Math.max(0, y)),
  ];
}

function addPoint(x, y) {
  const before = current.length;
  extendStroke(current, x, y, SPACING);
  if (current.length === before) return;
  paintStroke(current.slice(Math.max(0, before - 1)), BRUSH);
}

function lift() {
  if (!drawing) return;
  drawing = false;
  if (current.length) strokes.push(current);
  current = [];
}

function undo() {
  if (drawing || current.length) {
    drawing = false;
    current = [];
  } else if (strokes.length) {
    strokes.pop();
  }
  redraw();
  cancelRound();
}

function clearSheet() {
  drawing = false;
  current = [];
  strokes.length = 0;
  clearPixels();
  cancelRound();
}

function blur(image) {
  const out = new Float32Array(IN_DIM);
  for (let y = 0; y < INPUT; y++) {
    for (let x = 0; x < INPUT; x++) {
      let total = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = Math.min(INPUT - 1, Math.max(0, y + dy));
        for (let dx = -1; dx <= 1; dx++) {
          const xx = Math.min(INPUT - 1, Math.max(0, x + dx));
          total += image[yy * INPUT + xx];
        }
      }
      out[y * INPUT + x] = total / 9;
    }
  }
  return out;
}

function tensorFromCanvas() {
  const pixels = ctx.getImageData(0, 0, PAPER_W, PAPER_H).data;
  let count = 0;
  let minX = PAPER_W;
  let minY = PAPER_H;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < PAPER_H; y++) {
    for (let x = 0; x < PAPER_W; x++) {
      const i = (y * PAPER_W + x) * 4;
      const lum = 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
      const ink = Math.min(1, Math.max(0, (PAPER_LUM - lum) / PAPER_LUM));
      if (ink > 0.2) {
        count += 1;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (count < MIN_INK) return null;
  let x0 = minX;
  let y0 = minY;
  let x1 = maxX + 1;
  let y1 = maxY + 1;
  if (Math.max(x1 - x0, y1 - y0) < MIN_SIDE) return null;
  const pad = Math.max(4, Math.trunc(0.08 * Math.max(x1 - x0, y1 - y0)));
  x0 = Math.max(0, x0 - pad);
  y0 = Math.max(0, y0 - pad);
  x1 = Math.min(PAPER_W, x1 + pad);
  y1 = Math.min(PAPER_H, y1 + pad);
  const cropW = x1 - x0;
  const cropH = y1 - y0;
  const side = Math.max(cropW, cropH);

  const square = document.createElement("canvas");
  square.width = side;
  square.height = side;
  const squareCtx = square.getContext("2d");
  squareCtx.fillStyle = PAPER_CSS;
  squareCtx.fillRect(0, 0, side, side);
  squareCtx.imageSmoothingEnabled = false;
  squareCtx.drawImage(
    canvas,
    x0,
    y0,
    cropW,
    cropH,
    Math.floor((side - cropW) / 2),
    Math.floor((side - cropH) / 2),
    cropW,
    cropH
  );

  const inner = INPUT - 6;
  const fitted = document.createElement("canvas");
  fitted.width = INPUT;
  fitted.height = INPUT;
  const fittedCtx = fitted.getContext("2d");
  fittedCtx.fillStyle = PAPER_CSS;
  fittedCtx.fillRect(0, 0, INPUT, INPUT);
  fittedCtx.imageSmoothingEnabled = true;
  fittedCtx.imageSmoothingQuality = "high";
  const origin = Math.floor((INPUT - inner) / 2);
  fittedCtx.drawImage(square, origin, origin, inner, inner);

  const small = fittedCtx.getImageData(0, 0, INPUT, INPUT).data;
  const values = new Float32Array(IN_DIM);
  for (let i = 0; i < IN_DIM; i++) {
    const p = i * 4;
    const lum = 0.2126 * small[p] + 0.7152 * small[p + 1] + 0.0722 * small[p + 2];
    values[i] = Math.min(1, Math.max(0, (PAPER_LUM - lum) / PAPER_LUM));
  }
  const blurred = blur(values);
  let peak = 0;
  for (let i = 0; i < blurred.length; i++) if (blurred[i] > peak) peak = blurred[i];
  if (peak <= 0) return null;
  for (let i = 0; i < blurred.length; i++) blurred[i] /= peak;
  return blurred;
}

function answerFor(probs) {
  let best = 0;
  for (let i = 1; i < DIGITS; i++) if (probs[i] > probs[best]) best = i;
  if (probs[REJECT] >= probs[best] || probs[best] < CUTOFF) return OOC;
  return String(best);
}

function sampleIndexes(count, take) {
  const indexes = Array.from({ length: count }, (_, i) => i);
  for (let i = 0; i < take; i++) {
    const j = i + Math.floor(Math.random() * (count - i));
    const swap = indexes[i];
    indexes[i] = indexes[j];
    indexes[j] = swap;
  }
  return indexes.slice(0, take);
}

function bytesToBase64(bytes) {
  let text = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    text += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(text);
}

function base64ToBytes(text) {
  const raw = atob(text);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}

function floatsToBase64(array) {
  return bytesToBase64(new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
}

function base64ToFloats(text) {
  const bytes = base64ToBytes(text);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function base64ToInts(text) {
  const bytes = base64ToBytes(text);
  return new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function intsToBase64(array) {
  return bytesToBase64(new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
}

class Brain {
  constructor(weights, anchorX, anchorY) {
    this.W1 = weights.W1;
    this.b1 = weights.b1;
    this.W2 = weights.W2;
    this.b2 = weights.b2;
    this.W3 = weights.W3;
    this.b3 = weights.b3;
    this.anchorX = anchorX;
    this.anchorY = anchorY;
    this.replayX = [];
    this.replayY = [];
  }

  read(ink) {
    return answerFor(this._probsOne(ink));
  }

  _probsOne(ink) {
    const z1 = new Float32Array(H1);
    const h1 = new Float32Array(H1);
    for (let a = 0; a < H1; a++) {
      let sum = this.b1[a];
      const row = a * IN_DIM;
      for (let b = 0; b < IN_DIM; b++) sum += ink[b] * this.W1[row + b];
      z1[a] = sum;
      h1[a] = sum > 0 ? sum : 0;
    }
    const h2 = new Float32Array(H2);
    for (let a = 0; a < H2; a++) {
      let sum = this.b2[a];
      const row = a * H1;
      for (let b = 0; b < H1; b++) sum += h1[b] * this.W2[row + b];
      h2[a] = sum > 0 ? sum : 0;
    }
    const logits = new Float32Array(OUTPUTS);
    for (let a = 0; a < OUTPUTS; a++) {
      let sum = this.b3[a];
      const row = a * H2;
      for (let b = 0; b < H2; b++) sum += h2[b] * this.W3[row + b];
      logits[a] = sum;
    }
    return softmax(logits);
  }

  learn(ink, label) {
    const sample = Float32Array.from(ink);
    this.replayX.push(sample);
    this.replayY.push(label);
    if (this.replayX.length > REPLAY_MAX) {
      this.replayX.shift();
      this.replayY.shift();
    }

    const rows = [];
    const labels = [];
    for (let i = 0; i < LEARN_REPEAT; i++) {
      rows.push(sample);
      labels.push(label);
    }
    const older = this.replayY.length - 1;
    const replayTake = Math.min(REPLAY_BATCH, older);
    if (replayTake > 0) {
      for (const index of sampleIndexes(older, replayTake)) {
        rows.push(this.replayX[index]);
        labels.push(this.replayY[index]);
      }
    }
    const anchorTake = Math.min(ANCHOR_BATCH, this.anchorY.length);
    if (anchorTake > 0) {
      for (const index of sampleIndexes(this.anchorY.length, anchorTake)) {
        rows.push(this.anchorX.subarray(index * IN_DIM, (index + 1) * IN_DIM));
        labels.push(this.anchorY[index]);
      }
    }

    const batch = new Float32Array(rows.length * IN_DIM);
    rows.forEach((row, index) => batch.set(row, index * IN_DIM));
    const targets = Int32Array.from(labels);
    for (let step = 0; step < LEARN_STEPS; step++) this._sgd(batch, targets, LEARN_LR);
    this._save();
  }

  _forward(batch, count) {
    const z1 = new Float32Array(count * H1);
    const h1 = new Float32Array(count * H1);
    for (let i = 0; i < count; i++) {
      for (let a = 0; a < H1; a++) {
        let sum = this.b1[a];
        const w = a * IN_DIM;
        const x = i * IN_DIM;
        for (let b = 0; b < IN_DIM; b++) sum += batch[x + b] * this.W1[w + b];
        z1[i * H1 + a] = sum;
        h1[i * H1 + a] = sum > 0 ? sum : 0;
      }
    }
    const z2 = new Float32Array(count * H2);
    const h2 = new Float32Array(count * H2);
    for (let i = 0; i < count; i++) {
      for (let a = 0; a < H2; a++) {
        let sum = this.b2[a];
        const w = a * H1;
        const x = i * H1;
        for (let b = 0; b < H1; b++) sum += h1[x + b] * this.W2[w + b];
        z2[i * H2 + a] = sum;
        h2[i * H2 + a] = sum > 0 ? sum : 0;
      }
    }
    const logits = new Float32Array(count * OUTPUTS);
    for (let i = 0; i < count; i++) {
      for (let a = 0; a < OUTPUTS; a++) {
        let sum = this.b3[a];
        const w = a * H2;
        const x = i * H2;
        for (let b = 0; b < H2; b++) sum += h2[x + b] * this.W3[w + b];
        logits[i * OUTPUTS + a] = sum;
      }
    }
    return { z1, h1, z2, h2, logits };
  }

  _sgd(batch, labels, lr) {
    const count = labels.length;
    const { z1, h1, z2, h2, logits } = this._forward(batch, count);
    const probs = softmaxRows(logits, count, OUTPUTS);
    const dlogits = probs.slice();
    for (let i = 0; i < count; i++) dlogits[i * OUTPUTS + labels[i]] -= 1;
    for (let i = 0; i < dlogits.length; i++) dlogits[i] /= count;

    const dW3 = new Float32Array(OUTPUTS * H2);
    const db3 = new Float32Array(OUTPUTS);
    for (let i = 0; i < count; i++) {
      for (let a = 0; a < OUTPUTS; a++) {
        const grad = dlogits[i * OUTPUTS + a];
        db3[a] += grad;
        const row = a * H2;
        const hidden = i * H2;
        for (let b = 0; b < H2; b++) dW3[row + b] += grad * h2[hidden + b];
      }
    }

    const dz2 = new Float32Array(count * H2);
    for (let i = 0; i < count; i++) {
      for (let b = 0; b < H2; b++) {
        let sum = 0;
        for (let a = 0; a < OUTPUTS; a++) sum += dlogits[i * OUTPUTS + a] * this.W3[a * H2 + b];
        dz2[i * H2 + b] = z2[i * H2 + b] > 0 ? sum : 0;
      }
    }

    const dW2 = new Float32Array(H2 * H1);
    const db2 = new Float32Array(H2);
    for (let i = 0; i < count; i++) {
      for (let a = 0; a < H2; a++) {
        const grad = dz2[i * H2 + a];
        db2[a] += grad;
        const row = a * H1;
        const hidden = i * H1;
        for (let b = 0; b < H1; b++) dW2[row + b] += grad * h1[hidden + b];
      }
    }

    const dz1 = new Float32Array(count * H1);
    for (let i = 0; i < count; i++) {
      for (let b = 0; b < H1; b++) {
        let sum = 0;
        for (let a = 0; a < H2; a++) sum += dz2[i * H2 + a] * this.W2[a * H1 + b];
        dz1[i * H1 + b] = z1[i * H1 + b] > 0 ? sum : 0;
      }
    }

    const dW1 = new Float32Array(H1 * IN_DIM);
    const db1 = new Float32Array(H1);
    for (let i = 0; i < count; i++) {
      for (let a = 0; a < H1; a++) {
        const grad = dz1[i * H1 + a];
        db1[a] += grad;
        const row = a * IN_DIM;
        const input = i * IN_DIM;
        for (let b = 0; b < IN_DIM; b++) dW1[row + b] += grad * batch[input + b];
      }
    }

    step(this.W1, dW1, lr);
    step(this.b1, db1, lr);
    step(this.W2, dW2, lr);
    step(this.b2, db2, lr);
    step(this.W3, dW3, lr);
    step(this.b3, db3, lr);
  }

  _save() {
    try {
      const replayX = new Float32Array(this.replayX.length * IN_DIM);
      this.replayX.forEach((row, index) => replayX.set(row, index * IN_DIM));
      const payload = {
        v: 1,
        W1: floatsToBase64(this.W1),
        b1: floatsToBase64(this.b1),
        W2: floatsToBase64(this.W2),
        b2: floatsToBase64(this.b2),
        W3: floatsToBase64(this.W3),
        b3: floatsToBase64(this.b3),
        replayX: floatsToBase64(replayX),
        replayY: intsToBase64(Int32Array.from(this.replayY)),
      };
      localStorage.setItem(STORE_KEY, JSON.stringify(payload));
    } catch (err) {
      /* The round still counts. The factory brain in the repo stays as it was. */
    }
  }
}

function step(param, grad, lr) {
  for (let i = 0; i < param.length; i++) param[i] -= lr * grad[i];
}

function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const value = Math.exp(logits[i] - max);
    out[i] = value;
    sum += value;
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

function softmaxRows(logits, count, width) {
  const out = new Float32Array(logits.length);
  for (let i = 0; i < count; i++) {
    const row = softmax(logits.subarray(i * width, (i + 1) * width));
    out.set(row, i * width);
  }
  return out;
}

function readArray(view, buffer, offset, kind) {
  const ndim = view.getUint32(offset, true);
  offset += 4;
  const dims = [];
  for (let i = 0; i < ndim; i++) {
    dims.push(view.getUint32(offset, true));
    offset += 4;
  }
  const count = dims.reduce((total, dim) => total * dim, 1);
  const bytes = count * 4;
  const copy = buffer.slice(offset, offset + bytes);
  const data = kind === "i32" ? new Int32Array(copy) : new Float32Array(copy);
  return { data, offset: offset + bytes };
}

function parseBrain(buffer) {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== "REED") throw new Error("Reed's factory brain is unreadable.");
  const view = new DataView(buffer);
  if (view.getUint32(4, true) !== 1) throw new Error("Reed's factory brain is unreadable.");
  let offset = 8;
  const names = ["W1", "b1", "W2", "b2", "W3", "b3", "anchorX"];
  const arrays = {};
  for (const name of names) {
    const read = readArray(view, buffer, offset, "f32");
    arrays[name] = read.data;
    offset = read.offset;
  }
  const labels = readArray(view, buffer, offset, "i32");
  return new Brain(arrays, arrays.anchorX, labels.data);
}

function applyLearned(model) {
  const raw = localStorage.getItem(STORE_KEY);
  if (!raw) return;
  try {
    const saved = JSON.parse(raw);
    const W1 = base64ToFloats(saved.W1);
    const b1 = base64ToFloats(saved.b1);
    const W2 = base64ToFloats(saved.W2);
    const b2 = base64ToFloats(saved.b2);
    const W3 = base64ToFloats(saved.W3);
    const b3 = base64ToFloats(saved.b3);
    if (
      W1.length !== H1 * IN_DIM ||
      b1.length !== H1 ||
      W2.length !== H2 * H1 ||
      b2.length !== H2 ||
      W3.length !== OUTPUTS * H2 ||
      b3.length !== OUTPUTS
    ) {
      return;
    }
    model.W1 = W1;
    model.b1 = b1;
    model.W2 = W2;
    model.b2 = b2;
    model.W3 = W3;
    model.b3 = b3;
    const replayX = base64ToFloats(saved.replayX);
    const replayY = base64ToInts(saved.replayY);
    const rows = Math.floor(replayX.length / IN_DIM);
    model.replayX = [];
    model.replayY = [];
    for (let i = 0; i < rows && i < replayY.length; i++) {
      model.replayX.push(replayX.subarray(i * IN_DIM, (i + 1) * IN_DIM));
      model.replayY.push(replayY[i]);
    }
  } catch (err) {
    /* A bad save falls back to the factory brain. */
  }
}

function guess() {
  if (!brain) return;
  lift();
  pending = tensorFromCanvas();
  line = pending ? brain.read(pending) : OOC;
  mode = "answer";
  showChrome();
}

const arcadeVisit = String(Date.now());

function noteArcade(score) {
  try {
    const player = (localStorage.getItem("melted-arcade-player") || "").trim();
    if (!player || score <= 0) return;
    const raw = localStorage.getItem("melted-arcade-slips") || "[]";
    const slips = JSON.parse(raw);
    const list = Array.isArray(slips) ? slips : [];
    const row = { game: "guess", score, player, note: "", at: Date.now(), visit: arcadeVisit };
    const prev = list.findIndex((item) => item && item.visit === arcadeVisit);
    if (prev >= 0) list[prev] = row;
    else list.push(row);
    localStorage.setItem("melted-arcade-slips", JSON.stringify(list.slice(-40)));
  } catch (err) {
    /* The sheet still counts. The arcade will miss this sitting. */
  }
}

function rememberBest() {
  try {
    const saved = Number(localStorage.getItem(BEST_KEY));
    const best = Number.isFinite(saved) ? Math.max(0, Math.floor(saved)) : 0;
    if (rightCount > best) localStorage.setItem(BEST_KEY, String(rightCount));
  } catch (err) {
    /* The tally on the sheet still moves. */
  }
}

function markRight() {
  if (mode !== "answer") return;
  rightCount += 1;
  rememberBest();
  noteArcade(rightCount);
  finishRound();
}

function markWrong() {
  if (mode !== "answer") return;
  mode = "pick";
  showChrome();
}

function pick(label) {
  if (mode !== "pick") return;
  if (pending && brain) {
    try {
      brain.learn(pending, label);
    } catch (err) {
      /* Teaching failed to save. The sheet still clears. */
    }
  }
  line = label === REJECT ? OOC : String(label);
  finishRound();
}

function onKey(event) {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.key === "Enter") {
    event.preventDefault();
    guess();
  } else if (event.key === "z" || event.key === "Z") {
    undo();
  } else if (event.key === "c" || event.key === "C") {
    clearSheet();
  } else if (mode === "pick" && event.key >= "0" && event.key <= "9") {
    pick(Number(event.key));
  }
}

function wire() {
  const digits = document.getElementById("digits");
  for (let label = 0; label <= 12; label++) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = String(label);
    button.addEventListener("click", () => pick(label));
    digits.appendChild(button);
  }
  document.getElementById("ooc").addEventListener("click", () => pick(REJECT));
  undoBtn.addEventListener("click", undo);
  clearBtn.addEventListener("click", clearSheet);
  guessBtn.addEventListener("click", guess);
  rightBtn.addEventListener("click", markRight);
  wrongBtn.addEventListener("click", markWrong);
  window.addEventListener("keydown", onKey);
  window.addEventListener("pagehide", () => noteArcade(rightCount));

  canvas.addEventListener("pointerdown", (event) => {
    if (mode !== "draw" || event.button !== 0) return;
    canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
    line = "Draw a number.";
    showLine();
    drawing = true;
    current = [];
    const [x, y] = sheetPos(event);
    addPoint(x, y);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drawing) return;
    const [x, y] = sheetPos(event);
    addPoint(x, y);
  });
  canvas.addEventListener("pointerup", lift);
  canvas.addEventListener("pointercancel", lift);
}

async function boot() {
  clearPixels();
  wire();
  guessBtn.disabled = true;
  try {
    const response = await fetch("brain.bin");
    if (!response.ok) throw new Error("missing");
    brain = parseBrain(await response.arrayBuffer());
    applyLearned(brain);
    line = "Draw a number.";
    guessBtn.disabled = false;
  } catch (err) {
    line = "Reed has no factory brain. Open this page from a local server in the docs folder.";
  }
  showChrome();
}

boot();
