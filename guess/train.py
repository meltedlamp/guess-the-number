"""Build the factory brain from generated sheets and write guess/weights.npz."""

import argparse

import numpy as np
import pygame

from .brain import IN_DIM, Brain, answer_for
from .ink import INPUT, make_sample, paper_for
from .settings import DIGITS, OOC, OUTPUTS, REJECT

PER_DIGIT = 420
PER_REJECT = 1260
VAL_EACH = 36
EPOCHS = 36
BATCH = 128
ADAM_LR = 0.004


def _collect(label, count, rng):
    rows = []
    tries = 0
    limit = count * 8
    while len(rows) < count and tries < limit:
        tries += 1
        sample = make_sample(label, rng)
        if sample is not None:
            rows.append(sample)
    if len(rows) < count:
        raise RuntimeError(f"only drew {len(rows)} of {count} sheets for {label}")
    return np.stack(rows).astype(np.float32)


def build_set(per_digit, per_reject, rng):
    xs = []
    ys = []
    for label in range(DIGITS):
        print(f"  drawing {label}", flush=True)
        xs.append(_collect(label, per_digit, rng))
        ys.append(np.full(per_digit, label, dtype=np.int64))
    print("  drawing reject", flush=True)
    xs.append(_collect(REJECT, per_reject, rng))
    ys.append(np.full(per_reject, REJECT, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def _shift(batch, rng):
    """Nudge the 28×28 sheet by a pixel or two. The crop is not perfect."""
    images = batch.reshape(-1, INPUT, INPUT).copy()
    out = np.zeros_like(images)
    for i, image in enumerate(images):
        dy = int(rng.integers(-2, 3))
        dx = int(rng.integers(-2, 3))
        src_y0 = max(0, -dy)
        src_x0 = max(0, -dx)
        dst_y0 = max(0, dy)
        dst_x0 = max(0, dx)
        height = INPUT - abs(dy)
        width = INPUT - abs(dx)
        out[i, dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = image[
            src_y0 : src_y0 + height, src_x0 : src_x0 + width
        ]
    return out.reshape(-1, IN_DIM)


class _Adam:
    def __init__(self, params, lr):
        self.lr = lr
        self.t = 0
        self.m = [np.zeros_like(param) for param in params]
        self.v = [np.zeros_like(param) for param in params]

    def step(self, params, grads):
        self.t += 1
        b1_hat = 1 - 0.9**self.t
        b2_hat = 1 - 0.999**self.t
        for index, (param, grad) in enumerate(zip(params, grads)):
            self.m[index] = 0.9 * self.m[index] + 0.1 * grad
            self.v[index] = 0.999 * self.v[index] + 0.001 * (grad * grad)
            param -= self.lr * (self.m[index] / b1_hat) / (np.sqrt(self.v[index] / b2_hat) + 1e-8)


def fit(brain, train_x, train_y, val_x, val_y, rng):
    params = [brain.W1, brain.b1, brain.W2, brain.b2, brain.W3, brain.b3]
    adam = _Adam(params, ADAM_LR)
    order = np.arange(len(train_y))
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        total = 0.0
        seen = 0
        for start in range(0, len(order), BATCH):
            pick = order[start : start + BATCH]
            batch = _shift(train_x[pick], rng)
            noise = rng.normal(0, 0.02, batch.shape).astype(np.float32)
            batch = np.clip(batch + noise, 0, 1)
            grads, loss = brain.gradients(batch, train_y[pick])
            adam.step(params, grads)
            total += loss * len(pick)
            seen += len(pick)
        if epoch == 0 or epoch == EPOCHS - 1 or epoch % 5 == 4:
            acc = _exact_accuracy(brain, val_x, val_y)
            print(f"epoch {epoch + 1:02d}  loss {total / seen:.3f}  val {acc:.3f}", flush=True)


def _exact_accuracy(brain, data_x, data_y):
    """Class id match, before the out-of-context cutoff. Used while training."""
    probs = brain._probs(data_x)
    pred = np.argmax(probs, axis=1)
    return float(np.mean(pred == data_y))


def report(brain, data_x, data_y):
    probs = brain._probs(data_x)
    raw = np.argmax(probs, axis=1)
    said = [answer_for(row) for row in probs]
    digit_hit = 0
    digit_n = 0
    digit_ooc = 0
    reject_ooc = 0
    reject_n = 0
    confused = {}
    for truth, guess, top in zip(data_y, said, raw):
        truth = int(truth)
        if truth == REJECT:
            reject_n += 1
            if guess == OOC:
                reject_ooc += 1
            else:
                confused[f"reject->{guess}"] = confused.get(f"reject->{guess}", 0) + 1
        else:
            digit_n += 1
            if guess == str(truth):
                digit_hit += 1
            elif guess == OOC:
                digit_ooc += 1
            else:
                key = f"{truth}->{guess}"
                confused[key] = confused.get(key, 0) + 1
    print(
        f"digits read right {digit_hit}/{digit_n}  "
        f"digits refused {digit_ooc}/{digit_n}  "
        f"reject refused {reject_ooc}/{reject_n}"
    )
    top = sorted(confused.items(), key=lambda item: item[1], reverse=True)[:8]
    if top:
        print("misses:", ", ".join(f"{name} {count}" for name, count in top))
    return digit_hit / max(1, digit_n), reject_ooc / max(1, reject_n)


def _anchors(data_x, data_y, rng, each=12):
    xs = []
    ys = []
    for label in range(OUTPUTS):
        pool = np.flatnonzero(data_y == label)
        take = min(each, len(pool))
        pick = rng.choice(pool, size=take, replace=False)
        xs.append(data_x[pick])
        ys.append(data_y[pick])
    return np.concatenate(xs), np.concatenate(ys)


def _preview(path, rng):
    papers = []
    labels = []
    for label in list(range(DIGITS)) + [REJECT, REJECT, REJECT]:
        papers.append(paper_for(label, rng))
        labels.append(label)
    cell_w, cell_h = 160, 120
    cols = 8
    rows = (len(papers) + cols - 1) // cols
    sheet = pygame.Surface((cols * cell_w, rows * cell_h + 28))
    sheet.fill((8, 14, 18))
    font = pygame.font.SysFont("segoeui", 16)
    for index, (paper, label) in enumerate(zip(papers, labels)):
        thumb = pygame.transform.smoothscale(paper, (cell_w - 8, cell_h - 24))
        col, row = index % cols, index // cols
        x, y = col * cell_w + 4, row * cell_h + 4
        sheet.blit(thumb, (x, y))
        name = OOC if label == REJECT else str(label)
        sheet.blit(font.render(name, True, (236, 240, 242)), (x, y + cell_h - 22))
    pygame.image.save(sheet, path)
    print(path)


def main():
    parser = argparse.ArgumentParser(description="Train Reed's factory brain.")
    parser.add_argument("--preview", action="store_true", help="Save a sheet of sample drawings and stop.")
    args = parser.parse_args()
    pygame.init()
    rng = np.random.default_rng(7)
    if args.preview:
        _preview("preview.png", rng)
        return
    print("drawing sheets", flush=True)
    train_x, train_y = build_set(PER_DIGIT, PER_REJECT, rng)
    val_x, val_y = build_set(VAL_EACH, VAL_EACH, np.random.default_rng(11))
    print(f"train {len(train_y)}  val {len(val_y)}", flush=True)
    brain = Brain.create(np.random.default_rng(3))
    fit(brain, train_x, train_y, val_x, val_y, np.random.default_rng(5))
    digit_acc, reject_acc = report(brain, val_x, val_y)
    anchor_x, anchor_y = _anchors(train_x, train_y, np.random.default_rng(9))
    brain.anchor_x = anchor_x
    brain.anchor_y = anchor_y
    brain.save_factory(anchor_x, anchor_y)
    from .brain import FACTORY

    print(f"saved {FACTORY}")
    if digit_acc < 0.8 or reject_acc < 0.55:
        print("warning: the factory brain is still weak on the held-out sheets")


if __name__ == "__main__":
    main()
