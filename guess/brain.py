"""Reed's network. Four layers, NumPy only, kept on this computer when it learns."""

import os
from pathlib import Path

import numpy as np

from .settings import CUTOFF, DIGITS, H1, H2, INPUT, OOC, OUTPUTS, REJECT

IN_DIM = INPUT * INPUT
FACTORY = Path(__file__).resolve().parent / "weights.npz"
REPLAY_MAX = 32
LEARN_STEPS = 8
LEARN_LR = 0.05
LEARN_REPEAT = 4
ANCHOR_BATCH = 16
REPLAY_BATCH = 8


def answer_for(probs):
    """A digit, or out of context when the reject class wins or the digit is unsure."""
    probs = np.asarray(probs, dtype=np.float64)
    best = int(np.argmax(probs[:DIGITS]))
    if float(probs[REJECT]) >= float(probs[best]) or float(probs[best]) < CUTOFF:
        return OOC
    return str(best)


def _softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _user_dirs():
    found = []
    base = os.environ.get("LOCALAPPDATA")
    if base:
        found.append(Path(base) / "GuessTheNumber")
    found.append(Path.home() / ".guess-the-number")
    return found


def _he(rng, fan_in, fan_out):
    scale = np.sqrt(2.0 / fan_in)
    return (rng.standard_normal((fan_out, fan_in)) * scale).astype(np.float32)


class Brain:
    def __init__(self, weights, anchor_x, anchor_y, replay_x, replay_y):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = weights
        self.anchor_x = np.asarray(anchor_x, dtype=np.float32)
        self.anchor_y = np.asarray(anchor_y, dtype=np.int64)
        self.replay_x = np.asarray(replay_x, dtype=np.float32).reshape(-1, IN_DIM)
        self.replay_y = np.asarray(replay_y, dtype=np.int64).reshape(-1)

    @classmethod
    def create(cls, rng):
        weights = (
            _he(rng, IN_DIM, H1),
            np.zeros(H1, dtype=np.float32),
            _he(rng, H1, H2),
            np.zeros(H2, dtype=np.float32),
            _he(rng, H2, OUTPUTS),
            np.zeros(OUTPUTS, dtype=np.float32),
        )
        empty_x = np.zeros((0, IN_DIM), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.int64)
        return cls(weights, empty_x, empty_y, empty_x, empty_y)

    @classmethod
    def load(cls):
        factory = _read_npz(FACTORY)
        if factory is None or "anchor_x" not in factory.files or not _weights_ok(factory):
            if factory is not None:
                factory.close()
            raise FileNotFoundError(
                "Reed has no factory brain. From this folder run: python -m guess.train"
            )
        try:
            anchor_x = np.array(factory["anchor_x"], dtype=np.float32)
            anchor_y = np.array(factory["anchor_y"], dtype=np.int64)
            factory_weights = _weights_of(factory)
        finally:
            factory.close()

        learned = None
        for folder in _user_dirs():
            path = folder / "brain.npz"
            if not path.exists():
                continue
            data = _read_npz(path)
            if data is not None and _weights_ok(data):
                learned = data
                break
            if data is not None:
                data.close()
        if learned is None:
            weights = factory_weights
            replay_x = np.zeros((0, IN_DIM), dtype=np.float32)
            replay_y = np.zeros((0,), dtype=np.int64)
        else:
            try:
                weights = _weights_of(learned)
                if "replay_x" in learned.files:
                    replay_x = np.array(learned["replay_x"], dtype=np.float32)
                    replay_y = np.array(learned["replay_y"], dtype=np.int64)
                else:
                    replay_x = np.zeros((0, IN_DIM), dtype=np.float32)
                    replay_y = np.zeros((0,), dtype=np.int64)
            finally:
                learned.close()
        return cls(weights, anchor_x, anchor_y, replay_x, replay_y)

    def save_factory(self, anchor_x, anchor_y):
        np.savez_compressed(
            FACTORY,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
            anchor_x=np.asarray(anchor_x, dtype=np.float32),
            anchor_y=np.asarray(anchor_y, dtype=np.int64),
        )

    def save_user(self):
        payload = dict(
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
            replay_x=self.replay_x,
            replay_y=self.replay_y,
        )
        errors = []
        for folder in _user_dirs():
            try:
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "brain.npz", **payload)
                return
            except OSError as err:
                errors.append(err)
        if errors:
            raise errors[-1]

    def probabilities(self, ink):
        batch = np.asarray(ink, dtype=np.float32).reshape(1, IN_DIM)
        return self._probs(batch)[0]

    def read(self, ink):
        return answer_for(self.probabilities(ink))

    def learn(self, ink, label, rng=None, save=True):
        """A few steps on this drawing, plus recent labels and the original examples."""
        if rng is None:
            rng = np.random.default_rng()
        sample = np.asarray(ink, dtype=np.float32).reshape(IN_DIM)
        label = int(label)
        self.replay_x = np.concatenate([self.replay_x, sample.reshape(1, IN_DIM)], axis=0)[-REPLAY_MAX:]
        self.replay_y = np.concatenate(
            [self.replay_y, np.array([label], dtype=np.int64)]
        )[-REPLAY_MAX:]

        parts_x = [np.repeat(sample.reshape(1, IN_DIM), LEARN_REPEAT, axis=0)]
        parts_y = [np.full(LEARN_REPEAT, label, dtype=np.int64)]
        replay_n = min(REPLAY_BATCH, len(self.replay_y) - 1)
        if replay_n > 0:
            # The newest row is this drawing, already repeated above.
            older = len(self.replay_y) - 1
            take = min(replay_n, older)
            if take > 0:
                pick = rng.choice(older, size=take, replace=False)
                parts_x.append(self.replay_x[pick])
                parts_y.append(self.replay_y[pick])
        anchor_n = min(ANCHOR_BATCH, len(self.anchor_y))
        if anchor_n > 0:
            pick = rng.choice(len(self.anchor_y), size=anchor_n, replace=False)
            parts_x.append(self.anchor_x[pick])
            parts_y.append(self.anchor_y[pick])
        batch_x = np.concatenate(parts_x, axis=0)
        batch_y = np.concatenate(parts_y, axis=0)
        for _ in range(LEARN_STEPS):
            self.sgd_step(batch_x, batch_y, LEARN_LR)
        if save:
            self.save_user()

    def _probs(self, batch):
        _z1, _h1, _z2, _h2, logits = self._forward(batch)
        return _softmax(logits)

    def _forward(self, batch):
        z1 = batch @ self.W1.T + self.b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.W2.T + self.b2
        h2 = np.maximum(z2, 0)
        logits = h2 @ self.W3.T + self.b3
        return z1, h1, z2, h2, logits

    def sgd_step(self, batch, labels, lr):
        grads, _loss = self.gradients(batch, labels)
        self.W1 -= lr * grads[0]
        self.b1 -= lr * grads[1]
        self.W2 -= lr * grads[2]
        self.b2 -= lr * grads[3]
        self.W3 -= lr * grads[4]
        self.b3 -= lr * grads[5]

    def gradients(self, batch, labels):
        batch = np.asarray(batch, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        count = batch.shape[0]
        z1, h1, z2, h2, logits = self._forward(batch)
        probs = _softmax(logits)
        dlogits = probs.copy()
        dlogits[np.arange(count), labels] -= 1.0
        dlogits /= count
        picked = np.clip(probs[np.arange(count), labels], 1e-8, 1.0)
        loss = float(-np.mean(np.log(picked)))

        dW3 = dlogits.T @ h2
        db3 = dlogits.sum(axis=0)
        dh2 = dlogits @ self.W3
        dz2 = dh2 * (z2 > 0)

        dW2 = dz2.T @ h1
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2
        dz1 = dh1 * (z1 > 0)

        dW1 = dz1.T @ batch
        db1 = dz1.sum(axis=0)
        grads = [
            dW1.astype(np.float32),
            db1.astype(np.float32),
            dW2.astype(np.float32),
            db2.astype(np.float32),
            dW3.astype(np.float32),
            db3.astype(np.float32),
        ]
        return grads, loss


def _read_npz(path):
    try:
        return np.load(path)
    except (OSError, ValueError, EOFError):
        return None


def _weights_ok(data):
    try:
        return (
            data["W1"].shape == (H1, IN_DIM)
            and data["b1"].shape == (H1,)
            and data["W2"].shape == (H2, H1)
            and data["b2"].shape == (H2,)
            and data["W3"].shape == (OUTPUTS, H2)
            and data["b3"].shape == (OUTPUTS,)
        )
    except (KeyError, ValueError):
        return False


def _weights_of(data):
    return (
        data["W1"].astype(np.float32),
        data["b1"].astype(np.float32),
        data["W2"].astype(np.float32),
        data["b2"].astype(np.float32),
        data["W3"].astype(np.float32),
        data["b3"].astype(np.float32),
    )
