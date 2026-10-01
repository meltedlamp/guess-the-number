"""Strokes on the sheet, and the 28×28 ink the network actually sees."""

import math

import numpy as np
import pygame

from .settings import (
    INK,
    INPUT,
    MIN_INK,
    MIN_SIDE,
    PAPER,
    PAPER_LUM,
    REJECT,
)

SAMPLE_W, SAMPLE_H = 440, 320
FONTS = (
    "segoeui",
    "arial",
    "calibri",
    "consolas",
    "georgia",
    "verdana",
    "trebuchetms",
    "comicsansms",
    "timesnewroman",
    "couriernew",
    "tahoma",
    "impact",
)


def paint_stroke(surface, points, radius, color=INK):
    """Draw a solid marker stroke. Gaps between samples are filled."""
    if not points:
        return
    raw = [(int(p[0]), int(p[1])) for p in points]
    pts = [raw[0]]
    for point in raw[1:]:
        if point != pts[-1]:
            pts.append(point)
    if len(pts) == 1:
        pygame.draw.circle(surface, color, pts[0], radius)
        return
    width = max(1, radius * 2 - 1)
    pygame.draw.lines(surface, color, False, pts, width)
    for point in pts:
        pygame.draw.circle(surface, color, point, radius)


def extend_stroke(stroke, x, y, spacing):
    """Append (x, y), adding points along the way when the cursor jumps."""
    if not stroke:
        stroke.append((x, y))
        return
    x0, y0 = stroke[-1]
    dist = math.hypot(x - x0, y - y0)
    if dist < spacing:
        return
    steps = max(1, int(dist / spacing))
    for step in range(1, steps + 1):
        t = step / steps
        stroke.append((x0 + (x - x0) * t, y0 + (y - y0) * t))


def tensor_from_surface(surface):
    """Crop the ink, pad it square, and scale it to 784 floats in 0..1.

    Returns None when the sheet is blank or only a tap. Callers say
    out of context without asking the network.
    """
    rgb = pygame.surfarray.array3d(surface).astype(np.float32)
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    ink = np.clip((PAPER_LUM - lum) / PAPER_LUM, 0.0, 1.0)
    mask = ink > 0.2
    if int(mask.sum()) < MIN_INK:
        return None
    xs, ys = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if max(x1 - x0, y1 - y0) < MIN_SIDE:
        return None
    width, height = surface.get_size()
    pad = max(4, int(0.08 * max(x1 - x0, y1 - y0)))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width, x1 + pad)
    y1 = min(height, y1 + pad)
    crop = surface.subsurface(pygame.Rect(x0, y0, x1 - x0, y1 - y0)).copy()
    side = max(crop.get_width(), crop.get_height())
    square = pygame.Surface((side, side))
    square.fill(PAPER)
    square.blit(
        crop,
        ((side - crop.get_width()) // 2, (side - crop.get_height()) // 2),
    )
    inner = INPUT - 6
    small = pygame.transform.smoothscale(square, (inner, inner))
    fitted = pygame.Surface((INPUT, INPUT))
    fitted.fill(PAPER)
    fitted.blit(small, ((INPUT - inner) // 2, (INPUT - inner) // 2))
    small_rgb = pygame.surfarray.array3d(fitted).astype(np.float32)
    small_lum = (
        0.2126 * small_rgb[:, :, 0]
        + 0.7152 * small_rgb[:, :, 1]
        + 0.0722 * small_rgb[:, :, 2]
    )
    values = np.clip((PAPER_LUM - small_lum) / PAPER_LUM, 0.0, 1.0).T
    values = _blur(values)
    peak = float(values.max())
    if peak <= 0:
        return None
    values = values / peak
    return values.reshape(-1).astype(np.float32)


def _blur(image):
    padded = np.pad(image, 1, mode="edge")
    total = np.zeros_like(image)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
    return total / 9.0


def make_sample(label, rng):
    """One training sheet for a digit 0–12 or the reject class."""
    surface = pygame.Surface((SAMPLE_W, SAMPLE_H))
    surface.fill(PAPER)
    if label == REJECT:
        _draw_reject(surface, rng)
    elif rng.random() < 0.68:
        _draw_number_strokes(surface, label, rng)
    else:
        _draw_font(surface, str(label), rng)
    return tensor_from_surface(surface)


def paper_for(label, rng):
    """The sheet make_sample draws, before it is cropped. For previews."""
    surface = pygame.Surface((SAMPLE_W, SAMPLE_H))
    surface.fill(PAPER)
    if label == REJECT:
        _draw_reject(surface, rng)
    elif rng.random() < 0.68:
        _draw_number_strokes(surface, label, rng)
    else:
        _draw_font(surface, str(label), rng)
    return surface


def _draw_number_strokes(surface, number, rng):
    chars = [int(ch) for ch in str(number)]
    params = _warp_params(rng)
    rects = _layout(len(chars), rng, surface.get_width(), surface.get_height())
    for digit, rect in zip(chars, rects):
        variant = TEMPLATES[digit][int(rng.integers(0, len(TEMPLATES[digit])))]
        lines, _params = _warp(variant, rng, params)
        radius = max(4, int(rng.uniform(0.045, 0.11) * rect.h))
        _paint_norm(surface, lines, rect, radius)


def _draw_font(surface, text, rng):
    pygame.font.init()
    name = str(rng.choice(FONTS))
    size = int(rng.uniform(72, 168))
    bold = bool(rng.integers(0, 2))
    font = pygame.font.SysFont(name, size, bold=bold)
    image = font.render(text, True, INK)
    if image.get_width() < 2 or image.get_height() < 2:
        return
    stretch_x = float(rng.uniform(0.75, 1.2))
    stretch_y = float(rng.uniform(0.8, 1.15))
    image = pygame.transform.smoothscale(
        image,
        (
            max(2, int(image.get_width() * stretch_x)),
            max(2, int(image.get_height() * stretch_y)),
        ),
    )
    image = pygame.transform.rotate(image, float(rng.uniform(-16, 16)))
    max_w = surface.get_width() - 24
    max_h = surface.get_height() - 24
    if image.get_width() > max_w or image.get_height() > max_h:
        scale = min(max_w / image.get_width(), max_h / image.get_height())
        image = pygame.transform.smoothscale(
            image,
            (max(2, int(image.get_width() * scale)), max(2, int(image.get_height() * scale))),
        )
    x = int(rng.integers(8, max(9, surface.get_width() - image.get_width() - 8)))
    y = int(rng.integers(8, max(9, surface.get_height() - image.get_height() - 8)))
    surface.blit(image, (x, y))


def _draw_reject(surface, rng):
    kind = int(rng.integers(0, 4))
    if kind == 0:
        rect = pygame.Rect(24, 24, SAMPLE_W - 48, SAMPLE_H - 48)
        radius = int(rng.integers(5, 14))
        _paint_norm(surface, _scribble(rng), rect, radius)
    elif kind == 1:
        letter = str(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")))
        _draw_font(surface, letter, rng)
    elif kind == 2:
        number = int(rng.integers(13, 40) if rng.random() < 0.8 else rng.integers(40, 100))
        if rng.random() < 0.55:
            _draw_number_strokes(surface, number, rng)
        else:
            _draw_font(surface, str(number), rng)
    else:
        rect = pygame.Rect(36, 30, SAMPLE_W - 72, SAMPLE_H - 60)
        radius = int(rng.integers(6, 15))
        _paint_norm(surface, _shape(rng), rect, radius)


def _paint_norm(surface, lines, rect, radius):
    for line in lines:
        points = [(rect.x + x * rect.w, rect.y + y * rect.h) for x, y in line]
        paint_stroke(surface, points, radius)


def _layout(count, rng, width, height):
    gap = float(rng.uniform(0.03, 0.1)) * width
    margin = 30
    usable = width - margin * 2 - gap * (count - 1)
    cell_w = usable / count
    cell_h = height - 64
    top = 32 + float(rng.uniform(-8, 8))
    x = margin + float(rng.uniform(-6, 6))
    rects = []
    for _ in range(count):
        rects.append(pygame.Rect(int(x), int(top), int(cell_w), int(cell_h)))
        x += cell_w + gap
    return rects


def _warp_params(rng):
    return {
        "shear": float(rng.uniform(-0.28, 0.28)),
        "rot": float(rng.uniform(-0.22, 0.22)),
        "sx": float(rng.uniform(0.88, 1.12)),
        "sy": float(rng.uniform(0.88, 1.14)),
    }


def _warp(lines, rng, params=None):
    if params is None:
        params = _warp_params(rng)
    cos_r = math.cos(params["rot"])
    sin_r = math.sin(params["rot"])
    warped = []
    for line in lines:
        points = []
        for x, y in line:
            x += float(rng.uniform(-0.012, 0.012))
            y += float(rng.uniform(-0.012, 0.012))
            x = 0.5 + (x - 0.5) * params["sx"]
            y = 0.5 + (y - 0.5) * params["sy"]
            x = x + params["shear"] * (y - 0.5)
            dx, dy = x - 0.5, y - 0.5
            x = 0.5 + cos_r * dx - sin_r * dy
            y = 0.5 + sin_r * dx + cos_r * dy
            points.append((x, y))
        warped.append(points)
    return warped, params


def _scribble(rng):
    lines = []
    for _ in range(int(rng.integers(1, 4))):
        x = float(rng.uniform(0.08, 0.92))
        y = float(rng.uniform(0.08, 0.92))
        points = [(x, y)]
        for _step in range(int(rng.integers(5, 16))):
            x = min(0.96, max(0.04, x + float(rng.uniform(-0.2, 0.2))))
            y = min(0.96, max(0.04, y + float(rng.uniform(-0.2, 0.2))))
            points.append((x, y))
        lines.append(points)
    return lines


def _shape(rng):
    kind = int(rng.integers(0, 4))
    if kind == 0:
        return [[(0.16, 0.16), (0.84, 0.84)], [(0.84, 0.16), (0.16, 0.84)]]
    if kind == 1:
        points = []
        for i in range(42):
            theta = i / 5.5
            radius = 0.04 + i * 0.01
            points.append((0.5 + radius * math.cos(theta), 0.5 + radius * math.sin(theta)))
        return [points]
    if kind == 2:
        return [[(0.18, 0.22), (0.82, 0.22), (0.82, 0.78), (0.18, 0.78), (0.18, 0.22)]]
    return [[(0.08 + i * 0.08, 0.5 + 0.18 * math.sin(i * 0.7)) for i in range(12)]]


def _ellipse(cx, cy, rx, ry, n=36):
    points = []
    for i in range(n + 1):
        theta = -math.pi / 2 + math.tau * i / n
        points.append((cx + rx * math.cos(theta), cy + ry * math.sin(theta)))
    return points


def _cubic(p0, p1, p2, p3, n=14):
    points = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        points.append(
            (
                u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0],
                u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1],
            )
        )
    return points


def _join(*parts):
    points = []
    for part in parts:
        points.extend(part)
    return points


def _vflip(lines):
    return [[(x, 1.0 - y) for x, y in line] for line in lines]


def _templates():
    six = [
        _join(
            _cubic((0.70, 0.30), (0.66, 0.06), (0.16, 0.12), (0.28, 0.50)),
            _ellipse(0.48, 0.66, 0.26, 0.24),
        )
    ]
    digits = {
        0: [
            [_ellipse(0.50, 0.50, 0.28, 0.38)],
            [_ellipse(0.50, 0.50, 0.34, 0.32)],
        ],
        1: [
            [[(0.50, 0.08), (0.50, 0.92)]],
            [
                [(0.34, 0.28), (0.54, 0.08), (0.54, 0.90)],
                [(0.32, 0.90), (0.74, 0.90)],
            ],
        ],
        2: [
            [
                _join(
                    _cubic((0.22, 0.32), (0.20, 0.08), (0.78, 0.06), (0.76, 0.34)),
                    _cubic((0.76, 0.34), (0.74, 0.52), (0.36, 0.62), (0.20, 0.84)),
                    [(0.20, 0.84), (0.82, 0.88)],
                )
            ]
        ],
        3: [
            [
                _join(
                    _cubic((0.26, 0.20), (0.58, 0.02), (0.90, 0.20), (0.50, 0.46)),
                    _cubic((0.50, 0.46), (0.90, 0.50), (0.88, 0.94), (0.28, 0.82)),
                )
            ]
        ],
        4: [
            [
                [(0.64, 0.10), (0.64, 0.92)],
                [(0.18, 0.58), (0.84, 0.58)],
                [(0.18, 0.58), (0.56, 0.12)],
            ],
            [
                [(0.30, 0.12), (0.16, 0.60), (0.84, 0.60)],
                [(0.58, 0.10), (0.58, 0.92)],
            ],
        ],
        5: [
            [
                _join(
                    [(0.76, 0.14), (0.28, 0.14), (0.26, 0.44)],
                    _cubic((0.26, 0.44), (0.58, 0.34), (0.88, 0.52), (0.68, 0.80)),
                    _cubic((0.68, 0.80), (0.54, 0.98), (0.22, 0.90), (0.24, 0.68)),
                )
            ]
        ],
        6: [six],
        7: [
            [[(0.16, 0.16), (0.84, 0.14), (0.36, 0.92)]],
            [
                [(0.16, 0.16), (0.84, 0.14), (0.36, 0.92)],
                [(0.40, 0.48), (0.70, 0.42)],
            ],
        ],
        8: [
            [
                _ellipse(0.50, 0.32, 0.24, 0.20),
                _ellipse(0.50, 0.68, 0.27, 0.22),
            ]
        ],
        9: [
            _vflip(six),
            [
                _ellipse(0.48, 0.32, 0.26, 0.22),
                [(0.72, 0.36), (0.58, 0.92)],
            ],
        ],
    }
    return digits


TEMPLATES = _templates()
