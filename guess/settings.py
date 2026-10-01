"""Window, colours, and the line between a number and out of context."""

WIDTH, HEIGHT = 960, 730
FPS = 60
TITLE = "Guess the Number"

PAPER_W, PAPER_H = 680, 390
BRUSH = 12

INPUT = 28
DIGITS = 13  # 0 through 12
REJECT = 13
OUTPUTS = 14
H1 = 128
H2 = 64
# Best digit below this, or the reject class ahead of it, is not an answer.
CUTOFF = 0.55

# A tap or a blank sheet never becomes a stretched blot.
MIN_INK = 40
MIN_SIDE = 36

OOC = "out of context"

BG = (8, 14, 18)
PAPER = (244, 239, 228)
INK = (28, 26, 24)
FRAME = (58, 46, 36)
FRAME_EDGE = (214, 198, 170)
TEXT = (236, 240, 242)
MUTED = (154, 170, 164)
GOLD = (255, 214, 120)
OOC_COLOR = (232, 196, 160)
BUTTON = (14, 26, 32)
BUTTON_EDGE = (46, 68, 74)

PAPER_LUM = 0.2126 * PAPER[0] + 0.7152 * PAPER[1] + 0.0722 * PAPER[2]
