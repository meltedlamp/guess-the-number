"""The sheet, Reed's line, and the buttons that teach a wrong guess."""

import pygame

from .brain import REJECT, Brain
from .ink import extend_stroke, paint_stroke, tensor_from_surface
from .settings import (
    BG,
    BRUSH,
    BUTTON,
    BUTTON_EDGE,
    FRAME,
    FRAME_EDGE,
    GOLD,
    HEIGHT,
    INK,
    MUTED,
    OOC,
    OOC_COLOR,
    PAPER,
    PAPER_H,
    PAPER_W,
    TEXT,
    TITLE,
    WIDTH,
)


class Game:
    def __init__(self, brain):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 28)
        self.small = pygame.font.SysFont("segoeui", 18)
        self.medium = pygame.font.SysFont("segoeui", 32)
        self.big = pygame.font.SysFont("segoeui", 48, bold=True)
        self.button_font = pygame.font.SysFont("segoeui", 22, bold=True)
        self.label_font = pygame.font.SysFont("segoeui", 18, bold=True)
        self.brain = brain
        self.paper = pygame.Surface((PAPER_W, PAPER_H)).convert()
        self.paper_rect = pygame.Rect((WIDTH - PAPER_W) // 2, 120, PAPER_W, PAPER_H)
        self._layout()
        self.strokes = []
        self.current = []
        self.drawing = False
        self.mode = "draw"
        self.line = "Draw a number."
        self.pending = None
        self.right_count = 0
        self.running = True
        self._clear_sheet()

    def _layout(self):
        y = 526
        self.undo_rect = pygame.Rect(36, y, 110, 46)
        self.clear_rect = pygame.Rect(156, y, 110, 46)
        self.guess_rect = pygame.Rect(276, y, 130, 46)
        self.wrong_rect = pygame.Rect(WIDTH - 36 - 110, y, 110, 46)
        self.right_rect = pygame.Rect(self.wrong_rect.x - 120, y, 110, 46)
        gap = 6
        width = 58
        count = 13
        total = count * width + (count - 1) * gap
        x = (WIDTH - total) // 2
        self.label_rects = [
            pygame.Rect(x + index * (width + gap), 618, width, 40) for index in range(count)
        ]
        self.ooc_rect = pygame.Rect(0, 666, 240, 40)
        self.ooc_rect.centerx = WIDTH // 2

    def run(self):
        while self.running:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    self._key(event.key)
                    if not self.running:
                        return
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._press(event.pos)
                elif event.type == pygame.MOUSEMOTION and self.drawing:
                    self._add_point(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self._lift()
            self._draw()
            pygame.display.flip()

    def _key(self, key):
        if key == pygame.K_ESCAPE:
            self.running = False
            return
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._guess()
        elif key == pygame.K_z:
            self._undo()
        elif key == pygame.K_c:
            self._clear()
        elif self.mode == "pick" and pygame.K_0 <= key <= pygame.K_9:
            self._pick(key - pygame.K_0)
        elif self.mode == "pick" and pygame.K_KP0 <= key <= pygame.K_KP9:
            self._pick(key - pygame.K_KP0)

    def _press(self, pos):
        if self.drawing:
            self._lift()
        if self.undo_rect.collidepoint(pos):
            self._undo()
            return
        if self.clear_rect.collidepoint(pos):
            self._clear()
            return
        if self.guess_rect.collidepoint(pos):
            self._guess()
            return
        if self.mode == "answer":
            if self.right_rect.collidepoint(pos):
                self._right()
                return
            if self.wrong_rect.collidepoint(pos):
                self.mode = "pick"
                return
        if self.mode == "pick":
            for label, rect in enumerate(self.label_rects):
                if rect.collidepoint(pos):
                    self._pick(label)
                    return
            if self.ooc_rect.collidepoint(pos):
                self._pick(REJECT)
                return
        if self.mode == "draw" and self.paper_rect.collidepoint(pos):
            self.line = "Draw a number."
            self.drawing = True
            self.current = []
            self._add_point(pos)

    def _add_point(self, pos):
        x = min(max(pos[0] - self.paper_rect.x, 0), PAPER_W - 1)
        y = min(max(pos[1] - self.paper_rect.y, 0), PAPER_H - 1)
        before = len(self.current)
        extend_stroke(self.current, x, y, BRUSH * 0.45)
        if len(self.current) == before:
            return
        start = max(0, before - 1)
        paint_stroke(self.paper, self.current[start:], BRUSH)

    def _lift(self):
        if not self.drawing:
            return
        self.drawing = False
        if self.current:
            self.strokes.append(self.current)
        self.current = []

    def _redraw(self):
        self._clear_sheet()
        for stroke in self.strokes:
            paint_stroke(self.paper, stroke, BRUSH)
        if self.current:
            paint_stroke(self.paper, self.current, BRUSH)

    def _clear_sheet(self):
        self.paper.fill(PAPER)

    def _undo(self):
        if self.drawing or self.current:
            self.drawing = False
            self.current = []
        elif self.strokes:
            self.strokes.pop()
        self._redraw()
        self._cancel_round()

    def _clear(self):
        self.drawing = False
        self.current = []
        self.strokes = []
        self._clear_sheet()
        self._cancel_round()

    def _cancel_round(self):
        self.mode = "draw"
        self.pending = None
        self.line = "Draw a number."

    def _guess(self):
        self._lift()
        self.pending = tensor_from_surface(self.paper)
        if self.pending is None:
            self.line = OOC
        else:
            self.line = self.brain.read(self.pending)
        self.mode = "answer"

    def _right(self):
        if self.mode != "answer":
            return
        self.right_count += 1
        self._finish_round()

    def _pick(self, label):
        if self.mode != "pick":
            return
        if self.pending is not None:
            try:
                self.brain.learn(self.pending, label)
            except OSError:
                pass
        self.line = OOC if label == REJECT else str(label)
        self._finish_round()

    def _finish_round(self):
        self.mode = "draw"
        self.pending = None
        self.strokes = []
        self.current = []
        self.drawing = False
        self._clear_sheet()

    def _draw(self):
        self.screen.fill(BG)
        name = self.font.render("Reed", True, GOLD)
        self.screen.blit(name, (36, 22))
        tally = self.small.render(f"{self.right_count} right", True, MUTED)
        self.screen.blit(tally, tally.get_rect(right=WIDTH - 36, top=30))
        self._draw_line()
        frame = self.paper_rect.inflate(10, 10)
        pygame.draw.rect(self.screen, FRAME, frame, border_radius=8)
        pygame.draw.rect(self.screen, FRAME_EDGE, frame, 2, border_radius=8)
        self.screen.blit(self.paper, self.paper_rect)
        self._button(self.undo_rect, "Undo")
        self._button(self.clear_rect, "Clear")
        self._button(self.guess_rect, "Guess", primary=True)
        if self.mode == "answer":
            self._button(self.right_rect, "Right", kind="yes")
            self._button(self.wrong_rect, "Wrong", kind="no")
            self._hint("Right if that is the number. Wrong if it is not.", 600)
        elif self.mode == "pick":
            self._hint("What is it?", 596)
            for label, rect in enumerate(self.label_rects):
                self._button(rect, str(label), font=self.label_font)
            self._button(self.ooc_rect, OOC, font=self.label_font)
        else:
            self._hint("Draw on the sheet. Enter guesses.", 600)

    def _draw_line(self):
        if self.line == OOC:
            image = self.medium.render(self.line, True, OOC_COLOR)
        elif self.line.isdigit():
            image = self.big.render(self.line, True, TEXT)
        else:
            image = self.medium.render(self.line, True, MUTED)
        self.screen.blit(image, (36, 58))

    def _hint(self, text, y):
        image = self.small.render(text, True, MUTED)
        self.screen.blit(image, image.get_rect(center=(WIDTH // 2, y)))

    def _button(self, rect, label, primary=False, kind="normal", font=None):
        hover = rect.collidepoint(pygame.mouse.get_pos())
        if primary:
            fill = (48, 40, 28)
            border = GOLD
        elif kind == "yes":
            fill = (22, 42, 36)
            border = (120, 190, 150)
        elif kind == "no":
            fill = (46, 32, 30)
            border = (196, 140, 120)
        else:
            fill = BUTTON
            border = BUTTON_EDGE
        if hover:
            fill = tuple(min(255, channel + 16) for channel in fill)
        pygame.draw.rect(self.screen, fill, rect, border_radius=10)
        pygame.draw.rect(self.screen, border if not hover else TEXT, rect, 2, border_radius=10)
        face = (font or self.button_font).render(label, True, TEXT)
        self.screen.blit(face, face.get_rect(center=rect.center))


def _missing(message):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(TITLE)
    font = pygame.font.SysFont("segoeui", 22)
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                return
        screen.fill(BG)
        screen.blit(font.render(message, True, TEXT), (36, 80))
        pygame.display.flip()
        clock.tick(30)


def main():
    try:
        brain = Brain.load()
    except FileNotFoundError as err:
        _missing(str(err))
        return
    Game(brain).run()
