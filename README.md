# Guess the Number

A small drawing game written in Python with [Pygame](https://www.pygame.org/).
You draw a number on a sheet. An agent named Reed reads it with a neural network of its own and says what it sees.

The number is from 0 to 12. 10, 11, and 12 are both digits in that one drawing.
A scribble, a letter, a stray tap, or a number past 12 comes back as **out of context**.

Play it in the browser: [https://meltedlamp.github.io/guess-the-number/](https://meltedlamp.github.io/guess-the-number/)

## Features

- **You draw, Reed reads.** The line under Reed's name is the guess: a number, or the words out of context.
- **A sheet.** Draw with the mouse. Fast strokes stay solid. Undo drops the last stroke. Clear wipes the sheet.
- **0 through 12 only.** Reed has no answer for anything else.
- **Right and wrong.** Right counts a hit and leaves the network as it is. Wrong opens 0–12 and out of context, and the label you pick is what Reed studies.
- **It keeps what it learns.** Corrections stay on this computer. The factory brain shipped with the game is left alone.
- **A session tally.** How many guesses you marked right.

## Getting started

You need **Python 3.8+**.

```bash
git clone https://github.com/meltedlamp/guess-the-number.git
cd guess-the-number
pip install -r requirements.txt
python game.py
```

To train the factory brain again, then refresh the browser copy:

```bash
python -m guess.train
python -m guess.export_web
```

## In the browser

Open [https://meltedlamp.github.io/guess-the-number/](https://meltedlamp.github.io/guess-the-number/). Drawing, guessing, and teaching Reed work the same way. Corrections stay in this browser. The factory brain in the repo is left alone.

## Controls

| Action | Keys / mouse |
| --- | --- |
| Draw | Mouse on the sheet |
| Guess | `Enter`, or **Guess** |
| Undo | `Z`, or **Undo** |
| Clear | `C`, or **Clear** |
| Agree | **Right** |
| Correct Reed | **Wrong**, then `0`–`9` or a label. 10, 11, 12, and out of context are on the labels |
| Quit | `Esc` |

## How a round goes

- Draw one number and press **Guess**. An empty sheet is out of context without a guess.
- If Reed is right, press **Right**. The tally moves.
- If Reed is wrong, press **Wrong** and pick the real label. Reed takes a few small steps on that drawing, plus a short memory of your recent labels, so it does not forget 0–12.
- Clear handwriting should hit. A scrawl, a letter, or 13 and above should come back out of context.

## Project structure

```
guess-the-number/
├── game.py              # launcher: python game.py
├── guess/               # the game package (also runs with: python -m guess)
│   ├── game.py          # window, sheet, and buttons
│   ├── ink.py           # strokes and the 28×28 sheet the network sees
│   ├── brain.py         # the network, a learning step, load and save
│   ├── train.py         # draw practice sheets and write the factory brain
│   ├── export_web.py    # pack the factory brain into docs/brain.bin
│   ├── settings.py      # window size, colours, and the confidence cutoff
│   └── weights.npz      # the factory brain, so the first drawing can be guessed
├── .github/workflows/pages.yml  # publishes docs/ to GitHub Pages
├── docs/                # the browser page (GitHub Pages)
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── brain.bin        # factory weights and the examples Reed studies
├── requirements.txt     # pygame and numpy
└── README.md
```
