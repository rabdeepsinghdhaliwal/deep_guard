# Deep-Guard Stage 1 — "The Bouncer" — Execution Directive

This document assumes you have never done any of this before. Every
command is exact. Copy them character-for-character. If a step's
output doesn't match what's described, stop and read the
Troubleshooting section (Part 7) before continuing to the next step.

**The end result:** a website at `http://127.0.0.1:8000` on your own
Ubuntu machine, where you upload a photo or video, click a button, and
see a percentage telling you how likely the model thinks it's
AI-manipulated.

---

## Part 0 — The shape of what you're about to do (2 minutes, read only)

There are three phases, done in this order:

1. **Train in the cloud (Google Colab).** You will open a notebook file
   in your browser, upload a small credentials file, and click "Run"
   repeatedly. Google's computer does the heavy lifting. At the end,
   a file called `deepguard_bouncer.pth` downloads to your computer.
   This takes roughly 30–60 minutes of mostly-waiting.
2. **Move that file onto your Ubuntu machine.** One folder, one file.
3. **Start the local server and open the website.** A handful of
   terminal commands, run once.

You will not write or edit any code yourself. You are only running
what has already been written, in the order given.

---

## Part 1 — Get your Kaggle API key (`kaggle.json`)

You need this so the Colab notebook is allowed to download the
training dataset on your behalf.

1. Go to **kaggle.com** in your browser. If you don't have an account,
   click **Register** and create one (it's free).
2. Once logged in, click your profile picture in the top-right corner,
   then click **Settings**.
3. Scroll down to the section called **API**.
4. Click the button **Create New Token**.
5. Your browser will immediately download a file named `kaggle.json`.
   It will land in your Downloads folder. **Do not open or edit this
   file.** Do not rename it.
6. Leave it in Downloads for now — you'll upload it directly from
   there in Part 2.

---

## Part 2 — Run the Colab training notebook

1. Go to **colab.research.google.com** in your browser.
2. Click **File → Upload notebook** (top-left menu).
3. Click **Browse** and select the file
   `colab_notebook/DeepGuard_Bouncer_Training.ipynb` from the project
   folder you were given. It will open in a new Colab tab.
4. **Turn on the GPU before running anything:** click the **Runtime**
   menu → **Change runtime type** → under "Hardware accelerator" choose
   **T4 GPU** → click **Save**.
5. You will now run every cell from top to bottom. The easiest way:
   click the menu **Runtime → Run all**. Colab will ask "This notebook
   was not authored by Google" — click **Run anyway**.
6. Partway through, a cell will show a small **Choose Files** button
   (this is the credentials-upload cell from Part 1). Click it,
   navigate to your Downloads folder, and select `kaggle.json`.
   Training cannot proceed until you do this — the notebook will wait
   for you.
7. After that, everything runs automatically. You'll see progress
   printed under each cell — dataset image counts, then training
   progress for two phases, then a test accuracy number, then a
   confusion matrix. **You do not need to understand every printed
   line to know it's working** — just confirm no cell shows a red
   error box. A red box means something needs fixing before the next
   cell will succeed (see Part 7).
8. Near the very end, a cell runs `files.download(...)`. Your browser
   will prompt a file download — **allow it**. A file named
   `deepguard_bouncer.pth` (a few tens of MB) will land in your
   Downloads folder. This is the trained model. Keep track of where
   it lands.
9. **Expected total time:** roughly 5 minutes of setup/waiting for
   downloads, plus 20–45 minutes of training, depending on which GPU
   Colab assigns you.

You will know Part 2 succeeded when: `deepguard_bouncer.pth` exists in
your Downloads folder, and the last two cells' output showed a test
accuracy number and a "Saved final checkpoint to..." message with no
red error boxes anywhere above it.

---

## Part 3 — Set up the local project folder structure on Ubuntu

Open **VS Code**. Open its integrated terminal: menu **Terminal → New
Terminal**, or the shortcut `` Ctrl+` ``.

You should already have the `deepguard-bouncer` project folder (the one
containing this README) somewhere on your machine — e.g. if it's in
your Downloads, move it somewhere sensible first:

```bash
mkdir -p ~/projects
mv ~/Downloads/deepguard-bouncer ~/projects/
cd ~/projects/deepguard-bouncer
```

Confirm you're in the right place — this command should print a
listing that includes `app`, `colab_notebook`, `models`, and this
`README.md`:

```bash
ls
```

Now move the trained weights file from Part 2 into the `models/`
folder, **with this exact filename**:

```bash
mv ~/Downloads/deepguard_bouncer.pth ~/projects/deepguard-bouncer/models/deepguard_bouncer.pth
```

Confirm it landed correctly:

```bash
ls -la models/
```

You should see `deepguard_bouncer.pth` listed with a non-zero file
size. If it says "No such file or directory," the file is not in
Downloads under that exact name — check Part 7.

---

## Part 4 — Create the virtual environment and install dependencies

Still inside `~/projects/deepguard-bouncer` in your VS Code terminal,
run these commands **one at a time**, waiting for each to finish:

```bash
python3 -m venv venv
```

This creates an isolated Python environment inside a new `venv/`
folder, so these dependencies never conflict with anything else on
your system. It will take a few seconds and print nothing on success.

```bash
source venv/bin/activate
```

Your terminal prompt should now show `(venv)` at the start of the
line. **Every command from here on assumes you see that `(venv)`
prefix.** If you close the terminal and reopen it later, you must run
this `source` command again before continuing.

```bash
cd app
pip install -r requirements.txt
```

This installs FastAPI, PyTorch, OpenCV, and everything else the
backend needs. It will print a lot of text and take a few minutes —
PyTorch is a large download. Let it finish completely.

You'll know it succeeded if the last line is something like
`Successfully installed ...` with no red `ERROR:` text above it.

---

## Part 5 — Run the server and open the website

Still inside `~/projects/deepguard-bouncer/app`, with `(venv)` showing
in your prompt:

```bash
uvicorn main:app --reload
```

You should see output ending in a line like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Leave this terminal window open and running.** This is your server —
closing the terminal or pressing `Ctrl+C` stops the website.

Now open a browser and go to:

```
http://127.0.0.1:8000
```

You should see the Deep-Guard interface: a dark upload panel with
corner brackets, and a small status line at the bottom of the page.
That status line is your first diagnostic — it should read something
like `model loaded · running on cpu` with a green dot. If it shows a
red dot with an error message, see Part 7 before doing anything else.

**To stop the server later:** click into that terminal and press
`Ctrl+C`. To start it again in a future session: reopen the terminal,
`cd` into the project's `app` folder, run `source ../venv/bin/activate`
(or `source venv/bin/activate` if you're already in the project root),
then `uvicorn main:app --reload` again.

---

## Part 6 — Use it, and sanity-check both directions

1. Drag a photo onto the panel (or click it to open a file picker),
   then click **Run analysis**. Within a second or two you'll see a
   percentage and a verdict.
2. To sanity-check the model is actually discriminating and not just
   guessing one answer every time, test it against **both** kinds of
   input:
   - An ordinary photo you took yourself (a phone photo of anything
     with a face works well) — this should score low on "manipulated."
   - A GAN-generated face — the website **thispersondoesnotexist.com**
     generates a fresh one on every reload; download one and upload it
     to Deep-Guard — this should score high on "manipulated," since
     StyleGAN-style images are exactly what the training dataset's
     "fake" class was built from.
3. If both directions come out looking reasonable, Stage 1 is working
   correctly.

---

## Part 7 — Troubleshooting

**`nvidia-smi` fails in Colab, or training is extremely slow.**
You're on a CPU runtime. Runtime → Change runtime type → GPU → Save,
then Runtime → Restart runtime, then Run all again from the top.

**The `kaggle.json` upload cell says "No file named kaggle.json was uploaded."**
Your browser may have appended a number to the filename on a repeat
download (e.g. `kaggle(1).json`). Find it in Downloads and rename it
back to exactly `kaggle.json` before re-running that cell.

**A Colab cell shows a red error box.**
Read the last line of the red text — it usually names the exact
problem. Common ones:
- `403 Forbidden` on the Kaggle download cell: your `kaggle.json` is
  invalid or expired — go back to kaggle.com/settings and generate a
  fresh token, then re-run from Step 3 of Part 2.
- `RuntimeError: CUDA out of memory`: rare on this dataset/model
  combination, but if it happens, reduce `BATCH_SIZE = 64` to `32` in
  the "Data loaders" cell and re-run from that cell onward.

**`ls` in Part 3 doesn't show `app`, `colab_notebook`, etc.**
You're not inside the project folder. Run `pwd` to see where you
actually are, then `cd` to the correct path.

**`mv: cannot stat '~/Downloads/deepguard_bouncer.pth': No such file or directory`**
The file either downloaded somewhere other than `~/Downloads`
(check your browser's download settings/history) or the Colab download
step didn't actually complete — go back to Part 2, Step 8.

**`pip install -r requirements.txt` shows red `ERROR:` text.**
Confirm `(venv)` is showing in your prompt — if it isn't, run
`source venv/bin/activate` from the project root first, then retry the
`pip install` command. If it still fails, copy the exact error text —
it almost always names a missing system library, most commonly
resolved with:
```bash
sudo apt update && sudo apt install -y python3-dev build-essential
```
then re-run the `pip install` command.

**Opening `http://127.0.0.1:8000` shows "This site can't be reached."**
The `uvicorn` command isn't running, or errored before it started
listening. Look at the terminal where you ran it — if it shows a
Python traceback instead of the "Uvicorn running on..." line, copy the
last few lines of the error.

**The website loads, but the status line at the bottom is red / says "model not loaded."**
The server can't find `models/deepguard_bouncer.pth`. Re-check Part 3 —
the file must be at exactly `deepguard-bouncer/models/deepguard_bouncer.pth`,
spelled exactly that way, not inside a further subfolder.

**Uploading a file shows "Couldn't reach the analysis service."**
The `uvicorn` server stopped running (check its terminal window) or
crashed mid-request (check that terminal for a traceback).

---

## Part 8 — Where this fits, and what comes next

This is **Phase 1** of the full Deep-Guard project. It stands alone —
you now have a working, demoable deepfake-probability checker. Later
phases (the fingerprinting engine, the Merkle-linked provenance
ledger, the public Block Explorer) build *around* this component
without requiring you to rebuild anything delivered here: this
FastAPI endpoint becomes one internal call in the larger pipeline
instead of the final answer shown to the user.

See the main project blueprint document for the full phase breakdown
and how each later phase integrates with what you just built.
