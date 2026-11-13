# Repo Instructions for Team Members

Read this once, then keep it open while you work. Details of the project and your tasks are in docs/whisper_to_voice_team_guide.pdf.

## 1. One time setup (30 minutes)

1. Install Python 3.10 or newer and Git. On Windows use Git Bash or the VS Code terminal. On Mac use Terminal.
2. Clone the repo, then open the folder in VS Code:

        git clone https://github.com/YOUR_USERNAME/whisper_to_voice.git
        cd whisper_to_voice

3. Tell git who you are (once per laptop):

        git config user.name "Your Name"
        git config user.email "you@example.com"

4. Install the libraries:

        pip install -r requirements.txt

   If pip is not found, use pip3 on Mac or py followed by the same words on Windows.
5. Check the repo runs. This should fail with NotImplementedError. That is correct, it means the stubs are in place:

        python run_offline.py data/raw/shared_test.wav results/out.wav

6. Microphone permission. Mac: System Settings, Privacy and Security, Microphone, allow Terminal and VS Code. Windows: Settings, Privacy, Microphone, allow desktop apps.

## 2. Who edits which files

* Hafsa, Module A: src/framing.py, src/detect.py, src/stream.py, run_live.py
* Akhila, Module B: src/lpc.py
* Raks, Module C: src/pitch.py, src/synth.py, run_offline.py

Do not edit a teammate's file. If you find a bug in theirs, message them with the input that breaks it and what you expected. This is what keeps merge conflicts away.

Shared files (README.md, INSTRUCTIONS.md, requirements.txt, src/mocks.py): change only after telling the group. If you add a library, add it to requirements.txt and tell everyone to run pip install again.

## 3. Daily routine

Start of every work session:

    git pull

End of every work session (at least once a day):

    git add .
    git commit -m "A3 frame_energy and zcr validated"
    git push origin main

Write a short message that names the task ID from the PDF, for example "B4 levinson matches solve_toeplitz". Never push code that breaks the import of your own file, because it stops everyone else.

If the push is rejected, someone pushed before you. Run git pull, then push again. Because everyone edits different files, this resolves on its own.

If git pull reports a conflict, you edited a file that someone else also edited. Open the file, look for the lines marked with angle brackets and equals signs, keep the correct version, delete the markers, then add, commit and push. Ask a teammate if unsure.

To throw away your uncommitted changes to one file: git restore src/lpc.py

## 4. How to fill in a stub

1. Open your file. Each function has the right name and signature and currently raises NotImplementedError.
2. Replace the raise line with your implementation. Keep the name, the argument names and defaults, and the output shape exactly as in the README contract.
3. Keep the code flat: plain functions, no classes, no inline comments.
4. Test it alone on data/raw/shared_test.wav. For modules that need another module's output, use src/mocks.py:

        from src.mocks import mock_frames, mock_energy, mock_labels, mock_lpc

5. Run the check listed for that task in the PDF task table (compare with librosa or scipy, plot, listen). A task is done only when its check passes.
6. Commit and push.

## 5. Shared test clip (do this first)

One person records a whispered clip of two sentences, 16 kHz mono, in a quiet room, headset mic about 3 cm from the mouth, headphones on:

    python record.py data/raw/shared_test.wav 8

Listen to it, then commit and push it. This is the one audio file that is committed. Everyone tests on this clip so results are comparable.

Your own recordings follow the naming spk01_s03_whisper.wav and spk01_s03_normal.wav (speaker, sentence number, style). Sentence numbers match data/raw/transcripts.txt. These stay on your laptop, git ignores them.

## 6. Handoff files (due end of Saturday 14 Nov)

After your module works on the shared test clip, save its outputs and push them. Hafsa:

    import numpy as np
    np.save("data/handoff/labels.npy", labels)
    np.save("data/handoff/energy.npy", energy)

Akhila:

    np.save("data/handoff/A.npy", A)
    np.save("data/handoff/G.npy", G)

Rough values are fine. Raks loads these to test synthesis on real inputs without importing anyone's code. Push updated versions whenever your results improve.

## 7. What never goes into git

* Your own recordings, large audio, downloaded datasets (git already ignores data/raw except shared_test.wav and transcripts.txt)
* Anything in results/ (figures, wavs, tables). Share these on the group chat or a shared drive
* Virtual environment folders and notebook checkpoints
* Passwords, tokens or personal details

## 8. Notebooks

Validation notebooks go in notebooks/ and must run from top to bottom on a fresh clone using only files in the repo. Clear outputs before committing large ones. Name them after the module, for example notebooks/b_lpc_validation.ipynb.

## 9. Integration day (Sun 15 Nov)

* Everyone pulls the latest main before the session
* Raks runs python run_offline.py data/raw/shared_test.wav results/out.wav and shares the screen
* Shape or name mismatches are fixed by the file owner during the session, not by someone else
* After a successful run, push a tag message in the commit like "offline pipeline works" so everyone knows the baseline

## 10. If something goes wrong

* ModuleNotFoundError: run pip install again, or you are in the wrong folder. Run commands from the repo root
* No sound or crackle in live tests: headphones only, close other apps using the mic, do not plot inside the audio callback
* Cannot push (permission denied): ask Raks to confirm you are added as a collaborator on the repo
* Lost work: git log shows past commits, ask the group before running any command you do not understand
* Stuck on a concept: use the follow up prompts at the end of your prompt page in the PDF
