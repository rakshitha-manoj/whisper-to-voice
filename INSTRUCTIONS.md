# Repo Instructions for Team Members

Read this once, then keep it open while you work. Details of the project and your tasks are in docs/whisper_to_voice_team_guide.pdf.

## 1. How we use branches

* main is the shared, known good version. Nobody works directly on main except for the cases in section 7 and section 8.
* Each person works on their own branch and combines everything on main at fixed merge points.
* Branches (already created, you only switch to yours):
  * module_a, Hafsa
  * module_b, Akhila
  * module_c, Raks

Always check which branch you are on before you commit:

    git branch

The branch with a star is the current one. If it is not yours, switch before you edit anything.

## 2. One time setup (30 minutes)

1. Install Python 3.10 or newer and Git. On Windows use Git Bash or the VS Code terminal. On Mac use Terminal.
2. Clone the repo and open the folder in VS Code:

        git clone https://github.com/YOUR_USERNAME/whisper_to_voice.git
        cd whisper_to_voice

3. Tell git who you are (once per laptop):

        git config user.name "Your Name"
        git config user.email "you@example.com"

4. Switch to your branch (use module_a, module_b or module_c):

        git fetch origin
        git switch module_a

5. Install the libraries:

        pip install -r requirements.txt

   If pip is not found, use pip3 on Mac or py followed by the same words on Windows.
6. Check the repo runs. All modules are implemented now, so the test suite should pass (no recordings needed, it uses a synthetic clip). Full testing guide: docs/TESTING.md.

        python -m pytest tests -q

   Once data/raw/shared_test.wav has been recorded and committed (section 7), this also works:

        python run_offline.py data/raw/shared_test.wav results/out.wav

7. Microphone permission. Mac: System Settings, Privacy and Security, Microphone, allow Terminal and VS Code. Windows: Settings, Privacy, Microphone, allow desktop apps.

## 3. Who edits which files

* Hafsa, module_a: src/framing.py, src/detect.py, src/stream.py, run_live.py
* Akhila, module_b: src/lpc.py
* Raks, module_c: src/pitch.py, src/synth.py, run_offline.py

Do not edit a teammate's file. If you find a bug in theirs, message them with the input that breaks it and what you expected. This is what keeps merges clean.

Shared files (README.md, INSTRUCTIONS.md, requirements.txt, src/mocks.py) are changed on main only, after telling the group. See section 8.

## 4. Daily routine on your branch

Start of every work session. First get the latest shared files from main:

    git fetch origin
    git merge origin/main

End of every work session (at least once a day):

    git add .
    git commit -m "A3 frame_energy and zcr validated"
    git push origin module_a

Replace module_a with your own branch. Write a short message that names the task ID from the PDF, for example "B4 levinson matches solve_toeplitz". Never push code that breaks the import of your own file.

If git merge origin/main reports a conflict, you edited a shared file or someone changed your file. Open the file, look for the lines marked with angle brackets and equals signs, keep the correct version, delete the markers, then run git add and git commit. Ask the group if unsure.

To throw away your uncommitted changes to one file: git restore src/lpc.py

## 5. Merge schedule

* Until Sat 14 Nov night: everyone works only on their own branch, using src/mocks.py for inputs from teammates.
* Sat 14 Nov night, first merge: Raks merges all three branches into main (commands in section 9). After that, everyone runs the start of session commands from section 4 so their branch contains the others' real code.
* Sun 15 Nov, integration session: everyone is on a call. Raks runs the offline pipeline from main. When something does not fit, the file owner fixes it on their own branch, pushes, and Raks merges again.
* Mon 16 Nov: live version work stays on module_a and module_c. Merge into main only when it runs.
* Thu 19 Nov: freeze. The last working commit on main is the demo version. Only bug fixes after that.

Until Saturday you cannot see teammates' code. That is expected. Use src/mocks.py and the handoff files (section 7).

## 6. How to fill in a stub

1. Open your file. Each function has the right name and signature and currently raises NotImplementedError.
2. Replace the raise line with your implementation. Keep the name, the argument names and defaults, and the output shape exactly as in the README contract.
3. Keep the code flat: plain functions, no classes, no inline comments.
4. Test it alone on data/raw/shared_test.wav. For modules that need another module's output, use src/mocks.py:

        from src.mocks import mock_frames, mock_energy, mock_labels, mock_lpc

5. Run the check listed for that task in the PDF task table (compare with librosa or scipy, plot, listen). A task is done only when its check passes.
6. Commit and push to your branch.

## 7. Shared test clip and handoff files (directly on main)

These are separate files that cannot conflict, so they go straight to main. First commit your own work on your branch, then:

    git switch main
    git pull origin main
    git add data/raw/shared_test.wav
    git commit -m "Add shared test clip"
    git push origin main
    git switch module_a

Use your own branch name in the last line.

**Shared test clip.** One person (Raks) records a whispered clip of two sentences, 16 kHz mono, quiet room, headset mic about 3 cm from the mouth, headphones on:

    python record.py data/raw/shared_test.wav 8

Listen to it, then push it using the steps above. This is the one audio file that is committed. Your own recordings follow the naming spk01_s03_whisper.wav and spk01_s03_normal.wav (speaker, sentence number, style) and stay on your laptop, git ignores them.

**Handoff files, due end of Sat 14 Nov.** Once your module works on the shared test clip, save its outputs. Hafsa:

    import numpy as np
    np.save("data/handoff/labels.npy", labels)
    np.save("data/handoff/energy.npy", energy)

Akhila:

    import numpy as np
    np.save("data/handoff/A.npy", A)
    np.save("data/handoff/G.npy", G)

Then push them to main with the same steps, replacing the add and commit lines with:

    git add data/handoff
    git commit -m "Add handoff files"

Rough values are fine. Raks loads these to test synthesis on real inputs without waiting for the code merge. Push updated versions whenever your results improve.

## 8. Changing shared files

For README.md, INSTRUCTIONS.md, requirements.txt and src/mocks.py: tell the group first, then change them on main only.

    git switch main
    git pull origin main
    git add requirements.txt
    git commit -m "Add jiwer"
    git push origin main
    git switch module_a

Everyone then gets the change through git merge origin/main at the start of their next session. If you add a library, tell everyone to run pip install again.

## 9. Raks: merging all branches into main

    git switch main
    git pull origin main
    git fetch origin
    git merge origin/module_a
    git merge origin/module_b
    git merge module_c
    git push origin main
    git switch module_c

Tell the group when it is done. Everyone then runs git fetch origin and git merge origin/main on their own branch.

## 10. What never goes into git

* Your own recordings, large audio and downloaded datasets (git already ignores data/raw except shared_test.wav and transcripts.txt)
* Anything in results/ (figures, wavs, tables). Share these on the group chat or a shared drive
* Virtual environment folders and notebook checkpoints
* Passwords, tokens or personal details

## 11. Notebooks

Validation notebooks go in notebooks/ and must run from top to bottom on a fresh clone using only files in the repo. Clear outputs before committing large ones. Name them after the module, for example notebooks/b_lpc_validation.ipynb.

## 12. If something goes wrong

* Committed on the wrong branch: tell the group before doing anything else. It is fixable
* ModuleNotFoundError: run pip install again, or you are in the wrong folder. Run commands from the repo root
* Push rejected: you are probably pushing a branch that someone else also pushed. Check git branch, then ask the group
* Cannot push (permission denied): ask Raks to confirm you are added as a collaborator on the repo
* Switching branch is blocked by uncommitted changes: commit them on your branch first
* No sound or crackle in live tests: headphones only, close other apps using the mic, do not plot inside the audio callback
* Lost work: git log shows past commits, ask the group before running any command you do not understand
* Stuck on a concept: use the follow up prompts at the end of your prompt page in the PDF