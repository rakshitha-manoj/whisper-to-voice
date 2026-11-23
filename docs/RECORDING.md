# Recording guide

For Hafsa, Akhila and Raks. You do not need to know anything about audio. Set aside about one hour and take breaks.

## What you will record

Each person records all 20 sentences twice: whispered and in a normal voice. That is 40 files per person.

| Speaker id | Person |
|---|---|
| 01 | Hafsa |
| 02 | Akhila |
| 03 | Raks |

Only Hafsa also records the shared clip (`data/raw/shared_test.wav`: whispered sentences 01 and 02 back to back, about 12 seconds).

The script names and saves every file for you. You never type a file name.

## One-time setup

1. Open a terminal in the project folder (commit your own work first), then `git switch main` and `git pull origin main`. Check that `tools/record_session.py` exists.
2. If you use a virtual environment, activate it. Mac: `source .venv/bin/activate`. Windows PowerShell: `.venv\Scripts\Activate.ps1`; Git Bash: `source .venv/Scripts/activate` (use your folder name if it is not `.venv`). Then run `python -m pip install -r requirements.txt` (Mac without a venv: `python3 -m pip install -r requirements.txt`).
3. Microphone permission.
   * Mac: the first time you record, macOS asks whether Terminal (or VS Code) may use the microphone: click Allow. If you clicked Don't Allow, go to System Settings, Privacy and Security, Microphone, switch on Terminal or VS Code (whichever you run the script from) and restart it. Without permission the Mac records pure silence and every take says "Too quiet".
   * Windows: Settings, Privacy and security, Microphone: turn on "Microphone access" and "Let desktop apps access your microphone".
4. Use a wired headset if you can. Bluetooth earbuds (AirPods and similar) switch to a low-quality call mode when their mic is used and apply their own noise suppression, which removes whispers. Plug the headset in and set it as the default input and output in the system sound settings (Windows: Settings, System, Sound, Input and Output. Mac: System Settings, Sound, Input and Output). If you cannot, run `python tools/record_session.py --list-devices` and add `--device N` (N = number of your headset microphone; on Windows choose one marked MME) to every command below.
5. Turn off audio enhancements and noise suppression for the microphone (Windows: Settings, System, Sound, your mic, Audio enhancements off; also switch off any "noise cancellation" in Teams, Zoom or headset software. Mac: use "Standard" mic mode, not Voice Isolation). Noise suppression erases whispers.
6. Input volume: in the same sound settings set the microphone input volume to about 70 to 80 percent to start. If every whisper says "Too quiet", raise it; if normal sentences say "Clipping", lower it. Do not change it in the middle of the session.

The script prints which input and output device it uses when it starts. Check that it is your headset. If it stops with "cannot record or play 16 kHz mono", pick another device as the message says.

## Room and microphone

* Quiet room, door closed, fans and air conditioning off, phone on silent.
* Headset microphone 2 to 5 cm from your mouth, slightly to the side so that breath does not blow straight onto it.
* Wear headphones (the script plays each take back, and you do not want the speakers feeding into the mic).
* Keep the same position and distance for the whole session.

## How to whisper and how to speak

**Whisper:** no voice at all, only breath. Put your fingers on your throat: you should feel no buzz. Keep a normal speaking pace and say the sentence like you mean it. Do not over-articulate.

**Normal:** your usual voice and pitch, as if you were saying it to a friend. Do not read it in a flat list voice.

For both styles:

* Press Enter, stay silent through the countdown and "Recording... stay silent", and start only when "...speak now" appears (half a second into the recording). The first 0.3 s of every take is used to measure the room noise.
* Stay silent after the last word until the recording ends. Do not clear your throat or say "okay".
* If you stumble, choose redo. Do not try to fix it in the next take.
* Four sentences are questions (07, 11, 16 and 18). Say them naturally, with a rising question tone in the normal version. The system makes a falling pitch contour and cannot produce a rising question tone. This is a known limitation, but record them naturally anyway.

## Commands

Windows: `python`. Mac: `python3` (use whichever works for you). Run from the project folder.

    python tools/record_session.py --speaker 02

For each sentence the script shows the text and the style, counts down 3, 2, 1, records 5 seconds, plays the take back and asks:

    [k]eep  [r]edo  [p]lay again  [s]kip  [q]uit

Only a kept take is saved. You can quit at any time (q or Ctrl-C) and continue later: files that exist are skipped.

| What you want | Command |
|---|---|
| Start or resume | `python tools/record_session.py --speaker 02` |
| Only whispered or only normal | add `--styles whisper` or `--styles normal` |
| Start from sentence 12 | add `--from 12` |
| Redo one sentence (both styles) | `python tools/record_session.py --speaker 02 --only 12 --redo` |
| Redo one style of one sentence | `--only 12 --styles whisper --redo` |
| Longer takes | add `--seconds 7` |
| Choose microphone or headphones | `--list-devices`, then `--device N` and `--output-device M` |
| Shared clip (Hafsa only) | `python tools/record_session.py --shared` (then it prints the git commands to commit it) |
| Make the zip | add `--zip`, or answer y when asked at the end |

At the end the script prints how many of the 40 files exist and which are missing, and offers to create `spk02_recordings.zip` in the project folder.

## Warnings after a take

The script checks each take and may print a warning. You can still keep the take, but usually you should redo it.

| Warning | Meaning | What to do |
|---|---|---|
| Clipping | The signal hit the maximum, which distorts it. | Move the mic a bit further away, speak or whisper a bit softer. Redo. |
| Too quiet | Almost nothing was recorded: wrong or muted mic, no microphone permission (Mac), or the input volume is very low. | Check the device the script printed, the permission and the input volume; move the mic closer. Redo. |
| Noise or speech in the first 0.3 s | You started too early, or the room noise is loud compared to your voice. If it appears on every take, the room or the mic is the problem. | Wait for "...speak now". Make the room quieter or move the mic closer. Redo. |
| Sound in the last 0.3 s: may be cut off | You were still speaking when the recording ended, or you made a noise. | Start sooner or speak a little faster; use `--seconds 7` for long sentences. Redo. |

No warning does not guarantee a good take. Listen to the playback.

## The 20 sentences

| Id | Type | Sentence |
|---|---|---|
| 01 | Harvard | The birch canoe slid on the smooth planks. |
| 02 | Harvard | Glue the sheet to the dark blue background. |
| 03 | Harvard | It is easy to tell the depth of a well. |
| 04 | Harvard | These days a chicken leg is a rare dish. |
| 05 | Harvard | Rice is often served in round bowls. |
| 06 | Harvard | The juice of lemons makes fine punch. |
| 07 | conversational (question) | Could you please pass me the salt? |
| 08 | conversational | I think we should leave a little earlier tomorrow. |
| 09 | conversational | The meeting has been moved to Friday afternoon. |
| 10 | conversational | She forgot her umbrella at the bus stop. |
| 11 | conversational (question) | Do you want to grab some coffee after class? |
| 12 | conversational | My phone battery is almost dead again. |
| 13 | conversational | We need to finish the project before the weekend. |
| 14 | conversational | Please switch off the lights when you leave the room. |
| 15 | conversational | The library is closed on Sunday mornings. |
| 16 | conversational (question) | Have you seen my keys anywhere? |
| 17 | conversational | He usually walks his dog along the river. |
| 18 | conversational (question) | Shall we order pizza for dinner tonight? |
| 19 | conversational | The train was late because of the heavy rain. |
| 20 | conversational | Thank you so much for helping me with this. |

## Sending your recordings

1. Make the zip (`--zip`). It is called `spk02_recordings.zip` (your id) and contains your 40 WAV files.
2. Send it to Raks through Google Drive, or WhatsApp as a **document** (not as audio or media), so the audio is not recompressed.
3. WAV files never go into git. Git ignores them. The only audio file that is committed is `shared_test.wav`.

## For Raks: running the evaluation

1. Unzip each zip into `data/raw/`. Check that there are 120 files named `spkNN_sMM_whisper.wav` or `_normal.wav` (3 x 40), plus `shared_test.wav` from main.
2. Run everything: `python run_eval.py`. Results go to `results/`.
3. Split by speaker, as the team guide says: tune the settings (base pitch, warp, order) on spk01 and spk03 (`python run_eval.py --speakers 01 03`) and report the final numbers on the held-out speaker spk02 (`python run_eval.py --speakers 02`). Use `--results results/spk02` if you do not want one run to overwrite the other. `--speakers` ignores `shared_test.wav` unless you add `shared`.
4. Known open point: the base pitch is a fixed 120 Hz, which sounds male. A per-speaker base pitch (a slider setting for now) is not estimated automatically. Say so in the viva.
