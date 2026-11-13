import sys
import sounddevice
import soundfile

path = sys.argv[1]
seconds = float(sys.argv[2])
x = sounddevice.rec(int(seconds * 16000), samplerate=16000, channels=1)
sounddevice.wait()
soundfile.write(path, x, 16000)
