from thebeat import SoundStimulus
import matplotlib.pyplot as plt
# r1 = SoundStimulus.from_wav(filepath="Recordings (rhythm + 12-8 beat R1).wav", name="R1 + 12-8 beat")
# fig, ax = r1.plot_waveform(title="Waveform of R1 + 12-8 beat")
# fig.set_facecolor('blue')
# plt.show()

from extractor import extract_onsets
__, __, iois = extract_onsets("Recordings (rhythm alone R1).wav")
print(iois)