# -*- coding: utf-8 -*-
"""
Created on Sat Jun  19 15:18:29 2024

@author: UK
"""
import torch
import torchaudio

from transformers import AutoProcessor, AutoModelForCTC
import matplotlib.pyplot as plt

processor = AutoProcessor.from_pretrained("m3hrdadfi/wav2vec2-large-xlsr-turkish")
model = AutoModelForCTC.from_pretrained("m3hrdadfi/wav2vec2-large-xlsr-turkish")

from pvrecorder import PvRecorder
import wave
import struct

recorder = PvRecorder(device_index=0, frame_length=512)
audio = []

try:
    recorder.start()

    while True:
        print("Dinliyorum...")
        frame = recorder.read()
        audio.extend(frame)
except KeyboardInterrupt:
    print("Komut alındı.")
    recorder.stop()
    with wave.open("ses.wav", 'w') as f:
        f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
        f.writeframes(struct.pack("h" * len(audio), *audio))
finally:
    recorder.delete()

waveform, sample_rate = torchaudio.load("ses.wav")
waveform_resampled = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(waveform.t().numpy())
plt.title('Waveform (Ses Dosyası Dalgaformu)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
spe = torchaudio.transforms.Spectrogram()(waveform_resampled)
spe_ch1 = spe[0, :, :]
plt.imshow(spe_ch1.log2().numpy(), aspect='auto', cmap='viridis')
plt.title('Spectrogram')
plt.xlabel('Zaman')
plt.ylabel('Frekans')

plt.tight_layout()
plt.show()

with torch.no_grad():
    logits = model(waveform_resampled).logits

output_ids = torch.argmax(logits, dim=-1)
command = processor.batch_decode(output_ids)

print("Komutunuz:", command)


