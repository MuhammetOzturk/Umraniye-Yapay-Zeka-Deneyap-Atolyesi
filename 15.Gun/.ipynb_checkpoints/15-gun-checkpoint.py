import time
import wave
import struct
import threading
from pynput import keyboard
from pvrecorder import PvRecorder

import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC

#argmax'a ve struct'a bak

class SesKaydedici:
    def __init__(self):
        self.ses = []
        self.is_parcacigi = None
        self.kaydedici = PvRecorder(frame_length=512)

        self.processor = AutoProcessor.from_pretrained("m3hrdadfi/wav2vec2-large-xlsr-turkish")
        self.model = AutoModelForCTC.from_pretrained("m3hrdadfi/wav2vec2-large-xlsr-turkish")

    def ses_kaydet(self):
        print("Kayıt başlatılıyor")
        self.kaydedici.start()
        while self.kaydedici.is_recording:
            self.ses.extend(self.kaydedici.read())
        #print(self.ses)
        with wave.open("ses.wav", 'w') as f:
            f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
            f.writeframes(struct.pack("h" * len(self.ses), *self.ses))
        self.ses = []
        print("Kayıt bitti")


    def on_press(self, key):
        try:
            if key.char == 'a' and self.is_parcacigi is None:
                self.is_parcacigi = threading.Thread(
                    target=self.ses_kaydet,
                    args=(),
                    daemon=True
                )
                self.is_parcacigi.start()
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key == keyboard.Key.esc:
                print("Çıkış yapılıyor.")
                return False
            
            if key.char == 'a':
                self.kaydedici.stop()
                self.is_parcacigi = None
                time.sleep(1)
                self.texte_cevir()
        except AttributeError:
            pass

    def texte_cevir(self):
        waveform, sample_rate = torchaudio.load("ses.wav")
        waveform_resampled = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        with torch.no_grad():
            logits = self.model(waveform_resampled).logits

        #argmax'a bak
        output_ids = torch.argmax(logits, dim=-1)
        command = self.processor.batch_decode(output_ids)

        print("Komutunuz:", command)

    def baslat(self):
        with keyboard.Listener(
                suppress=True,
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()



#if __name__ == "__main__":
kaydedici = SesKaydedici()
print("Kayit modu hazir")
kaydedici.baslat()
