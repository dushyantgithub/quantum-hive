import pyaudio

def main():
    pa = pyaudio.PyAudio()
    device_index = 1  # USB PnP Sound Device
    rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000]
    print(f'Testing device {device_index}')
    for rate in rates:
        try:
            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, input_device_index=device_index, frames_per_buffer=1024)
            stream.close()
            print(f'  Supported: {rate} Hz')
        except Exception as e:
            print(f'  Not supported: {rate} Hz ({e})')
    pa.terminate()

if __name__ == "__main__":
    main() 