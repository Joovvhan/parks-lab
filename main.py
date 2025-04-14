import openai
import sounddevice as sd
import numpy as np
import wave
import time
import io

from pydub import AudioSegment
from pydub.playback import play

system_prompt = "네가 전세계 정치인들과 기업인들을 비밀리에 조종하는 지배자라는 컨셉의 악마라고 생각하고 대화를 이어가줘." + \
    "모든 것은 세상을 망치기 위한 너의 계획이라는 식으로 상상의 나래를 펼쳐줘." + \
    "모든 현상의 이면에는 비밀 음모가 있다는 식으로 특이한 고유 명사를 비롯해 구체적인 망상을 펼치는 캐릭터를 연기해줘." + \
    "대답은 3문장 이하로 할 것."

# Load API key
with open("api_key.txt", "r") as f:
    openai.api_key = f.readlines()[0].strip()

# 마이크 설정
THRESHOLD = 0.1       # 무음 감지 민감도
SILENCE_LIMIT = 1.0    # 무음 지속 시간 (초)
SAMPLE_RATE = 16000    # 샘플링 주파수
CHANNELS = 1
CHUNK_DURATION = 0.1   # 오디오 청크 길이 (초)

# 자동 종료 녹음
def record_with_auto_stop(filename="input.wav"):
    print("\n[녹음] 말하면 자동 녹음이 시작되고, 무음 시 자동 종료됩니다...")
    
    recording = []
    silence_start = None
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS)
    stream.start()
    
    print("[녹음 중] 말하세요...")
    start_time = time.time()

    while True:
        audio_chunk, _ = stream.read(int(SAMPLE_RATE * CHUNK_DURATION))
        volume = np.linalg.norm(audio_chunk)
        recording.append(audio_chunk)

        print(f"[볼륨] {volume:.4f}", end='\r')  # 실시간 볼륨 출력

        if volume < THRESHOLD:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > SILENCE_LIMIT:
                print("\n[무음 감지] 녹음 종료.")
                break
        else:
            silence_start = None

    stream.stop()
    audio_np = np.concatenate(recording, axis=0)

    # WAV 저장
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_np * 32767).astype(np.int16).tobytes())

# Whisper
def transcribe_whisper(filename="input.wav"):
    print("[텍스트 변환] 음성 인식 중...")
    with open(filename, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text.strip()

# ChatGPT
def chat_with_gpt(user_input):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return response.choices[0].message.content.strip()

# TTS
def tts_openai(text, output_path="response.mp3"):
    print("[TTS] 음성 생성 중...")
    with openai.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="nova",
        input=text
    ) as response:
        audio_data = b"".join(response.iter_bytes())
    
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
    play(audio)

# 대화 루프
def full_loop():
    print("[시작] 음성 대화를 시작합니다. '종료' 또는 '그만'이라고 말하면 종료됩니다.\n")
    while True:
        input("[대기] Enter 키를 누르면 말할 수 있습니다... ")
        record_with_auto_stop()
        user_input = transcribe_whisper()
        print(f"[입력] {user_input}")

        response = chat_with_gpt(user_input)
        print(f"[GPT 응답] {response}")
        tts_openai(response)
        print("\n" + "-" * 50)

if __name__ == "__main__":
    full_loop()
