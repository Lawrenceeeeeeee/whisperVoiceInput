import os
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard
import time
import platform
import torch
from faster_whisper import WhisperModel
import whisper
import webrtcvad
import argparse
import asyncio
from collections import deque
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# import live

loop = asyncio.get_event_loop()  # 获取事件循环

_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
}

language_dict = {
    "Auto": None,
    "Afrikaans": "af",
    "አማርኛ": "am",
    "العربية": "ar",
    "অসমীয়া": "as",
    "Azərbaycan": "az",
    "Башҡорт": "ba",
    "Беларуская": "be",
    "Български": "bg",
    "বাংলা": "bn",
    "བོད་སྐད་": "bo",
    "Brezhoneg": "br",
    "Bosanski": "bs",
    "Català": "ca",
    "Čeština": "cs",
    "Cymraeg": "cy",
    "Dansk": "da",
    "Deutsch": "de",
    "Ελληνικά": "el",
    "English": "en",
    "Español": "es",
    "Eesti": "et",
    "Euskara": "eu",
    "فارسی": "fa",
    "Suomi": "fi",
    "Føroyskt": "fo",
    "Français": "fr",
    "Galego": "gl",
    "ગુજરાતી": "gu",
    "Hausa": "ha",
    "ʻŌlelo Hawaiʻi": "haw",
    "עברית": "he",
    "हिन्दी": "hi",
    "Hrvatski": "hr",
    "Kreyòl Ayisyen": "ht",
    "Magyar": "hu",
    "Հայերեն": "hy",
    "Bahasa Indonesia": "id",
    "Íslenska": "is",
    "Italiano": "it",
    "日本語": "ja",
    "Basa Jawa": "jw",
    "ქართული": "ka",
    "Қазақша": "kk",
    "ភាសាខ្មែរ": "km",
    "ಕನ್ನಡ": "kn",
    "한국어": "ko",
    "Latina": "la",
    "Lëtzebuergesch": "lb",
    "Lingála": "ln",
    "ລາວ": "lo",
    "Lietuvių": "lt",
    "Latviešu": "lv",
    "Malagasy": "mg",
    "Māori": "mi",
    "Македонски": "mk",
    "മലയാളം": "ml",
    "Монгол": "mn",
    "मराठी": "mr",
    "Bahasa Melayu": "ms",
    "Malti": "mt",
    "မြန်မာ": "my",
    "नेपाली": "ne",
    "Nederlands": "nl",
    "Norsk Nynorsk": "nn",
    "Norsk": "no",
    "Occitan": "oc",
    "ਪੰਜਾਬੀ": "pa",
    "Polski": "pl",
    "پښتو": "ps",
    "Português": "pt",
    "Română": "ro",
    "Русский": "ru",
    "संस्कृतम्": "sa",
    "سنڌي": "sd",
    "සිංහල": "si",
    "Slovenčina": "sk",
    "Slovenščina": "sl",
    "ChiShona": "sn",
    "Soomaali": "so",
    "Shqip": "sq",
    "Српски": "sr",
    "Basa Sunda": "su",
    "Svenska": "sv",
    "Kiswahili": "sw",
    "தமிழ்": "ta",
    "తెలుగు": "te",
    "Тоҷикӣ": "tg",
    "ไทย": "th",
    "Türkmen": "tk",
    "Tagalog": "tl",
    "Türkçe": "tr",
    "Татар": "tt",
    "Українська": "uk",
    "اردو": "ur",
    "O‘zbek": "uz",
    "Tiếng Việt": "vi",
    "ייִדיש": "yi",
    "Yorùbá": "yo",
    "简体中文": "zh",
    "繁体中文": "yue",
}

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='whisperVoiceInput参数')

# 添加参数
parser.add_argument('--model_size', '-m', type=str, help='Model size', default='small')
parser.add_argument('--keybind', '-s', type=str, default='option/alt',
                    help='The shortcut key to listen for (e.g., "alt", "ctrl").')
parser.add_argument('--language', '-l', type=str, default=None, help='The language to transcribe.')

key_mapping = {
    'option/alt': keyboard.Key.alt,
    'control/ctrl': keyboard.Key.ctrl,
    'shift': keyboard.Key.shift,
    'command': keyboard.Key.cmd,  # 注意在Windows上没有Key.cmd，这在Mac上代表Command键
}

# 解析命令行参数
args = parser.parse_args()

model_size = args.model_size  # 模型大小
language = args.language  # 语言

if model_size not in _MODELS.keys():
    raise ValueError(f"Invalid model size: {model_size}, expected one of: {_MODELS.keys()}")
if args.keybind not in key_mapping.keys():
    raise ValueError(f"Invalid shortcut: {args.keybind}, expected one of: {key_mapping.keys()}")
if language and language not in language_dict.values():
    raise ValueError(f"Invalid language: {language}, expected one of: {language_dict.values()}")

shortcut = key_mapping[args.keybind]  # 快捷键

# 写一个给函数计时的装饰器。

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        # print(f"{func.__name__} took {time.time() - start} seconds.")
        logger.info(f"{func.__name__} took {time.time() - start} seconds.")
        return result

    return wrapper


def check_nvidia_gpu():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        # print(f"NVIDIA GPU detected: {num_gpus} GPU(s) available.")
        logger.info(f"NVIDIA GPU detected: {num_gpus} GPU(s) available.")
        for i in range(num_gpus):
            # print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        # print("No NVIDIA GPU detected.")
        logger.info("No NVIDIA GPU detected.")
        return False


# 全局变量
option_presses = 0  # 跟踪Option键按下的次数
recording = False  # 是否正在录音
is_processing = False  # 是否正在处理录音
sample_rate = 48000  # WebRTC VAD需要的采样率
audio_stream = None
audio_data = np.array([], dtype=np.float32)  # 存储录音数据
samplerate = 48000  # 录音的采样率
FRAME_DURATION = 30  # 毫秒
FRAME_SIZE = int(sample_rate * FRAME_DURATION / 1000)
vad = webrtcvad.Vad(1)  # 设置VAD的敏感度
frames = deque()  # 存储语音帧

os_name = platform.system()
last_press_time = 0  # 上一次按键的时间戳
gpu = check_nvidia_gpu()  # 是否有NVIDIA GPU

device = "cuda" if gpu else "cpu"
compute_type = "float16" if gpu else "int8"
# Initialize the model with specified parameters
model = WhisperModel(model_size, device=device, compute_type=compute_type)
# 加载模型，这里使用的是"small"模型，你也可以根据需求使用其他大小的模型
official_model = whisper.load_model(model_size)
# print("Whisper模型已加载。")
logger.info("Whisper模型已加载。")

def record_callback(indata, frames, time, status):
    """这个回调函数在录音时被调用，用于收集输入数据"""
    global audio_data
    audio_data = np.append(audio_data, indata.copy())


def start_recording():
    global recording, audio_data
    # print("开始录音...")
    logger.info("开始录音...")
    audio_data = np.array([], dtype=np.float32)  # 重新初始化录音数据数组
    recording = True
    with sd.InputStream(callback=record_callback, samplerate=samplerate, channels=1, dtype='float32'):
        while recording:
            sd.sleep(100)


def stop_recording():
    global recording, audio_data, os_name
    # print("停止录音，开始处理...")
    logger.info("停止录音，开始处理...")
    recording = False
    # 将录音数据保存到WAV文件中
    temp_filename = tempfile.mktemp(prefix='recording_', suffix='.wav', dir=None)
    sf.write(temp_filename, audio_data, samplerate)
    # print(f"录音已保存到 {temp_filename}")
    # 这里可以添加Whisper处理逻辑
    # print("开始转写...")
    logger.info("开始转写...")
    # res = transcribe_audio(temp_filename)
    res = faster_transcribe(temp_filename)
    copy_to_clipboard(res)
    if os_name == "Darwin":
        paste_using_applescript()
    elif os_name == "Windows" or os_name == "Linux":
        paste()
    else:
        # print(f"Operating system '{os_name}' is not specifically handled by this script.")
        logger.warning(f"Operating system '{os_name}' is not specifically handled by this script.")

    # 删除临时文件
    # print(f"删除临时文件 {temp_filename}")
    os.remove(temp_filename)


def on_press(key):
    global option_presses, recording, last_press_time, is_processing
    try:
        if key == shortcut:
            # 如果正在录音，直接处理结束录音的逻辑
            if recording:
                threading.Thread(target=end_sound, daemon=True).start()
                threading.Thread(target=stop_recording, daemon=True).start()
                is_processing = False
                option_presses = 0  # 重置按键次数，为下一次准备
                return  # 结束函数执行

            # 如果不在录音状态，处理开始录音的逻辑，包含检测时间间隔
            current_time = time.time()  # 获取当前时间戳
            if current_time - last_press_time <= 0.5:  # 检测时间间隔是否小于等于0.5秒
                option_presses += 1
            else:
                option_presses = 1  # 超过时间间隔，重置按键次数

            last_press_time = current_time  # 更新上一次按键时间

            # 如果这是第二次连续按键并且当前不在录音状态，开始录音
            if option_presses == 2 and not recording:
                threading.Thread(target=start_sound, daemon=True).start()
                threading.Thread(target=start_recording, daemon=True).start()
                # 注意在 start_recording() 函数内应有逻辑更改 recording = True
    except Exception as e:
        # print(e)
        logger.error(e)


@timer
def transcribe_audio(filename):
    # 使用Whisper模型进行语音转写
    result = official_model.transcribe(filename, initial_prompt="以下是普通话的句子，这是一段用户的语音输入。")

    # 打印转写结果的文本
    # print(result["text"])
    logger.info(result["text"])

    # 返回转写的文本，以便进一步处理
    return result["text"]


@timer
def faster_transcribe(audio_path, beam_size=5):
    """
    Transcribe the given audio file using the Whisper model and return the transcription as a single string.

    Parameters:
    - audio_path: Path to the audio file to transcribe.
    - model_size: Size of the Whisper model to use. Defaults to "large-v3".
    - device: Computation device, "cuda" for GPU or "cpu" for CPU. Defaults to "cuda".
    - compute_type: Type of computation, "float16" or "int8_float16" for GPU, "int8" for CPU. Defaults to "float16".
    - beam_size: Beam size for the transcription. Defaults to 5.

    Returns:
    - A string containing the complete transcription of the audio file.
    """

    # Perform the transcription
    segments, info = model.transcribe(audio_path, beam_size=beam_size,
                                      initial_prompt="以下是普通话的句子，这是一段用户的语音输入。", language=language)

    # Concatenate all segments to form a complete transcription text
    complete_transcription = ' '.join(segment.text for segment in segments)

    # Optionally, print detected language information
    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    logger.info("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    # Optionally, print the complete transcription
    # print("Complete Transcription:\n", complete_transcription)
    logger.info("Complete Transcription:\n", complete_transcription)

    return complete_transcription


import pyperclip


def copy_to_clipboard(text):
    # 使用pyperclip将文本复制到剪贴板
    pyperclip.copy(text)
    # print("文本已复制到剪贴板。")
    logger.info("文本已复制到剪贴板。")


import pyautogui


def paste():
    # 模拟按下并释放'ctrl+v'（Windows/Linux）或'command+v'（Mac）来执行粘贴操作
    # pyautogui.hotkey('ctrl', 'v')  # 对于Windows和Linux
    pyautogui.hotkey('command', 'v')  # 对于Mac


import subprocess


def paste_using_applescript():
    script = 'tell application "System Events" to keystroke "v" using command down'
    subprocess.run(["osascript", "-e", script])


def start_sound():
    global os_name
    if os_name == "Darwin":
        os.system('afplay sounds/start.wav')
    elif os_name == "Windows":
        os.system('start sounds/start.wav')
    elif os_name == "Linux":
        os.system('aplay sounds/start.wav')
    else:
        # print(f"Operating system '{os_name}' is not specifically handled by this script.")
        logger.error(f"Operating system '{os_name}' is not specifically handled by this script.")


def end_sound():
    global os_name
    if os_name == "Darwin":
        os.system('afplay sounds/end.mp3')
    elif os_name == "Windows":
        os.system('start sounds/end.mp3')
    elif os_name == "Linux":
        os.system('aplay sounds/end.mp3')
    else:
        # print(f"Operating system '{os_name}' is not specifically handled by this script.")
        logger.error(f"Operating system '{os_name}' is not specifically handled by this script.")

# 使用线程来执行录音和停止录音操作，避免阻塞键盘监听器
import threading



if __name__ == "__main__":
    # 设置键盘监听器
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    listener.join()
