import os
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard


# 全局变量
option_presses = 0  # 跟踪Option键按下的次数
recording = False  # 是否正在录音
audio_data = np.array([], dtype=np.float32)  # 存储录音数据
samplerate = 44100  # 录音的采样率


def record_callback(indata, frames, time, status):
    """这个回调函数在录音时被调用，用于收集输入数据"""
    global audio_data
    audio_data = np.append(audio_data, indata.copy())


def start_recording():
    global recording, audio_data
    print("开始录音...")
    audio_data = np.array([], dtype=np.float32)  # 重新初始化录音数据数组
    recording = True
    with sd.InputStream(callback=record_callback, samplerate=samplerate, channels=1, dtype='float32'):
        while recording:
            sd.sleep(100)


def stop_recording():
    global recording, audio_data
    print("停止录音，开始处理...")
    recording = False
    # 将录音数据保存到WAV文件中
    temp_filename = tempfile.mktemp(prefix='recording_', suffix='.wav', dir=None)
    sf.write(temp_filename, audio_data, samplerate)
    print(f"录音已保存到 {temp_filename}")
    # 这里可以添加Whisper处理逻辑
    print("开始转写...")
    res = transcribe_audio(temp_filename)
    copy_to_clipboard(res)
    paste()
    # 删除临时文件
    print(f"删除临时文件 {temp_filename}")
    os.remove(temp_filename)


def on_press(key):
    global option_presses, recording
    try:
        if key == keyboard.Key.alt:
            option_presses += 1
            if option_presses == 2 and not recording:
                # 使用线程来避免阻塞键盘监听
                threading.Thread(target=start_recording, daemon=True).start()
            elif option_presses == 3 and recording:
                stop_recording()
                option_presses = 0  # 重置按键次数，为下一次录音准备
    except Exception as e:
        print(e)


import whisper


def transcribe_audio(filename):
    # 加载模型，这里使用的是"small"模型，你也可以根据需求使用其他大小的模型
    model = whisper.load_model("small")

    # 使用Whisper模型进行语音转写
    result = model.transcribe(filename, initial_prompt="这是一段普通话语音")

    # 打印转写结果的文本
    print(result["text"])

    # 返回转写的文本，以便进一步处理
    return result["text"]


import pyperclip

def copy_to_clipboard(text):
    # 使用pyperclip将文本复制到剪贴板
    pyperclip.copy(text)
    print("文本已复制到剪贴板。")

import pyautogui

def paste():
    # 模拟按下并释放'ctrl+v'（Windows/Linux）或'command+v'（Mac）来执行粘贴操作
    # pyautogui.hotkey('ctrl', 'v')  # 对于Windows和Linux
    pyautogui.hotkey('command', 'v')  # 对于Mac



# 使用线程来执行录音和停止录音操作，避免阻塞键盘监听器
import threading

# 设置监听器监听键盘事件
listener = keyboard.Listener(on_press=on_press)
listener.start()
listener.join()
