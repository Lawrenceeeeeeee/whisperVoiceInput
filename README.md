# Whisper-Voice-Input

基于Whisper的语音输入脚本

## 依赖项

该项目使用以下Python库：

- numpy
- sounddevice
- soundfile
- pynput
- whisper
- pyperclip
- pyautogui

您可以使用pip安装这些依赖项：

```bash
pip install -r requirements.txt
```

Whisper的具体安装过程参见OpenAI的官方仓库[whisper](https://github.com/openai/whisper)

## 使用方法

在设置中开启权限,并将脚本放在后台运行. 脚本监听'option/alt'键, 连续按下两次开始录音，再按一次停止录音。然后使用Whisper库将录制的音频转录，并将转录复制到剪贴板, 并自动粘贴到光标处。

初次启动时需要安装模型,需要等待一段时间。

## 许可证

该项目在MIT许可证下授权 - 请查看LICENSE.md文件以获取详细信息。

