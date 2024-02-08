import sys
import subprocess
import main
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QComboBox, QPushButton, QDialog
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5 import QtCore
from PyQt5.QtGui import QFont
import json

class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super(LoadingDialog, self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Loading")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("模型加载中，请稍后……（初次启动或者使用新模型时等待时间可能较长）"))
        self.setLayout(layout)

class OutputWatcherThread(QThread):
    outputReceived = pyqtSignal(str)

    def __init__(self, process):
        super().__init__()
        self.process = process

    def run(self):
        while True:
            output = self.process.stdout.readline()
            if output:
                self.outputReceived.emit(output.strip())
            else:
                break


class VoiceInputManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.process = None  # 用于存储当前运行的main.py进程
        self.loading_dialog = None
        self.output_watcher = None  # 存储用于监视输出的线程
        self.settings = self.load_settings()  # 加载预设
        if not self.settings:  # 如果预设为空，设置默认值
            self.settings = {"Model Size": "small", "Language": "简体中文", "Keybind": "option/alt"}
        self.initUI()
        self.start_main_script(self.settings['Model Size'], main.language_dict[self.settings['Language']])  # 应用启动时自动运行main.py

    def initUI(self):
        self.setWindowTitle('Whisper Voice Input 控制台')
        self.setFixedWidth(400)
        self.setWindowFlags(QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()

        # 模型大小下拉菜单
        self.model_size_combo = QComboBox(self)
        self.model_size_combo.addItems(['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'])
        self.model_size_combo.setCurrentText(self.settings['Model Size'])
        self.layout.addWidget(QLabel('Whisper模型规格:', self))
        self.layout.addWidget(self.model_size_combo)

        # 语言下拉菜单
        self.language_combo = QComboBox(self)
        self.language_combo.addItems(main.language_dict.keys())  # 添加更多语言选项
        self.language_combo.setCurrentText(self.settings['Language'])

        self.layout.addWidget(QLabel('识别语言（多语种请选Auto）:', self))
        self.layout.addWidget(self.language_combo)

        # 快捷键
        self.keybind_combo = QComboBox(self)
        self.keybind_combo.addItems(main.key_mapping.keys())
        self.keybind_combo.setCurrentText(self.settings['Keybind'])
        self.layout.addWidget(QLabel('快捷键（双击）:', self))
        self.layout.addWidget(self.keybind_combo)


        # 添加一个QLabel用于显示main.py的最后一行输出
        self.output_label = QLabel("", self)
        font = QFont('Arial', 12, QFont.StyleItalic)
        self.output_label.setFont(font)
        self.output_label.setStyleSheet("color: gray;")
        self.layout.addWidget(self.output_label)
        # self.setLayout(self.layout)

        self.central_widget.setLayout(self.layout)

        # 重启脚本的按钮
        self.restart_button = QPushButton('Apply Settings and Restart', self)
        self.restart_button.clicked.connect(self.restart_main_script)
        self.layout.addWidget(self.restart_button)

        self.central_widget.setLayout(self.layout)

    def start_main_script(self, model_size='small', language='zh'):
        if self.process:
            self.process.terminate()  # 如果已有进程在运行，则终止它
        # 使用subprocess启动main.py，并传递模型大小和语言作为参数
        self.process = subprocess.Popen(['python', 'main.py', f'--model_size={model_size}', f'--language={language}'],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                                        universal_newlines=True)
        if self.output_watcher:
            self.output_watcher.terminate()
            self.output_watcher.wait()

        self.output_watcher = OutputWatcherThread(self.process)
        self.output_watcher.outputReceived.connect(self.check_output)
        self.output_watcher.start()

        # 更新UI状态
        self.restart_button.setEnabled(False)

    def check_output(self, output):
        if output:
            # Get the width of the window
            window_width = self.width()

            # Estimate the width of a single character. This will depend on the font and its size.
            # Here, we assume each character is approximately 10 pixels wide. Adjust this value as needed.
            char_width = 10

            # Calculate the maximum length
            max_length = window_width // char_width

            if len(output) > max_length:
                output = output[:max_length] + '...'

            self.output_label.setText(output)  # 更新QLabel以显示最新输出
            if "Whisper模型已加载。" in output:
                self.restart_button.setEnabled(True)


    def restart_main_script(self):
        model_size = self.model_size_combo.currentText()
        language = self.language_combo.currentText()
        keybind = self.keybind_combo.currentText()
        self.save_settings({"Model Size": model_size, "Language": language, "Keybind": keybind})
        # 强制结束当前进程和监视线程
        if self.process:
            self.process.terminate()  # 终止当前正在运行的进程
            self.process.wait()  # 等待进程结束
        if hasattr(self, 'output_watcher') and self.output_watcher.isRunning():
            self.output_watcher.terminate()  # 强制结束线程
            self.output_watcher.wait()  # 等待线程结束

        # 重启main.py进程
        self.start_main_script(model_size, main.language_dict[language])


    def save_settings(self, settings, filename='settings.json', encoding='utf-8'):
        with open(filename, 'w') as f:
            json.dump(settings, f)

    # 加载预设
    def load_settings(self, filename='settings.json', encoding='utf-8'):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}  # 返回一个空字典，如果文件不存在

    def closeEvent(self, event):
        # 强制结束进程和监视线程
        if self.process:
            self.process.terminate()
            self.process.wait()
        if hasattr(self, 'output_watcher') and self.output_watcher.isRunning():
            self.output_watcher.terminate()
            self.output_watcher.wait()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VoiceInputManager()
    window.show()
    sys.exit(app.exec_())
