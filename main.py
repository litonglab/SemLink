from PyQt6 import QtWidgets
import sys
from App import App  # 假设你的App类定义在app.py文件中

def main():
    app = QtWidgets.QApplication(sys.argv)  # 创建QApplication实例
    window = App()  # 创建App类的实例
    window.show()  # 显示窗口
    sys.exit(app.exec())  # 启动事件循环，并在窗口关闭后退出

if __name__ == '__main__':
    main()
