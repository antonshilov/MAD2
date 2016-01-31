import sys
# Импортируем минимальный набор виджетов
from PyQt5 import QtWidgets
# Импортируем созданный нами класс со слотами
from gui import MainWindowSlots


# Создаём ещё один класс, наследуясь от класса со слотами
class MainWindow(MainWindowSlots):
    # При инициализации класса нам необходимо выпонить некоторые операции
    def __init__(self, form):
        # Сконфигурировать интерфейс методом из базового класса Ui_Form
        self.setupUi(form)
        # Подключить созданные нами слоты к виджетам
        self.connect_slots()

    # Подключаем слоты к виджетам
    def connect_slots(self):
        self.calc_btn.clicked.connect(self.calc)
        return None


if __name__ == '__main__':
    # Создаём экземпляр приложения
    app = QtWidgets.QApplication(sys.argv)
    # Создаём базовое окно, в котором будет отображаться наш UI
    window = QtWidgets.QMainWindow()
    # Создаём экземпляр нашего UI
    ui = MainWindow(window)
    # Отображаем окно
    window.show()
    # Обрабатываем нажатие на кнопку окна "Закрыть"
    sys.exit(app.exec_())
