# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Stock_Screener1.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import re

class Ui_MainWindow(object):
    def __init__(self):
        self.bar_patterns = ["Choose Pattern", "Green VS Red Bars", "Two Crows", "Three Black Crows",
        "Three Inside Up/Down", "Three-Line Strike", "Three Outside Up/Down",
        "Three Stars In The South", "Three Advancing White Soldiers", "Abandoned Baby",
        "Advance Block", "Belt-hold", "Breakaway", "Closing Marubozu", "Concealing Baby Swallow",
        "Counterattack", "Dark Cloud Cover", "Doji", "Doji Star", "Dragonfly Doji", 
        "Engulfing Pattern", "Evening Doji Star", "Evening Star", "Up/Down-gap side-by-side white lines", 
        "Gravestone Doji", "Hammer", "Hanging Man", "Harami Pattern", "Harami Cross Pattern", 
        "High-Wave Candle", "Hikkake Pattern", "Modified Hikkake Pattern", "Homing Pigeon", 
        "Identical Three Crows", "In-Neck Pattern", "Inverted Hammer", "Kicking", 
        "Ladder Bottom", "Long Legged Doji", "Long Line Candle", "Marubozu", "Matching Low", 
        "Mat Hold", "Morning Doji Star", "Morning Star", "On-Neck Pattern", "Piercing Pattern", 
        "Rickshaw Man", "Rising/Falling Three Methods", "Separating Lines", "Shooting Star",
        "Short Line Candle", "Spinning Top", "Stalled Pattern", "Stick Sandwich", "Takuri",
        "Tasuki Gap", "Thrusting Pattern", "Tristar Pattern", "Unique 3 River", "Upside Gap Two Crows",
        "Upside/Downside Gap Three Methods"]
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1126, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1281, 701))
        self.tabWidget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_2.setGeometry(QtCore.QRect(40, 10, 113, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_2 = QtWidgets.QLabel(self.tab_3)
        self.label_2.setGeometry(QtCore.QRect(0, 10, 47, 14))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.tab_3)
        self.label_3.setGeometry(QtCore.QRect(0, 50, 47, 14))
        self.label_3.setObjectName("label_3")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.tab_3)
        self.textBrowser_3.setGeometry(QtCore.QRect(80, 51, 121, 20))
        self.textBrowser_3.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.label_4 = QtWidgets.QLabel(self.tab_3)
        self.label_4.setGeometry(QtCore.QRect(0, 110, 47, 14))
        self.label_4.setObjectName("label_4")
        self.textBrowser_4 = QtWidgets.QTextBrowser(self.tab_3)
        self.textBrowser_4.setGeometry(QtCore.QRect(80, 110, 121, 20))
        self.textBrowser_4.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_4.setObjectName("textBrowser_4")
        self.graphicsView = QtWidgets.QGraphicsView(self.tab_3)
        self.graphicsView.setGeometry(QtCore.QRect(0, 340, 381, 311))
        self.graphicsView.setObjectName("graphicsView")
        self.label_5 = QtWidgets.QLabel(self.tab_3)
        self.label_5.setGeometry(QtCore.QRect(10, 310, 47, 14))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.tab_3)
        self.label_6.setGeometry(QtCore.QRect(80, 310, 61, 16))
        self.label_6.setObjectName("label_6")
        self.lineEdit = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit.setGeometry(QtCore.QRect(40, 310, 31, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.label = QtWidgets.QLabel(self.tab_3)
        self.label.setGeometry(QtCore.QRect(0, 80, 61, 16))
        self.label.setObjectName("label")
        self.textBrowser_5 = QtWidgets.QTextBrowser(self.tab_3)
        self.textBrowser_5.setGeometry(QtCore.QRect(80, 81, 121, 20))
        self.textBrowser_5.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_5.setObjectName("textBrowser_5")
        self.label_7 = QtWidgets.QLabel(self.tab_3)
        self.label_7.setGeometry(QtCore.QRect(0, 140, 47, 14))
        self.label_7.setObjectName("label_7")
        self.textBrowser_6 = QtWidgets.QTextBrowser(self.tab_3)
        self.textBrowser_6.setGeometry(QtCore.QRect(80, 140, 121, 20))
        self.textBrowser_6.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_6.setObjectName("textBrowser_6")
        self.label_8 = QtWidgets.QLabel(self.tab_3)
        self.label_8.setGeometry(QtCore.QRect(0, 170, 47, 14))
        self.label_8.setObjectName("label_8")
        self.textBrowser_7 = QtWidgets.QTextBrowser(self.tab_3)
        self.textBrowser_7.setGeometry(QtCore.QRect(80, 170, 121, 20))
        self.textBrowser_7.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_7.setObjectName("textBrowser_7")
        self.label_9 = QtWidgets.QLabel(self.tab_3)
        self.label_9.setGeometry(QtCore.QRect(0, 200, 81, 16))
        self.label_9.setObjectName("label_9")
        self.textBrowser_8 = QtWidgets.QTextBrowser(self.tab_3)
        self.textBrowser_8.setGeometry(QtCore.QRect(80, 200, 121, 20))
        self.textBrowser_8.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_8.setObjectName("textBrowser_8")
        self.label_10 = QtWidgets.QLabel(self.tab_3)
        self.label_10.setGeometry(QtCore.QRect(0, 230, 61, 16))
        self.label_10.setObjectName("label_10")
        self.textBrowser_9 = QtWidgets.QTextBrowser(self.tab_3)
        self.textBrowser_9.setGeometry(QtCore.QRect(80, 230, 121, 20))
        self.textBrowser_9.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_9.setObjectName("textBrowser_9")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_3.setGeometry(QtCore.QRect(50, 10, 113, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.label_11 = QtWidgets.QLabel(self.tab)
        self.label_11.setGeometry(QtCore.QRect(10, 10, 47, 14))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.tab)
        self.label_12.setGeometry(QtCore.QRect(170, 10, 71, 16))
        self.label_12.setObjectName("label_12")
        self.comboBox = QtWidgets.QComboBox(self.tab)
        self.comboBox.setGeometry(QtCore.QRect(10, 40, 221, 22))
        self.comboBox.setObjectName("comboBox")
        for i in self.bar_patterns:
            self.comboBox.addItem("")
        self.spinBox = QtWidgets.QSpinBox(self.tab)
        self.spinBox.setGeometry(QtCore.QRect(90, 80, 42, 22))
        self.spinBox.setObjectName("spinBox")
        self.label_14 = QtWidgets.QLabel(self.tab)
        self.label_14.setGeometry(QtCore.QRect(140, 80, 31, 16))
        self.label_14.setObjectName("label_14")
        self.label_13 = QtWidgets.QLabel(self.tab)
        self.label_13.setGeometry(QtCore.QRect(10, 80, 81, 21))
        self.label_13.setObjectName("label_13")
        self.label_15 = QtWidgets.QLabel(self.tab)
        self.label_15.setGeometry(QtCore.QRect(10, 110, 331, 281))
        self.label_15.setAutoFillBackground(False)
        self.label_15.setText("")
        self.label_15.setPixmap(QtGui.QPixmap("img/bar-patterns/choose-pattern.jpg"))
        self.label_15.setScaledContents(True)
        self.label_15.setObjectName("label_15")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1126, 22))
        self.menubar.setObjectName("menubar")
        self.menuMain = QtWidgets.QMenu(self.menubar)
        self.menuMain.setObjectName("menuMain")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionSettings = QtWidgets.QAction(MainWindow)
        self.actionSettings.setObjectName("actionSettings")
        self.actionAppendix = QtWidgets.QAction(MainWindow)
        self.actionAppendix.setObjectName("actionAppendix")
        self.menuMain.addAction(self.actionSettings)
        self.menuMain.addAction(self.actionAppendix)
        self.menubar.addAction(self.menuMain.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Stock Screener"))
        self.label_2.setText(_translate("MainWindow", "Ticker"))
        self.label_3.setText(_translate("MainWindow", "Volume"))
        self.label_4.setText(_translate("MainWindow", "Close"))
        self.label_5.setText(_translate("MainWindow", "Last"))
        self.label_6.setText(_translate("MainWindow", "Days OHLC"))
        self.label.setText(_translate("MainWindow", "10-Avg Vol"))
        self.label_7.setText(_translate("MainWindow", "QRG"))
        self.label_8.setText(_translate("MainWindow", "EPS"))
        self.label_9.setText(_translate("MainWindow", "Dividend Yield"))
        self.label_10.setText(_translate("MainWindow", "Market Cap"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Stock Info"))
        self.label_11.setText(_translate("MainWindow", "Ticker"))
        self.label_12.setText(_translate("MainWindow", "Loading status"))
        for i in range(len(self.bar_patterns)):
            self.comboBox.setItemText(i, _translate("MainWindow", self.bar_patterns[i]))
        self.label_14.setText(_translate("MainWindow", "days"))
        self.label_13.setText(_translate("MainWindow", "Analysis Range"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Analysis"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Create Algo"))
        self.menuMain.setTitle(_translate("MainWindow", "File"))
        self.actionSettings.setText(_translate("MainWindow", "Settings"))
        self.actionAppendix.setText(_translate("MainWindow", "Appendix"))
        
        self.comboBox.currentIndexChanged.connect(self.on_change)
        
    def on_change(self, newIndex):
        pattern = self.comboBox.currentText()
        pattern = re.sub(r'/', '-', pattern).lower()
        pattern = re.sub(' ', '-', pattern).lower()
        print(pattern)
        self.label_15.setPixmap(QtGui.QPixmap("img/bar-patterns/{}.jpg".format(pattern)))
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

