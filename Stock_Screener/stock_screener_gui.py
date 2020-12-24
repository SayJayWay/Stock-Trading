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
        self.stockInfoTab = QtWidgets.QWidget()
        self.stockInfoTab.setObjectName("tab_3")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.stockInfoTab)
        self.lineEdit_2.setGeometry(QtCore.QRect(40, 10, 113, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.stockInfoTickerLabel = QtWidgets.QLabel(self.stockInfoTab)
        self.stockInfoTickerLabel.setGeometry(QtCore.QRect(0, 10, 47, 14))
        self.stockInfoTickerLabel.setObjectName("stockInfoTickerLabel")
        self.stockInfoVolumeLabel = QtWidgets.QLabel(self.stockInfoTab)
        self.stockInfoVolumeLabel.setGeometry(QtCore.QRect(0, 50, 47, 14))
        self.stockInfoVolumeLabel.setObjectName("stockInfoVolumeLabel")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.stockInfoTab)
        self.textBrowser_3.setGeometry(QtCore.QRect(80, 51, 121, 20))
        self.textBrowser_3.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.stockInfoCloseLabel = QtWidgets.QLabel(self.stockInfoTab)
        self.stockInfoCloseLabel.setGeometry(QtCore.QRect(0, 110, 47, 14))
        self.stockInfoCloseLabel.setObjectName("stockInfoCloseLabel")
        self.textBrowser_4 = QtWidgets.QTextBrowser(self.stockInfoTab)
        self.textBrowser_4.setGeometry(QtCore.QRect(80, 110, 121, 20))
        self.textBrowser_4.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_4.setObjectName("textBrowser_4")
        self.graphicsView = QtWidgets.QGraphicsView(self.stockInfoTab)
        self.graphicsView.setGeometry(QtCore.QRect(0, 340, 381, 311))
        self.graphicsView.setObjectName("graphicsView")
        self.stockInfoLastLabel = QtWidgets.QLabel(self.stockInfoTab)
        self.stockInfoLastLabel.setGeometry(QtCore.QRect(10, 310, 47, 14))
        self.stockInfoLastLabel.setObjectName("stockInfoLastLabel")
        self.stockInfoDaysOHLCLabel = QtWidgets.QLabel(self.stockInfoTab)
        self.stockInfoDaysOHLCLabel.setGeometry(QtCore.QRect(80, 310, 61, 16))
        self.stockInfoDaysOHLCLabel.setObjectName("stockInfoDaysOHLCLabel")
        self.lineEdit = QtWidgets.QLineEdit(self.stockInfoTab)
        self.lineEdit.setGeometry(QtCore.QRect(40, 310, 31, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.label = QtWidgets.QLabel(self.stockInfoTab)
        self.label.setGeometry(QtCore.QRect(0, 80, 61, 16))
        self.label.setObjectName("label")
        self.textBrowser_5 = QtWidgets.QTextBrowser(self.stockInfoTab)
        self.textBrowser_5.setGeometry(QtCore.QRect(80, 81, 121, 20))
        self.textBrowser_5.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_5.setObjectName("textBrowser_5")
        self.stockInfoQRGLabel = QtWidgets.QLabel(self.stockInfoTab)
        self.stockInfoQRGLabel.setGeometry(QtCore.QRect(0, 140, 47, 14))
        self.stockInfoQRGLabel.setObjectName("stockInfoQRGLabel")
        self.textBrowser_6 = QtWidgets.QTextBrowser(self.stockInfoTab)
        self.textBrowser_6.setGeometry(QtCore.QRect(80, 140, 121, 20))
        self.textBrowser_6.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_6.setObjectName("textBrowser_6")
        self.stockInfoEPSLabel = QtWidgets.QLabel(self.stockInfoTab)
        self.stockInfoEPSLabel.setGeometry(QtCore.QRect(0, 170, 47, 14))
        self.stockInfoEPSLabel.setObjectName("stockInfoEPSLabel")
        self.textBrowser_7 = QtWidgets.QTextBrowser(self.stockInfoTab)
        self.textBrowser_7.setGeometry(QtCore.QRect(80, 170, 121, 20))
        self.textBrowser_7.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_7.setObjectName("textBrowser_7")
        self.stockInfoDividendYieldLabel = QtWidgets.QLabel(self.stockInfoTab)
        self.stockInfoDividendYieldLabel.setGeometry(QtCore.QRect(0, 200, 81, 16))
        self.stockInfoDividendYieldLabel.setObjectName("stockInfoDividendYieldLabel")
        self.textBrowser_8 = QtWidgets.QTextBrowser(self.stockInfoTab)
        self.textBrowser_8.setGeometry(QtCore.QRect(80, 200, 121, 20))
        self.textBrowser_8.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_8.setObjectName("textBrowser_8")
        self.stockInfoMarketCapLabel = QtWidgets.QLabel(self.stockInfoTab)
        self.stockInfoMarketCapLabel.setGeometry(QtCore.QRect(0, 230, 61, 16))
        self.stockInfoMarketCapLabel.setObjectName("stockInfoMarketCapLabel")
        self.textBrowser_9 = QtWidgets.QTextBrowser(self.stockInfoTab)
        self.textBrowser_9.setGeometry(QtCore.QRect(80, 230, 121, 20))
        self.textBrowser_9.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_9.setObjectName("textBrowser_9")
        self.tabWidget.addTab(self.stockInfoTab, "")
        self.analysisTab = QtWidgets.QWidget()
        self.analysisTab.setObjectName("tab")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.analysisTab)
        self.lineEdit_3.setGeometry(QtCore.QRect(50, 10, 113, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.analysisTabTickerLabel = QtWidgets.QLabel(self.analysisTab)
        self.analysisTabTickerLabel.setGeometry(QtCore.QRect(10, 10, 47, 14))
        self.analysisTabTickerLabel.setObjectName("analysisTabTickerLabel")
        self.analysisLoadingStatusLabel = QtWidgets.QLabel(self.analysisTab)
        self.analysisLoadingStatusLabel.setGeometry(QtCore.QRect(170, 10, 71, 16))
        self.analysisLoadingStatusLabel.setObjectName("analysisLoadingStatusLabel")
        self.comboBox = QtWidgets.QComboBox(self.analysisTab)
        self.comboBox.setGeometry(QtCore.QRect(10, 40, 221, 22))
        self.comboBox.setObjectName("comboBox")
        for i in self.bar_patterns:
            self.comboBox.addItem("")
        self.spinBox = QtWidgets.QSpinBox(self.analysisTab)
        self.spinBox.setGeometry(QtCore.QRect(90, 80, 42, 22))
        self.spinBox.setObjectName("spinBox")
        self.label_14 = QtWidgets.QLabel(self.analysisTab)
        self.label_14.setGeometry(QtCore.QRect(140, 80, 31, 16))
        self.label_14.setObjectName("label_14")
        self.analysisAnalysisRangeLabel = QtWidgets.QLabel(self.analysisTab)
        self.analysisAnalysisRangeLabel.setGeometry(QtCore.QRect(10, 80, 81, 21))
        self.analysisAnalysisRangeLabel.setObjectName("analysisAnalysisRangeLabel")
        self.label_15 = QtWidgets.QLabel(self.analysisTab)
        self.label_15.setGeometry(QtCore.QRect(10, 110, 331, 281))
        self.label_15.setAutoFillBackground(False)
        self.label_15.setText("")
        self.label_15.setPixmap(QtGui.QPixmap("img/bar-patterns/choose-pattern.jpg"))
        self.label_15.setScaledContents(True)
        self.label_15.setObjectName("label_15")
        self.tabWidget.addTab(self.analysisTab, "")
        self.createAlgoTab = QtWidgets.QWidget()
        self.createAlgoTab.setObjectName("tab_2")
        self.tabWidget.addTab(self.createAlgoTab, "")
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
        
        # Naming the tabs
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.stockInfoTab), _translate("MainWindow", "Stock Info"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.analysisTab), _translate("MainWindow", "Analysis"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.createAlgoTab), _translate("MainWindow", "Create Algo"))
        
        # Labels in "Stock Info" tab
        self.stockInfoTickerLabel.setText(_translate("MainWindow", "Ticker"))
        self.stockInfoVolumeLabel.setText(_translate("MainWindow", "Volume"))
        self.stockInfoCloseLabel.setText(_translate("MainWindow", "Close"))
        self.stockInfoLastLabel.setText(_translate("MainWindow", "Last"))
        self.stockInfoDaysOHLCLabel.setText(_translate("MainWindow", "Days OHLC"))
        self.label.setText(_translate("MainWindow", "10-Avg Vol"))
        self.stockInfoQRGLabel.setText(_translate("MainWindow", "QRG"))
        self.stockInfoEPSLabel.setText(_translate("MainWindow", "EPS"))
        self.stockInfoDividendYieldLabel.setText(_translate("MainWindow", "Dividend Yield"))
        self.stockInfoMarketCapLabel.setText(_translate("MainWindow", "Market Cap"))
        
        # Labels in "Analysis" tab
        self.analysisTabTickerLabel.setText(_translate("MainWindow", "Ticker"))
        self.analysisLoadingStatusLabel.setText(_translate("MainWindow", "Loading status"))
        # self.analysisLoadingStatusLabel.setText(_translate("MainWindow", "days"))
        
        for i in range(len(self.bar_patterns)):
            self.comboBox.setItemText(i, _translate("MainWindow", self.bar_patterns[i]))
        
        self.analysisAnalysisRangeLabel.setText(_translate("MainWindow", "Analysis Range"))

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

