# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './ui/den.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_DeepFakeDetection(object):
    def setupUi(self, DeepFakeDetection):
        DeepFakeDetection.setObjectName("DeepFakeDetection")
        DeepFakeDetection.resize(910, 850)
        DeepFakeDetection.setMinimumSize(QtCore.QSize(910, 850))
        DeepFakeDetection.setMaximumSize(QtCore.QSize(910, 850))
        DeepFakeDetection.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        DeepFakeDetection.setFocusPolicy(QtCore.Qt.StrongFocus)
        DeepFakeDetection.setStyleSheet("background-color: rgb(4, 18, 27);\n"
"color: rgb(255, 255, 255);\n"
"opacity: 200;\n"
"border-color:  rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(DeepFakeDetection)
        self.centralwidget.setStyleSheet("background-color: rgb(4, 18, 27);\n"
"QMenuBar#menubar:hover{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    color: rgb(2, 10, 16);\n"
"}")
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 916, 828))
        self.tabWidget.setStyleSheet("background-color: rgb(4, 18, 27);\n"
"color: rgb(255, 255, 255);\n"
"border-color:NONE;\n"
"color: rgb(2, 10, 16);\n"
"opacity: 200;\n"
"")
        self.tabWidget.setObjectName("tabWidget")
        self.tabCikarim = QtWidgets.QWidget()
        self.tabCikarim.setObjectName("tabCikarim")
        self.widgetAll = QtWidgets.QWidget(self.tabCikarim)
        self.widgetAll.setGeometry(QtCore.QRect(0, 0, 911, 781))
        self.widgetAll.setStyleSheet("background-color: rgb(4, 18, 27);")
        self.widgetAll.setObjectName("widgetAll")
        self.widgetVideo = QtWidgets.QWidget(self.widgetAll)
        self.widgetVideo.setGeometry(QtCore.QRect(10, 10, 881, 461))
        self.widgetVideo.setStyleSheet("background-color: rgb(4, 18, 27);\n"
"background-color: rgb(2, 10, 16);\n"
"border-radius:20px;\n"
"")
        self.widgetVideo.setObjectName("widgetVideo")
        self.scrollArea = QtWidgets.QScrollArea(self.widgetAll)
        self.scrollArea.setGeometry(QtCore.QRect(50, 580, 281, 111))
        self.scrollArea.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(65, 199, 199);\n"
"")
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 267, 97))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.check_box_vision_transformer = QtWidgets.QCheckBox(self.scrollAreaWidgetContents)
        self.check_box_vision_transformer.setGeometry(QtCore.QRect(10, 10, 191, 31))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.check_box_vision_transformer.setFont(font)
        self.check_box_vision_transformer.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.check_box_vision_transformer.setStyleSheet("color: rgb(255, 255, 255);")
        self.check_box_vision_transformer.setObjectName("check_box_vision_transformer")
        self.check_box_cnn = QtWidgets.QCheckBox(self.scrollAreaWidgetContents)
        self.check_box_cnn.setGeometry(QtCore.QRect(10, 50, 131, 31))
        self.check_box_cnn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.check_box_cnn.setStyleSheet("color: rgb(255, 255, 255);")
        self.check_box_cnn.setObjectName("check_box_cnn")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.widgetButtons = QtWidgets.QWidget(self.widgetAll)
        self.widgetButtons.setGeometry(QtCore.QRect(30, 490, 641, 61))
        self.widgetButtons.setStyleSheet("QPushButton#pushButtonOynat{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(20,47,78,219), stop:1 rgba(85,98,112,226));\n"
"}\n"
"QPushButton#pushButtonOynat:hover{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(40,67,98,219), stop:1 rgba(105,118,132,226));\n"
"}\n"
"QPushButton#pushButtonOynat:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color:rgba(105,118,132,200);\n"
"}\n"
"QPushButton#pushButtonDurdur{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(20,47,78,219), stop:1 rgba(85,98,112,226));\n"
"}\n"
"QPushButton#pushButtonDurdur:hover{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(40,67,98,219), stop:1 rgba(105,118,132,226));\n"
"}\n"
"QPushButton#pushButtonDurdur:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color:rgba(105,118,132,200);\n"
"}\n"
"QPushButton#pushButtonDuraklat{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(20,47,78,219), stop:1 rgba(85,98,112,226));\n"
"}\n"
"QPushButton#pushButtonDuraklat:hover{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(40,67,98,219), stop:1 rgba(105,118,132,226));\n"
"}\n"
"QPushButton#pushButtonDuraklat:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color:rgba(105,118,132,200);\n"
"}\n"
"QPushButton#pushButtonInference{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(20,47,78,219), stop:1 rgba(85,98,112,226));\n"
"}\n"
"QPushButton#pushButtonInference:hover{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(40,67,98,219), stop:1 rgba(105,118,132,226));\n"
"}\n"
"QPushButton#pushButtonInference:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color:rgba(105,118,132,200);\n"
"}\n"
"border: 1px solid black;\n"
"border-radius:5px;\n"
"\n"
"")
        self.widgetButtons.setObjectName("widgetButtons")
        self.pushButtonOynat = QtWidgets.QPushButton(self.widgetButtons)
        self.pushButtonOynat.setGeometry(QtCore.QRect(20, 10, 81, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButtonOynat.setFont(font)
        self.pushButtonOynat.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButtonOynat.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(65, 199, 199);")
        self.pushButtonOynat.setObjectName("pushButtonOynat")
        self.pushButtonDuraklat = QtWidgets.QPushButton(self.widgetButtons)
        self.pushButtonDuraklat.setGeometry(QtCore.QRect(120, 10, 81, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButtonDuraklat.setFont(font)
        self.pushButtonDuraklat.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButtonDuraklat.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(65, 199, 199);")
        self.pushButtonDuraklat.setObjectName("pushButtonDuraklat")
        self.pushButtonDurdur = QtWidgets.QPushButton(self.widgetButtons)
        self.pushButtonDurdur.setGeometry(QtCore.QRect(220, 10, 81, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButtonDurdur.setFont(font)
        self.pushButtonDurdur.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButtonDurdur.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(65, 199, 199);")
        self.pushButtonDurdur.setObjectName("pushButtonDurdur")
        self.pushButtonInference = QtWidgets.QPushButton(self.widgetButtons)
        self.pushButtonInference.setGeometry(QtCore.QRect(360, 10, 81, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButtonInference.setFont(font)
        self.pushButtonInference.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButtonInference.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(65, 199, 199);")
        self.pushButtonInference.setObjectName("pushButtonInference")
        self.groupBoxSonuc = QtWidgets.QGroupBox(self.widgetAll)
        self.groupBoxSonuc.setGeometry(QtCore.QRect(540, 490, 361, 291))
        self.groupBoxSonuc.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(65, 199, 199);")
        self.groupBoxSonuc.setTitle("")
        self.groupBoxSonuc.setObjectName("groupBoxSonuc")
        self.labelFake = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelFake.setGeometry(QtCore.QRect(10, 10, 51, 121))
        self.labelFake.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelFake.setStyleSheet("border: 2px solid;\n"
"color: rgb(255, 0, 0);\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"border-color: red;")
        self.labelFake.setScaledContents(False)
        self.labelFake.setAlignment(QtCore.Qt.AlignCenter)
        self.labelFake.setObjectName("labelFake")
        self.labelReal = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelReal.setGeometry(QtCore.QRect(10, 160, 51, 121))
        self.labelReal.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelReal.setStyleSheet("border: 2px solid;\n"
"padding: 5px;\n"
"color: rgb(0, 170, 0);\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: green;")
        self.labelReal.setScaledContents(False)
        self.labelReal.setAlignment(QtCore.Qt.AlignCenter)
        self.labelReal.setObjectName("labelReal")
        self.labelFake1 = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelFake1.setGeometry(QtCore.QRect(160, 10, 91, 51))
        self.labelFake1.setStyleSheet("border: 2px solid;\n"
"color: rgb(255, 255, 255);\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"border-color: red;")
        self.labelFake1.setText("")
        self.labelFake1.setObjectName("labelFake1")
        self.labelFake2 = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelFake2.setGeometry(QtCore.QRect(260, 10, 91, 51))
        self.labelFake2.setStyleSheet("border: 2px solid;\n"
"color: rgb(255, 255, 255);\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"border-color: red;")
        self.labelFake2.setText("")
        self.labelFake2.setObjectName("labelFake2")
        self.labelFake3 = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelFake3.setGeometry(QtCore.QRect(160, 80, 91, 51))
        self.labelFake3.setStyleSheet("border: 2px solid;\n"
"color: rgb(255, 255, 255);\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"border-color: red;")
        self.labelFake3.setText("")
        self.labelFake3.setObjectName("labelFake3")
        self.labelFake4 = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelFake4.setGeometry(QtCore.QRect(260, 80, 91, 51))
        self.labelFake4.setStyleSheet("border: 2px solid;\n"
"color: rgb(255, 255, 255);\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"border-color: red;")
        self.labelFake4.setText("")
        self.labelFake4.setObjectName("labelFake4")
        self.labelReal1 = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelReal1.setGeometry(QtCore.QRect(160, 170, 91, 51))
        self.labelReal1.setStyleSheet("border: 2px solid;\n"
"padding: 5px;\n"
"color: rgb(0, 170, 0);\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: green;")
        self.labelReal1.setText("")
        self.labelReal1.setObjectName("labelReal1")
        self.labelReal2 = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelReal2.setGeometry(QtCore.QRect(260, 170, 91, 51))
        self.labelReal2.setStyleSheet("border: 2px solid;\n"
"padding: 5px;\n"
"color: rgb(0, 170, 0);\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: green;")
        self.labelReal2.setText("")
        self.labelReal2.setObjectName("labelReal2")
        self.labelReal3 = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelReal3.setGeometry(QtCore.QRect(160, 230, 91, 51))
        self.labelReal3.setStyleSheet("border: 2px solid;\n"
"padding: 5px;\n"
"color: rgb(0, 170, 0);\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: green;")
        self.labelReal3.setText("")
        self.labelReal3.setObjectName("labelReal3")
        self.labelReal4 = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelReal4.setGeometry(QtCore.QRect(260, 230, 91, 51))
        self.labelReal4.setStyleSheet("border: 2px solid;\n"
"padding: 5px;\n"
"color: rgb(0, 170, 0);\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: green;")
        self.labelReal4.setText("")
        self.labelReal4.setObjectName("labelReal4")
        self.labelFake_result = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelFake_result.setEnabled(True)
        self.labelFake_result.setGeometry(QtCore.QRect(70, 50, 81, 51))
        self.labelFake_result.setVisible(False)
        font = QtGui.QFont()
        font.setPointSize(7)
        self.labelFake_result.setFont(font)
        self.labelFake_result.setObjectName("labelFake_result")
        self.labelReal_result = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelReal_result.setGeometry(QtCore.QRect(70, 200, 81, 51))
        self.labelReal_result.setVisible(False)
        font = QtGui.QFont()
        font.setPointSize(7)
        self.labelReal_result.setFont(font)
        self.labelReal_result.setObjectName("labelReal_result")
        self.scrollArea.raise_()
        self.widgetButtons.raise_()
        self.widgetVideo.raise_()
        self.groupBoxSonuc.raise_()
        self.tabWidget.addTab(self.tabCikarim, "")
        self.tabEgitim = QtWidgets.QWidget()
        self.tabEgitim.setObjectName("tabEgitim")
        self.tabWidget.addTab(self.tabEgitim, "")
        DeepFakeDetection.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(DeepFakeDetection)
        self.statusbar.setObjectName("statusbar")
        DeepFakeDetection.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(DeepFakeDetection)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 910, 21))
        self.menubar.setObjectName("menubar")
        self.menuDosya = QtWidgets.QMenu(self.menubar)
        self.menuDosya.setObjectName("menuDosya")
        DeepFakeDetection.setMenuBar(self.menubar)
        self.actionAc = QtWidgets.QAction(DeepFakeDetection)
        self.actionAc.setObjectName("actionAc")
        self.actionKapat = QtWidgets.QAction(DeepFakeDetection)
        self.actionKapat.setObjectName("actionKapat")
        self.menuDosya.addAction(self.actionAc)
        self.menuDosya.addAction(self.actionKapat)
        self.menubar.addAction(self.menuDosya.menuAction())

        self.retranslateUi(DeepFakeDetection)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(DeepFakeDetection)

    def retranslateUi(self, DeepFakeDetection):
        _translate = QtCore.QCoreApplication.translate
        DeepFakeDetection.setWindowTitle(_translate("DeepFakeDetection", "Deep Fake Detection"))
        self.check_box_vision_transformer.setText(_translate("DeepFakeDetection", "Conv. Cross ViT EfficientNet B0"))
        self.check_box_cnn.setText(_translate("DeepFakeDetection", "Selim EfficientNet B7"))
        self.pushButtonOynat.setText(_translate("DeepFakeDetection", "OYNAT"))
        self.pushButtonDuraklat.setText(_translate("DeepFakeDetection", "DURAKLAT"))
        self.pushButtonDurdur.setText(_translate("DeepFakeDetection", "DURDUR"))
        self.pushButtonInference.setText(_translate("DeepFakeDetection", "Çıkarım"))
        self.labelFake.setText(_translate("DeepFakeDetection", "FAKE"))
        self.labelReal.setText(_translate("DeepFakeDetection", "REAL"))
        self.labelFake_result.setText(_translate("DeepFakeDetection", "ViT -> %{0:0.3f}"))
        self.labelReal_result.setText(_translate("DeepFakeDetection", "CNN -> %0.543"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabCikarim), _translate("DeepFakeDetection", "Çıkarım"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabEgitim), _translate("DeepFakeDetection", "Eğitim"))
        self.menuDosya.setTitle(_translate("DeepFakeDetection", "Dosya"))
        self.actionAc.setText(_translate("DeepFakeDetection", "Aç"))
        self.actionKapat.setText(_translate("DeepFakeDetection", "Kapat"))