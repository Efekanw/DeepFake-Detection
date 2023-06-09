# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/den.ui'
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
        self.tabWidget.setStyleSheet("color: rgb(255, 255, 255);\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(65, 199, 199);\n"
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
        self.scrollArea.setGeometry(QtCore.QRect(50, 590, 281, 131))
        self.scrollArea.setStyleSheet("color:rgb(89, 184, 148);\n"
"border: 2px solid;\n"
"opacity: 200;\n"
"border-color: rgb(89, 184, 148);\n"
"")
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 277, 127))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.check_box_vision_transformer = QtWidgets.QCheckBox(self.scrollAreaWidgetContents)
        self.check_box_vision_transformer.setGeometry(QtCore.QRect(10, 30, 191, 31))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.check_box_vision_transformer.setFont(font)
        self.check_box_vision_transformer.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.check_box_vision_transformer.setStyleSheet("color:rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"opacity: 200;\n"
"border-color: rgb(89, 184, 148);\n"
"")
        self.check_box_vision_transformer.setObjectName("check_box_vision_transformer")
        self.check_box_cnn = QtWidgets.QCheckBox(self.scrollAreaWidgetContents)
        self.check_box_cnn.setGeometry(QtCore.QRect(10, 70, 131, 31))
        self.check_box_cnn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.check_box_cnn.setStyleSheet("color:rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"opacity: 200;\n"
"border-color: rgb(89, 184, 148);\n"
"")
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
"border-color: rgb(65, 199, 199);\n"
"background-color: rgb(4, 18, 27);")
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
        self.groupBoxSonuc.setGeometry(QtCore.QRect(540, 490, 341, 291))
        self.groupBoxSonuc.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(65, 199, 199);")
        self.groupBoxSonuc.setTitle("")
        self.groupBoxSonuc.setObjectName("groupBoxSonuc")
        self.labelFake = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelFake.setGeometry(QtCore.QRect(10, 10, 91, 121))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelFake.setFont(font)
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
        self.labelReal.setGeometry(QtCore.QRect(10, 160, 91, 121))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.labelReal.setFont(font)
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
        self.labelFake_result = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelFake_result.setEnabled(True)
        self.labelFake_result.setGeometry(QtCore.QRect(120, 40, 211, 71))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.labelFake_result.setFont(font)
        self.labelFake_result.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"opacity: 200;\n"
"border-color:  rgb(89, 184, 148);")
        self.labelFake_result.setScaledContents(False)
        self.labelFake_result.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.labelFake_result.setObjectName("labelFake_result")
        self.labelReal_result = QtWidgets.QLabel(self.groupBoxSonuc)
        self.labelReal_result.setGeometry(QtCore.QRect(120, 190, 211, 71))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.labelReal_result.setFont(font)
        self.labelReal_result.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"opacity: 200;\n"
"border-color:  rgb(89, 184, 148);")
        self.labelReal_result.setObjectName("labelReal_result")
        self.scrollArea.raise_()
        self.widgetButtons.raise_()
        self.widgetVideo.raise_()
        self.groupBoxSonuc.raise_()
        self.tabWidget.addTab(self.tabCikarim, "")
        self.tabEgitim = QtWidgets.QWidget()
        self.tabEgitim.setObjectName("tabEgitim")
        self.groupBox = QtWidgets.QGroupBox(self.tabEgitim)
        self.groupBox.setGeometry(QtCore.QRect(10, 20, 891, 301))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.labelEpochViT = QtWidgets.QLabel(self.groupBox)
        self.labelEpochViT.setGeometry(QtCore.QRect(20, 120, 91, 21))
        self.labelEpochViT.setStyleSheet("color: rgb(255, 255, 255);")
        self.labelEpochViT.setObjectName("labelEpochViT")
        self.labelPatienceViT = QtWidgets.QLabel(self.groupBox)
        self.labelPatienceViT.setGeometry(QtCore.QRect(20, 180, 121, 21))
        self.labelPatienceViT.setStyleSheet("color: rgb(255, 255, 255);")
        self.labelPatienceViT.setObjectName("labelPatienceViT")
        self.labelBatchViT = QtWidgets.QLabel(self.groupBox)
        self.labelBatchViT.setGeometry(QtCore.QRect(320, 120, 131, 21))
        self.labelBatchViT.setStyleSheet("color: rgb(255, 255, 255);")
        self.labelBatchViT.setObjectName("labelBatchViT")
        self.labelLearningRateViT = QtWidgets.QLabel(self.groupBox)
        self.labelLearningRateViT.setGeometry(QtCore.QRect(620, 120, 131, 21))
        self.labelLearningRateViT.setStyleSheet("color: rgb(255, 255, 255);")
        self.labelLearningRateViT.setObjectName("labelLearningRateViT")
        self.labelWeightDecayViT = QtWidgets.QLabel(self.groupBox)
        self.labelWeightDecayViT.setGeometry(QtCore.QRect(320, 180, 131, 21))
        self.labelWeightDecayViT.setStyleSheet("color: rgb(255, 255, 255);")
        self.labelWeightDecayViT.setObjectName("labelWeightDecayViT")
        self.labelGamaViT = QtWidgets.QLabel(self.groupBox)
        self.labelGamaViT.setGeometry(QtCore.QRect(620, 180, 131, 21))
        self.labelGamaViT.setStyleSheet("color: rgb(255, 255, 255);")
        self.labelGamaViT.setObjectName("labelGamaViT")
        self.labelStepSizeViT = QtWidgets.QLabel(self.groupBox)
        self.labelStepSizeViT.setGeometry(QtCore.QRect(20, 240, 131, 21))
        self.labelStepSizeViT.setStyleSheet("color: rgb(255, 255, 255);")
        self.labelStepSizeViT.setObjectName("labelStepSizeViT")
        self.lineEditEpochViT = QtWidgets.QLineEdit(self.groupBox)
        self.lineEditEpochViT.setGeometry(QtCore.QRect(160, 120, 121, 21))
        self.lineEditEpochViT.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"opacity: 200;\n"
"border-color:  rgb(255, 255, 255);")
        self.lineEditEpochViT.setObjectName("lineEditEpochViT")
        self.lineEditPatienceViT = QtWidgets.QLineEdit(self.groupBox)
        self.lineEditPatienceViT.setGeometry(QtCore.QRect(160, 180, 121, 21))
        self.lineEditPatienceViT.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"opacity: 200;\n"
"border-color:  rgb(255, 255, 255);")
        self.lineEditPatienceViT.setObjectName("lineEditPatienceViT")
        self.lineEditBatchViT = QtWidgets.QLineEdit(self.groupBox)
        self.lineEditBatchViT.setGeometry(QtCore.QRect(460, 120, 121, 21))
        self.lineEditBatchViT.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"opacity: 200;\n"
"border-color:  rgb(255, 255, 255);\n"
"")
        self.lineEditBatchViT.setObjectName("lineEditBatchViT")
        self.lineEditLearningRateViT = QtWidgets.QLineEdit(self.groupBox)
        self.lineEditLearningRateViT.setGeometry(QtCore.QRect(760, 120, 121, 21))
        self.lineEditLearningRateViT.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"opacity: 200;\n"
"border-color:  rgb(255, 255, 255);\n"
"")
        self.lineEditLearningRateViT.setObjectName("lineEditLearningRateViT")
        self.lineEditWeightDecayViT = QtWidgets.QLineEdit(self.groupBox)
        self.lineEditWeightDecayViT.setGeometry(QtCore.QRect(460, 180, 121, 21))
        self.lineEditWeightDecayViT.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"opacity: 200;\n"
"border-color:  rgb(255, 255, 255);")
        self.lineEditWeightDecayViT.setObjectName("lineEditWeightDecayViT")
        self.lineEditGamaViT = QtWidgets.QLineEdit(self.groupBox)
        self.lineEditGamaViT.setGeometry(QtCore.QRect(760, 180, 121, 21))
        self.lineEditGamaViT.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"opacity: 200;\n"
"border-color:  rgb(255, 255, 255);\n"
"")
        self.lineEditGamaViT.setObjectName("lineEditGamaViT")
        self.lineEditStepSizeViT = QtWidgets.QLineEdit(self.groupBox)
        self.lineEditStepSizeViT.setGeometry(QtCore.QRect(160, 240, 121, 21))
        self.lineEditStepSizeViT.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"opacity: 200;\n"
"border-color:  rgb(255, 255, 255);")
        self.lineEditStepSizeViT.setObjectName("lineEditStepSizeViT")
        self.labelModelName = QtWidgets.QLabel(self.groupBox)
        self.labelModelName.setGeometry(QtCore.QRect(320, 240, 131, 21))
        self.labelModelName.setStyleSheet("color: rgb(255, 255, 255);")
        self.labelModelName.setObjectName("labelModelName")
        self.lineEditModelName = QtWidgets.QLineEdit(self.groupBox)
        self.lineEditModelName.setGeometry(QtCore.QRect(460, 240, 121, 21))
        self.lineEditModelName.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"opacity: 200;\n"
"border-color:  rgb(255, 255, 255);")
        self.lineEditModelName.setObjectName("lineEditModelName")
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_4.setGeometry(QtCore.QRect(60, 10, 771, 81))
        self.groupBox_4.setStyleSheet("QPushButton#pushButton_labelUpload{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(20,47,78,219), stop:1 rgba(85,98,112,226));\n"
"}\n"
"QPushButton#pushButton_labelUpload:hover{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(40,67,98,219), stop:1 rgba(105,118,132,226));\n"
"}\n"
"QPushButton#pushButton_labelUpload:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color:rgba(105,118,132,200);\n"
"}\n"
"QPushButton#pushButton_Egitim_veriseti{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(20,47,78,219), stop:1 rgba(85,98,112,226));\n"
"}\n"
"QPushButton#pushButton_Egitim_veriseti:hover{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(40,67,98,219), stop:1 rgba(105,118,132,226));\n"
"}\n"
"QPushButton#pushButton_Egitim_veriseti:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color:rgba(105,118,132,200);\n"
"}\n"
"QPushButton#pushButton_Validation_veriseti{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(20,47,78,219), stop:1 rgba(85,98,112,226));\n"
"}\n"
"QPushButton#pushButton_Validation_veriseti:hover{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(40,67,98,219), stop:1 rgba(105,118,132,226));\n"
"}\n"
"QPushButton#pushButton_Validation_veriseti:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color:rgba(105,118,132,200);\n"
"}\n"
"QPushButton#pushButton_Test_veriseti{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(20,47,78,219), stop:1 rgba(85,98,112,226));\n"
"}\n"
"QPushButton#pushButton_Test_veriseti:hover{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(40,67,98,219), stop:1 rgba(105,118,132,226));\n"
"}\n"
"QPushButton#pushButton_Test_veriseti:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color:rgba(105,118,132,200);\n"
"}\n"
"border: 1px solid black;\n"
"border-radius:5px;\n"
"\n"
"")
        self.groupBox_4.setTitle("")
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_Test_veriseti = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_Test_veriseti.setGeometry(QtCore.QRect(570, 20, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_Test_veriseti.setFont(font)
        self.pushButton_Test_veriseti.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_Test_veriseti.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(65, 199, 199);")
        self.pushButton_Test_veriseti.setObjectName("pushButton_Test_veriseti")
        self.pushButton_Egitim_veriseti = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_Egitim_veriseti.setGeometry(QtCore.QRect(180, 20, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_Egitim_veriseti.setFont(font)
        self.pushButton_Egitim_veriseti.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_Egitim_veriseti.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(65, 199, 199);")
        self.pushButton_Egitim_veriseti.setObjectName("pushButton_Egitim_veriseti")
        self.pushButton_Validation_veriseti = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_Validation_veriseti.setGeometry(QtCore.QRect(370, 20, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_Validation_veriseti.setFont(font)
        self.pushButton_Validation_veriseti.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_Validation_veriseti.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(65, 199, 199);")
        self.pushButton_Validation_veriseti.setObjectName("pushButton_Validation_veriseti")
        self.pushButton_labelUpload = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_labelUpload.setGeometry(QtCore.QRect(30, 20, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_labelUpload.setFont(font)
        self.pushButton_labelUpload.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_labelUpload.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(65, 199, 199);")
        self.pushButton_labelUpload.setObjectName("pushButton_labelUpload")
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_5.setGeometry(QtCore.QRect(760, 220, 120, 80))
        self.groupBox_5.setStyleSheet("QPushButton#pushButtonEgitim{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(20,47,78,219), stop:1 rgba(85,98,112,226));\n"
"}\n"
"QPushButton#pushButtonEgitim:hover{\n"
"    background-color: qlineargradient(spread:pad, x:1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(40,67,98,219), stop:1 rgba(105,118,132,226));\n"
"}\n"
"QPushButton#pushButtonEgitim:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color:rgba(105,118,132,200);\n"
"}\n"
"border: 1px solid black;\n"
"border-radius:5px;")
        self.groupBox_5.setTitle("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.pushButtonEgitim = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButtonEgitim.setGeometry(QtCore.QRect(0, 20, 121, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButtonEgitim.setFont(font)
        self.pushButtonEgitim.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButtonEgitim.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(65, 199, 199);")
        self.pushButtonEgitim.setObjectName("pushButtonEgitim")
        self.groupBox_2 = QtWidgets.QGroupBox(self.tabEgitim)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 340, 891, 71))
        self.groupBox_2.setStyleSheet("border-color:rgb(89, 184, 148)")
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.labelAccuracy = QtWidgets.QLabel(self.groupBox_2)
        self.labelAccuracy.setGeometry(QtCore.QRect(20, 20, 91, 31))
        self.labelAccuracy.setStyleSheet("color: rgb(255, 255, 255);")
        self.labelAccuracy.setObjectName("labelAccuracy")
        self.labelF1Score = QtWidgets.QLabel(self.groupBox_2)
        self.labelF1Score.setGeometry(QtCore.QRect(320, 20, 91, 31))
        self.labelF1Score.setStyleSheet("color: rgb(255, 255, 255);")
        self.labelF1Score.setObjectName("labelF1Score")
        self.labelLoss = QtWidgets.QLabel(self.groupBox_2)
        self.labelLoss.setGeometry(QtCore.QRect(620, 20, 91, 31))
        self.labelLoss.setStyleSheet("color: rgb(255, 255, 255);")
        self.labelLoss.setObjectName("labelLoss")
        self.labelRocCurves = QtWidgets.QLabel(self.groupBox_2)
        self.labelRocCurves.setGeometry(QtCore.QRect(780, 290, 91, 31))
        self.labelRocCurves.setStyleSheet("color: rgb(255, 255, 255);")
        self.labelRocCurves.setObjectName("labelRocCurves")
        self.labelAccuracySonuc = QtWidgets.QLabel(self.groupBox_2)
        self.labelAccuracySonuc.setEnabled(True)
        self.labelAccuracySonuc.setGeometry(QtCore.QRect(160, 20, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.labelAccuracySonuc.setFont(font)
        self.labelAccuracySonuc.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(89, 184, 148);")
        self.labelAccuracySonuc.setText("")
        self.labelAccuracySonuc.setObjectName("labelAccuracySonuc")
        self.labelF1ScoreSonuc = QtWidgets.QLabel(self.groupBox_2)
        self.labelF1ScoreSonuc.setEnabled(True)
        self.labelF1ScoreSonuc.setGeometry(QtCore.QRect(460, 20, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.labelF1ScoreSonuc.setFont(font)
        self.labelF1ScoreSonuc.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(89, 184, 148);")
        self.labelF1ScoreSonuc.setText("")
        self.labelF1ScoreSonuc.setObjectName("labelF1ScoreSonuc")
        self.labelLossSonuc = QtWidgets.QLabel(self.groupBox_2)
        self.labelLossSonuc.setEnabled(True)
        self.labelLossSonuc.setGeometry(QtCore.QRect(760, 20, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.labelLossSonuc.setFont(font)
        self.labelLossSonuc.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(89, 184, 148);")
        self.labelLossSonuc.setText("")
        self.labelLossSonuc.setObjectName("labelLossSonuc")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tabEgitim)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 440, 891, 341))
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.labelRocCurves_2 = QtWidgets.QLabel(self.groupBox_3)
        self.labelRocCurves_2.setGeometry(QtCore.QRect(460, 10, 91, 31))
        self.labelRocCurves_2.setStyleSheet("color: rgb(255, 255, 255);")
        self.labelRocCurves_2.setObjectName("labelRocCurves_2")
        self.label_roc = QtWidgets.QLabel(self.groupBox_3)
        self.label_roc.setGeometry(QtCore.QRect(450, 50, 431, 281))
        self.label_roc.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(89, 184, 148);")
        self.label_roc.setText("")
        self.label_roc.setAlignment(QtCore.Qt.AlignCenter)
        self.label_roc.setObjectName("label_roc")
        self.label_conf = QtWidgets.QLabel(self.groupBox_3)
        self.label_conf.setGeometry(QtCore.QRect(10, 50, 431, 281))
        self.label_conf.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(89, 184, 148);")
        self.label_conf.setText("")
        self.label_conf.setAlignment(QtCore.Qt.AlignCenter)
        self.label_conf.setObjectName("label_conf")
        self.labelConfisionMatrix = QtWidgets.QLabel(self.groupBox_3)
        self.labelConfisionMatrix.setGeometry(QtCore.QRect(20, 10, 91, 31))
        self.labelConfisionMatrix.setStyleSheet("color: rgb(255, 255, 255);")
        self.labelConfisionMatrix.setObjectName("labelConfisionMatrix")
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
        self.actionVeri_seti_sec = QtWidgets.QAction(DeepFakeDetection)
        self.actionVeri_seti_sec.setObjectName("actionVeri_seti_sec")
        self.actionTraining_veriseti_sec = QtWidgets.QAction(DeepFakeDetection)
        self.actionTraining_veriseti_sec.setObjectName("actionTraining_veriseti_sec")
        self.actionValidation_veriseti_sec = QtWidgets.QAction(DeepFakeDetection)
        self.actionValidation_veriseti_sec.setObjectName("actionValidation_veriseti_sec")
        self.actionTest_veriseti_sec = QtWidgets.QAction(DeepFakeDetection)
        self.actionTest_veriseti_sec.setObjectName("actionTest_veriseti_sec")
        self.menuDosya.addAction(self.actionAc)
        self.menuDosya.addAction(self.actionKapat)
        self.menubar.addAction(self.menuDosya.menuAction())

        self.retranslateUi(DeepFakeDetection)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(DeepFakeDetection)

    def retranslateUi(self, DeepFakeDetection):
        _translate = QtCore.QCoreApplication.translate
        DeepFakeDetection.setWindowTitle(_translate("DeepFakeDetection", "Deep Fake Detection"))
        self.check_box_vision_transformer.setText(_translate("DeepFakeDetection", "Conv. Cross ViT EfficientNet B0"))
        self.check_box_cnn.setText(_translate("DeepFakeDetection", "Selim EfficientNet B7"))
        self.pushButtonOynat.setText(_translate("DeepFakeDetection", "OYNAT"))
        self.pushButtonDuraklat.setText(_translate("DeepFakeDetection", "DURAKLAT"))
        self.pushButtonDurdur.setText(_translate("DeepFakeDetection", "DURDUR"))
        self.pushButtonInference.setText(_translate("DeepFakeDetection", "ÇIKARIM"))
        self.labelFake.setText(_translate("DeepFakeDetection", "FAKE"))
        self.labelReal.setText(_translate("DeepFakeDetection", "REAL"))
        self.labelFake_result.setText(_translate("DeepFakeDetection", "Sahte Doğruluk Oranı"))
        self.labelReal_result.setText(_translate("DeepFakeDetection", "Gerçek Doğruluk Oranı"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabCikarim), _translate("DeepFakeDetection", "Çıkarım"))
        self.labelEpochViT.setText(_translate("DeepFakeDetection", "Epoch"))
        self.labelPatienceViT.setText(_translate("DeepFakeDetection", "Patience"))
        self.labelBatchViT.setText(_translate("DeepFakeDetection", "Batch size"))
        self.labelLearningRateViT.setText(_translate("DeepFakeDetection", "Learning rate"))
        self.labelWeightDecayViT.setText(_translate("DeepFakeDetection", "Weight-decay"))
        self.labelGamaViT.setText(_translate("DeepFakeDetection", "Gama"))
        self.labelStepSizeViT.setText(_translate("DeepFakeDetection", "Step size"))
        self.lineEditEpochViT.setText(_translate("DeepFakeDetection", "2"))
        self.lineEditPatienceViT.setText(_translate("DeepFakeDetection", "5"))
        self.lineEditBatchViT.setText(_translate("DeepFakeDetection", "8"))
        self.lineEditLearningRateViT.setText(_translate("DeepFakeDetection", "0.01"))
        self.lineEditWeightDecayViT.setText(_translate("DeepFakeDetection", "0.000001"))
        self.lineEditGamaViT.setText(_translate("DeepFakeDetection", "0.1"))
        self.lineEditStepSizeViT.setText(_translate("DeepFakeDetection", "15"))
        self.labelModelName.setText(_translate("DeepFakeDetection", "Model İsmi"))
        self.lineEditModelName.setText(_translate("DeepFakeDetection", "vit_01"))
        self.pushButton_Test_veriseti.setText(_translate("DeepFakeDetection", "TEST VERİ SETİ YÜKLE"))
        self.pushButton_Egitim_veriseti.setText(_translate("DeepFakeDetection", "EĞİTİM VERİ SETİ YÜKLE"))
        self.pushButton_Validation_veriseti.setText(_translate("DeepFakeDetection", "DOĞRULAMA VERİ SETİ YÜKLE"))
        self.pushButton_labelUpload.setText(_translate("DeepFakeDetection", "LABEL YÜKLE"))
        self.pushButtonEgitim.setText(_translate("DeepFakeDetection", "EĞİTİM"))
        self.labelAccuracy.setText(_translate("DeepFakeDetection", "Accuracy"))
        self.labelF1Score.setText(_translate("DeepFakeDetection", "F1 Score"))
        self.labelLoss.setText(_translate("DeepFakeDetection", "Loss"))
        self.labelRocCurves.setText(_translate("DeepFakeDetection", "Roc curves"))
        self.labelRocCurves_2.setText(_translate("DeepFakeDetection", "Roc curves"))
        self.labelConfisionMatrix.setText(_translate("DeepFakeDetection", "Confision Matrix"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabEgitim), _translate("DeepFakeDetection", "Eğitim"))
        self.menuDosya.setTitle(_translate("DeepFakeDetection", "Dosya"))
        self.actionAc.setText(_translate("DeepFakeDetection", "Aç"))
        self.actionKapat.setText(_translate("DeepFakeDetection", "Kapat"))
        self.actionVeri_seti_sec.setText(_translate("DeepFakeDetection", "Veri Seti Seç"))
        self.actionTraining_veriseti_sec.setText(_translate("DeepFakeDetection", "Eğitim Veri Seti Seç"))
        self.actionValidation_veriseti_sec.setText(_translate("DeepFakeDetection", "Validation Veri Seti Seç"))
        self.actionTest_veriseti_sec.setText(_translate("DeepFakeDetection", "Test Veri Seti Seç"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    DeepFakeDetection = QtWidgets.QMainWindow()
    ui = Ui_DeepFakeDetection()
    ui.setupUi(DeepFakeDetection)
    DeepFakeDetection.show()
    sys.exit(app.exec_())
