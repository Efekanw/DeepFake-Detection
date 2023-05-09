# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './ui/Userregister.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_UserRegister(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(437, 688)
        Dialog.setMinimumSize(QtCore.QSize(437, 688))
        Dialog.setMaximumSize(QtCore.QSize(437, 688))
        Dialog.setStyleSheet("background-color: rgb(54, 54, 54);")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(-10, -10, 451, 691))
        self.widget.setStyleSheet("background-color: rgb(4, 18, 27);")
        self.widget.setObjectName("widget")
        self.push_button_kayit_ol = QtWidgets.QPushButton(self.widget)
        self.push_button_kayit_ol.setGeometry(QtCore.QRect(80, 460, 301, 41))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(17)
        font.setBold(False)
        font.setWeight(50)
        self.push_button_kayit_ol.setFont(font)
        self.push_button_kayit_ol.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.push_button_kayit_ol.setStyleSheet("color: rgb(231, 231, 231);\n"
"border: 2px solid rgb(65, 199, 199);\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\\n")
        self.push_button_kayit_ol.setObjectName("push_button_kayit_ol")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.widget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(80, 280, 301, 51))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.line_edit_username = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(15)
        self.line_edit_username.setFont(font)
        self.line_edit_username.setStyleSheet("color: rgb(231, 231, 231);\n"
"\n"
"border: None;\n"
"border-bottom-color: white;\n"
"border-radius: 10px;\n"
"padding: 0 8px;\n"
"background: rgb(4, 18, 27);\n"
"selection-background-color: darkgray;")
        self.line_edit_username.setObjectName("line_edit_username")
        self.verticalLayout.addWidget(self.line_edit_username)
        self.frame = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.frame.setStyleSheet("border: 2px solid white;")
        self.frame.setFrameShape(QtWidgets.QFrame.HLine)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setLineWidth(1)
        self.frame.setObjectName("frame")
        self.verticalLayout.addWidget(self.frame)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.widget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(80, 380, 301, 44))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.line_edit_password = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.line_edit_password.setFont(font)
        self.line_edit_password.setStyleSheet("color: rgb(231, 231, 231);\n"
"\n"
"border: None;\n"
"border-bottom-color: white;\n"
"border-radius: 10px;\n"
"padding: 0 8px;\n"
"background: rgb(4, 18, 27);\n"
"selection-background-color: darkgray;")
        self.line_edit_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.line_edit_password.setObjectName("line_edit_password")
        self.verticalLayout_2.addWidget(self.line_edit_password)
        self.frame_2 = QtWidgets.QFrame(self.verticalLayoutWidget_2)
        self.frame_2.setStyleSheet("border: 2px solid  rgb(65, 199, 199);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_2.setLineWidth(1)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_2.addWidget(self.frame_2)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.widget)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(80, 330, 301, 51))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.line_edit_mail = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(15)
        self.line_edit_mail.setFont(font)
        self.line_edit_mail.setStyleSheet("color: rgb(231, 231, 231);\n"
"\n"
"border: None;\n"
"border-bottom-color: white;\n"
"border-radius: 10px;\n"
"padding: 0 8px;\n"
"background: rgb(4, 18, 27);\n"
"selection-background-color: darkgray;")
        self.line_edit_mail.setText("")
        self.line_edit_mail.setObjectName("line_edit_mail")
        self.verticalLayout_3.addWidget(self.line_edit_mail)
        self.frame_3 = QtWidgets.QFrame(self.verticalLayoutWidget_3)
        self.frame_3.setStyleSheet("border: 2px solid white;")
        self.frame_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_3.setLineWidth(1)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_3.addWidget(self.frame_3)
        self.push_button_cikis = QtWidgets.QPushButton(self.widget)
        self.push_button_cikis.setGeometry(QtCore.QRect(80, 560, 301, 41))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(17)
        font.setBold(False)
        font.setWeight(50)
        self.push_button_cikis.setFont(font)
        self.push_button_cikis.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.push_button_cikis.setStyleSheet("color: rgb(231, 231, 231);\n"
"border: 2px solid rgb(65, 199, 199);\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\\n")
        self.push_button_cikis.setObjectName("push_button_cikis")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(110, 30, 231, 231))
        self.label.setStyleSheet("border-image: url(:/newPrefix/icons/deepfake.png);\n"
"border-radius: 30px;")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(40, 390, 31, 31))
        self.label_2.setStyleSheet("border-image: url(:/newPrefix/icons/lock_32x32.png);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(40, 340, 31, 31))
        self.label_3.setStyleSheet("border-image: url(:/newPrefix/icons/mail_32x32.png);")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setGeometry(QtCore.QRect(40, 290, 31, 31))
        self.label_4.setStyleSheet("border-image: url(:/newPrefix/icons/user_32x32.png);")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.push_button_geri = QtWidgets.QPushButton(self.widget)
        self.push_button_geri.setGeometry(QtCore.QRect(80, 510, 301, 41))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(17)
        font.setBold(False)
        font.setWeight(50)
        self.push_button_geri.setFont(font)
        self.push_button_geri.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.push_button_geri.setStyleSheet("color: rgb(231, 231, 231);\n"
"border: 2px solid rgb(65, 199, 199);\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\\n")
        self.push_button_geri.setObjectName("push_button_geri")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Kayıt Ol"))
        self.push_button_kayit_ol.setText(_translate("Dialog", "Kayıt Ol"))
        self.line_edit_username.setPlaceholderText(_translate("Dialog", "Kullanıcı Adı"))
        self.line_edit_password.setPlaceholderText(_translate("Dialog", "Şifre"))
        self.line_edit_mail.setPlaceholderText(_translate("Dialog", "Mail"))
        self.push_button_cikis.setText(_translate("Dialog", "Çıkış"))
        self.push_button_geri.setText(_translate("Dialog", "Geri"))
import icons_rc