# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/Userlogin.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_UserLogin(object):
    def setupUi(self, UserLogin):
        UserLogin.setObjectName("UserLogin")
        UserLogin.resize(460, 689)
        UserLogin.setMinimumSize(QtCore.QSize(460, 689))
        UserLogin.setMaximumSize(QtCore.QSize(460, 689))
        self.widget = QtWidgets.QWidget(UserLogin)
        self.widget.setGeometry(QtCore.QRect(-10, -10, 571, 801))
        self.widget.setObjectName("widget")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(10, 10, 461, 691))
        self.label_3.setStyleSheet("background-color: rgb(4, 18, 27);")
        self.label_3.setLineWidth(1)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.push_button_giris_yap = QtWidgets.QPushButton(self.widget)
        self.push_button_giris_yap.setGeometry(QtCore.QRect(90, 520, 301, 41))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(17)
        font.setBold(False)
        font.setWeight(50)
        self.push_button_giris_yap.setFont(font)
        self.push_button_giris_yap.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.push_button_giris_yap.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 2px solid;\n"
"padding: 5px;\n"
"border-radius: 10px;\n"
"opacity: 200;\n"
"border-color: rgb(65, 199, 199);\n"
"")
        self.push_button_giris_yap.setObjectName("push_button_giris_yap")
        self.push_button_kayit_ol = QtWidgets.QPushButton(self.widget)
        self.push_button_kayit_ol.setGeometry(QtCore.QRect(90, 570, 301, 41))
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
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(90, 389, 301, 51))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.line_edit_kullanici_adi = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(15)
        self.line_edit_kullanici_adi.setFont(font)
        self.line_edit_kullanici_adi.setStyleSheet("color: rgb(231, 231, 231);\n"
"\n"
"border: None;\n"
"border-bottom-color: white;\n"
"border-radius: 10px;\n"
"padding: 0 8px;\n"
"background: rgb(4, 18, 27);\n"
"selection-background-color: darkgray;")
        self.line_edit_kullanici_adi.setObjectName("line_edit_kullanici_adi")
        self.verticalLayout.addWidget(self.line_edit_kullanici_adi)
        self.frame = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.frame.setStyleSheet("border: 2px solid white;")
        self.frame.setFrameShape(QtWidgets.QFrame.HLine)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setLineWidth(1)
        self.frame.setObjectName("frame")
        self.verticalLayout.addWidget(self.frame)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.widget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(90, 450, 301, 44))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.line_edit_sifre = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.line_edit_sifre.setFont(font)
        self.line_edit_sifre.setStyleSheet("color: rgb(231, 231, 231);\n"
"\n"
"border: None;\n"
"border-bottom-color: white;\n"
"border-radius: 10px;\n"
"padding: 0 8px;\n"
"background: rgb(4, 18, 27);\n"
"selection-background-color: darkgray;")
        self.line_edit_sifre.setEchoMode(QtWidgets.QLineEdit.Password)
        self.line_edit_sifre.setObjectName("line_edit_sifre")
        self.verticalLayout_2.addWidget(self.line_edit_sifre)
        self.frame_2 = QtWidgets.QFrame(self.verticalLayoutWidget_2)
        self.frame_2.setStyleSheet("border: 2px solid  rgb(65, 199, 199);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_2.setLineWidth(1)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_2.addWidget(self.frame_2)
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(50, 400, 31, 31))
        self.label.setStyleSheet("background-image: url(:/newPrefix/icons/user_32x32.png);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(50, 450, 31, 31))
        self.label_2.setStyleSheet("background-image: url(:/newPrefix/icons/lock_32x32.png);\n"
"color:  rgb(74, 223, 223);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setGeometry(QtCore.QRect(130, 90, 231, 231))
        self.label_4.setStyleSheet("border-image: url(:/newPrefix/icons/deepfake.png);\n"
"border-radius: 30px;")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")

        self.retranslateUi(UserLogin)
        QtCore.QMetaObject.connectSlotsByName(UserLogin)

    def retranslateUi(self, UserLogin):
        _translate = QtCore.QCoreApplication.translate
        UserLogin.setWindowTitle(_translate("UserLogin", "Kullanıcı Girişi"))
        self.push_button_giris_yap.setText(_translate("UserLogin", "Giriş Yap"))
        self.push_button_kayit_ol.setText(_translate("UserLogin", "Kayıt Ol"))
        self.line_edit_kullanici_adi.setPlaceholderText(_translate("UserLogin", "Kullanıcı Adı"))
        self.line_edit_sifre.setPlaceholderText(_translate("UserLogin", "Şifre"))
import ui.icons_rc



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    UserLogin = QtWidgets.QDialog()
    ui = Ui_UserLogin()
    ui.setupUi(UserLogin)
    UserLogin.show()
    sys.exit(app.exec_())
