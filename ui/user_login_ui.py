# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/Userlogin.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_UserLogin(object):
    def setupUi(self, UserLogin):
        UserLogin.setObjectName("UserLogin")
        UserLogin.resize(548, 431)
        UserLogin.setMinimumSize(QtCore.QSize(548, 431))
        UserLogin.setMaximumSize(QtCore.QSize(548, 431))
        self.widget = QtWidgets.QWidget(UserLogin)
        self.widget.setGeometry(QtCore.QRect(-10, -10, 571, 451))
        self.widget.setObjectName("widget")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(20, 20, 531, 411))
        self.label.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:0, stop:0 rgba(0, 0, 0, 9), stop:0 rgba(0, 0, 0, 50), stop: 0.835227 rgba(0, 0, 0, 75));\n"
"border-radius:30px;")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(20, 20, 531, 411))
        self.label_2.setStyleSheet("border-image: url(:/images/Learn-Facial-Recognition-scaled.jpg);\n"
"border-radius:30px;")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(30, 30, 511, 391))
        self.label_3.setStyleSheet("background-color: rgba(0,0,0,100);\n"
"border-radius:30px;")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_Giris = QtWidgets.QLabel(self.widget)
        self.label_Giris.setGeometry(QtCore.QRect(110, 30, 351, 81))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_Giris.setFont(font)
        self.label_Giris.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_Giris.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Giris.setObjectName("label_Giris")
        self.line_edit_kullanici_adi = QtWidgets.QLineEdit(self.widget)
        self.line_edit_kullanici_adi.setGeometry(QtCore.QRect(140, 140, 251, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.line_edit_kullanici_adi.setFont(font)
        self.line_edit_kullanici_adi.setStyleSheet("background-color: rgba(0,0,0,0);\n"
"border:none;\n"
"border-bottom: 2px solid rgba(105,118,132,255);\n"
"color: rgba(255,255,255,230);\n"
"padding-bottom:7px;")
        self.line_edit_kullanici_adi.setObjectName("line_edit_kullanici_adi")
        self.line_edit_sifre = QtWidgets.QLineEdit(self.widget)
        self.line_edit_sifre.setGeometry(QtCore.QRect(140, 200, 251, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.line_edit_sifre.setFont(font)
        self.line_edit_sifre.setStyleSheet("background-color: rgba(0,0,0,0);\n"
"border:none;\n"
"border-bottom: 2px solid rgba(105,118,132,255);\n"
"color: rgba(255,255,255,230);\n"
"padding-bottom:7px;")
        self.line_edit_sifre.setEchoMode(QtWidgets.QLineEdit.Password)
        self.line_edit_sifre.setObjectName("line_edit_sifre")
        self.push_button_giris_yap = QtWidgets.QPushButton(self.widget)
        self.push_button_giris_yap.setGeometry(QtCore.QRect(140, 270, 251, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.push_button_giris_yap.setFont(font)
        self.push_button_giris_yap.setStyleSheet("QPushButton #pushButton{\n"
"    background-color: qlineargradient(spread:pad, x1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(20,47,78,219) stop:1 rgba(85,98,112,226));\n"
"    color:rgba(255,255,255,210);\n"
"border-radius:5px;\n"
"}\n"
"QPushButton #pushButton:hover{\n"
"    background-color: qlineargradient(spread:pad, x1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(40,67,98,219) stop:1 rgba(105,118,132,226));\n"
"}\n"
"QPushButton #pushButton:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color:rgba(105,118,132,200);\n"
"}")
        self.push_button_giris_yap.setObjectName("push_button_giris_yap")
        self.push_button_kayit_ol = QtWidgets.QPushButton(self.widget)
        self.push_button_kayit_ol.setGeometry(QtCore.QRect(140, 330, 251, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.push_button_kayit_ol.setFont(font)
        self.push_button_kayit_ol.setStyleSheet("QPushButton #pushButton{\n"
"    background-color: qlineargradient(spread:pad, x1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(20,47,78,219) stop:1 rgba(85,98,112,226));\n"
"    color:rgba(255,255,255,210);\n"
"border-radius:5px;\n"
"}\n"
"QPushButton #pushButton:hover{\n"
"    background-color: qlineargradient(spread:pad, x1:0, y1:0.505682, x2:1, y2:0.477, stop:0 rgba(40,67,98,219) stop:1 rgba(105,118,132,226));\n"
"}\n"
"QPushButton #pushButton:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color:rgba(105,118,132,200);\n"
"}")
        self.push_button_kayit_ol.setObjectName("push_button_kayit_ol")

        self.retranslateUi(UserLogin)
        QtCore.QMetaObject.connectSlotsByName(UserLogin)

    def retranslateUi(self, UserLogin):
        _translate = QtCore.QCoreApplication.translate
        UserLogin.setWindowTitle(_translate("UserLogin", "Kullanıcı Girişi"))
        self.label_Giris.setText(_translate("UserLogin", "GİRİŞ"))
        self.line_edit_kullanici_adi.setPlaceholderText(_translate("UserLogin", "Kullanıcı Adı"))
        self.line_edit_sifre.setPlaceholderText(_translate("UserLogin", "Şifre"))
        self.push_button_giris_yap.setText(_translate("UserLogin", "Giriş Yap"))
        self.push_button_kayit_ol.setText(_translate("UserLogin", "Kayıt Ol"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    UserLogin = QtWidgets.QDialog()
    ui = Ui_UserLogin()
    ui.setupUi(UserLogin)
    UserLogin.show()
    sys.exit(app.exec_())
