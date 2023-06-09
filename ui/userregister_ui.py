# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Userregister.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(550, 506)
        Dialog.setMinimumSize(QtCore.QSize(550, 506))
        Dialog.setMaximumSize(QtCore.QSize(550, 506))
        Dialog.setStyleSheet("background-color: rgb(54, 54, 54);")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(10, 10, 531, 481))
        self.widget.setObjectName("widget")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(20, 20, 531, 411))
        self.label.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:0, stop:0 rgba(0, 0, 0, 9), stop:0 rgba(0, 0, 0, 50), stop: 0.835227 rgba(0, 0, 0, 75));\n"
"border-radius:30px;")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(20, 20, 621, 411))
        self.label_2.setStyleSheet("border-image: url(:/images/Learn-Facial-Recognition-scaled.jpg);\n"
"border-radius:30px;")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(30, 10, 471, 451))
        self.label_3.setStyleSheet("background-color: rgba(0,0,0,100);\n"
"border-radius:30px;")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_Kayitol = QtWidgets.QLabel(self.widget)
        self.label_Kayitol.setGeometry(QtCore.QRect(110, 40, 311, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_Kayitol.setFont(font)
        self.label_Kayitol.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_Kayitol.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Kayitol.setObjectName("label_Kayitol")
        self.lineEditKullaniciAdi = QtWidgets.QLineEdit(self.widget)
        self.lineEditKullaniciAdi.setGeometry(QtCore.QRect(110, 230, 311, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.lineEditKullaniciAdi.setFont(font)
        self.lineEditKullaniciAdi.setStyleSheet("background-color: rgba(0,0,0,0);\n"
"border:none;\n"
"border-bottom: 2px solid rgba(105,118,132,255);\n"
"color: rgba(255,255,255,230);\n"
"padding-bottom:7px;")
        self.lineEditKullaniciAdi.setObjectName("lineEditKullaniciAdi")
        self.lineEditSifre = QtWidgets.QLineEdit(self.widget)
        self.lineEditSifre.setGeometry(QtCore.QRect(110, 290, 311, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.lineEditSifre.setFont(font)
        self.lineEditSifre.setStyleSheet("background-color: rgba(0,0,0,0);\n"
"border:none;\n"
"border-bottom: 2px solid rgba(105,118,132,255);\n"
"color: rgba(255,255,255,230);\n"
"padding-bottom:7px;")
        self.lineEditSifre.setObjectName("lineEditSifre")
        self.pushButtonKayitOl = QtWidgets.QPushButton(self.widget)
        self.pushButtonKayitOl.setGeometry(QtCore.QRect(110, 410, 311, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.pushButtonKayitOl.setFont(font)
        self.pushButtonKayitOl.setStyleSheet("QPushButton #pushButton{\n"
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
        self.pushButtonKayitOl.setObjectName("pushButtonKayitOl")
        self.lineEditAd = QtWidgets.QLineEdit(self.widget)
        self.lineEditAd.setGeometry(QtCore.QRect(110, 110, 311, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.lineEditAd.setFont(font)
        self.lineEditAd.setStyleSheet("background-color: rgba(0,0,0,0);\n"
"border:none;\n"
"border-bottom: 2px solid rgba(105,118,132,255);\n"
"color: rgba(255,255,255,230);\n"
"padding-bottom:7px;")
        self.lineEditAd.setObjectName("lineEditAd")
        self.lineEditSoyad = QtWidgets.QLineEdit(self.widget)
        self.lineEditSoyad.setGeometry(QtCore.QRect(110, 170, 311, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.lineEditSoyad.setFont(font)
        self.lineEditSoyad.setStyleSheet("background-color: rgba(0,0,0,0);\n"
"border:none;\n"
"border-bottom: 2px solid rgba(105,118,132,255);\n"
"color: rgba(255,255,255,230);\n"
"padding-bottom:7px;")
        self.lineEditSoyad.setObjectName("lineEditSoyad")
        self.lineEditMail = QtWidgets.QLineEdit(self.widget)
        self.lineEditMail.setGeometry(QtCore.QRect(110, 350, 311, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.lineEditMail.setFont(font)
        self.lineEditMail.setStyleSheet("background-color: rgba(0,0,0,0);\n"
"border:none;\n"
"border-bottom: 2px solid rgba(105,118,132,255);\n"
"color: rgba(255,255,255,230);\n"
"padding-bottom:7px;")
        self.lineEditMail.setObjectName("lineEditMail")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Kayıt Ol"))
        self.label_Kayitol.setText(_translate("Dialog", "KAYIT OL"))
        self.lineEditKullaniciAdi.setPlaceholderText(_translate("Dialog", "Kullanıcı Adı"))
        self.lineEditSifre.setPlaceholderText(_translate("Dialog", "Şifre"))
        self.pushButtonKayitOl.setText(_translate("Dialog", "Kayıt Ol"))
        self.lineEditAd.setPlaceholderText(_translate("Dialog", "Ad"))
        self.lineEditSoyad.setPlaceholderText(_translate("Dialog", "Soyad"))
        self.lineEditMail.setPlaceholderText(_translate("Dialog", "Mail"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
