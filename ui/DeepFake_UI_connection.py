from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from ui.facedetection_ui import Ui_DeepFakeDetection
from ui.user_login_ui import Ui_UserLogin
from ui.user_register_ui import Ui_UserRegister
from PyQt5.QtCore import QThread, pyqtSignal
from cross__efficient__vit.cross__efficient__vit.vision_transformer import visionTransformerPredict, visionTransformerTrain
from CNN.predict import inference
import os
import shutil
from database import db_functions
import hashlib
import binascii
from PyQt5.QtGui import QRegExpValidator, QImage, QPixmap
import pathlib
import time
import ui.icons_rc

class TrainThread(QThread):
    trainCompleted = pyqtSignal(dict)
    progressUpdate = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.num_epochs = None
        self.training_params = None
        self.model_name = None
        self.patience = None
        self.file_path = None
        self.train_path = None
        self.val_path = None
        self.test_path = None
        self.userid = None
        self.connection = None

    def run(self):
        try:
            result_test = visionTransformerTrain(num_epochs=self.num_epochs, training_params=self.training_params, model_name=self.model_name, train_path=self.train_path, val_path=self.val_path, test_path=self.test_path, userid=self.userid)
            metric_id = db_functions.get_id_by_metric_name(self.connection, result_test["metric_name"])
            conf_image = db_functions.retrieve_conf_image(self.connection, metric_id)
            roc_image = db_functions.retrieve_roc_image(self.connection, metric_id)
            result_test["conf_image"] = conf_image
            result_test["roc_image"] = roc_image
        except:
            print("train_test")
        self.trainCompleted.emit(result_test)


class InferenceThread(QThread):
    finished = pyqtSignal(float)
    inferenceCompleted = pyqtSignal(dict)
    progressUpdate = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.parameter = None
        self.file_path = None

    def closest_to_zero_or_one(self, num1, num2):
        zero_dist1 = abs(num1 - 0)
        one_dist1 = abs(num1 - 1)
        zero_dist2 = abs(num2 - 0)
        one_dist2 = abs(num2 - 1)

        min_dist = min(zero_dist1, one_dist1, zero_dist2, one_dist2)

        if min_dist == zero_dist1:
            return num1
        elif min_dist == one_dist1:
            return num1
        elif min_dist == zero_dist2:
            return num2
        else:
            return num2

    def run(self):
        try:
            start_time = time.time()
            result_dict = {}
            result_str = ''
            model_cnn_result = -1
            model_vt_result = -1
            if self.parameter == 0:
                model_vt_result = visionTransformerPredict(self.file_path)
                result = float(model_vt_result)
                if result >= 0.5:
                    result_str = "Sahte"
                else:
                    result_str = "Gerçek"
            elif self.parameter == 1:
                model_cnn_result = inference(self.file_path)
                model_cnn_result = model_cnn_result["label"]
                if model_cnn_result >= 0.5:
                    result_str = "Sahte"
                else:
                    result_str = "Gerçek"
            elif self.parameter == 2:
                model_vt_result = visionTransformerPredict(self.file_path)
                model_cnn_result = inference(self.file_path)
                model_cnn_result = model_cnn_result["label"]
                #result = str(model_vt_result)  +','+ str(model_cnn_result["label"])
                if model_cnn_result >= 0.5 and float(model_vt_result) >= 0.5:
                    result_str = "Sahte"
                elif model_cnn_result < 0.5 and float(model_vt_result) < 0.5:
                    result_str = "Gerçek"
                elif self.closest_to_zero_or_one(model_cnn_result, float(model_vt_result)) >= 0.5:
                    result_str = "Sahte"
                else:
                    result_str = "Gerçek"
            else:
                return 0
            result_dict = {"result": result_str, "cnn": model_cnn_result, "vt": float(model_vt_result)}
            end_time = time.time()
            total_time = end_time - start_time
        except:
            print("run")
        print(total_time)
        self.inferenceCompleted.emit(result_dict)


class LoginWindow(QMainWindow):
    def __init__(self, connection):
        super().__init__()
        self.ui = Ui_UserLogin()
        self.ui.setupUi(self)
        self.connection = connection
        self.window_main = None
        self.window_register = None
        self.make_connection()

    def login(self):
        try:
            username = self.ui.line_edit_kullanici_adi.text()
            password = self.ui.line_edit_sifre.text()
            print(username)
            print(password)
            userid = db_functions.check_login(self.connection, username, password)
            print(userid)
            self.window_main = MainWindow(self.connection, userid)
            if userid != False:
                if self.window_main.isVisible():
                    self.window_main.hide()
                else:
                    self.close()
                    self.window_main.show()
            else:
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setWindowTitle("Hata")
                msgBox.setText("Kullanıcı Adı, Parola Hatalı")
                msgBox.setDefaultButton(QMessageBox.Ok)
                msgBox.exec()

        except:
            print("login error")

    def register_window(self):
        self.window_register = RegisterWindow(self.connection)
        if self.window_register.isVisible():
            self.window_register.hide()
        else:
            self.close()
            self.window_register.show()

    def make_connection(self):
        self.ui.push_button_giris_yap.clicked.connect(self.login)
        self.ui.push_button_kayit_ol.clicked.connect(self.register_window)




class RegisterWindow(QMainWindow):
    def __init__(self, connection):
        super().__init__()
        self.ui = Ui_UserRegister()
        self.ui.setupUi(self)
        self.connection = connection
        self.window_login = None
        self.le_password = self.ui.line_edit_password
        self.le_password.validator()
        regex = QRegExp(".{8,}")  # En az 8 karakter uzunluğunda parola
        validator = QRegExpValidator(regex)
        self.le_password.setValidator(validator)
        self.make_connection()

    def register(self):
        warning = 0
        username = self.ui.line_edit_username.text()
        password = self.ui.line_edit_password.text()
        mail = self.ui.line_edit_mail.text()

        list_control = [1 if(text.isspace() or text == '') else 0 for text in [username, password, mail]]
        msg = ''
        for index, i in enumerate(list_control):
            if index == 0 and i == 1:
                msg += "\n-Lütfen Kullanıcı Adınızı Giriniz"
            elif index == 1 and i == 1:
                msg += "\n-Lütfen Parolanızı Giriniz"
            elif index == 2 and i == 1:
                msg += "\n-Lütfen Mailinizi Giriniz"
        if msg != '':
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowTitle("Uyarı")
            msgBox.setText(msg)
            msgBox.setDefaultButton(QMessageBox.Ok)
            msgBox.exec()
        else:
            salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
            # Verilen parolayı hashle
            hashed_password = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt, 100000)
            hashed_password = binascii.hexlify(hashed_password)
            # Salt ile birleştirerek hashli parolayı oluştur
            hash_result = (salt + hashed_password).decode('ascii')

            # hashed_password = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt, 100000)
            # hashed_password = binascii.hexlify(hashed_password)
            db_functions.register(self.connection, username, hash_result, mail)
            self.window_login = LoginWindow(self.connection)
            if self.window_login.isVisible():
                self.window_login.hide()
            else:
                self.close()
                self.window_login.show()

    def make_connection(self):
        self.ui.push_button_kayit_ol.clicked.connect(self.register)
        self.ui.push_button_geri.clicked.connect(self.back)

    def back(self):
        self.window_login = LoginWindow(self.connection)
        if self.window_login.isVisible():
            self.window_login.hide()
        else:
            self.close()
            self.window_login.show()



class MainWindow(QMainWindow):

    def __init__(self, connection, userid):
        super().__init__()
        self.ui = Ui_DeepFakeDetection()
        self.ui.setupUi(self)
        self.connection = connection
        self.userid = userid
        #self.show()
        self.setup()
        self.makeConnections()
        self.inferenceThread = InferenceThread()
        self.trainThread = TrainThread()
        self.inferenceThread.inferenceCompleted.connect(self.showInferenceResult)
        self.trainThread.trainCompleted.connect(self.showTrainResult)
        self.inferenceThread.progressUpdate.connect(self.updateProgressBar)
        self.file_path = ''
        self.train_path = ''
        self.val_path = ''
        self.test_path = ''


    def showTrainResult(self, result_test):
        self.ui.labelAccuracySonuc.clear()
        self.ui.labelLossSonuc.clear()
        self.ui.labelF1ScoreSonuc.clear()
        self.ui.label_roc.clear()
        self.ui.label_conf.clear()
        self.ui.labelAccuracySonuc.setText(str(result_test["accuracy"]))
        self.ui.labelLossSonuc.setText(str(result_test["loss"]))
        self.ui.labelF1ScoreSonuc.setText(str(result_test["f1"]))
        try:
            qimage = QImage.fromData(result_test["conf_image"])
            pixmap = QPixmap.fromImage(qimage)
            self.ui.label_conf.setPixmap(pixmap)
            self.ui.label_conf.setScaledContents(True)
            qimage2 = QImage.fromData(result_test["roc_image"])
            pixmap2 = QPixmap.fromImage(qimage2)
            self.ui.label_roc.setPixmap(pixmap2)
            self.ui.label_roc.setScaledContents(True)
            #self.ui.labelConfisionMatrix = result_test["conf_image"]
        except Exception as e:
            print(e)

    def updateProgressBar(self, value):
        self.ui.progressBarVisionTransformer.setValue(value)

    def setup(self):
        self.videoOutput = self.makeVideoWidget()
        self.mediaPlayer = self.makeMediaPlayer()

    def makeMediaPlayer(self):
        mediaPlayer = QMediaPlayer(self)
        mediaPlayer.setVideoOutput(self.videoOutput)
        return mediaPlayer

    def makeVideoWidget(self):
        videoOutput = QVideoWidget(self)
        vbox = QVBoxLayout()
        vbox.addWidget(videoOutput)
        self.ui.widgetVideo.setLayout(vbox)
        return videoOutput

    def makeConnections(self):
        self.ui.actionKapat.triggered.connect(self.close)
        self.ui.actionAc.triggered.connect(self.onActionAcTriggered)
        self.ui.pushButtonOynat.clicked.connect(self.mediaPlayer.play)
        self.ui.pushButtonDuraklat.clicked.connect(self.mediaPlayer.pause)
        self.ui.pushButtonDurdur.clicked.connect(self.mediaPlayer.stop)
        self.ui.pushButtonInference.clicked.connect(self.runInference)
        self.ui.pushButton_Egitim_veriseti.clicked.connect(self.onActionTrainAc)
        self.ui.pushButton_Validation_veriseti.clicked.connect(self.onActionValAc)
        self.ui.pushButton_Test_veriseti.clicked.connect(self.onActionTestAc)
        self.ui.pushButtonEgitim.clicked.connect(self.runTrain)
        self.ui.pushButton_labelUpload.clicked.connect(self.onActionLabelAc)

    def onActionLabelAc(self):
        path = QFileDialog.getExistingDirectory(self, "Etiket Dosyalarını Seç", "/")
        print(path)
        path = pathlib.Path(path)
        file_list = os.listdir(path)
        #target_folder = "/run"
        pwd = pathlib.Path.absolute(pathlib.Path(os.path.dirname(__file__))).parent
        target_folder = pathlib.Path.joinpath(pwd, 'run')
        #target_folder = pwd +  '/run'
        # Her bir dosya ve dizin için kopyalama işlemi yapılır
        for file_name in file_list:
            source_path = pathlib.Path.joinpath(path, file_name)
            #source_path = path + '/' + file_name
            #target_path = target_folder + '/' + file_name
            target_path = pathlib.Path.joinpath(target_folder, file_name)

            # Eğer kaynak öğe bir dosya ise, hedef klasöre kopyalanır
            if os.path.isfile(source_path):
                shutil.copy2(source_path, target_path)
                print(f"{source_path} dosyası kopyalandı.")

            # Eğer kaynak öğe bir klasör ise, hedef klasöre rekurzif olarak kopyalanır
            elif os.path.isdir(source_path):
                shutil.copytree(source_path, target_path)
                print(f"{source_path} klasörü ve içeriği kopyalandı.")

    def onActionTestAc(self):
        path = QFileDialog.getExistingDirectory(self, "Test Veri Seti Seç", "/")
        print(path)
        filepath = path
        if filepath == "":
            return
        self.test_path = filepath
        print(filepath)

    def onActionAcTriggered(self):
        path = QFileDialog.getOpenFileName(self, "Video Aç", "/")
        filepath = path[0]
        if filepath == "":
            return
        self.mediaPlayer.setMedia(QMediaContent(QUrl(filepath)))
        self.mediaPlayer.play()
        self.file_path = filepath

    def onActionValAc(self):
        path = QFileDialog.getExistingDirectory(self, "Validation Veri Seti Seç", "/")
        print(path)
        filepath = path
        if filepath == "":
            return
        self.val_path = filepath
        print(filepath)

    def onActionTrainAc(self):
        path = QFileDialog.getExistingDirectory(self, "Eğitim Veri Seti Seç", "/")
        print(path)
        filepath = path
        if filepath == "":
            return
        self.train_path = filepath
        print(filepath)

    def runInference(self):
        self.clear_folder()
        self.ui.labelFake_result.clear()
        self.ui.labelReal_result.clear()
        self.inferenceThread.file_path = self.file_path
        if not self.ui.check_box_vision_transformer.isChecked() and not self.ui.check_box_cnn.isChecked():
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowTitle("Hata")
            msgBox.setText("Lütfen Model Seçiniz")
            # msgBox.setInformativeText("Lütfen daha sonra tekrar deneyin.")
            msgBox.setDefaultButton(QMessageBox.Ok)
            msgBox.exec()
        else:
            if self.ui.check_box_vision_transformer.isChecked() and  self.ui.check_box_cnn.isChecked():
                self.inferenceThread.parameter = 2
            elif self.ui.check_box_cnn.isChecked():
                self.inferenceThread.parameter = 1
            elif self.ui.check_box_vision_transformer.isChecked():
                self.inferenceThread.parameter = 0
            self.inferenceThread.start()

    def clear_folder(self):
        folder = os.path.join('run_test')
        try:
            print(os.getcwd())
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete  Reason: ' + str(e))

    def showInferenceResult(self, result_dict):
        #self.ui.labelResult.setText(result)
        result_fake = ''
        result_real = ''
        self.ui.labelFake_result.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.ui.labelReal_result.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.ui.labelFake.setStyleSheet("background-color: rgb(4, 18, 27);color: red; border: 2px solid;color: rgb(255, 0, 0);padding: 5px;border-radius: 10px;border-color: red;")
        self.ui.labelReal.setStyleSheet("background-color: rgb(4, 18, 27);color: green; border: 2px solid;padding: 5px;color: rgb(0, 170, 0);border-radius: 10px;opacity: 200;border-color: green;")

        if result_dict["cnn"] >= 0.5 and result_dict["cnn"] != float(-1):
            result_dict["cnn"] = (result_dict["cnn"] * 100)
            result_fake += "CNN -> %{0:0.2f}\n".format(result_dict["cnn"])
        elif result_dict["cnn"] != float(-1):
            result_dict["cnn"] = 100 - (result_dict["cnn"] * 100)
            result_dict["vt"] = (result_dict["vt"] * 100)
            result_real += "CNN -> %{0:0.2f}\n".format(result_dict["cnn"])

        if result_dict["vt"] >= 0.5 and result_dict["vt"] != float(-1):
            result_dict["vt"] = (result_dict["vt"] * 100)
            result_fake += "ViT -> %{0:0.2f}\n".format(result_dict["vt"])
        elif result_dict["vt"] != float(-1):
            result_dict["vt"] = 100 - (result_dict["vt"] * 100)
            result_real += "ViT -> %{0:0.2f}\n".format(result_dict["vt"])

        if result_fake != '':
            self.ui.labelFake_result.setText(result_fake)
            self.ui.labelFake_result.setVisible(True)
        if result_real != '':
            self.ui.labelReal_result.setText(result_real)
            self.ui.labelReal_result.setVisible(True)

        if result_dict["result"] == "Sahte":
            self.ui.labelFake.setStyleSheet("background-color: red;color: white;")
        else:
            self.ui.labelReal.setStyleSheet("background-color: green;color: white;")

    def runTrain(self):
        flag = 1
        msg = ''
        if self.train_path == '':
            msg = "Eğitim Veri Kümesi Seçiniz"
            flag = 0
        if self.val_path == '':
            msg = "Doğrulama Veri Kümesi Seçiniz"
            flag = 0
        if self.test_path == '':
            msg = "Test Veri Kümesi Seçiniz"
            flag = 0
        if (self.ui.lineEditEpochViT.text() == '' or self.ui.lineEditLearningRateViT.text() == '' or
            self.ui.lineEditPatienceViT.text() == '' or self.ui.lineEditGamaViT.text() == '' or
            self.ui.lineEditBatchViT.text() == '' or  self.ui.lineEditStepSizeViT.text() == '' or
            self.ui.lineEditWeightDecayViT.text() == '' or self.ui.lineEditModelName.text() == ''):
            msg = "Model parametreleri boş olmamalıdır."
            flag = 0
        else:
            if float(self.ui.lineEditEpochViT.text()).is_integer():
                if int(self.ui.lineEditEpochViT.text()) <= 0:
                    msg = "Epoch değeri 0'dan büyük  olmalıdır"
                    flag = 0
            else:
                msg = "Epoch değeri tam sayı olmalıdır"
                flag = 0
            if float(self.ui.lineEditLearningRateViT.text()) <= 0 or float(self.ui.lineEditLearningRateViT.text()) >=  1:
                msg = "Learning Rate değeri 0 il 1 arasında olmalıdır"
                flag = 0
            if float(self.ui.lineEditPatienceViT.text()).is_integer():
                if int(self.ui.lineEditPatienceViT.text()) <= 0:
                    msg = "Patience değeri 0'dan büyük tam sayı olmalıdır"
                    flag = 0
            else:
                msg = "Patience değeri tam sayı olmalıdır"
                flag = 0
            if float(self.ui.lineEditGamaViT.text()) <= 0 or float(self.ui.lineEditGamaViT.text()) >=  1:
                msg = "Gamma değeri 0 ile 1 arasında olmalıdır"
                flag = 0
            if float(self.ui.lineEditBatchViT.text()).is_integer():
                if int(self.ui.lineEditBatchViT.text()) <= 0:
                    msg = "Batch size değeri 0'dan büyük tam sayı olmalıdır"
                    flag = 0
            else:
                msg = "Batch size değeri tam sayı olmalıdır"
                flag = 0
            if float(self.ui.lineEditStepSizeViT.text()).is_integer():
                if int(self.ui.lineEditStepSizeViT.text()) <= 0:
                    msg = "Step size değeri 0'dan büyük tam sayı olmalıdır"
                    flag = 0
            else:
                msg = "Step size değeri tam sayı olmalıdır"
                flag = 0
            if float(self.ui.lineEditWeightDecayViT.text()) <= 0 or float(self.ui.lineEditWeightDecayViT.text()) >=  1:
                msg = "Weight Decay değeri 0 ile 1 arasında olmalıdır"
                flag = 0
            if db_functions.check_metric_exists(self.connection, self.ui.lineEditModelName.text()):
                msg = "Bu model isminde bir model zaten mevcut"
                flag = 0
        if flag == 1:
            self.trainThread.model_name = self.ui.lineEditModelName.text() + '.pth'
            self.trainThread.num_epochs = int(self.ui.lineEditEpochViT.text())
            self.trainThread.patience = int(self.ui.lineEditPatienceViT.text())
            training_params = {"bs": int(self.ui.lineEditBatchViT.text()),
                               "lr": float(self.ui.lineEditLearningRateViT.text()),
                               "weight_decay": float(self.ui.lineEditWeightDecayViT.text()),
                               "gamma": float(self.ui.lineEditGamaViT.text()),
                               "step_size": int(self.ui.lineEditStepSizeViT.text())
                               }
            self.trainThread.training_params = training_params
            self.trainThread.train_path = self.train_path
            self.trainThread.val_path = self.val_path
            self.trainThread.test_path = self.test_path
            self.trainThread.start()
            self.trainThread.userid = self.userid
            self.trainThread.connection = self.connection
        else:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowTitle("Hata")
            msgBox.setText(msg)
            msgBox.setDefaultButton(QMessageBox.Ok)
            msgBox.exec()
