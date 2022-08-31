import sys
import os
import time

from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui
from PyQt5.QtCore import pyqtSlot, QEventLoop
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox
import torch 

from utils.stdout_redirect import StdoutRedirect
from utils.preprocess import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
form_class=uic.loadUiType("main.ui")[0]
  

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.title='커널 취약점 패치 분류 프로그램 v1.0'
        self.setupUi(self)
        self.setWindowTitle(self.title)
        
        # related data
        self.data_filepath = ''
        self.tokenizer_path = './vulpatch-tokenizer'
        self.dataset = None
        self.data_loadbtn.clicked.connect(self.data_load_clicked)
        self.data_updatebtn.clicked.connect(self.data_update_clicked)
        self.data_updatebtn.setDisabled(True)
        self.result_box.setDisabled(True)

        # related model
        self.model_path = ''
        self.model = None
        self.classifier_loadbtn.setText('불러온 데이터 분류하기')
        self.classifier_loadbtn.clicked.connect(self.classifier_load_clicked)
        self.model_box.setDisabled(True)
        
        # related classification
        self.input_box = self.input_box.toPlainText()
        self.classify_btn.clicked.connect(self.classify_clicked)
        
        self.stdout = StdoutRedirect()
        self.stdout.start()
        self.stdout.printOccur.connect(lambda x: self.append_text(x))
        
    
    def append_text(self, msg):
        self.textBrowser.moveCursor(QtGui.QTextCursor.End)
        self.textBrowser.insertPlainText(msg)
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
    
    def set_text(self, msg):
        self.textBrowser.moveCursor(QtGui.QTextCursor.End)
        self.textBrowser.setText(msg)
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

    @pyqtSlot()
    def data_update_clicked(self):
        from utils.parse_syzbot import parse_patch
        self.status.setText("데이터를 수집중입니다..........")
        filepath = parse_patch()
        self.data_load_clicked(filepath)

    @pyqtSlot()
    def data_load_clicked(self, filepath=None):
        self.model_box.setDisabled(True)
        self.data_filename.setText(f'파일명:  ')
        self.model_correct.setText(f'정확도: ')
        self.status.setText("데이터를 불러오는 중입니다..........")
        if not filepath:
            fname=QFileDialog.getOpenFileName()
        else:
            fname = [filepath]
        
        if fname[0] and fname[0] != '':
            filepath = fname[0]
            self.data_filepath = os.path.abspath(filepath)
            filename=os.path.basename(fname[0])
            self.data_filename.setText(f'파일명:  {filename}')
    
            self.stdout.printOccur.connect(lambda x: self.set_text(x))
            data = load_data(self.data_filepath)
            if not data:
                self.status.setText("파일 형식이 올바르지 않습니다.")
                return 
            self.model_box.setDisabled(False)
            self.status.setText("데이터를 불러왔습니다.")
            self.tokenizer = load_tokenizer(self.tokenizer_path)
            print(self.tokenizer)
        else:
            self.status.setText("데이터를 불러오지 못했습니다.")
    
    
    @pyqtSlot()
    def classifier_load_clicked(self, filepath='./utils/data/model.pt'):
        self.status.setText("모델을 불러오는 중입니다..........")
        from utils.classify import _load, load_checkpoint, calc_accuracy
        from utils.model import LSTM
        self.model_filename.setText(f'모델 파일명:  ')
        if not self.model:
            if not filepath:
                fname = QFileDialog.getOpenFileName()
            else:
                fname = [filepath]
            if fname[0]:
                filepath = fname[0]
                filename=os.path.basename(fname[0])
                self.model_filename.setText(f'모델 파일명:  {filename}')
                vocab_size = self.tokenizer.vocab_size
                self.model = LSTM(vocab_size=vocab_size, embedding_dim=128, hidden_size=128).to(device)
                try:
                    _load(filepath, self.model)
                except Exception as e:
                    import torch.optim as optim
                    optimizer = optim.Adam(self.model.parameters(), lr=0.001)
                    load_checkpoint(filepath, self.model, optimizer)

        self.status.setText("모델을 불러왔습니다.")
        self.listWidget.clear()
        time.sleep(3)
        self.status.setText("정확도를 측정중입니다...")
        count, total, acc, vuls = calc_accuracy(self.model, self.data_filepath, self.tokenizer)
        self.status.setText("정확도 측정을 완료하였습니다.")
        self.model_correct.setText(f'정확도: {acc*100:0.2f}%({count}/{total})')
        for vul in vuls:
            item = QListWidgetItem(vul)
            self.listWidget.addItem(item)
    
    @pyqtSlot()
    def classify_clicked(self):
        # classify git patch message 
        return 
    
    
if __name__=="__main__":
    app=QApplication(sys.argv)
    myWindow=MyWindow()
    myWindow.show()
    app.exec_()
    