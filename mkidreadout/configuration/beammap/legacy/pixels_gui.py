# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pixels_gui.ui'
#
# Created: Mon Jul 11 18:01:51 2016
#      by: PyQt4 UI code generator 4.9.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_pixels_gui(object):
    def setupUi(self, pixels_gui):
        pixels_gui.setObjectName(_fromUtf8("pixels_gui"))
        pixels_gui.resize(1200, 1100)
        self.centralwidget = QtGui.QWidget(pixels_gui)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.pixPlot = MPL_Widget(self.centralwidget)
        self.pixPlot.setGeometry(QtCore.QRect(0, 0, 1021, 931))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 245, 248))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 245, 248))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 245, 248))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 245, 248))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.pixPlot.setPalette(palette)
        self.pixPlot.setObjectName(_fromUtf8("pixPlot"))
        self.dirbtn = QtGui.QPushButton(self.centralwidget)
        self.dirbtn.setGeometry(QtCore.QRect(390, 938, 75, 23))
        self.dirbtn.setObjectName(_fromUtf8("dirbtn"))
        self.dirle = QtGui.QLabel(self.centralwidget)
        self.dirle.setGeometry(QtCore.QRect(480, 940, 321, 21))
        self.dirle.setFrameShape(QtGui.QFrame.StyledPanel)
        self.dirle.setText(_fromUtf8(""))
        self.dirle.setObjectName(_fromUtf8("dirle"))
        self.savebtn = QtGui.QPushButton(self.centralwidget)
        self.savebtn.setGeometry(QtCore.QRect(1060, 570, 75, 23))
        self.savebtn.setObjectName(_fromUtf8("savebtn"))
        self.scalele = QtGui.QLineEdit(self.centralwidget)
        self.scalele.setGeometry(QtCore.QRect(1040, 240, 113, 20))
        self.scalele.setObjectName(_fromUtf8("scalele"))
        self.anglele = QtGui.QLineEdit(self.centralwidget)
        self.anglele.setGeometry(QtCore.QRect(1040, 190, 113, 20))
        self.anglele.setObjectName(_fromUtf8("anglele"))
        self.xoffle = QtGui.QLineEdit(self.centralwidget)
        self.xoffle.setGeometry(QtCore.QRect(1040, 300, 113, 20))
        self.xoffle.setObjectName(_fromUtf8("xoffle"))
        self.yoffle = QtGui.QLineEdit(self.centralwidget)
        self.yoffle.setGeometry(QtCore.QRect(1040, 350, 113, 20))
        self.yoffle.setObjectName(_fromUtf8("yoffle"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(1070, 280, 46, 13))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(1070, 330, 46, 13))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(1080, 220, 46, 13))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(1080, 170, 46, 13))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.dblbtn = QtGui.QPushButton(self.centralwidget)
        self.dblbtn.setGeometry(QtCore.QRect(1060, 460, 75, 23))
        self.dblbtn.setObjectName(_fromUtf8("dblbtn"))
        self.hidebtn = QtGui.QPushButton(self.centralwidget)
        self.hidebtn.setGeometry(QtCore.QRect(1060, 500, 75, 23))
        self.hidebtn.setObjectName(_fromUtf8("hidebtn"))
        pixels_gui.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(pixels_gui)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        pixels_gui.setStatusBar(self.statusbar)

        self.retranslateUi(pixels_gui)
        QtCore.QMetaObject.connectSlotsByName(pixels_gui)

    def retranslateUi(self, pixels_gui):
        pixels_gui.setWindowTitle(QtGui.QApplication.translate("pixels_gui", "Pixel Assignment", None, QtGui.QApplication.UnicodeUTF8))
        self.dirbtn.setText(QtGui.QApplication.translate("pixels_gui", "Config", None, QtGui.QApplication.UnicodeUTF8))
        self.savebtn.setText(QtGui.QApplication.translate("pixels_gui", "Save", None, QtGui.QApplication.UnicodeUTF8))
        self.scalele.setText(QtGui.QApplication.translate("pixels_gui", "10", None, QtGui.QApplication.UnicodeUTF8))
        self.anglele.setText(QtGui.QApplication.translate("pixels_gui", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.xoffle.setText(QtGui.QApplication.translate("pixels_gui", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.yoffle.setText(QtGui.QApplication.translate("pixels_gui", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("pixels_gui", "X Offset", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("pixels_gui", "Y Offset", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("pixels_gui", "Scale", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("pixels_gui", "Angle", None, QtGui.QApplication.UnicodeUTF8))
        self.dblbtn.setText(QtGui.QApplication.translate("pixels_gui", "Show Doubles", None, QtGui.QApplication.UnicodeUTF8))
        self.hidebtn.setText(QtGui.QApplication.translate("pixels_gui", "Hide Doubles", None, QtGui.QApplication.UnicodeUTF8))

from mpl_pyqt4_widget import MPL_Widget
