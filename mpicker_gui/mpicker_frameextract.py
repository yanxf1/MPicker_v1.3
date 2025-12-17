# Copyright (C) 2024  Xiaofeng Yan, Shudong Li
# Xueming Li Lab, Tsinghua University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from PyQt5.QtWidgets import  *
from PyQt5.QtCore import QThread,pyqtSignal
import numpy as np
import mrcfile
import os, sys, warnings
import Mpicker_core_gui
import threading

class Mpicker_FrameExtract(QFrame):
    def __init__(self, *__args):
        super(Mpicker_FrameExtract, self).__init__(*__args)

    def setParameters(self, UI):
        self.UI = UI
        self.set_Button_ExtractSurface()
        # self.UI.spinBox_Thickness.valueChanged.connect(self.SpinBox_Thickness)
        self.UI.doubleSpinBox_Fillvalue.valueChanged.connect(self.SpinBox_Fillvalue)
        self.UI.spinBox_Order.valueChanged.connect(self.SpinBox_Order)
        self.UI.spinBox_RBFsample.valueChanged.connect(self.SpinBox_RBFsample)
        self.UI.radioButton_choseRBF.toggled.connect(self.radioButton_Change)
        self.batch_mode = False
        self.Extract_Label_List = []

    def set_Button_ExtractSurface(self):
        self.UI.Button_ExtractSurface.clicked.connect(self.Extract_Surface)
        menu = QMenu(self.UI.Button_ExtractSurface)
        actionA = QAction("Extract All Selected", menu)
        actionA.triggered.connect(self.Extract_Surface_Batch)
        font = self.UI.Button_ExtractSurface.font()
        actionA.setFont(font)
        menu.addAction(actionA)
        self.UI.Button_ExtractSurface.setMenu(menu)

    def Extract_Surface(self):
        if self.UI.MRC_path is None:
            QMessageBox.warning(self, "Input Warning", "Please Input a Mask -> "
                                                       "and then Find Surface -> "
                                                       "final Extract Surface"
                                                       "\n", QMessageBox.Ok)
            return
        self.UI_tabWidget_currentWidget_objectName = self.UI.tabWidget.currentWidget().objectName()
        self.UI_tomo_surfcurrent = self.UI.tomo_surfcurrent
        self.UI_tomo_manual_surfcurrent = self.UI.tomo_manual_surfcurrent
        self.UI_SURF_path = self.UI.SURF_path
        self.UI_ManualTxt_path = self.UI.ManualTxt_path
        self.batch_mode = False
        self.aux_Extract_Surface()

    def Extract_Surface_Batch(self):
        if self.UI.MRC_path is None:
            QMessageBox.warning(self, "Input Warning", "Please Input a Mask -> "
                                                       "and then Find Surface -> "
                                                       "final Extract Surface"
                                                       "\n", QMessageBox.Ok)
            return
        self.Extract_Label_List = []
        self.UI_tabWidget_currentWidget_objectName = self.UI.tabWidget.currentWidget().objectName()
        self.UI_tomo_surfcurrent = self.UI.tomo_surfcurrent
        self.UI_tomo_manual_surfcurrent = self.UI.tomo_manual_surfcurrent
        self.UI_SURF_path = self.UI.SURF_path
        self.UI_ManualTxt_path = self.UI.ManualTxt_path
        self.batch_mode = True

        if self.UI_tabWidget_currentWidget_objectName == "tab_auto":
            for i in range(self.UI.scrollArea_Select_vbox.count()):
                widget = self.UI.scrollArea_Select_vbox.itemAt(i)
                if widget is None:
                    continue
                chosed_label = widget.layout().itemAt(0).widget()
                chosed_checkbox = widget.layout().itemAt(1).widget()
                if chosed_checkbox.isChecked():
                    self.Extract_Label_List.append(chosed_label.text())
        elif self.UI_tabWidget_currentWidget_objectName == "tab_manual":
            for i in range(self.UI.scrollArea_manualsurf_vbox.count()):
                widget = self.UI.scrollArea_manualsurf_vbox.itemAt(i)
                if widget is None:
                    continue
                chosed_label = widget.layout().itemAt(0).widget()
                chosed_checkbox = widget.layout().itemAt(1).widget()
                if chosed_checkbox.isChecked():
                    self.Extract_Label_List.append(chosed_label.text())

        if len(self.Extract_Label_List) == 0:
            QMessageBox.warning(self, "Batch Extraction", "No surface has been selected for extraction.",
                                QMessageBox.Ok)
            return

        self.Thread_extract_surface_batch = QThread_Extract_Batch()
        self.Thread_extract_surface_batch.setParameters(self)
        self.Thread_extract_surface_batch.finished.connect(self.Reset_Button_batch)
        self.Thread_extract_surface_batch.load_progress.connect(self.load_progress_batch)
        self.UI.label_progress.setText("Extracting Surface Batch ...")
        self.UI.progressBar.setValue(0)
        self.UI.Button_ExtractSurface.setEnabled(False)
        self.UI.Button_FindSurface.setEnabled(False)
        try:
            self.Thread_extract_surface_batch.start()
        except Exception as e:
            self.Reset_Button_batch()
            QMessageBox.warning(self, "Batch Extraction Error", str(e), QMessageBox.Ok)

    def load_progress_batch(self, val):
        self.aux_Extract_Surface()
        total = len(self.Extract_Label_List)
        self.UI.label_progress.setText(f"Extracting Surface Batch ... {val}/{total}")
        if total > 0:
            percent = int(((val-1)/total) * 100 + 0.5)
        else:
            percent = 0
        self.UI.progressBar.setValue(percent)

    def Reset_Button_batch(self):
        self.UI.progressBar.setValue(100)
        self.UI.Button_ExtractSurface.setEnabled(True)
        self.UI.Button_FindSurface.setEnabled(True)
        self.UI.label_progress.setText("Extract Surface Done")
        
    def aux_Extract_Surface(self):
        if not self.batch_mode:
            self.UI.progressBar.setValue(0)
        self.Thread_extract_surface = QThread_Extract()
        self.Thread_extract_surface.setParameters(self)
        if not self.batch_mode:
            self.Thread_extract_surface.load_progress.connect(self.load_progress)
        self.Thread_extract_surface.finished.connect(self.Reset_Button)
        self.Thread_extract_surface.error_signal.connect(self.raiseerror)
        if self.UI_tabWidget_currentWidget_objectName == "tab_auto":
            if self.UI_SURF_path is None:
                return
            chosed_label    = self.UI.scrollArea_Select_vbox.itemAt(self.UI_tomo_surfcurrent).layout().itemAt(0).widget()
            chosed_number   = chosed_label.text().split(". ")[0]
            surf_number     = os.path.basename(self.UI_SURF_path).split("_")[1]
            if chosed_number == surf_number:
                if not self.batch_mode:
                    self.UI.label_progress.setText("Extracting Surface...Please Wait")
                    self.UI.Button_ExtractSurface.setEnabled(False)
                    self.UI.Button_FindSurface.setEnabled(False)
                try:
                    self.Thread_extract_surface.start()
                except:
                    str_e = str(self.Thread_extract_surface.exception)
                    self.Reset_Button()
                    QMessageBox.warning(self, "Extraction Error",
                                            str_e,
                                            QMessageBox.Ok)
            else:
                Warning_text = "Surface has not been found yet. " \
                                "Please Find Surface before Extract Surface. "
                QMessageBox.warning(self, "Extraction Error", Warning_text, QMessageBox.Ok)
        elif self.UI_tabWidget_currentWidget_objectName == "tab_manual":
            if self.UI_ManualTxt_path is not None:
                if not self.batch_mode:
                    self.UI.label_progress.setText("Extracting Surface...Please Wait")
                    self.UI.Button_ExtractSurface.setEnabled(False)
                    self.UI.Button_FindSurface.setEnabled(False)
                try:
                    self.Thread_extract_surface.start()
                except:
                    str_e = str(self.Thread_extract_surface.exception)
                    self.Reset_Button()
                    QMessageBox.warning(self, "Extraction Error",
                                            str_e,
                                            QMessageBox.Ok)
            else:
                Warning_text = "Surface has not been found yet. " \
                                "Please Select Points before Extract Surface. "
                QMessageBox.warning(self, "Extraction Error", Warning_text, QMessageBox.Ok)

    def raiseerror(self, str_e):
        if os.path.exists(self.UI.extract_config_path):
            os.remove(self.UI.extract_config_path)
        self.UI.tomo_addresult = None
        self.Thread_extract_surface.terminate()
        QMessageBox.warning(self, "Extraction Error", str_e, QMessageBox.Ok)

    def load_progress(self,val):
        self.UI.progressBar.setValue(val)

    def Reset_Button(self):
        if self.UI.tomo_addresult is not None:
            # add widget in scrollArea for the new flattened tomo
            auto_surfcurrent = None
            if self.UI_tabWidget_currentWidget_objectName == "tab_auto":
                auto_surfcurrent = (True, self.UI_tomo_surfcurrent)
            elif self.UI_tabWidget_currentWidget_objectName == "tab_manual":
                auto_surfcurrent = (False, self.UI_tomo_manual_surfcurrent)
            self.UI.Init_Scroll_Mrc(auto_surfcurrent)
        if self.batch_mode:
            # this job finished, then start next job
            self.Thread_extract_surface_batch.continue_event.set()
        else:
            self.UI.Button_ExtractSurface.setEnabled(True)
            self.UI.Button_FindSurface.setEnabled(True)
            self.UI.label_progress.setText("Extract Surface Done")

    def SpinBox_Order(self):
        value = self.UI.spinBox_Order.value()
        if value >= 0 and value <= 99:
            self.UI.spinBox_Order.setValue(value)
        elif value < 0:
            self.UI.spinBox_Order.setValue(0)
        else:
            self.UI.spinBox_Order.setValue(99)

    def SpinBox_Fillvalue(self):
        value = self.UI.doubleSpinBox_Fillvalue.value()
        if value >= -90 and value <= 90:
            self.UI.doubleSpinBox_Fillvalue.setValue(value)
        elif value < -90:
            self.UI.doubleSpinBox_Fillvalue.setValue(-90)
        else:
            self.UI.doubleSpinBox_Fillvalue.setValue(90)

    def SpinBox_RBFsample(self):
        value = self.UI.spinBox_RBFsample.value()
        if value >= 1 and value < 999:
            self.UI.spinBox_RBFsample.setValue(value)
        elif value < 0:
            self.UI.spinBox_RBFsample.setValue(1)
        else:
            self.UI.spinBox_RBFsample.setValue(999)

    # def SpinBox_Thickness(self):
    #     value = self.UI.spinBox_Thickness.value()
    #     if value >= 0 and value < 50:
    #         self.UI.spinBox_Thickness.setValue(value)
    #     elif value < 0:
    #         self.UI.spinBox_Thickness.setValue(0)
    #     else:
    #         self.UI.spinBox_Thickness.setValue(50)

    def radioButton_Change(self):
        if self.UI.radioButton_choseRBF.isChecked():
            self.UI.label_RBFsample.setEnabled(True)
            self.UI.spinBox_RBFsample.setEnabled(True)
            self.UI.label_Smooth.setEnabled(True)
            self.UI.doubleSpinBox_Smoothfactor.setEnabled(True)
            self.UI.label_Order.setEnabled(False)
            self.UI.spinBox_Order.setEnabled(False)
        else:
            self.UI.label_RBFsample.setEnabled(False)
            self.UI.spinBox_RBFsample.setEnabled(False)
            self.UI.label_Smooth.setEnabled(False)
            self.UI.doubleSpinBox_Smoothfactor.setEnabled(False)
            self.UI.label_Order.setEnabled(True)
            self.UI.spinBox_Order.setEnabled(True)


class QThread_Extract(QThread):
    def __init__(self):
        super(QThread_Extract,self).__init__()
        self.progress = 0

    load_progress   = pyqtSignal(int)
    error_signal    = pyqtSignal(str)

    def setParameters(self, FrameExtract:Mpicker_FrameExtract):
        self.UI = FrameExtract.UI
        self.UI_tabWidget_currentWidget_objectName = FrameExtract.UI_tabWidget_currentWidget_objectName
        self.UI_tomo_surfcurrent = FrameExtract.UI_tomo_surfcurrent
        self.UI_tomo_manual_surfcurrent = FrameExtract.UI_tomo_manual_surfcurrent

    def show_progress(self, value):
        while self.progress < value:
            self.progress += 1
            self.load_progress.emit(self.progress)

    def run(self):
        if self.UI_tabWidget_currentWidget_objectName == "tab_auto":
            if self.UI.scrollArea_Select_vbox.itemAt(self.UI_tomo_surfcurrent) is None:
                self.emit_error("Seems no surface was selected.")
                return
            chosed_label = self.UI.scrollArea_Select_vbox.itemAt(self.UI_tomo_surfcurrent).layout().itemAt(0).widget()
            chosed_number = chosed_label.text().split(". ")[0]
            self.UI.make_number = 0
            for i in range(self.UI.scroll_hbox.count()):
                chosed_label = self.UI.scroll_hbox.itemAt(i).itemAt(0).widget()
                number = chosed_label.text().split("-")[0]
                make_number = int(chosed_label.text().split("-")[1])
                if number == chosed_number and self.UI.make_number<=make_number:
                    self.UI.make_number = make_number
            self.UI.make_number = self.UI.make_number + 1
            self.UI.namesuffix = "surface_"+chosed_number
            name = self.UI.namesuffix + "-" +str(self.UI.make_number)+"_"
        elif self.UI_tabWidget_currentWidget_objectName == "tab_manual":
            if self.UI.scrollArea_manualsurf_vbox.itemAt(self.UI_tomo_manual_surfcurrent) is None:
                self.emit_error("Seems no surface was selected.")
                return
            chosed_surf_label = self.UI.scrollArea_manualsurf_vbox.itemAt(self.UI_tomo_manual_surfcurrent).layout().itemAt(0).widget()
            chosed_surf_number = "M"+chosed_surf_label.text().split(". ")[0]
            self.UI.make_number = 0
            for i in range(self.UI.scroll_hbox.count()):
                chosed_label = self.UI.scroll_hbox.itemAt(i).itemAt(0).widget()
                number = chosed_label.text().split("-")[0]
                make_number = int(chosed_label.text().split("-")[1])
                if number == chosed_surf_number and self.UI.make_number <= make_number:
                    self.UI.make_number = make_number
            self.UI.make_number = self.UI.make_number + 1
            self.UI.namesuffix = "manual_"+chosed_surf_label.text().split(". ")[0]
            name = self.UI.namesuffix + "-" + str(self.UI.make_number) + "_"
        else:
            self.emit_error("Unknown tab widget object name: " + self.UI_tabWidget_currentWidget_objectName)
            return

        self.UI.result_folder = os.path.join(self.UI.Text_SavePath.toPlainText(), self.UI.namesuffix + "_" +
                                                os.path.splitext(os.path.basename(self.UI.MRC_path))[0])

        if self.UI.radioButton_choseRBF.isChecked():
            RBF_sample = self.UI.spinBox_RBFsample.value()
            self.UI.Result_path = os.path.join(self.UI.result_folder, name +"RBF_" + str(RBF_sample)  + "_thick_" + str(self.UI.spinBox_Thickness.value()) + "_result.mrc")
            self.UI.Convert_path = os.path.join(self.UI.result_folder, name + "RBF_" + str(RBF_sample)  + "_thick_" + str(self.UI.spinBox_Thickness.value()) + "_convert_coord.npy")
            self.UI.savetxt_path = os.path.join(self.UI.result_folder, name + "RBF_" + str(RBF_sample) + "_thick_" + str(self.UI.spinBox_Thickness.value()) + "_SelectPoints.txt")
            self.UI.saverbfcoord_path = os.path.join(self.UI.result_folder, name + "RBF_InterpCoords.txt")
        else:
            RBF_sample = None
            self.UI.Result_path = os.path.join(self.UI.result_folder, name + "POLY_" + str(self.UI.spinBox_Order.value())  + "_thick_" + str(self.UI.spinBox_Thickness.value()) + "_result.mrc")
            self.UI.Convert_path = os.path.join(self.UI.result_folder , name + "POLY_" + str(self.UI.spinBox_Order.value())  + "_thick_" + str(self.UI.spinBox_Thickness.value()) + "_convert_coord.npy")
            self.UI.savetxt_path = os.path.join(self.UI.result_folder , name + "POLY_" + str(self.UI.spinBox_Order.value())  + "_thick_" + str(self.UI.spinBox_Thickness.value()) + "_SelectPoints.txt")
            self.UI.saverbfcoord_path = None

        # Init_config_extract
        self.UI.extract_config_path = os.path.join(self.UI.result_folder,
                                        self.UI.namesuffix + "-" + str(self.UI.make_number) + ".config")
        if os.path.exists(self.UI.extract_config_path) is False:
            config_file = open(self.UI.extract_config_path, "w").close()
        self.UI.extract_config.read(self.UI.extract_config_path, encoding='utf-8')
        if self.UI.extract_config.has_section("Parameter") is False:
            self.UI.extract_config.add_section("Parameter")
        if self.UI.radioButton_choseRBF.isChecked():
            self.UI.extract_config.set("Parameter", "Method", "RBF")
        else:
            self.UI.extract_config.set("Parameter", "Method", "POLY")
        self.UI.extract_config.set("Parameter", "RBFSample", str(self.UI.spinBox_RBFsample.value()))
        self.UI.extract_config.set("Parameter", "PolyOrder", str(self.UI.spinBox_Order.value()))
        self.UI.extract_config.set("Parameter", "Thickness", str(self.UI.spinBox_Thickness.value()))
        self.UI.extract_config.set("Parameter", "CylinderOrder", str(self.UI.spinBox_Cylinderorder.value()))
        self.UI.extract_config.set("Parameter", "FillValue", str(self.UI.doubleSpinBox_Fillvalue.value()))
        self.UI.extract_config.set("Parameter", "smoothfactor",str(self.UI.doubleSpinBox_Smoothfactor.value()))
        self.UI.extract_config.set("Parameter", "expandratio",str(self.UI.ad_expandratio))
        warnings.filterwarnings("ignore")
        with open(self.UI.extract_config_path, "w") as config_file:
            self.UI.extract_config.write(config_file)
        warnings.filterwarnings("default")

        self.show_progress(20)

        try:
            result_path = self.UI.Result_path
            core_path = Mpicker_core_gui.__file__
            cmd = f'{sys.executable} {core_path} --mode flatten --config_tomo {self.UI.ini_config_path} --config_surf {self.UI.extract_config_path}'
            if self.UI.checkBox_show3d.isChecked():
                cmd += ' --show_3d'
            if self.UI.checkBox_showfitting.isChecked():
                cmd += ' --show_fit'
            if self.UI.advance_setting.checkBox_adprinttime.isChecked():
                cmd += ' --time'
            print(f'\033[0;35m{cmd}\033[0m')

            s = os.system(cmd)
            if s != 0:
                print("exit code", s)
                raise Exception("surface flatten failed, see terminal for details")
            self.load_progress.emit(75)

            with mrcfile.mmap(result_path, permissive=True) as mrc:
                self.UI.tomo_addresult = mrc.data[:, ::-1, :]

        except Exception as e:
            self.emit_error(e)

        self.show_progress(100)

    def emit_error(self, e):
        self.UI.tomo_addresult = None
        self.error_signal.emit(str(e))


class QThread_Extract_Batch(QThread):

    load_progress   = pyqtSignal(int)

    def setParameters(self, FrameExtract:Mpicker_FrameExtract):
        self.FrameExtract = FrameExtract
        self.UI = self.FrameExtract.UI
        self.Extract_Label_List = self.FrameExtract.Extract_Label_List
        self.result_folder = self.UI.result_folder # just for some check in Extract_Surface
        self.continue_event = threading.Event()

    def run(self):
        idx = 0
        if self.FrameExtract.UI_tabWidget_currentWidget_objectName == "tab_auto":
            for i in range(self.UI.scrollArea_Select_vbox.count()):
                if self.UI.scrollArea_Select_vbox.itemAt(i) is None:
                    continue
                chosed_label = self.UI.scrollArea_Select_vbox.itemAt(i).layout().itemAt(0).widget()
                if chosed_label.text() not in self.Extract_Label_List:
                    continue
                chosed_number = chosed_label.text().split(". ")[0]
                self.FrameExtract.UI_SURF_path = os.path.join(self.result_folder, "surface_"+chosed_number+"_surf.mrc.npz")
                self.FrameExtract.UI_tomo_surfcurrent = i
                idx += 1
                self.continue_event.clear() # clear the event for next iteration
                self.load_progress.emit(idx) # run single surface extraction
                self.continue_event.wait() # wait for gui update
        elif self.FrameExtract.UI_tabWidget_currentWidget_objectName == "tab_manual":
            for i in range(self.UI.scrollArea_manualsurf_vbox.count()):
                if self.UI.scrollArea_manualsurf_vbox.itemAt(i) is None:
                    continue
                chosed_label = self.UI.scrollArea_manualsurf_vbox.itemAt(i).layout().itemAt(0).widget()
                if chosed_label.text() not in self.Extract_Label_List:
                    continue
                chosed_number = chosed_label.text().split(". ")[0]
                self.FrameExtract.UI_ManualTxt_path = os.path.join(self.result_folder, "manual_"+chosed_number+"_surf.txt")
                self.FrameExtract.UI_tomo_manual_surfcurrent = i
                idx += 1
                self.continue_event.clear()
                self.load_progress.emit(idx)
                self.continue_event.wait()
