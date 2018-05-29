# -*- coding: utf-8 -*-
"""TODO: Add ability to slice away some parallellipipedal region
   TODO: Add ability to add radiometer cones to the plot as well as perform
         live integration of the field
"""
from __future__ import division
import pyqtgraph as pg
import pyqtgraph.functions
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph.dockarea as da
import numpy as np

# Matplotlib is used to render equations
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()

import generator

def renderTeX(string):
  fig,ax = plt.subplots()
  ax.set_axis_off()
  fig.text(0.25,0.75,string, size=20)
  buffer,shape = fig.canvas.print_to_buffer()
  arr = np.array(buffer,dtype=np.uint8)
  arr = np.reshape(arr, (shape[1],shape[0],4))

  # Clean up (otherwise each plt.subplot() could appear as a separate
  # window if plt.show() was called)
  plt.close() # Closes current figure (i.e. fig created above)

  # Find minimum bounding box
  bgcolor = arr[0,0,:]
  rows = np.sum(arr, axis=(1,2))
  inds = np.where(rows != shape[0]*np.sum(bgcolor))[0]
  rmin,rmax = inds[0], inds[-1]
  cols = np.sum(arr, axis=(0,2))
  inds = np.where(cols != shape[1]*np.sum(bgcolor))[0]
  cmin,cmax = inds[0], inds[-1]

  # Extract region within bounding box
  arr = arr[rmin:(rmax+1),cmin:(cmax+1),:]
  # Convert from RGBA to BGRA (which is ARGB backwards)
  arr = arr[:,:,[2,1,0,3]]
  
  # Make background white
  arr[np.all(arr == bgcolor, axis=2)] = [255,255,255,255]
  
  # Return a QImage
  return pyqtgraph.functions.makeQImage(arr, transpose=False)
  
class GUIwindow(QtGui.QMainWindow):
  def __init__(self):
    QtGui.QMainWindow.__init__(self)
    self.setWindowTitle('Turbulence visualization GUI')
    self.resize(800,600)

    dockArea = da.DockArea()
    self.setCentralWidget(dockArea)
    
    ## Create controlling dock
    controlDock = da.Dock('Parameters', size=(200,600))
    dockArea.addDock(controlDock, 'left')
    # Although controlDock functions like a LayoutWidget already,
    # the margins don't behave nicely. Hence we use an extra middle hand
    #layout = pg.LayoutWidget()
    layout = QtGui.QBoxLayout(QtGui.QBoxLayout.TopToBottom)
    dummyWidget = QtGui.QWidget()
    dummyWidget.setLayout(layout)
    controlDock.addWidget(dummyWidget)
    layout.setAlignment(QtCore.Qt.AlignTop)
    
    # Add size controller
    layout.addWidget(SizeWidget())
    
    # Add generator controller
    layout.addWidget(GeneratorChoiceWidget())
    
    ## Create plotting dock
    volDock = da.Dock('Volumetric plot', size=(600,800))
    dockArea.addDock(volDock, 'right')
    
    global volumetricPlot
    volumetricPlot = VolumetricPlot()
    self.volumetricPlot = volumetricPlot # Store reference
    volDock.addWidget(volumetricPlot,0,0)
    
    # Set up transparency slider
    Tslider = QtGui.QSlider()
    Tslider.setMinimum(0)
    Tslider.setMaximum(255)
    Tslider.setValue(127)
    Tslider.valueChanged.connect(volumetricPlot.transparencyChanged)
    Tslider.sliderReleased.connect(volumetricPlot.updateVolumeData)
    volDock.addWidget(Tslider,0,1)

    Sslider = QtGui.QSlider()
    Sslider.setMinimum(0)
    Sslider.setMaximum(volumetricPlot.Nz)
    Sslider.setValue(volumetricPlot.sliceHeight)
    Sslider.valueChanged.connect(volumetricPlot.sliceHeightChanged)
    Sslider.sliderReleased.connect(volumetricPlot.sliceVolumeData)
    volDock.addWidget(Sslider,0,2)
    
    Pslider = QtGui.QSlider()
    Pslider.setMinimum(10)
    Pslider.setMaximum(150)
    Pslider.setValue(100)
    Pslider.valueChanged.connect(volumetricPlot.powerParamChanged)
    Pslider.sliderReleased.connect(volumetricPlot.updateVolumeData)
    volDock.addWidget(Pslider,0,3)
    
    self.show()

class GaussParameterWidget(pg.LayoutWidget):
  def __init__(self):
    pg.LayoutWidget.__init__(self)
    
    currRow = 0
    img = renderTeX('$C(\\vec{r})\\propto \\mathrm{exp}(-r^2/a^2)' +
                    '$\n$' +
                    '\\Phi(\\vec{k}) \propto \\mathrm{exp}(-a^2 k^2/4)$')
    eqnLabel = QtGui.QLabel(alignment=QtCore.Qt.AlignHCenter)
    eqnLabel.setPixmap(QtGui.QPixmap.fromImage(img))
    self.addWidget(eqnLabel,row=currRow,col=0,colspan=2)
    currRow += 1
    
    self.addWidget(QtGui.QLabel('a:'), row=currRow, col=0)
    self.aSpinBox = pg.SpinBox(value=1.,bounds=[0,None],suffix='m',
                          siPrefix=True, dec=True, step=0.5, minStep=0.1)
    self.addWidget(self.aSpinBox, row=currRow, col=1)
    currRow += 1
    
    self.covRB = QtGui.QRadioButton('Covariance')
    self.specRB = QtGui.QRadioButton('Spectrum')
    csButtonGroup = QtGui.QButtonGroup()
    csButtonGroup.addButton(self.covRB)
    csButtonGroup.addButton(self.specRB)
    csButtonGroup.setExclusive(True)
    self.csButtonGroup = csButtonGroup
    self.addWidget(self.covRB, row=currRow, col=0)
    self.addWidget(self.specRB, row=currRow, col=1)
    currRow += 1
    
    genButton = QtGui.QPushButton('Generate field')
    genButton.clicked.connect(self.generate)
    self.addWidget(genButton, row=currRow, colspan=2)
    
  def getName(self):
    return 'Gauss'
    
  def generate(self):
    print 'Generate pressed'
    Nx,Ny,Nz = [b.value() for b in NspinBoxes]
    Lx,Ly,Lz = [b.value() for b in LspinBoxes]
    a = self.aSpinBox.value()
    
    
    if self.covRB.isChecked():
      gen = generator.HomogeneousGenerator(Nx,Ny,Nz,Lx,Ly,Lz,
        covariance=lambda x,y,z: np.exp(-(x*x+y*y+z*z)/(a*a)))
    elif self.specRB.isChecked():
      gen = generator.HomogeneousGenerator(Nx,Ny,Nz,Lx,Ly,Lz,
        spectrum=lambda kx,ky,kz: a**3/8/np.pi**(3/2)
                 *np.exp(-a*a*(kx*kx+ky*ky+kz*kz)/4))
    else:
      print 'Select covariance or spectrum'
      return
    
    volumetricPlot.setData(Nx,Ny,Nz,Lx,Ly,Lz,gen.get_current_field())
    del gen
    
class IshimaruParameterWidget(pg.LayoutWidget):
  def __init__(self):
    pg.LayoutWidget.__init__(self)
    
    currRow = 0
    
    img = renderTeX('$\\Phi(\\vec{k}) \propto \\frac{\\mathrm{exp}\\left(-(L_{\\mathrm{min}} k)^2\\right)}{\\left(1/{L_0}^2 + k^2\\right)^{-11/6}}$')
    eqnLabel = QtGui.QLabel(alignment=QtCore.Qt.AlignHCenter)
    eqnLabel.setPixmap(QtGui.QPixmap.fromImage(img))
    self.addWidget(eqnLabel,row=currRow,col=0,colspan=2)
    currRow += 1
    
    self.addWidget(QtGui.QLabel('L0:'), row=currRow, col=0)
    L0SpinBox = pg.SpinBox(value=2.,bounds=[0,None],suffix='m',
                          siPrefix=True, dec=True, step=1.0, minStep=0.1)
    self.addWidget(L0SpinBox,row=currRow,col=1)
    currRow += 1
    
    self.addWidget(QtGui.QLabel('Lmin:'), row=currRow, col=0)
    LminSpinBox = pg.SpinBox(value=0.1,bounds=[0,None],suffix='m',
                          siPrefix=True, dec=True, step=0.1, minStep=0.1)
    self.addWidget(LminSpinBox, row=currRow,col=1)
    currRow += 1
    
    genButton = QtGui.QPushButton('Generate field')
    self.addWidget(genButton, row=currRow, colspan=2)
    
    self.setHidden(True)
  
  def getName(self):
    return 'Ishimaru'
    
class GeneratorChoiceWidget(pg.LayoutWidget):
  def __init__(self):
    pg.LayoutWidget.__init__(self)
    
    self.lastInd = -1
    
    self.addWidget(QtGui.QLabel('Generator type:'), row=0, col=0)
    self.genControllers = [
        GaussParameterWidget(),
        IshimaruParameterWidget()
      ]
    
    comboBox = QtGui.QComboBox()
    for genControl in self.genControllers:
      comboBox.addItem(genControl.getName())
    comboBox.currentIndexChanged.connect(self.changeGenerator)
    
    self.addWidget(comboBox, row=0, col=1)

  def changeGenerator(self, ind):
    if self.lastInd != -1:
      self.layout.removeWidget(self.genControllers[self.lastInd])
      self.genControllers[self.lastInd].setHidden(True)
    self.addWidget(self.genControllers[ind], row=1, col=0, colspan=2)
    self.genControllers[ind].setHidden(False)
    self.lastInd = ind
    
class SizeWidget(pg.LayoutWidget):
  def __init__(self):
    pg.LayoutWidget.__init__(self)
  
    # Title row
    currRow = 0
    for i,txt in enumerate(['x','y','z']):
      self.addWidget(QtGui.QLabel(txt,
          alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom),
          row=currRow, col=i+1)
    currRow += 1
    
    # Resolution row
    self.addWidget(QtGui.QLabel('N:',
        alignment=QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter),
        row=currRow, col=0)
    global NspinBoxes
    NspinBoxes = [QtGui.QSpinBox(value=16, minimum=1,maximum=1024) 
                  for _ in range(3)]
    for i,sb in enumerate(NspinBoxes):
      self.addWidget(sb,row=currRow,col=i+1)
    currRow += 1
    
    # Physical size row
    self.addWidget(QtGui.QLabel('L:',
        alignment=QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter),
        row=currRow, col=0)
    global LspinBoxes
    LspinBoxes = [pg.SpinBox(value=16.,bounds=[0,None],suffix='m',
                  siPrefix=True, dec=True, step=0.5, minStep=0.1)
                 for _ in range(3) ]
    for i,sb in enumerate(LspinBoxes):
      self.addWidget(sb,row=currRow,col=i+1)
    currRow += 1
    
class VolumetricPlot(gl.GLViewWidget):
  def __init__(self):
    gl.GLViewWidget.__init__(self)
    #w.setBackgroundColor((135,206,235)) # skyblue from gnuplot
    self.setBackgroundColor((255,255,255)) # white, good for report
    #w.setBackgroundColor((255,255,255)) # black, also useful?
    
    Nx,Ny,Nz = 16,16,16
    Lx,Ly,Lz = 16,16,16
    self.data = 2*np.random.rand(Nx,Ny,Nz) - 1
    
    self.Nx,self.Ny,self.Nz,self.Lx,self.Ly,self.Lz = Nx,Ny,Nz,Lx,Ly,Lz
    
    self.transparency = 1.
    self.sliceHeight = Nz
    self.powerParameter = 1.0
    
    # Add explicit bounding box
    boundingBox = gl.GLBoxItem()
    boundingBox.scale(Lx, Ly, Lz)
    boundingBox.setColor([0,0,0]) # Paint it black
    self.addItem(boundingBox)
    self.boundingBox = boundingBox

    # Add a GL volume item, the main attraction
    # It works by drawing a configurable number of planes per small volume
    # Uses lot's of interpolation.
    volItem = gl.GLVolumeItem([])
    volItem.scale(Lx/Nx, Ly/Ny, Lz/Nz)
    self.addItem(volItem)
    self.volItem = volItem # store a reference
    
    self.updateVolumeData()
    
  def transparencyChanged(self, val):
    self.transparency = val/255

  def updateVolumeData(self):
    map = pg.ColorMap([-1,-0.1,0.1,1],
                [[0.,0.,1.,self.transparency],
                 [0.,0.,0.5,0.], [0.5,0.0,0.,0.],
                 [1.,0.,0.,self.transparency]],
                mode=[pg.ColorMap.HSV_POS]*4)

    d2 = map.mapToByte(np.sign(self.data)*np.abs(self.data)**self.powerParameter)
    d2[1:, 0:2, 0:2] = [255,0,0,255]
    d2[0:2, 1:, 0:2] = [0,255,0,255]
    d2[0:2, 0:2, 1:] = [0,0,255,255]
    d2[0,0,0] = [0,0,0,255]
    self.volItem.setData(d2[:,:,0:self.sliceHeight,:])

  def sliceHeightChanged(self, val):
    self.sliceHeight = val
    
  def sliceVolumeData(self):
    self.volItem.setData(d2[:,:,0:sliceHeight,:])
    
  def powerParamChanged(self, val):
    self.powerParameter = val/100
    
  def setData(s, Nx,Ny,Nz,Lx,Ly,Lz, data):
    """Note that self has been shortened to s"""
    s.boundingBox.scale(Lx/s.Lx,Ly/s.Ly,Lz/s.Lz, local=False)
    s.volItem.scale(Lx/Nx*s.Nx/s.Lx, Ly/Ny*s.Ny/s.Ly, Lz/Nz*s.Nz/s.Lz)
  
    s.Nx,s.Ny,s.Nz = Nx,Ny,Nz
    s.Lx,s.Ly,s.Lz = Lx,Ly,Lz
    s.data = data
    s.sliceHeight = Nz
    s.updateVolumeData()
    
# Create application
app = QtGui.QApplication([])
win = GUIwindow()
    
## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
