import pyqtgraph as pg
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QFrame, QLabel
from PyQt5.QtCore import Qt

class AnomalyGraph(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("Panel")
        self.layout = QVBoxLayout()
        
        self.header = QLabel("AI ANOMALY TELEMETRY")
        self.header.setObjectName("MetricLabel")
        self.layout.addWidget(self.header)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1E293B')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.1)
        self.plot_widget.setYRange(0, 1.0)
        
        # Hide axes for a cleaner "mini-graph" look or style them
        self.plot_widget.getAxis('left').setPen('#94A3B8')
        self.plot_widget.getAxis('bottom').setPen('#94A3B8')
        
        self.layout.addWidget(self.plot_widget)
        self.setLayout(self.layout)
        
        self.time_data = list(range(100))
        self.score_data = [0] * 100
        
        # Electric Blue Glow Pen
        self.pen = pg.mkPen(color='#3B82F6', width=3)
        self.data_line = self.plot_widget.plot(self.time_data, self.score_data, pen=self.pen)
        
    def update_graph(self, score):
        self.score_data = self.score_data[1:] + [score]
        self.data_line.setData(self.time_data, self.score_data)
        
        # Dynamic color change if anomaly detected
        if score > 0.5:
            self.pen.setColor('#EF4444')
        else:
            self.pen.setColor('#3B82F6')
        self.data_line.setPen(self.pen)

class MetricCard(QFrame):
    def __init__(self, label, value="0.0"):
        super().__init__()
        self.setObjectName("Panel")
        self.setFixedWidth(180)
        self.layout = QVBoxLayout()
        
        self.lbl_title = QLabel(label)
        self.lbl_title.setObjectName("MetricLabel")
        
        self.lbl_value = QLabel(value)
        self.lbl_value.setObjectName("MetricValue")
        
        self.layout.addWidget(self.lbl_title)
        self.layout.addWidget(self.lbl_value)
        self.setLayout(self.layout)
        
    def update(self, val):
        self.lbl_value.setText(str(val))
