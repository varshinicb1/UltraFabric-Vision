
# Professional UI Design Tokens
MAIN_STYLE = """
QMainWindow {
    background-color: #0F172A;
}

QWidget#CentralWidget {
    background-color: #0F172A;
}

QLabel {
    color: #E2E8F0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

QPushButton {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    color: #F8FAFC;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #334155;
    border-color: #64748B;
}

QPushButton#ActionBtn {
    background-color: #3B82F6;
    border: none;
}

QPushButton#ActionBtn:hover {
    background-color: #2563EB;
}

QPushButton#StopBtn {
    background-color: #EF4444;
    border: none;
}

QPushButton#StopBtn:hover {
    background-color: #DC2626;
}

QComboBox {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 6px;
    color: #F8FAFC;
    padding: 5px 15px;
}

QFrame#Panel {
    background-color: #1E293B;
    border-radius: 12px;
    border: 1px solid #334155;
}

QLabel#TitleLabel {
    font-size: 24px;
    font-weight: 800;
    color: #3B82F6;
    letter-spacing: 1px;
}

QLabel#StatusNormal {
    color: #10B981;
    font-weight: bold;
    font-size: 18px;
}

QLabel#StatusDefect {
    color: #EF4444;
    font-weight: bold;
    font-size: 18px;
}

QLabel#MetricValue {
    font-size: 20px;
    font-weight: 700;
    color: #F8FAFC;
}

QLabel#MetricLabel {
    font-size: 11px;
    color: #94A3B8;
    text-transform: uppercase;
    font-weight: 600;
}
"""
