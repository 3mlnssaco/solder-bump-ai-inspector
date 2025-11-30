"""
솔더 범프 결함 검출 시스템 - 메인 애플리케이션
"""

import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 애플리케이션 스타일 설정
    app.setStyle("Fusion")
    
    # 메인 윈도우 생성
    window = MainWindow()
    window.show()
    
    # 애플리케이션 실행
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
