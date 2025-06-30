from PyQt5.QtCore import (
    Qt,
    pyqtSignal,
)
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QTextEdit,
)


class BatchProcessDialog(QDialog):
    """Batch processing dialog class"""

    promptReady = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle(self.tr("Batch Process All Images"))
        self.setMinimumWidth(450)
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI interface"""
        self.setStyleSheet(
            """
            QDialog {
                background-color: white;
                border-radius: 8px;
            }
        """
        )

        # Main layout
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(24, 24, 24, 24)
        dialog_layout.setSpacing(20)

        # Instruction label
        instruction_label = QLabel(
            self.tr("Enter the prompt to apply to all images:")
        )
        instruction_label.setStyleSheet(
            """
            QLabel {
                font-size: 14px;
                color: #374151;
                font-weight: 500;
            }
        """
        )
        dialog_layout.addWidget(instruction_label)

        # Input box design
        self.batch_message_input = QTextEdit()
        self.batch_message_input.setPlaceholderText(
            self.tr(
                "Type your prompt here and use `@image` to reference the image."
            )
        )
        self.batch_message_input.setStyleSheet(
            """
            QTextEdit {
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                background-color: #F9FAFB;
                color: #1F2937;
                font-size: 14px;
                line-height: 1.5;
                padding: 12px;
            }
            QTextEdit:focus {
                border: 1px solid #6366F1;
            }
            QScrollBar:vertical {
                width: 8px;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                background: #D1D5DB;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """
        )
        self.batch_message_input.setAcceptRichText(False)
        self.batch_message_input.setMinimumHeight(160)
        self.batch_message_input.setMaximumHeight(200)
        self.batch_message_input.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        dialog_layout.addWidget(self.batch_message_input)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 8, 0, 0)
        button_layout.setSpacing(12)
        button_layout.addStretch()

        # Cancel button
        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.setStyleSheet(
            """
            QPushButton {
                background-color: white;
                color: #4B5563;
                border: 1px solid #E5E7EB;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 500;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #F9FAFB;
                border-color: #D1D5DB;
            }
            QPushButton:pressed {
                background-color: #F3F4F6;
            }
        """
        )
        cancel_btn.setMinimumHeight(36)
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)

        # Confirm button
        confirm_btn = QPushButton(self.tr("Confirm"))
        confirm_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4F46E5;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 500;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4338CA;
            }
            QPushButton:pressed {
                background-color: #3730A3;
            }
        """
        )
        confirm_btn.setMinimumHeight(36)
        confirm_btn.setCursor(Qt.PointingHandCursor)
        confirm_btn.clicked.connect(self.accept)

        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(confirm_btn)
        dialog_layout.addLayout(button_layout)

        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setWindowFlags(self.windowFlags() & ~Qt.FramelessWindowHint)

    def center_on_parent(self):
        """Center the dialog on the parent window"""
        if self.parent:
            center_point = self.parent.mapToGlobal(self.parent.rect().center())
            dialog_rect = self.rect()
            self.move(
                center_point.x() - dialog_rect.width() // 2,
                center_point.y() - dialog_rect.height() // 2,
            )

    def get_prompt(self):
        """Get the user input prompt"""
        return self.batch_message_input.toPlainText().strip()

    def exec_(self):
        """Override exec_ method to adjust position before showing the dialog"""
        self.adjustSize()
        self.center_on_parent()
        result = super().exec_()

        if result == QDialog.Accepted:
            prompt = self.get_prompt()
            if prompt:
                self.promptReady.emit(prompt)
                return prompt
        return ""
