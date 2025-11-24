# src/Exception.py

import sys
import traceback


def error_message_detail(error, error_detail=sys):
    """
    Creates a detailed error message with filename and line number.
    """
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    return (
        f"\nError occurred in script: {file_name}"
        f"\nLine number: {exc_tb.tb_lineno}"
        f"\nError message: {str(error)}"
    )


class CustomException(Exception):
    def __init__(self, error_message, error_detail=sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
