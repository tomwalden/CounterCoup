class IllegalMoveException(Exception):
    """Exception for illegal game moves in Coup"""

    def __init__(self, message):
        super().__init__(message)


class IllegalGameException(Exception):
    """Exception for illegal game setups"""

    def __init__(self, message):
        super().__init__(message)