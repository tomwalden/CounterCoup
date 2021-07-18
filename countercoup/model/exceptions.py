class IllegalMoveException(Exception):

    def __init__(self, message):
        super().__init__(message)


class IllegalGameException(Exception):

    def __init__(self, message):
        super().__init__(message)