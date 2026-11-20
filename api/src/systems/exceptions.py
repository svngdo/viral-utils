from src.exceptions import ConflictError


class SystemNameConflictError(ConflictError):
    def __init__(self):
        self.message = "A system with this name name already exists"


class SystemNotFoundError(ConflictError):
    def __init__(self):
        self.message = "System not found"


class NoSystemsFoundError(ConflictError):
    def __init__(self):
        self.message = "No systems found"
