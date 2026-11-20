from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class ConflictError(Exception):
    def __init__(self, message: str):
        self.message = message


class NotFoundError(Exception):
    def __init__(self, message: str):
        self.message = message


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ConflictError)
    async def conflict_error_handler(request: Request, exc: ConflictError):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": exc.message},
        )

    @app.exception_handler(NotFoundError)
    async def not_found_error_handler(request: Request, exc: NotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": exc.message},
        )
