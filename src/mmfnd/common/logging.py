from rich.console import Console
from rich.logging import RichHandler
import logging

def setup_logger(name: str = "mmfnd", level: int = logging.INFO) -> logging.Logger:
    console = Console()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    handler = RichHandler(console=console, rich_tracebacks=True)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
