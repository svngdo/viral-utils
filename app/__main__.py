from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)

setup_logging()

logger.info("Starting Application...")
