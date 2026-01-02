import logging
import logging.config
from pathlib import Path
from datetime import datetime
import json

def setup_logging(
    level: str = "INFO",
    experiment_name: str | None = None,
    config_file: Path | None = None,
) -> None:
    
    logs_dir = Path(__file__).resolve().parents[3] / "logs"
    logs_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp = experiment_name or "run"
    logfile = logs_dir / f"{ts}_{exp}.log"

    # Default to bundled JSON config if none is provided
    if config_file is None:
        config_file = Path(__file__).with_name("config.json")

    try:
        with config_file.open() as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback minimal config if JSON is missing/broken
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": "INFO",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "formatter": "standard",
                    "filename": str(logfile),
                    "level": "DEBUG",
                },
            },
            "root": {
                "handlers": ["console", "file"],
                "level": "DEBUG",
            },
        }

    level = level.upper()
    handlers = config.setdefault("handlers", {})

    console_handler = handlers.get("console")
    if console_handler is None:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": level,
        }
    else:
        console_handler["level"] = level

    file_handler = handlers.get("file")
    if file_handler is None:
        handlers["file"] = {
            "class": "logging.FileHandler",
            "formatter": "standard",
            "filename": str(logfile),
            "level": "DEBUG",
        }
    else:
        file_handler["filename"] = str(logfile)

    root = config.setdefault("root", {})
    root.setdefault("handlers", ["console", "file"])
    root.setdefault("level", "DEBUG")

    logging.config.dictConfig(config)