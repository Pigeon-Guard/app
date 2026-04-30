import asyncio
import logging
import os
import cv2
from pathlib import Path

from app.application import Application
from app.config import Config

def _setup_logging(cfg: Config):
    """Setup logging"""
    log_dir = os.path.dirname(cfg.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    handlers = [logging.FileHandler(cfg.log_file)]
    if cfg.log_console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def _create_directories(cfg: Config):
    """Create necessary directories"""
    os.makedirs(cfg.detection_folder, exist_ok=True)


async def run_daemon(app: Application):
    """Run in daemon mode"""
    await app.start()


async def run_interactive(app: Application):
    """Run in interactive mode"""
    # Start application in background
    task = asyncio.create_task(app.start())

    print("Pigeon detection started. Press 'q' to quit, 's' for status")

    loop = asyncio.get_event_loop()

    while app.running:
        # Read input in non-blocking way
        user_input = await loop.run_in_executor(None, input)
        user_input = user_input.strip().lower()

        if user_input == 'q':
            app.stop()
            break
        elif user_input == 's':
            status = app.get_status()
            print(f"Status: {status}")
        else:
            print("Commands: 'q' (quit), 's' (status)")

    # Wait for application to finish
    await task


async def run_image_detection(app: Application, image_path: str):
    """Run detection on a single image"""
    logger = logging.getLogger(__name__)

    # Load image
    if not Path(image_path).exists():
        logger.error(f"Image file not found: {image_path}")
        print(f"Error: Image file not found: {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f"Failed to load image: {image_path}")
        print(f"Error: Failed to load image: {image_path}")
        return

    logger.info(f"Processing image: {image_path}")

    # Run detection
    detection = await app.detect(frame)

    if detection:
        print(f"✓ Pigeon detected with {detection.confidence:.2%} confidence")
    else:
        print(f"✗ No pigeon detected")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Pigeon Detection")
    parser.add_argument("--env-file", default=".env", help="Environment file path (default: .env)")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--image", help="Run detection on a single image file")

    args = parser.parse_args()

    config = Config(env_file=args.env_file)
    _setup_logging(config)
    _create_directories(config)

    # Create application
    app = Application(config)

    try:
        if args.image:
            asyncio.run(run_image_detection(app, args.image))
        elif args.daemon:
            asyncio.run(run_daemon(app))
        else:
            asyncio.run(run_interactive(app))
    except KeyboardInterrupt:
        print("\nShutting down...")
        app.stop()