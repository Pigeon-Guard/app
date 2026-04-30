import asyncio
import logging
import os

from app import Application, Config

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Pigeon Detection")
    parser.add_argument("--env-file", default=".env", help="Environment file path (default: .env)")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")

    args = parser.parse_args()

    config = Config(env_file=args.env_file)
    _setup_logging(config)
    _create_directories(config)

    # Create application
    app = Application(config)

    try:
        if args.daemon:
            asyncio.run(run_daemon(app))
        else:
            asyncio.run(run_interactive(app))
    except KeyboardInterrupt:
        print("\nShutting down...")
        app.stop()