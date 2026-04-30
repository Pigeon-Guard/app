# Pigeon Guard

![Tests](https://github.com/YOUR_USERNAME/pigeon-guard/workflows/Run%20Unit%20Tests/badge.svg)

Detecting Pigeons in Images using Machine Learning

## Configuration

The application uses environment variables for configuration. Create a `.env` file in the project root:

```shell
cp .env.example .env
```

Then edit `.env` with your settings. See `.env.example` for all available options.

### Key Configuration Variables

- `PGUARD_MODEL_PATH`: Path to the detection model
- `PGUARD_STREAM_URL`: URL of the video stream
- `PGUARD_CONFIDENCE_THRESHOLD`: Detection confidence threshold (0.0-1.0)
- `PGUARD_PUSHOVER_ENABLED`: Enable/disable Pushover notifications
- `PGUARD_PUSHOVER_USER_KEY`: Your Pushover user key
- `PGUARD_PUSHOVER_API_TOKEN`: Your Pushover API token
- See `.env.example` for complete list

## Run detection on single image

```shell
./app.sh --image <path>
```

## Run continuous detection in interactive mode

```shell
./app.sh
```

## Run continuous detection in background

```shell
nohup ./app.sh --daemon &
```

## Using custom environment file

```shell
./app.sh --env-file .env.production
```

## Development

### Running Tests

Run the unit tests:

```shell
source .venv/bin/activate
python3 -m unittest discover tests
```

Run tests with verbose output:

```shell
python3 -m unittest discover -v tests
```

Run a specific test file:

```shell
python3 -m unittest tests.test_event_bus
```

### Test Coverage

The project includes comprehensive unit tests for:
- Event bus system
- Event handlers (detection and notification)
- Video stream observer
- Pushover notifier

Tests use mocks for external dependencies (cv2, requests, file I/O) and are fully automated via GitHub Actions.

See [tests/README.md](tests/README.md) for more details.