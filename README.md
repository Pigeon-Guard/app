# Pigeon Guard

![Tests](https://github.com/pigeon-guard/app/actions/workflows/ci.yml/badge.svg)

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

## Usage

## Raspberry Pi 5 with AI HAT+

**Prerequisites:**

- [Install required software packages for the AI HAT+](https://www.raspberrypi.com/documentation/computers/ai.html)
- [Install Docker on your Raspberry Pi 5](https://docs.docker.com/engine/install/debian/)

**Detection on single image**

```
docker run --rm \
    --network host \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib:/usr/lib:ro \
    -v /lib:/lib:ro \
    -v /usr/share:/usr/share:ro \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/.env.hailo:/app/.env \
    -v $(pwd)/data:/data \
    ghcr.io/pigeon-guard/app:latest --image /data/test-image.jpg
```

**Continuous detection in video stream**

```
docker run -d --restart always \
    --network host \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib:/usr/lib:ro \
    -v /lib:/lib:ro \
    -v /usr/share:/usr/share:ro \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/.env.hailo:/app/.env \
    ghcr.io/pigeon-guard/app:latest
```

## Other Systems

**Detection on single image**

```
docker run --rm \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/.env.x86:/app/.env \
    -v $HOME/Downloads:/data \
    ghcr.io/pigeon-guard/app:latest --image /data/test-image.jpg
```

**Continuous detection in video stream**

```
docker run \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/.env.x86:/app/.env \
    ghcr.io/pigeon-guard/app:latest
```

## Development

**Detection on single image**

```shell
./app.sh --image <path>
```

**Continuous detection in interactive mode**

```shell
./app.sh
```

**Custom environment file**

```shell
./app.sh --env-file .env.production
```

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