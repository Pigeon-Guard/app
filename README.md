# Pigeon Guard

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