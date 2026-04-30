import os
from dotenv import load_dotenv

def _getenv_bool(key: str, default: str = 'false') -> bool:
    """Get boolean environment variable"""
    return os.getenv(key, default).lower() in ['true', '1', 'yes', 'on']

def _getenv_int(key: str, default: str) -> int:
    """Get integer environment variable"""
    return int(os.getenv(key, default))

def _getenv_float(key: str, default: str) -> float:
    """Get float environment variable"""
    return float(os.getenv(key, default))

class Config:
    """Simple configuration class using environment variables"""

    def __init__(self, env_file: str = ".env"):
        # Load .env file
        load_dotenv(env_file, override=True)

        # Model
        self.model_path = os.getenv('PGUARD_MODEL_PATH', 'model/best.pt')
        self.confidence_threshold = _getenv_float('PGUARD_CONFIDENCE_THRESHOLD', '0.33')
        self.device = os.getenv('PGUARD_DEVICE', 'auto')

        # Stream
        self.stream_url = os.getenv('PGUARD_STREAM_URL', 'http://localhost:9081')
        self.frame_skip = _getenv_int('PGUARD_FRAME_SKIP', '5')
        self.reconnect_attempts = _getenv_int('PGUARD_RECONNECT_ATTEMPTS', '-1')
        self.reconnect_delay = _getenv_int('PGUARD_RECONNECT_DELAY', '5')

        # Detection
        self.cooldown_period = _getenv_int('PGUARD_COOLDOWN_PERIOD', '30')
        self.save_detections = _getenv_bool('PGUARD_SAVE_DETECTIONS', 'true')
        self.detection_folder = os.getenv('PGUARD_DETECTION_FOLDER', 'detections')
        self.image_format = os.getenv('PGUARD_IMAGE_FORMAT', 'jpg')
        self.image_quality = _getenv_int('PGUARD_IMAGE_QUALITY', '85')

        # Preprocessing
        self.input_width = _getenv_int('PGUARD_INPUT_WIDTH', '224')
        self.input_height = _getenv_int('PGUARD_INPUT_HEIGHT', '224')
        self.normalize = _getenv_bool('PGUARD_NORMALIZE', 'false')
        self.bgr_to_rgb = _getenv_bool('PGUARD_BGR_TO_RGB', 'true')

        # Pushover Notifications
        self.pushover_enabled = _getenv_bool('PGUARD_PUSHOVER_ENABLED', 'false')
        self.pushover_user_key = os.getenv('PGUARD_PUSHOVER_USER_KEY', '')
        self.pushover_api_token = os.getenv('PGUARD_PUSHOVER_API_TOKEN', '')

        # Logging
        self.log_level = os.getenv('PGUARD_LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('PGUARD_LOG_FILE', 'logs/pigeon_detection.log')
        self.log_console = _getenv_bool('PGUARD_LOG_CONSOLE', 'true')

        # Performance
        self.max_queue_size = _getenv_int('PGUARD_MAX_QUEUE_SIZE', '10')
        self.processing_threads = _getenv_int('PGUARD_PROCESSING_THREADS', '2')
        self.gpu_memory_fraction = _getenv_float('PGUARD_GPU_MEMORY_FRACTION', '0.8')

        # Advanced
        self.debug_mode = _getenv_bool('PGUARD_DEBUG_MODE', 'false')
        self.save_all_frames = _getenv_bool('PGUARD_SAVE_ALL_FRAMES', 'false')
        self.motion_threshold = _getenv_float('PGUARD_MOTION_THRESHOLD', '0.1')
        self.enable_face_blur = _getenv_bool('PGUARD_ENABLE_FACE_BLUR', 'false')
