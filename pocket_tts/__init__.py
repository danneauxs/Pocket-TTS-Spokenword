try:
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package
    beartype_this_package(conf=BeartypeConf(is_color=False))
except ImportError:
    # Beartype not available, skip runtime type checking
    pass

# Only import TTSModel if runtime dependencies are available
try:
    from pocket_tts.models.tts_model import TTSModel  # noqa: E402
    _tts_model_available = True
except ImportError:
    _tts_model_available = False
    TTSModel = None

# Public API - only expose TTSModel if available
if _tts_model_available:
    __all__ = ["TTSModel"]
else:
    __all__ = []
