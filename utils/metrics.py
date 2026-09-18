import time
from functools import wraps
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AITutorMetrics")

def measure_latency(func):
    """Decorator tự động đo thời gian thực thi của hàm (Latency)"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        logger.info(f"[PERFORMANCE] Hàm '{func.__name__}' thực thi mất: {latency_ms:.2f} ms")
        
        return result
    return wrapper