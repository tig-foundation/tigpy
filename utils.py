from functools import wraps
from datetime import datetime, timezone
import hashlib
import json
import base64
import logging
import time
import os

logging.basicConfig(
    format="[%(levelname)s] - %(message)s",
    level=logging.WARNING
)
logger = logging.getLogger("TIG")
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_nested(cls, d):
        if isinstance(d, dict):
            return cls({key: cls.from_nested(d[key]) for key in d})
        elif isinstance(d, list):
            return [cls.from_nested(l) for l in d]
        else:
            return d

def now() -> datetime:
    return datetime.now(timezone.utc).astimezone()

def md5Hex(txt) -> str:
    return hashlib.md5(txt.encode("utf-16")).hexdigest()
    
def md5Seed(txt) -> int:
    return int.from_bytes(
        hashlib.md5(txt.encode("utf-16")).digest()[-4:], "big"
    )

def base64Encode(txt) -> str:
    return base64.urlsafe_b64encode(txt.encode("utf-16")).decode()

def base64Decode(txt) -> int:
    return base64.urlsafe_b64decode(txt.encode()).decode("utf-16")

def minJsonDump(o):
    return json.dumps(o, separators=(',', ':'), sort_keys=True, default=str)

def loadJson(body: str):
    return AttrDict.from_nested(json.loads(body))

def timeit(func):
    if not os.getenv("TIMEIT"):
        return func
    # https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.debug(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper