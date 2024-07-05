import hashlib
from . import more_math


def get_base32_hash(x: bytes) -> str:
    hash_bytes = hashlib.sha256(x).digest()
    hash_int = int.from_bytes(hash_bytes, byteorder="little")
    hash_str32 = more_math.to_base32_str(hash_int, 51)
    return hash_str32


def get_base60_hash(x: bytes) -> str:
    hash_bytes = hashlib.sha256(x).digest()
    hash_int = int.from_bytes(hash_bytes, byteorder="little")
    hash_str60 = more_math.to_base60_str(hash_int, 42)
    return hash_str60


def test_hash():
    import random
    rng = random.Random(x=42)
    first = set()
    for i in range(1000):
        bytes_ = rng.randbytes(20)
        hash60 = get_base60_hash(bytes_)
        first.add(hash60[0])
    assert len(first) == 60
