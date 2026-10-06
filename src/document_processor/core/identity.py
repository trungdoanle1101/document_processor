import hashlib


HASH_ALGORITHM = "sha256" 
def compute_hash(file_bytes: bytes) -> str:
    digest = hashlib.new(HASH_ALGORITHM, file_bytes).hexdigest()
    
    return f"{HASH_ALGORITHM}:{digest}"