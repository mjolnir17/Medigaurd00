import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

CHUNK = 1024 * 1024


def save_upload(upload: UploadFile, max_bytes: int, suffix: str = "") -> Path:
    """Streams an upload to a temp file and enforces a size limit. The caller deletes the file."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    size = 0
    try:
        while chunk := upload.file.read(CHUNK):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, f"The file is larger than the {max_bytes // 1024 ** 2} MB limit.")
            tmp.write(chunk)
    except Exception:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise
    tmp.close()
    return Path(tmp.name)
