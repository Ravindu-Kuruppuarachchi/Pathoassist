"""Read single members of a large remote zip with HTTP range requests (disk is too tight to
download whole archives). The central directory is read once through a seekable HTTP file;
each member is then fetched as one byte range, with retries, and inflated locally.
"""
import io
import re
import struct
import time
import zipfile
import zlib

import requests


def _get_range(url, start, end, tries=8):
    for attempt in range(tries):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == end - start + 1:
                return r
        except requests.RequestException:
            pass
        time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(f"range {start}-{end} of {url} failed")


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (size from Content-Range)."""

    def __init__(self, url, block=2 ** 20):
        self.url, self.block, self.pos = url, block, 0
        cr = _get_range(url, 0, 0).headers["Content-Range"]
        self.size = int(re.search(r"/(\d+)", cr).group(1))
        self._cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def _blk(self, k):
        if k not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            s = k * self.block
            self._cache[k] = _get_range(self.url, s, min(self.size, s + self.block) - 1).content
        return self._cache[k]

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        out, p = bytearray(), self.pos
        while len(out) < n:
            k, o = divmod(p, self.block)
            chunk = self._blk(k)[o:o + n - len(out)]
            out += chunk
            p += len(chunk)
        b[:n] = out
        self.pos += n
        return n


class RemoteZip:
    def __init__(self, url):
        self.url = url
        self.infos = {i.filename: i for i in zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url))).infolist()}

    def namelist(self):
        return list(self.infos)

    def _range(self, start, end):
        return _get_range(self.url, start, end).content

    def _data_start(self, i):
        head = self._range(i.header_offset, i.header_offset + 29)
        n_name, n_extra = struct.unpack("<HH", head[26:30])
        return i.header_offset + 30 + n_name + n_extra

    def read(self, name):
        i = self.infos[name]
        start = self._data_start(i)
        data = self._range(start, start + i.compress_size - 1) if i.compress_size else b""
        if i.compress_type == zipfile.ZIP_STORED:
            return data
        if i.compress_type == zipfile.ZIP_DEFLATED:
            return zlib.decompressobj(-15).decompress(data)
        raise ValueError(f"unsupported compression {i.compress_type}")

    def extract_to(self, name, path, chunk=64 * 2 ** 20):
        """Stream a (large) member to a local file without holding it in memory."""
        i = self.infos[name]
        start = self._data_start(i)
        d = zlib.decompressobj(-15) if i.compress_type == zipfile.ZIP_DEFLATED else None
        with open(path, "wb") as f:
            for off in range(0, i.compress_size, chunk):
                b = self._range(start + off, start + min(off + chunk, i.compress_size) - 1)
                f.write(d.decompress(b) if d else b)
            if d:
                f.write(d.flush())
        return path
