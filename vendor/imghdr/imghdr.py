"""Recognize image file formats based on their first few bytes.

Compatibility shim: this module was removed from the Python standard library
in Python 3.13 (deprecated since 3.11, PEP 594).  It is vendored here so that
packages that still import it (e.g. img2dataset 1.x) continue to work.

This is a verbatim copy of the CPython 3.12 implementation.
"""

__all__ = ["what"]


def what(file, h=None):
    """Return the type of image contained in *file*.

    *file* may be a filename or a file-like object.  The optional *h*
    argument provides the first few bytes of the file to test; when supplied,
    *file* is not opened.

    Returns a string such as ``'jpeg'``, ``'png'``, ``'gif'``, etc., or
    ``None`` if the format is not recognised.
    """
    f = None
    try:
        if h is None:
            if isinstance(file, str):
                f = open(file, "rb")  # noqa: PTH123
                h = f.read(32)
            else:
                location = file.tell()
                h = file.read(32)
                file.seek(location)
        for tf in tests:
            res = tf(h, f)
            if res:
                return res
    finally:
        if f:
            f.close()
    return None


# ---------------------------------------------------------------------------
# Per-format tests — each returns the format string or None
# ---------------------------------------------------------------------------

tests = []


def test_rgbe(h, f):
    if h[:6] in (b"#?RGBE", b"#?XYZE"):
        return "rgbe"


tests.append(test_rgbe)


def test_png(h, f):
    if h[:8] == b"\211PNG\r\n\032\n":
        return "png"


tests.append(test_png)


def test_jpeg(h, f):
    """JPEG data in JFIF or Exif format."""
    if h[6:10] in (b"JFIF", b"Exif"):
        return "jpeg"


tests.append(test_jpeg)


def test_gif(h, f):
    if h[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"


tests.append(test_gif)


def test_tiff(h, f):
    if h[:2] in (b"MM", b"II"):
        return "tiff"


tests.append(test_tiff)


def test_rgb(h, f):
    if h[:2] == b"\001\332":
        return "rgb"


tests.append(test_rgb)


def test_pbm(h, f):
    if len(h) >= 3 and h[0] == ord(b"P") and h[1] in b"14" and h[2] in b" \t\r\n":
        return "pbm"


tests.append(test_pbm)


def test_pgm(h, f):
    if len(h) >= 3 and h[0] == ord(b"P") and h[1] in b"25" and h[2] in b" \t\r\n":
        return "pgm"


tests.append(test_pgm)


def test_ppm(h, f):
    if len(h) >= 3 and h[0] == ord(b"P") and h[1] in b"36" and h[2] in b" \t\r\n":
        return "ppm"


tests.append(test_ppm)


def test_rast(h, f):
    if h[:4] == b"\x59\xa6\x6a\x95":
        return "rast"


tests.append(test_rast)


def test_xbm(h, f):
    if h[:8] == b"#define ":
        return "xbm"


tests.append(test_xbm)


def test_bmp(h, f):
    if h[:2] == b"BM":
        return "bmp"


tests.append(test_bmp)


def test_webp(h, f):
    if h[:4] == b"RIFF" and h[8:12] == b"WEBP":
        return "webp"


tests.append(test_webp)


def test_exr(h, f):
    if h[:4] == b"\x76\x2f\x31\x01":
        return "exr"


tests.append(test_exr)
