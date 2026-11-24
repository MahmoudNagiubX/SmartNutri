import io
import email.message
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from smartnutri_backend.camera import (
    CameraService,
    CameraUnavailable,
    FileCameraService,
    ImageQualityError,
    IPSnapshotCameraService,
    MockCameraService,
    _NoRedirectHandler,
)


def clear_image(size=(640, 480), color=(120, 130, 140), orientation=None):
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, size[0] // 2, size[1] // 2), fill=(220, 40, 20))
    draw.ellipse((size[0] // 2, size[1] // 3, size[0] - 40, size[1] - 40), fill=(20, 180, 40))
    output = io.BytesIO()
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    image.save(output, format="JPEG", quality=92, exif=exif)
    return output.getvalue()


class SequenceCamera(CameraService):
    source = "file"
    simulated = False

    def __init__(self, sequence, **kwargs):
        super().__init__(**kwargs)
        self.sequence = list(sequence)

    def capture_raw(self):
        value = self.sequence.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class CameraServiceTests(unittest.TestCase):
    def make_service(self, cls=SequenceCamera, **kwargs):
        defaults = {
            "max_image_bytes": 100_000,
            "min_width": 320,
            "min_height": 240,
            "max_dimension": 4096,
            "min_brightness": 18,
            "max_brightness": 238,
            "min_edge_variance": 4,
            "max_clipped_fraction": 0.65,
            "capture_retries": 0,
            "burst_count": 1,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    def test_mock_camera_is_marked_simulated_and_quality_checked(self):
        capture = MockCameraService(capture_retries=0).capture_best()
        self.assertEqual(capture.source, "mock")
        self.assertTrue(capture.simulated)
        self.assertTrue(capture.quality.acceptable)
        self.assertGreater(capture.quality.width, 320)

    def test_exif_orientation_is_applied_before_quality_and_storage(self):
        camera = self.make_service(sequence=[clear_image((640, 480), orientation=6)])
        capture = camera.capture_best()
        self.assertEqual((capture.quality.width, capture.quality.height), (480, 640))

    def test_blurry_and_exposed_images_are_rejected(self):
        blurry = io.BytesIO()
        Image.new("RGB", (640, 480), (120, 120, 120)).save(blurry, format="JPEG")
        camera = self.make_service(sequence=[blurry.getvalue()])
        with self.assertRaises(ImageQualityError):
            camera.capture_best()

        dark = io.BytesIO()
        Image.new("RGB", (640, 480), (0, 0, 0)).save(dark, format="JPEG")
        camera = self.make_service(sequence=[dark.getvalue()])
        with self.assertRaises(ImageQualityError):
            camera.capture_best()

    def test_capture_retries_and_selects_usable_frame(self):
        blurry = io.BytesIO()
        Image.new("RGB", (640, 480), (120, 120, 120)).save(blurry, format="JPEG")
        camera = self.make_service(
            sequence=[blurry.getvalue(), clear_image()], capture_retries=1
        )
        self.assertTrue(camera.capture_best().quality.acceptable)

    def test_oversized_and_unsupported_images_are_rejected(self):
        oversized = self.make_service(sequence=[b"x" * 100_001])
        with self.assertRaises(ImageQualityError):
            oversized.capture_best()
        unsupported = self.make_service(sequence=[b"not an image"])
        with self.assertRaises(ImageQualityError):
            unsupported.capture_best()
        small = self.make_service(sequence=[clear_image((160, 120))])
        with self.assertRaises(ImageQualityError):
            small.capture_best()

    def test_file_camera_reads_only_configured_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "meal.jpg"
            path.write_bytes(clear_image())
            capture = FileCameraService(path).capture_best()
            self.assertEqual(capture.source, "file")
            self.assertTrue(capture.simulated)
            self.assertTrue(capture.quality.acceptable)

    def test_ip_snapshot_adapter_allows_lan_and_rejects_public_or_credential_urls(self):
        camera = IPSnapshotCameraService(
            "http://192.168.1.20:8080/photo.jpg",
            allowed_host="192.168.1.20:8080",
        )
        self.assertTrue(camera.status()["configured"])
        for url in (
            "http://8.8.8.8/photo.jpg",
            "http://user:password@192.168.1.20/photo.jpg",
            "http://192.168.1.21:8080/photo.jpg",
            "http://192.168.1.20:8081/photo.jpg",
            "http://localhost:8080/photo.jpg",
            "http://camera.local:8080/photo.jpg",
            "http://127.0.0.1:8080/photo.jpg",
            "http://192.168.1.20:8080/photo.jpg?token=secret",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                IPSnapshotCameraService(url, allowed_host="192.168.1.20:8080")
        with self.assertRaises(ValueError):
            IPSnapshotCameraService("http://192.168.1.20:8080/photo.jpg")

    def test_ip_snapshot_blocks_all_redirects(self):
        camera = IPSnapshotCameraService(
            "http://192.168.1.20:8080/photo.jpg",
            allowed_host="192.168.1.20:8080",
        )
        request = urllib.request.Request(camera.snapshot_url)
        with self.assertRaises(CameraUnavailable):
            _NoRedirectHandler().redirect_request(
                request, None, 302, "Found", {}, "http://8.8.8.8/redirected.jpg"
            )
        with self.assertRaises(CameraUnavailable):
            _NoRedirectHandler().redirect_request(
                request, None, 302, "Found", {}, "http://192.168.1.20:8080/other.jpg"
            )

    def test_ip_snapshot_checks_response_size_and_content_type(self):
        class FakeResponse:
            def __init__(self, data, content_type="image/jpeg"):
                self.data = data
                self.headers = email.message.Message()
                self.headers["Content-Type"] = content_type
                self.headers["Content-Length"] = str(len(data))

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def geturl(self):
                return "http://192.168.1.20:8080/photo.jpg"

            def read(self, _count):
                return self.data

        class FakeOpener:
            def __init__(self, response):
                self.response = response

            def open(self, *_args, **_kwargs):
                return self.response

        camera = IPSnapshotCameraService(
            "http://192.168.1.20:8080/photo.jpg",
            allowed_host="192.168.1.20:8080",
            max_image_bytes=8,
        )
        with patch(
            "smartnutri_backend.camera.urllib.request.build_opener",
            return_value=FakeOpener(FakeResponse(b"123456789")),
        ):
            with self.assertRaises(ImageQualityError):
                camera.capture_raw()
        with patch(
            "smartnutri_backend.camera.urllib.request.build_opener",
            return_value=FakeOpener(FakeResponse(b"image", "text/html")),
        ):
            with self.assertRaises(ImageQualityError):
                camera.capture_raw()


if __name__ == "__main__":
    unittest.main()
