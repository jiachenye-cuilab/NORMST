"""Native process locks and safe file publication on the running platform."""

import errno
import json
import multiprocessing
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from filelock import Timeout

from normst import artifacts


def _hold_lock(path, connection):
    with artifacts.writer_lock(path):
        connection.send("acquired")
        connection.recv()
    connection.send("released")
    connection.close()


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="normst artifacts ")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def _start_holder(self, path):
        context = multiprocessing.get_context("spawn")
        connection, child = context.Pipe()
        process = context.Process(target=_hold_lock, args=(path, child))
        process.start()
        child.close()

        def cleanup():
            if process.is_alive():
                process.terminate()
            process.join(10)
            connection.close()
            self.assertFalse(process.is_alive(), "lock holder did not stop")

        self.addCleanup(cleanup)
        self.assertTrue(connection.poll(20), "lock holder did not become ready")
        self.assertEqual(connection.recv(), "acquired")
        return process, connection

    def test_process_lock_is_nonblocking_and_released_on_exit(self):
        for terminate in (False, True):
            with self.subTest(terminate=terminate):
                path = self.folder / f"writer-{terminate}.lock"
                process, connection = self._start_holder(path)
                # Contention must make one attempt, never enter a polling sleep.
                with patch("filelock._api.time.sleep", side_effect=AssertionError("lock waited")):
                    with self.assertRaisesRegex(RuntimeError, "another writer"):
                        with artifacts.writer_lock(path):
                            self.fail("two processes acquired the same lock")
                if terminate:
                    process.terminate()
                else:
                    connection.send("release")
                    self.assertTrue(connection.poll(20))
                    self.assertEqual(connection.recv(), "released")
                process.join(10)
                self.assertFalse(process.is_alive())
                if not terminate:
                    self.assertEqual(process.exitcode, 0)
                with artifacts.writer_lock(path):
                    pass

    def test_lock_io_errors_propagate(self):
        # An invalid destination is an I/O error, not lock contention.
        with self.assertRaises(OSError):
            with artifacts.writer_lock(self.folder):
                self.fail("a directory cannot be a lock file")
        error = OSError(errno.EIO, "injected lock I/O failure")
        with patch("normst.artifacts.FileLock") as factory:
            factory.return_value.acquire.side_effect = error
            with self.assertRaises(OSError) as caught:
                with artifacts.writer_lock(self.folder / "writer.lock"):
                    self.fail("acquisition failed")
            self.assertIs(caught.exception, error)
            factory.return_value.release.assert_not_called()

    def test_lock_body_exception_is_preserved_and_releases_lock(self):
        path = self.folder / "writer.lock"
        error = Timeout("unrelated-operation.lock")
        with self.assertRaises(Timeout) as caught:
            with artifacts.writer_lock(path):
                raise error
        self.assertIs(caught.exception, error)
        with artifacts.writer_lock(path):
            pass

    def test_files_are_synced_with_writable_handles_closed_before_publication(self):
        real_open = Path.open
        real_fsync = artifacts.os.fsync
        real_replace = artifacts.os.replace
        real_rename = artifacts.os.rename
        handles, synced = [], set()

        def track_open(path, mode="r", *args, **kwargs):
            stream = real_open(path, mode, *args, **kwargs)
            if mode == "r+b":
                handles.append(stream)
            return stream

        def track_fsync(descriptor):
            for stream in handles:
                if not stream.closed and stream.fileno() == descriptor:
                    self.assertTrue(stream.writable())
                    synced.add(id(stream))
            return real_fsync(descriptor)

        def replace(source, destination):
            self.assertEqual(Path(source).parent, Path(destination).parent)
            self.assertTrue(handles)
            self.assertTrue(all(stream.closed for stream in handles))
            return real_replace(source, destination)

        def rename(source, destination):
            self.assertEqual(Path(source).parent, Path(destination).parent)
            self.assertTrue(all(stream.closed for stream in handles))
            return real_rename(source, destination)

        with patch.object(Path, "open", track_open), \
                patch.object(artifacts.os, "fsync", track_fsync), \
                patch.object(artifacts.os, "replace", replace), \
                patch.object(artifacts.os, "rename", rename):
            target = self.folder / "metadata.json"
            target.write_text("old", encoding="utf-8")
            artifacts.write_json(target, {"value": "new"})
            with artifacts.staged_directory(self.folder / "prediction") as staging:
                (staging / "result.bin").write_bytes(b"complete result")
        self.assertEqual(synced, {id(stream) for stream in handles})
        self.assertEqual(json.loads(target.read_text()), {"value": "new"})
        output = self.folder / "prediction"
        self.assertEqual((output / "result.bin").read_bytes(), b"complete result")
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")

    def test_atomic_file_io_failures_preserve_previous_file(self):
        target = self.folder / "metadata.json"
        target.write_bytes(b"previous complete artifact")
        for operation in ("fsync", "replace"):
            with self.subTest(operation=operation):
                error = OSError(errno.EIO, f"injected {operation} failure")
                with patch.object(artifacts.os, operation, side_effect=error):
                    with self.assertRaises(OSError) as caught:
                        artifacts.write_json(target, {"value": "new"})
                self.assertIs(caught.exception, error)
                self.assertEqual(target.read_bytes(), b"previous complete artifact")
                self.assertEqual(list(self.folder.glob(".metadata.json-*")), [])

    def test_directory_sync_platform_branch_and_io_errors(self):
        # Simulates platform dispatch only; it is not a native Windows test.
        with patch.object(artifacts, "os", SimpleNamespace(name="nt")):
            artifacts.sync_directory(self.folder)
        for operation in ("open", "fsync"):
            with self.subTest(operation=operation):
                error = OSError(errno.EIO, f"injected directory {operation} failure")
                posix = SimpleNamespace(name="posix", O_RDONLY=0, O_DIRECTORY=1,
                    open=Mock(return_value=42), fsync=Mock(), close=Mock())
                getattr(posix, operation).side_effect = error
                with patch.object(artifacts, "os", posix):
                    with self.assertRaises(OSError) as caught:
                        artifacts.sync_directory(self.folder)
                self.assertIs(caught.exception, error)
                if operation == "fsync":
                    posix.close.assert_called_once_with(42)
                else:
                    posix.close.assert_not_called()

    def test_staged_file_sync_failure_marks_failed_and_allows_retry(self):
        output = self.folder / "prediction"
        real_sync = artifacts._sync_file
        error = OSError(errno.EIO, "injected output sync failure")

        def sync(path):
            if Path(path).name == "result.bin":
                raise error
            real_sync(path)

        with patch.object(artifacts, "_sync_file", sync):
            with self.assertRaises(OSError) as caught:
                with artifacts.staged_directory(output) as staging:
                    (staging / "result.bin").write_bytes(b"result")
        self.assertIs(caught.exception, error)
        self.assertFalse(output.exists())
        self.assertEqual(json.loads((staging / "status.json").read_text())["status"], "failed")
        with artifacts.staged_directory(output) as retry:
            (retry / "result.bin").write_bytes(b"result")
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")
        self.assertEqual(json.loads((staging / "status.json").read_text())["status"], "failed")

    def test_staged_rename_error_preserves_failure_status(self):
        output = self.folder / "prediction"
        error = OSError(errno.EIO, "injected directory rename failure")
        with patch.object(artifacts.os, "rename", side_effect=error):
            with self.assertRaises(OSError) as caught:
                with artifacts.staged_directory(output) as staging:
                    (staging / "result.bin").write_bytes(b"result")
        self.assertIs(caught.exception, error)
        self.assertFalse(output.exists())
        self.assertEqual(json.loads((staging / "status.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
