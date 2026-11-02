"""Pure temporary-directory tests for reversible HOME -> /data storage switch.

No real /data, home folder, crontab, dataset or training is accessed.
"""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.ops.cutover_historical_storage_perseo_v1 import (
    activate_paths,rollback_paths,require,material_differences
)

class StorageCutoverTests(unittest.TestCase):
    def setup_roots(self,root):
        home=Path(root)/"home"
        data=Path(root)/"data"
        home.mkdir()
        data.mkdir()
        original=home/"ltd-storage"
        original.mkdir()
        (original/"frozen.txt").write_bytes(b"immutable")
        backup=home/"ltd-storage.precutover-SYNTHETIC"
        return original,data,backup

    def test_root_directory_mtime_only_is_ignored(self):
        material,ignored=material_differences(".d..t...... ./\n")
        self.assertEqual(material,[])
        self.assertEqual(ignored,[".d..t...... ./"])

    def test_actual_file_content_change_is_never_ignored(self):
        lines=".d..t...... ./\n>fc........ datasets/historical/data.csv\n"
        material,ignored=material_differences(lines)
        self.assertEqual(material,[">fc........ datasets/historical/data.csv"])
        self.assertEqual(len(ignored),1)

    def test_nested_directory_mtime_is_still_blocked(self):
        material,ignored=material_differences(
            ".d..t...... ./\n.d..t...... datasets/historical/\n")
        self.assertEqual(material,[".d..t...... datasets/historical/"])
        self.assertEqual(len(ignored),1)

    def test_deletion_symlink_or_permission_diffs_are_blocked(self):
        lines="*deleting datasets/obsolete.csv\n"
        lines+=".L...p..... datasets/link\n"
        lines+=".f...p..... datasets/data.csv\n"
        material,ignored=material_differences(lines)
        self.assertEqual(len(material),3)
        self.assertFalse(ignored)

    def test_activate_preserves_original_bytes_and_backup(self):
        with TemporaryDirectory() as tmp:
            source,dest,backup=self.setup_roots(tmp)
            activate_paths(source,dest,backup)
            self.assertTrue(source.is_symlink())
            self.assertEqual(source.resolve(),dest.resolve())
            self.assertTrue((backup/"frozen.txt").is_file())
            self.assertEqual((backup/"frozen.txt").read_bytes(),b"immutable")

    def test_rollback_recreates_real_home_path_without_deleting_data(self):
        with TemporaryDirectory() as tmp:
            source,dest,backup=self.setup_roots(tmp)
            activate_paths(source,dest,backup)
            rollback_paths(source,dest,backup)
            self.assertTrue(source.is_dir())
            self.assertFalse(source.is_symlink())
            self.assertEqual((source/"frozen.txt").read_bytes(),b"immutable")
            self.assertTrue(dest.is_dir())

    def test_backup_collision_stops_before_mutating_home(self):
        with TemporaryDirectory() as tmp:
            source,dest,backup=self.setup_roots(tmp)
            backup.mkdir()
            with self.assertRaisesRegex(RuntimeError,"BACKUP_ALREADY_EXISTS"):
                activate_paths(source,dest,backup)
            self.assertTrue(source.is_dir())
            self.assertFalse(source.is_symlink())

    def test_wrong_symlink_blocks_rollback_without_changes(self):
        with TemporaryDirectory() as tmp:
            source,dest,backup=self.setup_roots(tmp)
            other=Path(tmp)/"other"
            other.mkdir()
            source.rename(backup)
            source.symlink_to(other,target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError,"UNEXPECTED_ACTIVE_SYMLINK"):
                rollback_paths(source,dest,backup)
            self.assertTrue(source.is_symlink())
            self.assertEqual(source.resolve(),other.resolve())
            self.assertTrue((backup/"frozen.txt").exists())

    def test_no_double_activation(self):
        with TemporaryDirectory() as tmp:
            source,dest,backup=self.setup_roots(tmp)
            activate_paths(source,dest,backup)
            another=source.with_name("ltd-storage.precutover-SECOND")
            with self.assertRaisesRegex(RuntimeError,"BAD_SOURCE_BEFORE_SWITCH"):
                activate_paths(source,dest,another)
            self.assertTrue(backup.is_dir())

    def test_partial_link_creation_fails_and_restores_source(self):
        with TemporaryDirectory() as tmp:
            source,dest,backup=self.setup_roots(tmp)
            # Not a synthetic failure injection, but an actual invalid symlink
            # destination path; activation refuses if the target cannot resolve.
            unavailable=Path(tmp)/"missing-dest"
            with self.assertRaises(RuntimeError):
                activate_paths(source,unavailable,backup)
            self.assertTrue(source.is_dir())
            self.assertFalse(source.is_symlink())
            self.assertTrue((source/"frozen.txt").is_file())
            self.assertFalse(backup.exists())

if __name__=="__main__":
    unittest.main()
