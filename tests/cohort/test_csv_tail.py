"""Logical EOF checks and validated DuckDB rejection using tiny synthetic CSVs."""

from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import duckdb

from src.cohort.common import (
    CSV_TAIL_BYTES,
    _check_csv_tail,
    _csv_reader,
    _OutputRun,
    _validate_csv_rejects,
)


class CsvTailTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="csv_tail_test_")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "source.csv"
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def read_python(self) -> tuple[list, dict]:
        check = _check_csv_tail(self.path)
        with self.path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(_csv_reader(stream, check))
        return rows, check

    def read_duckdb(self) -> tuple[list, dict]:
        check = _check_csv_tail(self.path)
        with duckdb.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM read_csv(?, columns={'a':'VARCHAR','b':'VARCHAR',"
                "'c':'VARCHAR'}, auto_detect=false, header=true, delim=',', "
                "quote='\"', escape='\"', strict_mode=true, null_padding=false, "
                "ignore_errors=true, store_rejects=true, rejects_limit=0)",
                [str(self.path)],
            ).fetchall()
            rejects = connection.execute(
                "SELECT line_byte_position, csv_line FROM reject_errors"
            ).fetchall()
            _validate_csv_rejects(self.path, check, rejects)
        return rows, check

    def test_valid_endings_quotes_multiline_bom_and_empty_values(self) -> None:
        for newline in (b"\n", b"\r\n"):
            for ending in (b"", newline):
                for bom in (b"", b"\xef\xbb\xbf"):
                    with self.subTest(newline=newline, ending=ending, bom=bom):
                        data = (
                            bom
                            + newline.join(
                                [
                                    b"a,b,c",
                                    b'1,"two, commas",3',
                                    b'2,"multi' + newline + b'line ""quoted""",',
                                ]
                            )
                            + ending
                        )
                        self.path.write_bytes(data)
                        for reader in (self.read_python, self.read_duckdb):
                            rows, check = reader()
                            self.assertEqual(len(rows), 2)
                            self.assertEqual(check["status"], "complete")
                            self.assertEqual(check["skipped_rows"], 0)
                        self.assertEqual(self.path.read_bytes(), data)

    def test_only_incomplete_final_record_is_skipped(self) -> None:
        for newline in (b"\n", b"\r\n"):
            for tail in (b"4,5", b'4,"unterminated', b'4,"multi' + newline + b"line"):
                with self.subTest(newline=newline, tail=tail):
                    self.path.write_bytes(newline.join([b"a,b,c", b"1,2,3", tail]))
                    for reader in (self.read_python, self.read_duckdb):
                        rows, check = reader()
                        self.assertEqual(len(rows), 1)
                        self.assertEqual(check["status"], "incomplete")
                        self.assertEqual(check["skipped_rows"], 1)

    def test_middle_multiple_and_extra_fields_are_not_skipped(self) -> None:
        for data in (
            b"a,b,c\n4,5\n1,2,3",
            b"a,b,c\n4,5\n1,2,3\n4,5",
            b"a,b,c\n1,2,3\n4,5,6,7",
            b'a,b,c\n1,2,3\n4,"bad"x,6',
        ):
            with self.subTest(data=data):
                self.path.write_bytes(data)
                with self.assertRaises(ValueError):
                    self.read_duckdb()
        self.path.write_bytes(b"a,b,c\n4,5\n1,2,3")
        with self.assertRaisesRegex(csv.Error, "before EOF"):
            self.read_python()

    def test_large_quoted_suffix_is_unverified_until_stream_read(self) -> None:
        text = b"x" * CSV_TAIL_BYTES + b"\ninside a quoted field"
        self.path.write_bytes(b'a,b,c\n1,"' + text + b'",3')
        self.assertEqual(_check_csv_tail(self.path)["status"], "unverified")
        for reader in (self.read_python, self.read_duckdb):
            rows, check = reader()
            self.assertEqual(len(rows), 1)
            self.assertEqual(check["status"], "complete")

    def test_unverified_previews_preserve_header_and_last_line_in_report(self) -> None:
        header = '"first,column",second,third'
        last_line = '2,"review, with ""quotes""",3'
        data = (
            header
            + '\r\n1,"'
            + "x" * CSV_TAIL_BYTES
            + '\r\ninside a multiline field",3\r\n'
            + last_line
            + "\r\n"
        )
        self.path.write_bytes(b"\xef\xbb\xbf" + data.encode("utf-8"))
        outputs = self.path.parent / "outputs"
        with _OutputRun(outputs, 1) as run:
            check = run.check_csv(self.path)
            self.assertEqual(check["status"], "unverified")
            self.assertEqual(check["header_text"], header)
            self.assertEqual(check["last_line_text"], last_line)
            self.assertFalse(check["last_line_truncated"])
            run.write_csv_tail_report()
        with (outputs / "audit/step_01_csv_tail_checks.csv").open(newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["header_text"], header)
        self.assertEqual(row["last_line_text"], last_line)
        self.assertEqual(row["last_line_truncated"], "False")

        self.path.write_bytes(b'a,b,c\n1,"' + b"x" * (CSV_TAIL_BYTES + 20))
        check = _check_csv_tail(self.path)
        self.assertEqual(check["status"], "unverified")
        self.assertTrue(check["last_line_truncated"])
        self.assertEqual(check["last_line_text"], "x" * CSV_TAIL_BYTES)

    def test_missing_empty_and_header_only(self) -> None:
        self.assertEqual(_check_csv_tail(self.path)["status"], "missing")
        self.path.write_bytes(b"")
        self.assertEqual(_check_csv_tail(self.path)["status"], "unreadable")
        self.path.write_bytes(b"a,b,c\n")
        rows, check = self.read_python()
        self.assertEqual(rows, [])
        self.assertEqual(check["reason"], "header_only")


if __name__ == "__main__":
    unittest.main()
