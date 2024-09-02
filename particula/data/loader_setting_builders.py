"""Module for the builder classes for the general data loading settings."""

# pylint: disable=too-few-public-methods

from typing import Any, Dict
from particula.next.abc_builder import BuilderABC
from particula.data.mixin import (
    RelativeFolderMixin,
    FilenameRegexMixin,
    FileMinSizeBytesMixin,
    HeaderRowMixin,
    DataChecksMixin,
    DataColumnMixin,
    DataHeaderMixin,
    TimeColumnMixin,
    TimeFormatMixin,
    DelimiterMixin,
    TimeShiftSecondsMixin,
    TimezoneIdentifierMixin,
    ChecksCharactersMixin,
    ChecksCharCountsMixin,
    ChecksSkipRowsMixin,
    ChecksSkipEndMixin,
)


# pylint: disable=too-many-ancestors
class LoaderSetting1DBuilder(
    BuilderABC,
    RelativeFolderMixin,
    FilenameRegexMixin,
    FileMinSizeBytesMixin,
    HeaderRowMixin,
    DataChecksMixin,
    DataColumnMixin,
    DataHeaderMixin,
    TimeColumnMixin,
    TimeFormatMixin,
    DelimiterMixin,
    TimeShiftSecondsMixin,
    TimezoneIdentifierMixin,
):
    """Builder class for creating settings for loading and checking 1D data
    from CSV files."""

    def __init__(self):
        required_parameters = [
            "relative_data_folder",
            "filename_regex",
            "file_min_size_bytes",
            "header_row",
            "data_checks",
            "data_column",
            "data_header",
            "time_column",
            "time_format",
            "delimiter",
            "time_shift_seconds",
            "timezone_identifier",
        ]
        BuilderABC.__init__(self, required_parameters)
        RelativeFolderMixin.__init__(self)
        FilenameRegexMixin.__init__(self)
        FileMinSizeBytesMixin.__init__(self)
        HeaderRowMixin.__init__(self)
        DataChecksMixin.__init__(self)
        DataColumnMixin.__init__(self)
        DataHeaderMixin.__init__(self)
        TimeColumnMixin.__init__(self)
        TimeFormatMixin.__init__(self)
        DelimiterMixin.__init__(self)
        TimeShiftSecondsMixin.__init__(self)
        TimezoneIdentifierMixin.__init__(self)

    def build(self) -> Dict[str, Any]:
        """Build and return the settings dictionary for 1D data loading."""
        self.pre_build_check()
        return {
            "relative_data_folder": self.relative_data_folder,
            "filename_regex": self.filename_regex,
            "MIN_SIZE_BYTES": self.file_min_size_bytes,
            "data_loading_function": "general_1d_load",
            "header_row": self.header_row,
            "data_checks": self.data_checks,
            "data_column": self.data_column,
            "data_header": self.data_header,
            "time_column": self.time_column,
            "time_format": self.time_format,
            "delimiter": self.delimiter,
            "time_shift_seconds": self.time_shift_seconds,
            "timezone_identifier": self.timezone_identifier,
        }


class DataChecksBuilder(
    BuilderABC,
    ChecksCharactersMixin,
    ChecksCharCountsMixin,
    ChecksSkipRowsMixin,
    ChecksSkipEndMixin,
):
    """Builder class for constructing the data checks dictionary."""

    def __init__(self):
        required_parameters = [
            "characters",
            "char_counts",
            "skip_rows",
            "skip_end",
        ]
        BuilderABC.__init__(self, required_parameters)
        ChecksCharactersMixin.__init__(self)
        ChecksCharCountsMixin.__init__(self)
        ChecksSkipRowsMixin.__init__(self)
        ChecksSkipEndMixin.__init__(self)

    def build(self) -> Dict[str, Any]:
        """Build and return the data checks dictionary."""
        return {
            "characters": self.characters,
            "char_counts": self.char_counts,
            "skip_rows": self.skip_rows,
            "skip_end": self.skip_end,
        }
