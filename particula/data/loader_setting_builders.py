"""Module for the builder classes for the general data loading settings."""

# pylint: disable=too-few-public-methods

from typing import Any, Dict
from particula.next.abc_builder import BuilderABC
from particula.data import mixin


# pylint: disable=too-many-ancestors
class LoaderSetting1DBuilder(
    BuilderABC,
    mixin.RelativeFolderMixin,
    mixin.FilenameRegexMixin,
    mixin.FileMinSizeBytesMixin,
    mixin.HeaderRowMixin,
    mixin.DataChecksMixin,
    mixin.DataColumnMixin,
    mixin.DataHeaderMixin,
    mixin.TimeColumnMixin,
    mixin.TimeFormatMixin,
    mixin.DelimiterMixin,
    mixin.TimeShiftSecondsMixin,
    mixin.TimezoneIdentifierMixin,
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
        mixin.RelativeFolderMixin.__init__(self)
        mixin.FilenameRegexMixin.__init__(self)
        mixin.FileMinSizeBytesMixin.__init__(self)
        mixin.HeaderRowMixin.__init__(self)
        mixin.DataChecksMixin.__init__(self)
        mixin.DataColumnMixin.__init__(self)
        mixin.DataHeaderMixin.__init__(self)
        mixin.TimeColumnMixin.__init__(self)
        mixin.TimeFormatMixin.__init__(self)
        mixin.DelimiterMixin.__init__(self)
        mixin.TimeShiftSecondsMixin.__init__(self)
        mixin.TimezoneIdentifierMixin.__init__(self)

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
    mixin.ChecksCharactersMixin,
    mixin.ChecksCharCountsMixin,
    mixin.ChecksSkipRowsMixin,
    mixin.ChecksSkipEndMixin,
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
        mixin.ChecksCharactersMixin.__init__(self)
        mixin.ChecksCharCountsMixin.__init__(self)
        mixin.ChecksSkipRowsMixin.__init__(self)
        mixin.ChecksSkipEndMixin.__init__(self)

    def build(self) -> Dict[str, Any]:
        """Build and return the data checks dictionary."""
        return {
            "characters": self.characters,
            "char_counts": self.char_counts,
            "skip_rows": self.skip_rows,
            "skip_end": self.skip_end,
        }
