"""Mixins for the builder classes for the general data loading settings."""

# pylint: disable=too-few-public-methods

from typing import Union, List, Dict, Any


class RelativeFolderMixin:
    """Mixin class for setting the relative data folder."""

    def __init__(self):
        self.relative_data_folder = None

    def set_relative_data_folder(self, folder: str):
        """Set the relative data folder for the folder with the data loading.

        Args:
            folder (str): Relative path to the data folder.
                e.g. 'data_folder'. Where the data folder is located in
                project_path/data_folder.

        """
        self.relative_data_folder = folder
        return self


class FilenameRegexMixin:
    """Mixin class for setting the filename regex."""

    def __init__(self):
        self.filename_regex = None

    def set_filename_regex(self, regex: str):
        """Set the filename regex for the data files to load.

        Args:
            regex (str): Regular expression for the filenames, e.g.
                'data_*.csv'.

        References:
            [Explore Regex](https://regex101.com/)
            [Python Regex Doc](https://docs.python.org/3/library/re.html)
        """
        self.filename_regex = regex
        return self


class FileMinSizeBytesMixin:
    """Mixin class for setting the minimum file size in bytes."""

    def __init__(self):
        self.file_min_size_bytes = 10000

    def set_file_min_size_bytes(self, size: int):
        """Set the minimum file size in bytes for the data files to load.

        Args:
            size (int): Minimum file size in bytes. Default is 10000 bytes.
        """
        self.file_min_size_bytes = size
        return self


class HeaderRowMixin:
    """Mixin class for setting the header row."""

    def __init__(self):
        self.header_row = None

    def set_header_row(self, row: int):
        """Set the header row for the data files to load.

        Args:
            row (int): Row number for the header row in the data file, indexed
                from 0.
        """
        self.header_row = row
        return self


class DataChecksMixin:
    """Mixin class for setting the data checks."""

    def __init__(self):
        self.data_checks = None

    def set_data_checks(self, data_checks: Dict[str, Any]):
        """Dictionary of data checks to perform on the loaded data.

        Args:
            checks (dict): Dictionary of data checks to perform on the loaded
                data. The keys are the names of the checks, and the values are
                the parameters for the checks.
        """
        self.data_checks = data_checks
        return self


class DataColumnMixin:
    """Mixin class for setting the data column."""

    def __init__(self):
        self.data_column = None

    def set_data_column(self, data_columns: Union[List[str], List[int]]):
        """The data columns for the data files to load. Build with
        `DataChecksBuilder`.

        Args:
            data_columns: List of column numbers or names for the data columns
                to load from the data files. The columns are indexed from 0.
                e.g. [3, 5] or ['data 1', 'data 3'].
        """
        self.data_column = data_columns
        return self


class DataHeaderMixin:
    """Mixin class for setting the data header for the Stream."""

    def __init__(self):
        self.data_header = None

    def set_data_header(self, headers: List[str]):
        """Set the Stream headers corresponding to the data columns. This is
        to improve the readability of the Stream data. The headers should be
        in the same order as the data columns. These are also the same headers
        that will be written to the output file or csv.

        Args:
            headers: List of headers corresponding to the data
                columns to load. e.g. ['data-1[m/s]', 'data_3[L]'].
        """
        self.data_header = headers
        return self


class TimeColumnMixin:
    """Mixin class for setting the time column."""

    def __init__(self):
        self.time_column = None

    def set_time_column(self, columns: List[int]):
        """The time column for the data files to load. The time column is
        used to convert the time data to an Unix-Epoch timestamp.

        Args:
            columns: List of column indexes for the time columns to
                load from the data files. The columns are indexed from 0.
                e.g. [0] or [1, 2] to combine 1 and 2 columns.
        """
        self.time_column = columns
        return self


class TimeFormatMixin:
    """Mixin class for setting the time format."""

    def __init__(self):
        self.time_format = "%Y-%m-%dT%H:%M:%S"

    def set_time_format(self, time_format_str: str):
        """Set the time format for the time data in the data files.

        Args:
            time_format_str (str): Time format string for the time data in the
                data files. Default is ISO "%Y-%m-%dT%H:%M:%S".
                e.g. "%Y-%m-%dT%H:%M:%S" for '2021-01-01T12:00:00'.

        Examples:
            ``` py title="USA date format"
            time_format_str = "%m/%d/%Y %H:%M:%S"
            # e.g. '01/01/2021 12:00:00'
            ```

            ``` py title="European date format"
            time_format_str = "%d/%m/%Y %H:%M:%S"
            # e.g. '01/01/2021 12:00:00'
            ```

            ``` py title="ISO date format"
            time_format_str = "%Y-%m-%dT%H:%M:%S"
            # e.g. '2021-01-01T12:00:00'
            ```

            ``` py title="AM/PM time format"
            time_format_str = "%Y-%m-%d %I:%M:%S %p"
            # e.g. '2021-01-01 12:00:00 PM'
            ```

            ``` py title="Fractional seconds"
            time_format_str = "%Y-%m-%dT%H:%M:%S.%f"
            # e.g. '2021-01-01T12:00:00.123456'
            ```

        References:
            - [Python Docs](
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)
            - [Python Time Format](https://strftime.org/)
        """
        self.time_format = time_format_str
        return self


class DelimiterMixin:
    """Mixin class for setting the delimiter."""

    def __init__(self):
        self.delimiter = None

    def set_delimiter(self, delimiter: str):
        """Set the delimiter for the data files to load.

        Args:
            delimiter (str): Delimiter for the data columns in the data files.
                e.g. ',' for CSV files or '\t' for tab-separated files.
        """
        self.delimiter = delimiter
        return self


class TimeShiftSecondsMixin:
    """Mixin class for setting the time shift in seconds."""

    def __init__(self):
        self.time_shift_seconds = 0

    def set_time_shift_seconds(self, shift: int):
        """Set the time shift in seconds for the time data in the data files.
        This is helpful to match the time stamps of two data folders. This
        shift is applied to all files loaded with this builder.

        Args:
            shift (int): Time shift in seconds for the time data in the data
                files. Default is 0 seconds.
        """
        self.time_shift_seconds = shift
        return self


class TimezoneIdentifierMixin:
    """Mixin class for setting the timezone identifier."""

    def __init__(self):
        self.timezone_identifier = "UTC"

    def set_timezone_identifier(self, timezone: str):
        """Set the timezone identifier for the time data in the data files.
        The timezone shift is handled by the pytz library.

        Args:
            timezone (str): Timezone identifier for the time data in the data
                files. Default is 'UTC'.

        Examples:
            ``` py title="List of Timezones"
            timezone = "Europe/London"  # or "GMT"
            ```

            ``` py title="Mountain Timezone"
            timezone = "America/Denver"  # or "MST7MDT"
            ```

            ``` py title="ETH Zurich Timezone"
            timezone = "Europe/Zurich"  # or "CET"
            ```

        References:
            [List of Timezones](
            https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)
        """

        self.timezone_identifier = timezone
        return self


class ChecksCharactersMixin:
    """Mixin class for setting the character length range for data checks."""

    def __init__(self):
        self.characters = None

    def set_characters(self, characters: list[int]):
        """Set the character length range for the data checks. This is
        how many characters are expected a line of the data file, for it to
        be considered valid, and proceed with data parsing.

        Args:
            characters: List of one (or two) integers for the minimum (and
                maximum) number of characters expected in a line of the data
                file. e.g. [10, 100] for 10 to 100 characters. or [10] for
                10 or more characters.
        """
        self.characters = characters
        return self


class ChecksCharCountsMixin:
    """Mixin class for setting the character counts for data checks."""

    def __init__(self):
        self.char_counts = None

    def set_char_counts(self, char_counts: dict[str, int]):
        """Set the required character counts for the data checks. This is
        the number of times a character should appear in a line of the data
        file, for it to be considered valid, and proceed with data parsing.

        Args:
            char_counts: Dictionary of characters and their required counts
                for the data checks. The keys are the characters, and the
                values are the required counts. e.g. {",": 4, ":": 0}.

        Examples:
        ``` py title="Set number of commas"
        char_counts = {",": 4}
        # valid line: '1,2,3,4'
        # invalid line removed: '1,2,3'
        ```

        ``` py title="Filter out specific words"
        char_counts = {"Temp1 Error": 0}
        # valid line: '23.4, 0.1, 0.2, no error'
        # invalid line removed: '23.4, 0.1, 0.2, Temp1 Error'
        ```
        """
        self.char_counts = char_counts
        return self


class ChecksSkipRowsMixin:
    """Mixin class for setting the number of rows to skip at the beginning."""

    def __init__(self):
        self.skip_rows = 0

    def set_skip_rows(self, skip_rows: int):
        """Set the number of rows to skip at the beginning of the file.

        Args:
            skip_rows (int): Number of rows to skip at the beginning of the
                file.
        """
        self.skip_rows = skip_rows
        return self


class ChecksSkipEndMixin:
    """Mixin class for setting the number of rows to skip at the end."""

    def __init__(self):
        self.skip_end = 0

    def set_skip_end(self, skip_end: int):
        """Set the number of rows to skip at the end of the file.

        Args:
            skip_end (int): Number of rows to skip at the end of the file.
        """
        self.skip_end = skip_end
        return self
