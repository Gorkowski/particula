# Mixin

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Mixin

> Auto-generated documentation for [particula.data.mixin](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py) module.

## ChecksCharCountsMixin

[Show source in mixin.py:299](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L299)

Mixin class for setting the character counts for data checks.

#### Signature

```python
class ChecksCharCountsMixin:
    def __init__(self): ...
```

### ChecksCharCountsMixin().set_char_counts

[Show source in mixin.py:305](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L305)

Set the required character counts for the data checks. This is
the number of times a character should appear in a line of the data
file, for it to be considered valid, and proceed with data parsing.

#### Arguments

- `char_counts` - Dictionary of characters and their required counts
    for the data checks. The keys are the characters, and the
    values are the required counts. e.g. {",": 4, ":": 0}.

#### Examples

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

#### Signature

```python
def set_char_counts(self, char_counts: dict[str, int]): ...
```



## ChecksCharactersMixin

[Show source in mixin.py:278](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L278)

Mixin class for setting the character length range for data checks.

#### Signature

```python
class ChecksCharactersMixin:
    def __init__(self): ...
```

### ChecksCharactersMixin().set_characters

[Show source in mixin.py:284](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L284)

Set the character length range for the data checks. This is
how many characters are expected a line of the data file, for it to
be considered valid, and proceed with data parsing.

#### Arguments

- `characters` - List of one (or two) integers for the minimum (and
    maximum) number of characters expected in a line of the data
    file. e.g. [10, 100] for 10 to 100 characters. or [10] for
    10 or more characters.

#### Signature

```python
def set_characters(self, characters: list[int]): ...
```



## ChecksSkipEndMixin

[Show source in mixin.py:349](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L349)

Mixin class for setting the number of rows to skip at the end.

#### Signature

```python
class ChecksSkipEndMixin:
    def __init__(self): ...
```

### ChecksSkipEndMixin().set_skip_end

[Show source in mixin.py:355](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L355)

Set the number of rows to skip at the end of the file.

#### Arguments

- `skip_end` *int* - Number of rows to skip at the end of the file.

#### Signature

```python
def set_skip_end(self, skip_end: int): ...
```



## ChecksSkipRowsMixin

[Show source in mixin.py:332](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L332)

Mixin class for setting the number of rows to skip at the beginning.

#### Signature

```python
class ChecksSkipRowsMixin:
    def __init__(self): ...
```

### ChecksSkipRowsMixin().set_skip_rows

[Show source in mixin.py:338](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L338)

Set the number of rows to skip at the beginning of the file.

#### Arguments

- `skip_rows` *int* - Number of rows to skip at the beginning of the
    file.

#### Signature

```python
def set_skip_rows(self, skip_rows: int): ...
```



## DataChecksMixin

[Show source in mixin.py:81](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L81)

Mixin class for setting the data checks.

#### Signature

```python
class DataChecksMixin:
    def __init__(self): ...
```

### DataChecksMixin().set_data_checks

[Show source in mixin.py:87](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L87)

Dictionary of data checks to perform on the loaded data.

#### Arguments

- `checks` *dict* - Dictionary of data checks to perform on the loaded
    data. The keys are the names of the checks, and the values are
    the parameters for the checks.

#### Signature

```python
def set_data_checks(self, data_checks: Dict[str, Any]): ...
```



## DataColumnMixin

[Show source in mixin.py:99](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L99)

Mixin class for setting the data column.

#### Signature

```python
class DataColumnMixin:
    def __init__(self): ...
```

### DataColumnMixin().set_data_column

[Show source in mixin.py:105](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L105)

The data columns for the data files to load. Build with
`DataChecksBuilder`.

#### Arguments

- `data_columns` - List of column numbers or names for the data columns
    to load from the data files. The columns are indexed from 0.
    e.g. [3, 5] or ['data 1', 'data 3'].

#### Signature

```python
def set_data_column(self, data_columns: Union[List[str], List[int]]): ...
```



## DataHeaderMixin

[Show source in mixin.py:118](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L118)

Mixin class for setting the data header for the Stream.

#### Signature

```python
class DataHeaderMixin:
    def __init__(self): ...
```

### DataHeaderMixin().set_data_header

[Show source in mixin.py:124](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L124)

Set the Stream headers corresponding to the data columns. This is
to improve the readability of the Stream data. The headers should be
in the same order as the data columns. These are also the same headers
that will be written to the output file or csv.

#### Arguments

- `headers` - List of headers corresponding to the data
    columns to load. e.g. ['data-1[m/s]', 'data_3[L]'].

#### Signature

```python
def set_data_header(self, headers: List[str]): ...
```



## DelimiterMixin

[Show source in mixin.py:206](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L206)

Mixin class for setting the delimiter.

#### Signature

```python
class DelimiterMixin:
    def __init__(self): ...
```

### DelimiterMixin().set_delimiter

[Show source in mixin.py:212](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L212)

Set the delimiter for the data files to load.

#### Arguments

- `delimiter` *str* - Delimiter for the data columns in the data files.
    e.g. ',' for CSV files or '	' for tab-separated files.

#### Signature

```python
def set_delimiter(self, delimiter: str): ...
```



## FileMinSizeBytesMixin

[Show source in mixin.py:48](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L48)

Mixin class for setting the minimum file size in bytes.

#### Signature

```python
class FileMinSizeBytesMixin:
    def __init__(self): ...
```

### FileMinSizeBytesMixin().set_file_min_size_bytes

[Show source in mixin.py:54](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L54)

Set the minimum file size in bytes for the data files to load.

#### Arguments

- `size` *int* - Minimum file size in bytes. Default is 10000 bytes.

#### Signature

```python
def set_file_min_size_bytes(self, size: int): ...
```



## FilenameRegexMixin

[Show source in mixin.py:27](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L27)

Mixin class for setting the filename regex.

#### Signature

```python
class FilenameRegexMixin:
    def __init__(self): ...
```

### FilenameRegexMixin().set_filename_regex

[Show source in mixin.py:33](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L33)

Set the filename regex for the data files to load.

#### Arguments

- `regex` *str* - Regular expression for the filenames, e.g.
    'data_*.csv'.

#### References

[Explore Regex](https://regex101.com/)
[Python Regex Doc](https://docs.python.org/3/library/re.html)

#### Signature

```python
def set_filename_regex(self, regex: str): ...
```



## HeaderRowMixin

[Show source in mixin.py:64](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L64)

Mixin class for setting the header row.

#### Signature

```python
class HeaderRowMixin:
    def __init__(self): ...
```

### HeaderRowMixin().set_header_row

[Show source in mixin.py:70](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L70)

Set the header row for the data files to load.

#### Arguments

- `row` *int* - Row number for the header row in the data file, indexed
    from 0.

#### Signature

```python
def set_header_row(self, row: int): ...
```



## RelativeFolderMixin

[Show source in mixin.py:8](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L8)

Mixin class for setting the relative data folder.

#### Signature

```python
class RelativeFolderMixin:
    def __init__(self): ...
```

### RelativeFolderMixin().set_relative_data_folder

[Show source in mixin.py:14](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L14)

Set the relative data folder for the folder with the data loading.

#### Arguments

- `folder` *str* - Relative path to the data folder.
    e.g. 'data_folder'. Where the data folder is located in
    project_path/data_folder.

#### Signature

```python
def set_relative_data_folder(self, folder: str): ...
```



## TimeColumnMixin

[Show source in mixin.py:138](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L138)

Mixin class for setting the time column.

#### Signature

```python
class TimeColumnMixin:
    def __init__(self): ...
```

### TimeColumnMixin().set_time_column

[Show source in mixin.py:144](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L144)

The time column for the data files to load. The time column is
used to convert the time data to an Unix-Epoch timestamp.

#### Arguments

- `columns` - List of column indexes for the time columns to
    load from the data files. The columns are indexed from 0.
    e.g. [0] or [1, 2] to combine 1 and 2 columns.

#### Signature

```python
def set_time_column(self, columns: List[int]): ...
```



## TimeFormatMixin

[Show source in mixin.py:157](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L157)

Mixin class for setting the time format.

#### Signature

```python
class TimeFormatMixin:
    def __init__(self): ...
```

### TimeFormatMixin().set_time_format

[Show source in mixin.py:163](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L163)

Set the time format for the time data in the data files.

#### Arguments

- `time_format_str` *str* - Time format string for the time data in the
    data files. Default is ISO "%Y-%m-%dT%H:%M:%S".
    e.g. "%Y-%m-%dT%H:%M:%S" for '2021-01-01T12:00:00'.

#### Examples

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

#### References

- [Python Docs](
https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)
- [Python Time Format](https://strftime.org/)

#### Signature

```python
def set_time_format(self, time_format_str: str): ...
```



## TimeShiftSecondsMixin

[Show source in mixin.py:223](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L223)

Mixin class for setting the time shift in seconds.

#### Signature

```python
class TimeShiftSecondsMixin:
    def __init__(self): ...
```

### TimeShiftSecondsMixin().set_time_shift_seconds

[Show source in mixin.py:229](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L229)

Set the time shift in seconds for the time data in the data files.
This is helpful to match the time stamps of two data folders. This
shift is applied to all files loaded with this builder.

#### Arguments

- `shift` *int* - Time shift in seconds for the time data in the data
    files. Default is 0 seconds.

#### Signature

```python
def set_time_shift_seconds(self, shift: int): ...
```



## TimezoneIdentifierMixin

[Show source in mixin.py:242](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L242)

Mixin class for setting the timezone identifier.

#### Signature

```python
class TimezoneIdentifierMixin:
    def __init__(self): ...
```

### TimezoneIdentifierMixin().set_timezone_identifier

[Show source in mixin.py:248](https://github.com/Gorkowski/particula/blob/main/particula/data/mixin.py#L248)

Set the timezone identifier for the time data in the data files.
The timezone shift is handled by the pytz library.

#### Arguments

- `timezone` *str* - Timezone identifier for the time data in the data
    files. Default is 'UTC'.

#### Examples

``` py title="List of Timezones"
timezone = "Europe/London"  # or "GMT"
```

``` py title="Mountain Timezone"
timezone = "America/Denver"  # or "MST7MDT"
```

``` py title="ETH Zurich Timezone"
timezone = "Europe/Zurich"  # or "CET"
```

#### References

[List of Timezones](
https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)

#### Signature

```python
def set_timezone_identifier(self, timezone: str): ...
```
