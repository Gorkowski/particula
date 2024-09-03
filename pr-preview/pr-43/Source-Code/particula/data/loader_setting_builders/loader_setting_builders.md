# Loader Setting Builders

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Loader Setting Builders

> Auto-generated documentation for [particula.data.loader_setting_builders](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py) module.

## DataChecksBuilder

[Show source in loader_setting_builders.py:99](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L99)

Builder class for constructing the data checks dictionary.

#### Signature

```python
class DataChecksBuilder(
    BuilderABC,
    ChecksCharactersMixin,
    ChecksCharCountsMixin,
    ChecksSkipRowsMixin,
    ChecksSkipEndMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../next/abc_builder.md#builderabc)
- [ChecksCharCountsMixin](./mixin.md#checkscharcountsmixin)
- [ChecksCharactersMixin](./mixin.md#checkscharactersmixin)
- [ChecksSkipEndMixin](./mixin.md#checksskipendmixin)
- [ChecksSkipRowsMixin](./mixin.md#checksskiprowsmixin)

### DataChecksBuilder().build

[Show source in loader_setting_builders.py:121](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L121)

Build and return the data checks dictionary.

#### Signature

```python
def build(self) -> Dict[str, Any]: ...
```



## LoaderSetting1DBuilder

[Show source in loader_setting_builders.py:32](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L32)

Builder class for creating settings for loading and checking 1D data
from CSV files.

#### Signature

```python
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
    def __init__(self): ...
```

#### See also

- [BuilderABC](../next/abc_builder.md#builderabc)
- [DataChecksMixin](./mixin.md#datachecksmixin)
- [DataColumnMixin](./mixin.md#datacolumnmixin)
- [DataHeaderMixin](./mixin.md#dataheadermixin)
- [DelimiterMixin](./mixin.md#delimitermixin)
- [FileMinSizeBytesMixin](./mixin.md#fileminsizebytesmixin)
- [FilenameRegexMixin](./mixin.md#filenameregexmixin)
- [HeaderRowMixin](./mixin.md#headerrowmixin)
- [RelativeFolderMixin](./mixin.md#relativefoldermixin)
- [TimeColumnMixin](./mixin.md#timecolumnmixin)
- [TimeFormatMixin](./mixin.md#timeformatmixin)
- [TimeShiftSecondsMixin](./mixin.md#timeshiftsecondsmixin)
- [TimezoneIdentifierMixin](./mixin.md#timezoneidentifiermixin)

### LoaderSetting1DBuilder().build

[Show source in loader_setting_builders.py:79](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L79)

Build and return the settings dictionary for 1D data loading.

#### Signature

```python
def build(self) -> Dict[str, Any]: ...
```



## LoaderSettingSizerBuilder

[Show source in loader_setting_builders.py:159](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L159)

Builder class for creating settings for loading and checking sizer
1D and 2D data from CSV files.

#### Signature

```python
class LoaderSettingSizerBuilder(
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
    SizerDataReaderMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../next/abc_builder.md#builderabc)
- [DataChecksMixin](./mixin.md#datachecksmixin)
- [DataColumnMixin](./mixin.md#datacolumnmixin)
- [DataHeaderMixin](./mixin.md#dataheadermixin)
- [DelimiterMixin](./mixin.md#delimitermixin)
- [FileMinSizeBytesMixin](./mixin.md#fileminsizebytesmixin)
- [FilenameRegexMixin](./mixin.md#filenameregexmixin)
- [HeaderRowMixin](./mixin.md#headerrowmixin)
- [RelativeFolderMixin](./mixin.md#relativefoldermixin)
- [SizerDataReaderMixin](./mixin.md#sizerdatareadermixin)
- [TimeColumnMixin](./mixin.md#timecolumnmixin)
- [TimeFormatMixin](./mixin.md#timeformatmixin)
- [TimeShiftSecondsMixin](./mixin.md#timeshiftsecondsmixin)
- [TimezoneIdentifierMixin](./mixin.md#timezoneidentifiermixin)

### LoaderSettingSizerBuilder().build

[Show source in loader_setting_builders.py:212](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L212)

Build and return the two dictionaries for 1D and 2D sizer data
loading .

#### Signature

```python
def build(self) -> Tuple[Dict[str, Any], Dict[str, Any]]: ...
```



## SizerDataReaderBuilder

[Show source in loader_setting_builders.py:131](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L131)

Builder class for constructing the sizer data reader dictionary.

#### Signature

```python
class SizerDataReaderBuilder(
    BuilderABC,
    SizerConcentrationConvertFromMixin,
    SizerStartKeywordMixin,
    SizerEndKeywordMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../next/abc_builder.md#builderabc)
- [SizerConcentrationConvertFromMixin](./mixin.md#sizerconcentrationconvertfrommixin)
- [SizerEndKeywordMixin](./mixin.md#sizerendkeywordmixin)
- [SizerStartKeywordMixin](./mixin.md#sizerstartkeywordmixin)

### SizerDataReaderBuilder().build

[Show source in loader_setting_builders.py:149](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L149)

Build and return the sizer data reader dictionary.

#### Signature

```python
def build(self) -> Dict[str, Any]: ...
```
