# Loader Setting Builders

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Loader Setting Builders

> Auto-generated documentation for [particula.data.loader_setting_builders](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py) module.

## DataChecksBuilder

[Show source in loader_setting_builders.py:95](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L95)

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

[Show source in loader_setting_builders.py:117](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L117)

Build and return the data checks dictionary.

#### Signature

```python
def build(self) -> Dict[str, Any]: ...
```



## LoaderSetting1DBuilder

[Show source in loader_setting_builders.py:28](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L28)

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

[Show source in loader_setting_builders.py:75](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L75)

Build and return the settings dictionary for 1D data loading.

#### Signature

```python
def build(self) -> Dict[str, Any]: ...
```
