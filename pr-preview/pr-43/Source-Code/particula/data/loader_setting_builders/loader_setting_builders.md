# Loader Setting Builders

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Loader Setting Builders

> Auto-generated documentation for [particula.data.loader_setting_builders](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py) module.

## DataChecksBuilder

[Show source in loader_setting_builders.py:78](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L78)

Builder class for constructing the data checks dictionary.

#### Signature

```python
class DataChecksBuilder(
    BuilderABC,
    mixin.ChecksCharactersMixin,
    mixin.ChecksCharCountsMixin,
    mixin.ChecksSkipRowsMixin,
    mixin.ChecksSkipEndMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../next/abc_builder.md#builderabc)

### DataChecksBuilder().build

[Show source in loader_setting_builders.py:100](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L100)

Build and return the data checks dictionary.

#### Signature

```python
def build(self) -> Dict[str, Any]: ...
```



## LoaderSetting1DBuilder

[Show source in loader_setting_builders.py:11](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L11)

Builder class for creating settings for loading and checking 1D data
from CSV files.

#### Signature

```python
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
    def __init__(self): ...
```

#### See also

- [BuilderABC](../next/abc_builder.md#builderabc)

### LoaderSetting1DBuilder().build

[Show source in loader_setting_builders.py:58](https://github.com/Gorkowski/particula/blob/main/particula/data/loader_setting_builders.py#L58)

Build and return the settings dictionary for 1D data loading.

#### Signature

```python
def build(self) -> Dict[str, Any]: ...
```
