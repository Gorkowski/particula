# Stream

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Stream

> Auto-generated documentation for [particula.data.stream](https://github.com/Gorkowski/particula/blob/main/particula/data/stream.py) module.

## Stream

[Show source in stream.py:11](https://github.com/Gorkowski/particula/blob/main/particula/data/stream.py#L11)

#### Attributes

- `header`: `List[str]` - Initialize other fields as empty arrays: field(default_factory=list)


A class for consistent data storage and format.

#### Attributes

---------
header : List[str]
    A list of strings representing the header of the data stream.
data : np.ndarray
    A numpy array representing the data stream. The first dimension
    represents time and the second dimension represents the header.
time : np.ndarray
    A numpy array representing the time stream.
files : List[str]
    A list of strings representing the files containing the data stream.

#### Methods

-------
validate_inputs
    Validates the inputs to the Stream class.
datetime64 -> np.ndarray
    Returns an array of datetime64 objects representing the time stream.
    Useful for plotting, with matplotlib.dates.
return_header_dict -> dict
    Returns the header as a dictionary with keys as header elements and
    values as their indices.

#### Signature

```python
class Stream: ...
```

### Stream().__getitem__

[Show source in stream.py:57](https://github.com/Gorkowski/particula/blob/main/particula/data/stream.py#L57)

Allows for indexing of the data stream.

#### Arguments

----------
index : int or str
    The index of the data stream to return.

#### Returns

-------
np.ndarray
    The data stream at the specified index.

#### Signature

```python
def __getitem__(self, index: Union[int, str]): ...
```

### Stream().__len__

[Show source in stream.py:86](https://github.com/Gorkowski/particula/blob/main/particula/data/stream.py#L86)

Returns the length of the time stream.

#### Signature

```python
def __len__(self): ...
```

### Stream().__setitem__

[Show source in stream.py:71](https://github.com/Gorkowski/particula/blob/main/particula/data/stream.py#L71)

Allows for setting or adding of a row of data in the stream.

#### Arguments

index : The index of the data stream to set.
value : The data to set at the specified index.

future work maybe add a list option and iterate through the list

#### Signature

```python
def __setitem__(self, index: Union[int, str], value): ...
```

### Stream().datetime64

[Show source in stream.py:90](https://github.com/Gorkowski/particula/blob/main/particula/data/stream.py#L90)

Returns an array of datetime64 objects representing the time stream.
Useful for plotting, with matplotlib.dates.

#### Signature

```python
@property
def datetime64(self) -> np.ndarray: ...
```

### Stream().header_dict

[Show source in stream.py:98](https://github.com/Gorkowski/particula/blob/main/particula/data/stream.py#L98)

Returns the header as a dictionary with index (0, 1) as the keys
and the names as values.

#### Signature

```python
@property
def header_dict(self) -> dict: ...
```

### Stream().header_float

[Show source in stream.py:104](https://github.com/Gorkowski/particula/blob/main/particula/data/stream.py#L104)

Returns the header as a numpy array of floats.

#### Signature

```python
@property
def header_float(self) -> np.ndarray: ...
```

### Stream().validate_inputs

[Show source in stream.py:47](https://github.com/Gorkowski/particula/blob/main/particula/data/stream.py#L47)

Validates the inputs for the DataStream object.

#### Raises

    - `TypeError` - If header is not a list.
# this might be why I can't call Stream without inputs

#### Signature

```python
def validate_inputs(self): ...
```



## StreamAveraged

[Show source in stream.py:111](https://github.com/Gorkowski/particula/blob/main/particula/data/stream.py#L111)

A subclass of Stream with additional parameters related to averaging.

#### Attributes

- `average_interval` *float* - The size of the window used for averaging.
- `start_time` *float* - The start time for averaging.
- `stop_time` *float* - The stop time for averaging.
- `standard_deviation` *float* - The standard deviation of the data.

#### Signature

```python
class StreamAveraged(Stream): ...
```

#### See also

- [Stream](#stream)

### StreamAveraged().get_std

[Show source in stream.py:149](https://github.com/Gorkowski/particula/blob/main/particula/data/stream.py#L149)

Returns the standard deviation of the data.

#### Signature

```python
def get_std(self, index) -> np.ndarray: ...
```

### StreamAveraged().validate_averaging_params

[Show source in stream.py:131](https://github.com/Gorkowski/particula/blob/main/particula/data/stream.py#L131)

Validates the averaging parameters for the stream.

#### Raises

- `ValueError` - If average_window is not a positive number or if
start_time and stop_time are not numbers or if start_time is
greater than or equal to stop_time.

#### Signature

```python
def validate_averaging_params(self): ...
```
