# Lake

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Lake

> Auto-generated documentation for [particula.data.lake](../../../particula/data/lake.py) module.

- [Lake](#lake)
  - [Lake](#lake-1)
    - [Lake().__delitem__](#lake()__delitem__)
    - [Lake().__dir__](#lake()__dir__)
    - [Lake().__getattr__](#lake()__getattr__)
    - [Lake().__getitem__](#lake()__getitem__)
    - [Lake().__iter__](#lake()__iter__)
    - [Lake().__len__](#lake()__len__)
    - [Lake().__repr__](#lake()__repr__)
    - [Lake().__setitem__](#lake()__setitem__)
    - [Lake().add_stream](#lake()add_stream)
    - [Lake().items](#lake()items)
    - [Lake().keys](#lake()keys)
    - [Lake().summary](#lake()summary)
    - [Lake().values](#lake()values)

## Lake

[Show source in lake.py:10](../../../particula/data/lake.py#L10)

A class representing a lake which is a collection of streams.

#### Attributes

streams (Dict[str, Stream]): A dictionary to hold streams with their
names as keys.

#### Signature

```python
class Lake: ...
```

### Lake().__delitem__

[Show source in lake.py:91](../../../particula/data/lake.py#L91)

Remove a stream by name.
Example: del lake['stream_name']

#### Signature

```python
def __delitem__(self, key: str) -> None: ...
```

### Lake().__dir__

[Show source in lake.py:50](../../../particula/data/lake.py#L50)

List available streams.
Example: dir(lake)

#### Signature

```python
def __dir__(self) -> list: ...
```

### Lake().__getattr__

[Show source in lake.py:39](../../../particula/data/lake.py#L39)

Allow accessing streams as an attributes.

#### Raises

    - `AttributeError` - If the stream name is not in the lake.
- `Example` - lake.stream_name

#### Signature

```python
def __getattr__(self, name: str) -> Any: ...
```

### Lake().__getitem__

[Show source in lake.py:78](../../../particula/data/lake.py#L78)

Get a stream by name.
Example: lake['stream_name']

#### Signature

```python
def __getitem__(self, key: str) -> Any: ...
```

### Lake().__iter__

[Show source in lake.py:55](../../../particula/data/lake.py#L55)

Iterate over the streams in the lake.
Example: [stream.header for stream in lake]""

#### Signature

```python
def __iter__(self) -> Iterator[Any]: ...
```

### Lake().__len__

[Show source in lake.py:73](../../../particula/data/lake.py#L73)

Return the number of streams in the lake.
Example: len(lake)

#### Signature

```python
def __len__(self) -> int: ...
```

### Lake().__repr__

[Show source in lake.py:99](../../../particula/data/lake.py#L99)

Return a string representation of the lake.
Example: print(lake)

#### Signature

```python
def __repr__(self) -> str: ...
```

### Lake().__setitem__

[Show source in lake.py:83](../../../particula/data/lake.py#L83)

Set a stream by name.
Example: lake['stream_name'] = new_stream

#### Signature

```python
def __setitem__(self, key: str, value: Stream) -> None: ...
```

#### See also

- [Stream](./stream.md#stream)

### Lake().add_stream

[Show source in lake.py:19](../../../particula/data/lake.py#L19)

Add a stream to the lake.

#### Arguments

-----------
    - `stream` *Stream* - The stream object to be added.
    - `name` *str* - The name of the stream.

#### Raises

-------
    - `ValueError` - If the stream name is already in use or not a valid
    identifier.

#### Signature

```python
def add_stream(self, stream: Stream, name: str) -> None: ...
```

#### See also

- [Stream](./stream.md#stream)

### Lake().items

[Show source in lake.py:61](../../../particula/data/lake.py#L61)

Return an iterator over the key-value pairs.

#### Signature

```python
def items(self) -> Iterator[Tuple[Any, Any]]: ...
```

### Lake().keys

[Show source in lake.py:69](../../../particula/data/lake.py#L69)

Return an iterator over the keys.

#### Signature

```python
def keys(self) -> Iterator[Any]: ...
```

### Lake().summary

[Show source in lake.py:104](../../../particula/data/lake.py#L104)

    Return a string summary iterating over each stream
    and print Stream.header.
Example: lake.summary

#### Signature

```python
@property
def summary(self) -> None: ...
```

### Lake().values

[Show source in lake.py:65](../../../particula/data/lake.py#L65)

Return an iterator over the values.

#### Signature

```python
def values(self) -> Iterator[Any]: ...
```