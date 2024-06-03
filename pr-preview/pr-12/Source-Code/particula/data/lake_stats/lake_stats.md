# Lake Stats

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Lake Stats

> Auto-generated documentation for [particula.data.lake_stats](https://github.com/Gorkowski/particula/blob/main/particula/data/lake_stats.py) module.

## average_std

[Show source in lake_stats.py:10](https://github.com/Gorkowski/particula/blob/main/particula/data/lake_stats.py#L10)

"
Averages the data in a lake over a specified time interval.

#### Signature

```python
def average_std(
    lake: Lake,
    average_interval: Union[float, int] = 60,
    new_time_array: Optional[np.ndarray] = None,
    clone: bool = True,
) -> Lake: ...
```

#### See also

- [Lake](./lake.md#lake)
