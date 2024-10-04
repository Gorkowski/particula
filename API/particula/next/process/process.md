# Process

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Next](./index.md#next) / Process

> Auto-generated documentation for [particula.next.process](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py) module.

## MassCoagulation

[Show source in process.py:53](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py#L53)

MOCK-UP Runnable process that modifies an aerosol instance by
mass coagulation.

#### Signature

```python
class MassCoagulation(RunnableProcess):
    def __init__(self, other_setting2: Any): ...
```

#### See also

- [RunnableProcess](#runnableprocess)

### MassCoagulation().execute

[Show source in process.py:60](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py#L60)

#### Signature

```python
def execute(self, aerosol: Aerosol) -> Aerosol: ...
```

#### See also

- [Aerosol](./aerosol.md#aerosol)

### MassCoagulation().rate

[Show source in process.py:66](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py#L66)

#### Signature

```python
def rate(self, aerosol: Aerosol) -> float: ...
```

#### See also

- [Aerosol](./aerosol.md#aerosol)



## MassCondensation

[Show source in process.py:37](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py#L37)

MOCK-UP: Runnable process that modifies an aerosol instance by
mass condensation.

#### Signature

```python
class MassCondensation(RunnableProcess):
    def __init__(self, other_settings: Any): ...
```

#### See also

- [RunnableProcess](#runnableprocess)

### MassCondensation().execute

[Show source in process.py:43](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py#L43)

#### Signature

```python
def execute(self, aerosol: Aerosol) -> Aerosol: ...
```

#### See also

- [Aerosol](./aerosol.md#aerosol)

### MassCondensation().rate

[Show source in process.py:49](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py#L49)

#### Signature

```python
def rate(self, aerosol: Aerosol) -> float: ...
```

#### See also

- [Aerosol](./aerosol.md#aerosol)



## ProcessSequence

[Show source in process.py:70](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py#L70)

A sequence of processes to be executed in order.

#### Attributes

- processes (List[RunnableProcess]): A list of RunnableProcess objects.

#### Methods

- `-` *add_process* - Add a process to the sequence.
- `-` *execute* - Execute the sequence of processes on an aerosol instance.
- `-` *__or__* - Add a process to the sequence using the | operator.

#### Signature

```python
class ProcessSequence:
    def __init__(self): ...
```

### ProcessSequence().__or__

[Show source in process.py:95](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py#L95)

Add a process to the sequence using the | operator.

#### Signature

```python
def __or__(self, process: RunnableProcess): ...
```

#### See also

- [RunnableProcess](#runnableprocess)

### ProcessSequence().add_process

[Show source in process.py:84](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py#L84)

Add a process to the sequence.

#### Signature

```python
def add_process(self, process: RunnableProcess): ...
```

#### See also

- [RunnableProcess](#runnableprocess)

### ProcessSequence().execute

[Show source in process.py:88](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py#L88)

Execute the sequence of processes on an aerosol instance.

#### Signature

```python
def execute(self, aerosol: Aerosol) -> Aerosol: ...
```

#### See also

- [Aerosol](./aerosol.md#aerosol)



## RunnableProcess

[Show source in process.py:9](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py#L9)

Runnable process that can modify an aerosol instance.

#### Signature

```python
class RunnableProcess(ABC): ...
```

### RunnableProcess().__or__

[Show source in process.py:26](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py#L26)

Chain this process with another process using the | operator.

#### Signature

```python
def __or__(self, other): ...
```

### RunnableProcess().execute

[Show source in process.py:12](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py#L12)

Execute the process and modify the aerosol instance.

#### Arguments

- aerosol (Aerosol): The aerosol instance to modify.

#### Signature

```python
@abstractmethod
def execute(self, aerosol: Aerosol) -> Aerosol: ...
```

#### See also

- [Aerosol](./aerosol.md#aerosol)

### RunnableProcess().rate

[Show source in process.py:19](https://github.com/Gorkowski/particula/blob/main/particula/next/process.py#L19)

Return the rate of the process.

#### Arguments

- aerosol (Aerosol): The aerosol instance to modify.

#### Signature

```python
@abstractmethod
def rate(self, aerosol: Aerosol) -> float: ...
```

#### See also

- [Aerosol](./aerosol.md#aerosol)
