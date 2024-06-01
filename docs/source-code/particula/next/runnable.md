# Runnable

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Next](./index.md#next) / Runnable

> Auto-generated documentation for [particula.next.runnable](../../../../particula/next/runnable.py) module.

## Runnable

[Show source in runnable.py:10](../../../../particula/next/runnable.py#L10)

Runnable process that can modify an aerosol instance.

Parameters: None

#### Methods

- `-` *rate* - Return the rate of the process.
- `-` *execute* - Execute the process and modify the aerosol instance.
- `-` *__or__* - Chain this process with another process using the | operator.

#### Signature

```python
class Runnable(ABC): ...
```

### Runnable().__or__

[Show source in runnable.py:36](../../../../particula/next/runnable.py#L36)

Chain this process with another process using the | operator.

#### Signature

```python
def __or__(self, other: "Runnable"): ...
```

### Runnable().execute

[Show source in runnable.py:28](../../../../particula/next/runnable.py#L28)

Execute the process and modify the aerosol instance.

#### Arguments

- aerosol (Aerosol): The aerosol instance to modify.
- time_step (float): The time step for the process in seconds.

#### Signature

```python
@abstractmethod
def execute(self, aerosol: Aerosol, time_step: float) -> Aerosol: ...
```

#### See also

- [Aerosol](./aerosol.md#aerosol)

### Runnable().rate

[Show source in runnable.py:21](../../../../particula/next/runnable.py#L21)

Return the rate of the process.

#### Arguments

- aerosol (Aerosol): The aerosol instance to modify.

#### Signature

```python
@abstractmethod
def rate(self, aerosol: Aerosol) -> Any: ...
```

#### See also

- [Aerosol](./aerosol.md#aerosol)



## RunnableSequence

[Show source in runnable.py:45](../../../../particula/next/runnable.py#L45)

A sequence of processes to be executed in order.

#### Attributes

- processes (List[Runnable]): A list of RunnableProcess objects.

#### Methods

- `-` *add_process* - Add a process to the sequence.
- `-` *execute* - Execute the sequence of processes on an aerosol instance.
- `-` *__or__* - Add a process to the sequence using the | operator.

#### Signature

```python
class RunnableSequence:
    def __init__(self): ...
```

### RunnableSequence().__or__

[Show source in runnable.py:70](../../../../particula/next/runnable.py#L70)

Add a runnable to the sequence using the | operator.

#### Signature

```python
def __or__(self, process: Runnable): ...
```

#### See also

- [Runnable](#runnable)

### RunnableSequence().add_process

[Show source in runnable.py:59](../../../../particula/next/runnable.py#L59)

Add a process to the sequence.

#### Signature

```python
def add_process(self, process: Runnable): ...
```

#### See also

- [Runnable](#runnable)

### RunnableSequence().execute

[Show source in runnable.py:63](../../../../particula/next/runnable.py#L63)

Execute the sequence of runnables on an aerosol instance.

#### Signature

```python
def execute(self, aerosol: Aerosol, time_step: float) -> Aerosol: ...
```

#### See also

- [Aerosol](./aerosol.md#aerosol)