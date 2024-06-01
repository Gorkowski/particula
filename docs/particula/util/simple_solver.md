# SimpleSolver

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / SimpleSolver

> Auto-generated documentation for [particula.util.simple_solver](../../../particula/util/simple_solver.py) module.

- [SimpleSolver](#simplesolver)
  - [SimpleSolver](#simplesolver-1)
    - [SimpleSolver().prep_inputs](#simplesolver()prep_inputs)
    - [SimpleSolver().solution](#simplesolver()solution)
  - [ode_func](#ode_func)

## SimpleSolver

[Show source in simple_solver.py:33](../../../particula/util/simple_solver.py#L33)

a class to solve the ODE:

Need:
1. initial distribution
2. associated radius
3. associated coagulation kernel

Also:
1. desired time span in seconds (given unitless)

#### Signature

```python
class SimpleSolver:
    def __init__(self, **kwargs): ...
```

### SimpleSolver().prep_inputs

[Show source in simple_solver.py:66](../../../particula/util/simple_solver.py#L66)

strip units, etc.

#### Signature

```python
def prep_inputs(self): ...
```

### SimpleSolver().solution

[Show source in simple_solver.py:77](../../../particula/util/simple_solver.py#L77)

utilize scipy.integrate.odeint

#### Signature

```python
def solution(self): ...
```



## ode_func

[Show source in simple_solver.py:21](../../../particula/util/simple_solver.py#L21)

function to integrate

#### Signature

```python
def ode_func(_nums, _, _rads, _coag): ...
```