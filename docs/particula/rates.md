# Rates

[Particula Index](../README.md#particula-index) / [Particula](./index.md#particula) / Rates

> Auto-generated documentation for [particula.rates](../../particula/rates.py) module.

- [Rates](#rates)
  - [Rates](#rates-1)
    - [Rates()._coag_loss_gain](#rates()_coag_loss_gain)
    - [Rates().coagulation_gain](#rates()coagulation_gain)
    - [Rates().coagulation_loss](#rates()coagulation_loss)
    - [Rates().coagulation_rate](#rates()coagulation_rate)
    - [Rates().condensation_growth_rate](#rates()condensation_growth_rate)
    - [Rates().condensation_growth_speed](#rates()condensation_growth_speed)
    - [Rates().dilution_rate](#rates()dilution_rate)
    - [Rates().nucleation_rate](#rates()nucleation_rate)
    - [Rates().sum_rates](#rates()sum_rates)
    - [Rates().wall_loss_rate](#rates()wall_loss_rate)

## Rates

[Show source in rates.py:12](../../particula/rates.py#L12)

The class to calculate the rates

#### Signature

```python
class Rates:
    def __init__(self, particle=None, lazy=True): ...
```

### Rates()._coag_loss_gain

[Show source in rates.py:42](../../particula/rates.py#L42)

get both loss and gain

#### Signature

```python
def _coag_loss_gain(self): ...
```

### Rates().coagulation_gain

[Show source in rates.py:57](../../particula/rates.py#L57)

get coagulation gain rate

#### Signature

```python
def coagulation_gain(self): ...
```

### Rates().coagulation_loss

[Show source in rates.py:51](../../particula/rates.py#L51)

get the coagulation loss rate

#### Signature

```python
def coagulation_loss(self): ...
```

### Rates().coagulation_rate

[Show source in rates.py:63](../../particula/rates.py#L63)

get the coagulation rate by summing the loss and gain rates

#### Signature

```python
def coagulation_rate(self): ...
```

### Rates().condensation_growth_rate

[Show source in rates.py:75](../../particula/rates.py#L75)

condensation rate

#### Signature

```python
def condensation_growth_rate(self): ...
```

### Rates().condensation_growth_speed

[Show source in rates.py:69](../../particula/rates.py#L69)

condensation speed

#### Signature

```python
def condensation_growth_speed(self): ...
```

### Rates().dilution_rate

[Show source in rates.py:95](../../particula/rates.py#L95)

dilution rate

#### Signature

```python
def dilution_rate(self): ...
```

### Rates().nucleation_rate

[Show source in rates.py:86](../../particula/rates.py#L86)

nucleation rate

#### Signature

```python
def nucleation_rate(self): ...
```

### Rates().sum_rates

[Show source in rates.py:111](../../particula/rates.py#L111)

Sum rates, with options to disable individual rate terms.

#### Arguments

----------
coagulation : bool, optional
    does the coagulation calcuation, by default True
condensation : bool, optional
    does the condensation calculation, by default True
nucleation : bool, optional
    does the nucleation calculation, by default True
dilution : bool, optional
    does the dilution calculation, by default False
wall_loss : bool, optional
    does the wall loss calculation, by default False

#### Signature

```python
def sum_rates(
    self,
    coagulation=True,
    condensation=True,
    nucleation=True,
    dilution=False,
    wall_loss=False,
): ...
```

### Rates().wall_loss_rate

[Show source in rates.py:103](../../particula/rates.py#L103)

wall loss rate

#### Signature

```python
def wall_loss_rate(self): ...
```