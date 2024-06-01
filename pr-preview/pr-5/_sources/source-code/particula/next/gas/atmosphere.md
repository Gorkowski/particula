# Atmosphere

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Atmosphere

> Auto-generated documentation for [particula.next.gas.atmosphere](../../../../../particula/next/gas/atmosphere.py) module.

## Atmosphere

[Show source in atmosphere.py:8](../../../../../particula/next/gas/atmosphere.py#L8)

Represents a mixture of gas species, detailing properties such as
temperature, total pressure, and the list of gas species in the mixture.

#### Attributes

- temperature (float): The temperature of the gas mixture in Kelvin.
- total_pressure (float): The total pressure of the gas mixture in Pascals.
- species (List[GasSpecies]): A list of GasSpecies objects representing the
    species in the gas mixture.

#### Methods

- `-` *add_species* - Adds a gas species to the mixture.
- `-` *remove_species* - Removes a gas species from the mixture by index.

#### Signature

```python
class Atmosphere: ...
```

### Atmosphere().__getitem__

[Show source in atmosphere.py:55](../../../../../particula/next/gas/atmosphere.py#L55)

Returns the gas species at the given index.

#### Signature

```python
def __getitem__(self, index: int) -> GasSpecies: ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)

### Atmosphere().__iter__

[Show source in atmosphere.py:51](../../../../../particula/next/gas/atmosphere.py#L51)

Allows iteration over the species in the gas mixture.

#### Signature

```python
def __iter__(self): ...
```

### Atmosphere().__len__

[Show source in atmosphere.py:59](../../../../../particula/next/gas/atmosphere.py#L59)

Returns the number of species in the gas mixture.

#### Signature

```python
def __len__(self): ...
```

### Atmosphere().__str__

[Show source in atmosphere.py:63](../../../../../particula/next/gas/atmosphere.py#L63)

Returns a string representation of the Gas object.

#### Signature

```python
def __str__(self): ...
```

### Atmosphere().add_species

[Show source in atmosphere.py:28](../../../../../particula/next/gas/atmosphere.py#L28)

Adds a gas species to the mixture.

#### Arguments

- gas_species (GasSpecies): The GasSpecies object to be added to the
mixture.

#### Signature

```python
def add_species(self, gas_species: GasSpecies) -> None: ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)

### Atmosphere().remove_species

[Show source in atmosphere.py:38](../../../../../particula/next/gas/atmosphere.py#L38)

Removes a gas species from the mixture by index.

#### Arguments

- index (int): The index of the gas species to be removed from the
list.

#### Signature

```python
def remove_species(self, index: int) -> None: ...
```
