# Partial Pressure Function

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Partial Pressure Function

> Auto-generated documentation for [particula.next.gas.properties.partial_pressure_function](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/properties/partial_pressure_function.py) module.

## calculate_partial_pressure

[Show source in partial_pressure_function.py:10](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/properties/partial_pressure_function.py#L10)

Calculate the partial pressure of a gas from its concentration, molar mass,
and temperature.

#### Arguments

- concentration (float): Concentration of the gas in kg/m^3.
- molar_mass (float): Molar mass of the gas in kg/mol.
- temperature (float): Temperature in Kelvin.

#### Returns

- `-` *float* - Partial pressure of the gas in Pascals (Pa).

#### Signature

```python
def calculate_partial_pressure(
    concentration: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```
