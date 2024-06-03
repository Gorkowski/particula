# Vapor Pressure Factories

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Vapor Pressure Factories

> Auto-generated documentation for [particula.next.gas.vapor_pressure_factories](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_factories.py) module.

## vapor_pressure_factory

[Show source in vapor_pressure_factories.py:9](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_factories.py#L9)

Factory method to create a concrete VaporPressureStrategy object using
builders.

#### Arguments

----
- strategy (str): The strategy to use for vapor pressure calculations.
  - `Options` - "constant", "antoine", "clausius_clapeyron", "water_buck".
- parameters (dict): A dictionary containing the necessary parameters for
    the strategy. If no parameters are needed, this can be left as None.

#### Returns

-------
- vapor_pressure_strategy (VaporPressureStrategy): A concrete
  implementation of the VaporPressureStrategy built using the appropriate
  builder.

#### Signature

```python
def vapor_pressure_factory(
    strategy: str, parameters: ignore = None
) -> VaporPressureStrategy: ...
```

#### See also

- [VaporPressureStrategy](./vapor_pressure_strategies.md#vaporpressurestrategy)
