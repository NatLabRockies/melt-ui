# melt-ui

## Installation

Install Pixi, then create the locked MELT-UI environment:

```bash
pixi install --locked
```

## Running

```bash
pixi run start
```

## Development checks

```bash
pixi install -e dev --locked
pixi run -e dev lint
pixi run -e dev format-check
pixi run -e dev compile
pixi run -e dev precommit
```

NLR Software Record: SWR 26-071
