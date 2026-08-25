# modurl_logger

`modurl_logger` is a Rust library crate for aggregating ModuRL scalar metrics into terminal graphs and TensorBoard event files.

> [!WARNING]
> ModuRL is early in development. Public APIs may change between revisions.

## Add the dependencies

```toml
[dependencies]
candle-core = "0.11.0"
modurl_logger = "0.1.0"
```

## Example

Log two loss values and display them as a terminal graph:

```rust
use candle_core::{Device, Tensor};
use modurl_logger::{Error, Logger, TerminalLogger};

fn main() -> Result<(), Error> {
    let device = Device::Cpu;
    let first_loss = Tensor::new(0.8_f32, &device)?;
    let second_loss = Tensor::new(0.3_f32, &device)?;
    let mut logger = TerminalLogger::default();

    logger.log(100, &[("loss", &first_loss)])?;
    logger.log(200, &[("loss", &second_loss)])?;
    logger.display();
    Ok(())
}
```

`Logger::log` accepts named numeric tensors shaped `[]`. A larger timestep completes the preceding timestep. Call `finish` or `display` after the last entry so the logger processes the final timestep.

## Backends

- `TerminalLogger` stores completed series and plots them in the terminal. Use `with_live_updates` to redraw an interactive terminal during training.
- `TensorBoardLogger` writes TensorBoard event files to a selected directory. Call `finish` to flush the final event and report filesystem errors.

Open TensorBoard event files with:

```console
tensorboard --logdir path/to/runs
```

## Aggregation

Repeated values at one timestep can use mean, sum, minimum, maximum, or last-value reduction. `AggregationConfig` sets a default and supports overrides by metric name. Each reduction can also apply a rolling average across completed timesteps.

## Documentation

Build the API documentation locally:

```console
cargo doc -p modurl_logger --no-deps --open
```

See the [workspace examples](../examples/examples) for integration with ModuRL training agents.

## License

`modurl_logger` is available under the [MIT License](LICENSE).
