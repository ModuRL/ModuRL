use candle_core::{Device, Tensor};
use modurl_logger::{Aggregation, AggregationConfig, Error, Logger, TerminalLogger};

fn scalar(value: f32) -> Tensor {
    Tensor::new(value, &Device::Cpu).unwrap()
}

#[test]
fn reduces_repeated_values_and_uses_partial_rolling_windows() {
    let config = AggregationConfig::new(Aggregation::mean().with_rolling_window(2));
    let mut logger = TerminalLogger::new(config);

    let one = scalar(1.0);
    let three = scalar(3.0);
    let five = scalar(5.0);
    logger.log(10, &[("loss", &one)]).unwrap();
    logger.log(10, &[("loss", &three)]).unwrap();
    logger.log(20, &[("loss", &five)]).unwrap();
    logger.finish().unwrap();

    assert_eq!(logger.series("loss"), Some(&[(10, 2.0), (20, 3.5)][..]));
}

#[test]
fn applies_name_based_reduction_overrides() {
    let config = AggregationConfig::new(Aggregation::mean())
        .with_override("sum", Aggregation::sum())
        .with_override("min", Aggregation::min())
        .with_override("max", Aggregation::max())
        .with_override("last", Aggregation::last());
    let mut logger = TerminalLogger::new(config);
    let one = scalar(1.0);
    let three = scalar(3.0);

    logger
        .log(
            1,
            &[
                ("mean", &one),
                ("sum", &one),
                ("min", &one),
                ("max", &one),
                ("last", &one),
            ],
        )
        .unwrap();
    logger
        .log(
            1,
            &[
                ("mean", &three),
                ("sum", &three),
                ("min", &three),
                ("max", &three),
                ("last", &three),
            ],
        )
        .unwrap();
    logger.finish().unwrap();

    assert_eq!(logger.series("mean"), Some(&[(1, 2.0)][..]));
    assert_eq!(logger.series("sum"), Some(&[(1, 4.0)][..]));
    assert_eq!(logger.series("min"), Some(&[(1, 1.0)][..]));
    assert_eq!(logger.series("max"), Some(&[(1, 3.0)][..]));
    assert_eq!(logger.series("last"), Some(&[(1, 3.0)][..]));
}

#[test]
fn converts_numeric_scalar_tensors_to_f32() {
    let mut logger = TerminalLogger::default();
    let value = Tensor::new(7_i64, &Device::Cpu).unwrap();

    logger.log(4, &[("integer", &value)]).unwrap();
    logger.finish().unwrap();

    assert_eq!(logger.series("integer"), Some(&[(4, 7.0)][..]));
}

#[test]
fn rejects_non_scalar_tensors_without_advancing_the_timestep() {
    let mut logger = TerminalLogger::default();
    let vector = Tensor::new(&[1.0_f32, 2.0], &Device::Cpu).unwrap();
    let error = logger.log(10, &[("loss", &vector)]).unwrap_err();

    assert!(matches!(
        error,
        Error::NonScalar { metric, shape }
            if metric == "loss" && shape == [2]
    ));

    let value = scalar(3.0);
    logger.log(5, &[("loss", &value)]).unwrap();
    logger.finish().unwrap();
    assert_eq!(logger.series("loss"), Some(&[(5, 3.0)][..]));
}

#[test]
fn rejects_out_of_order_timesteps() {
    let mut logger = TerminalLogger::default();
    let value = scalar(1.0);
    logger.log(10, &[("loss", &value)]).unwrap();

    let error = logger.log(9, &[("loss", &value)]).unwrap_err();
    assert!(matches!(
        error,
        Error::OutOfOrderTimestep {
            current: 10,
            received: 9
        }
    ));
}

#[test]
fn terminal_display_handles_a_single_point() {
    let mut logger = TerminalLogger::default();
    let value = scalar(1.0);
    logger.log(10, &[("loss", &value)]).unwrap();

    logger.display();

    assert_eq!(logger.series("loss"), Some(&[(10, 1.0)][..]));
}

#[test]
fn live_updates_preserve_completed_series() {
    let mut logger = TerminalLogger::default().with_live_updates();
    let one = scalar(1.0);
    let two = scalar(2.0);
    logger.log(10, &[("loss", &one)]).unwrap();
    logger.log(20, &[("loss", &two)]).unwrap();
    logger.finish().unwrap();

    assert_eq!(logger.series("loss"), Some(&[(10, 1.0), (20, 2.0)][..]));
}
