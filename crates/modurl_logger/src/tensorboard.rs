use std::{
    error, fmt,
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, ErrorKind, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::Tensor;
use crc::{CRC_32_ISCSI, Crc};
use prost::{Message, Oneof};

use crate::{
    AggregationConfig, Error as MetricError, Logger,
    aggregation::{Aggregator, CompletedTimestep},
};

const FILE_VERSION: &str = "brain.Event:2";
const CRC_MASK_DELTA: u32 = 0xa282_ead8;
const CRC32C: Crc<u32> = Crc::<u32>::new(&CRC_32_ISCSI);
static FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// An error produced by a [`TensorBoardLogger`].
#[derive(Debug)]
pub enum TensorBoardError {
    /// A metric could not be accepted or aggregated.
    Metric(MetricError),
    /// A timestep could not be represented by TensorBoard's signed step type.
    TimestepOutOfRange { timestep: usize },
    /// The event file could not be created, written, or flushed.
    Io(io::Error),
}

impl fmt::Display for TensorBoardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metric(error) => error.fmt(formatter),
            Self::TimestepOutOfRange { timestep } => write!(
                formatter,
                "timestep {timestep} cannot be represented by TensorBoard"
            ),
            Self::Io(error) => error.fmt(formatter),
        }
    }
}

impl error::Error for TensorBoardError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Metric(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::TimestepOutOfRange { .. } => None,
        }
    }
}

impl From<MetricError> for TensorBoardError {
    fn from(error: MetricError) -> Self {
        Self::Metric(error)
    }
}

impl From<io::Error> for TensorBoardError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

pub type TensorBoardResult<T> = std::result::Result<T, TensorBoardError>;

/// Writes aggregated scalar metrics to a TensorBoard event file.
///
/// Each completed timestep becomes one event containing all metrics recorded
/// at that timestep. Values use the same aggregation and rolling-window rules
/// as [`crate::TerminalLogger`]. The event file can be viewed with:
///
/// ```text
/// tensorboard --logdir path/to/runs
/// ```
///
/// Call [`finish`](Self::finish) after the final log entry so the last
/// timestep is written and filesystem errors are reported. Dropping the
/// logger also attempts a best-effort finish.
#[derive(Debug)]
pub struct TensorBoardLogger {
    aggregator: Aggregator,
    writer: EventFileWriter<BufWriter<File>>,
    event_file: PathBuf,
}

impl TensorBoardLogger {
    /// Creates a TensorBoard event file inside `log_dir`.
    pub fn new(log_dir: impl AsRef<Path>, config: AggregationConfig) -> TensorBoardResult<Self> {
        let (file, event_file) = create_event_file(log_dir.as_ref())?;
        let writer = EventFileWriter::new(BufWriter::new(file))?;
        Ok(Self {
            aggregator: Aggregator::new(config),
            writer,
            event_file,
        })
    }

    /// Creates a TensorBoard logger using the default aggregation.
    pub fn with_default_aggregation(log_dir: impl AsRef<Path>) -> TensorBoardResult<Self> {
        Self::new(log_dir, AggregationConfig::default())
    }

    /// Returns the event file created by this logger.
    pub fn event_file(&self) -> &Path {
        &self.event_file
    }

    /// Completes the final timestep and flushes the event file.
    pub fn finish(&mut self) -> TensorBoardResult<()> {
        if let Some(completed) = self.aggregator.finish() {
            self.write_completed(completed)?;
        }
        self.writer.flush()?;
        Ok(())
    }

    fn write_completed(&mut self, completed: CompletedTimestep) -> TensorBoardResult<()> {
        let CompletedTimestep {
            timestep,
            mut metrics,
        } = completed;
        let timestep = tensorboard_timestep(timestep)?;
        metrics.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        if !metrics.is_empty() {
            self.writer.write_summary(timestep, metrics)?;
        }
        self.writer.flush()?;
        Ok(())
    }
}

impl Logger for TensorBoardLogger {
    type Error = TensorBoardError;

    fn log(&mut self, timestep: usize, metrics: &[(&str, &Tensor)]) -> TensorBoardResult<()> {
        tensorboard_timestep(timestep)?;
        if let Some(completed) = self.aggregator.log(timestep, metrics)? {
            self.write_completed(completed)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> TensorBoardResult<()> {
        Self::finish(self)
    }
}

impl Drop for TensorBoardLogger {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

fn tensorboard_timestep(timestep: usize) -> TensorBoardResult<i64> {
    i64::try_from(timestep).map_err(|_| TensorBoardError::TimestepOutOfRange { timestep })
}

fn create_event_file(log_dir: &Path) -> io::Result<(File, PathBuf)> {
    fs::create_dir_all(log_dir)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?
        .as_micros();
    let process = std::process::id();

    loop {
        let sequence = FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = log_dir.join(format!(
            "events.out.tfevents.{timestamp}.modurl.{process}.{sequence}"
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((file, path)),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
}

#[derive(Debug)]
struct EventFileWriter<W> {
    writer: W,
}

impl<W: Write> EventFileWriter<W> {
    fn new(writer: W) -> io::Result<Self> {
        Self::new_with_wall_time(writer, wall_time()?)
    }

    fn new_with_wall_time(writer: W, wall_time: f64) -> io::Result<Self> {
        let mut writer = Self { writer };
        writer.write_event(Event {
            wall_time,
            step: 0,
            what: Some(event::What::FileVersion(FILE_VERSION.to_owned())),
        })?;
        Ok(writer)
    }

    fn write_summary(&mut self, step: i64, metrics: Vec<(String, f32)>) -> io::Result<()> {
        self.write_summary_with_wall_time(step, metrics, wall_time()?)
    }

    fn write_summary_with_wall_time(
        &mut self,
        step: i64,
        metrics: Vec<(String, f32)>,
        wall_time: f64,
    ) -> io::Result<()> {
        let value = metrics
            .into_iter()
            .map(|(tag, value)| SummaryValue {
                tag,
                value: Some(value),
            })
            .collect();
        self.write_event(Event {
            wall_time,
            step,
            what: Some(event::What::Summary(Summary { value })),
        })
    }

    fn write_event(&mut self, event: Event) -> io::Result<()> {
        write_record(&mut self.writer, &event.encode_to_vec())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

fn wall_time() -> io::Result<f64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?
        .as_secs_f64())
}

fn write_record(mut writer: impl Write, data: &[u8]) -> io::Result<()> {
    let length = u64::try_from(data.len())
        .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "TensorBoard event is too large"))?
        .to_le_bytes();
    writer.write_all(&length)?;
    writer.write_all(&masked_crc32c(&length).to_le_bytes())?;
    writer.write_all(data)?;
    writer.write_all(&masked_crc32c(data).to_le_bytes())?;
    Ok(())
}

fn masked_crc32c(data: &[u8]) -> u32 {
    CRC32C
        .checksum(data)
        .rotate_right(15)
        .wrapping_add(CRC_MASK_DELTA)
}

#[derive(Clone, PartialEq, Message)]
struct Event {
    #[prost(double, tag = "1")]
    wall_time: f64,
    #[prost(int64, tag = "2")]
    step: i64,
    #[prost(oneof = "event::What", tags = "3, 5")]
    what: Option<event::What>,
}

mod event {
    use super::{Oneof, Summary};

    #[derive(Clone, PartialEq, Oneof)]
    pub(super) enum What {
        #[prost(string, tag = "3")]
        FileVersion(String),
        #[prost(message, tag = "5")]
        Summary(Summary),
    }
}

#[derive(Clone, PartialEq, Message)]
struct Summary {
    #[prost(message, repeated, tag = "1")]
    value: Vec<SummaryValue>,
}

#[derive(Clone, PartialEq, Message)]
struct SummaryValue {
    #[prost(string, tag = "1")]
    tag: String,
    #[prost(float, optional, tag = "2")]
    value: Option<f32>,
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use candle_core::{Device, Tensor};
    use prost::Message;

    use super::{
        CRC32C, Event, EventFileWriter, FILE_SEQUENCE, FILE_VERSION, TensorBoardLogger, event,
        masked_crc32c,
    };
    use crate::{Logger, TensorBoardError};

    #[test]
    fn uses_the_tensorflow_crc32c_algorithm() {
        assert_eq!(CRC32C.checksum(b"123456789"), 0xe306_9283);
        assert_eq!(masked_crc32c(b"123456789"), 0xc78a_b0e5);
    }

    #[test]
    fn writes_valid_records_and_preserves_zero_scalars() {
        let mut writer = EventFileWriter::new_with_wall_time(Vec::new(), 1.0).unwrap();
        writer
            .write_summary_with_wall_time(
                10,
                vec![("loss".to_owned(), 0.0), ("reward".to_owned(), 2.5)],
                2.0,
            )
            .unwrap();
        let records = read_records(&writer.writer);

        assert_eq!(records.len(), 2);
        assert_eq!(
            records[1],
            &[
                0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x10, 0x0a, 0x2a, 0x1c, 0x0a,
                0x0b, 0x0a, 0x04, b'l', b'o', b's', b's', 0x15, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x0d,
                0x0a, 0x06, b'r', b'e', b'w', b'a', b'r', b'd', 0x15, 0x00, 0x00, 0x20, 0x40,
            ]
        );
        let version = Event::decode(records[0]).unwrap();
        assert_eq!(version.wall_time, 1.0);
        assert_eq!(
            version.what,
            Some(event::What::FileVersion(FILE_VERSION.to_owned()))
        );

        let scalars = Event::decode(records[1]).unwrap();
        assert_eq!(scalars.wall_time, 2.0);
        assert_eq!(scalars.step, 10);
        let Some(event::What::Summary(summary)) = scalars.what else {
            panic!("expected a summary event");
        };
        assert_eq!(summary.value[0].tag, "loss");
        assert_eq!(summary.value[0].value, Some(0.0));
        assert_eq!(summary.value[1].tag, "reward");
        assert_eq!(summary.value[1].value, Some(2.5));
    }

    #[test]
    fn logger_writes_completed_steps_and_trait_finish_writes_the_last_step() {
        let directory = TestDirectory::new();
        let mut logger = TensorBoardLogger::with_default_aggregation(&directory.path).unwrap();
        let event_file = logger.event_file().to_owned();
        let one = Tensor::new(1.0_f32, &Device::Cpu).unwrap();
        let three = Tensor::new(3.0_f32, &Device::Cpu).unwrap();
        let five = Tensor::new(5.0_f32, &Device::Cpu).unwrap();

        logger.log(7, &[("loss", &one)]).unwrap();
        logger.log(7, &[("loss", &three)]).unwrap();
        logger.log(8, &[("loss", &five)]).unwrap();

        let data = fs::read(&event_file).unwrap();
        assert_eq!(read_records(&data).len(), 2);

        {
            let logger: &mut dyn Logger<Error = TensorBoardError> = &mut logger;
            logger.finish().unwrap();
        }

        let data = fs::read(event_file).unwrap();
        let records = read_records(&data);
        assert_eq!(records.len(), 3);
        let event = Event::decode(records[1]).unwrap();
        assert_eq!(event.step, 7);
        let Some(event::What::Summary(summary)) = event.what else {
            panic!("expected a summary event");
        };
        assert_eq!(summary.value.len(), 1);
        assert_eq!(summary.value[0].tag, "loss");
        assert_eq!(summary.value[0].value, Some(2.0));

        let event = Event::decode(records[2]).unwrap();
        assert_eq!(event.step, 8);
        let Some(event::What::Summary(summary)) = event.what else {
            panic!("expected a summary event");
        };
        assert_eq!(summary.value[0].value, Some(5.0));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_steps_that_tensorboard_cannot_represent() {
        let directory = TestDirectory::new();
        let mut logger = TensorBoardLogger::with_default_aggregation(&directory.path).unwrap();
        let value = Tensor::new(1.0_f32, &Device::Cpu).unwrap();

        let error = logger.log(usize::MAX, &[("loss", &value)]).unwrap_err();

        assert!(matches!(
            error,
            TensorBoardError::TimestepOutOfRange {
                timestep: usize::MAX
            }
        ));
    }

    fn read_records(mut data: &[u8]) -> Vec<&[u8]> {
        let mut records = Vec::new();
        while !data.is_empty() {
            assert!(data.len() >= 16);
            let length_bytes: [u8; 8] = data[..8].try_into().unwrap();
            let length = usize::try_from(u64::from_le_bytes(length_bytes)).unwrap();
            let length_crc = u32::from_le_bytes(data[8..12].try_into().unwrap());
            assert_eq!(length_crc, masked_crc32c(&length_bytes));
            assert!(data.len() >= 16 + length);
            let record = &data[12..12 + length];
            let data_crc = u32::from_le_bytes(data[12 + length..16 + length].try_into().unwrap());
            assert_eq!(data_crc, masked_crc32c(record));
            records.push(record);
            data = &data[16 + length..];
        }
        records
    }

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new() -> Self {
            let sequence = FILE_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "modurl-logger-test-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
