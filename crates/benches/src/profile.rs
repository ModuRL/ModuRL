use burn::tensor::Device;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    path::Path,
    sync::{Arc, Mutex},
    thread::ThreadId,
    time::{Duration, Instant},
};
use tracing::{Id, Subscriber};
use tracing_subscriber::{
    Layer, filter::filter_fn, layer::Context, prelude::*, registry::LookupSpan,
};

struct Frame {
    id: u64,
    name: &'static str,
    path: String,
    start: Instant,
    children: Duration,
}

#[derive(Default)]
struct Timing {
    calls: usize,
    total: Duration,
    own: Duration,
}

#[derive(Default)]
struct State {
    stacks: HashMap<ThreadId, Vec<Frame>>,
    thread_ids: HashMap<ThreadId, usize>,
    events: Vec<Value>,
    totals: BTreeMap<String, Timing>,
    error: Option<String>,
}

#[derive(Clone)]
struct Timings {
    state: Arc<Mutex<State>>,
    origin: Instant,
    sync_device: Option<Device>,
}

impl Timings {
    fn synchronize(&self) {
        if let Some(device) = &self.sync_device
            && let Err(error) = device.sync()
            && let Ok(mut state) = self.state.lock()
        {
            state.error = Some(error.to_string());
        }
    }

    fn save(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let state = self.state.lock().map_err(|error| error.to_string())?;
        if let Some(error) = &state.error {
            return Err(error.clone().into());
        }
        if state.stacks.values().any(|stack| !stack.is_empty()) {
            return Err("trace contains unfinished spans".into());
        }
        let summary: Vec<Value> = state
            .totals
            .iter()
            .map(|(name, timing)| {
                json!({
                    "path": name, "calls": timing.calls,
                    "inclusive_seconds": timing.total.as_secs_f64(),
                    "self_seconds": timing.own.as_secs_f64(),
                })
            })
            .collect();
        let report = json!({
            "traceEvents": state.events,
            "displayTimeUnit": "ms",
            "synchronized": self.sync_device.is_some(),
            "summary": summary,
        });
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_vec(&report)?)?;
        println!("Performance trace: {}", path.display());
        let mut rows: Vec<_> = state.totals.iter().collect();
        rows.sort_by_key(|row| std::cmp::Reverse(row.1.own));
        println!("Self seconds | Inclusive seconds | Calls | Span path");
        for (name, timing) in rows.into_iter().take(15) {
            println!(
                "{:.4} | {:.4} | {} | {}",
                timing.own.as_secs_f64(),
                timing.total.as_secs_f64(),
                timing.calls,
                name
            );
        }
        Ok(())
    }
}

impl<S: Subscriber + for<'lookup> LookupSpan<'lookup>> Layer<S> for Timings {
    fn on_enter(&self, id: &Id, context: Context<'_, S>) {
        self.synchronize();
        let Some(span) = context.span(id) else {
            return;
        };
        let name = span.metadata().name();
        let start = Instant::now();
        if let Ok(mut state) = self.state.lock() {
            let thread = std::thread::current().id();
            let next_id = state.thread_ids.len();
            state.thread_ids.entry(thread).or_insert(next_id);
            let stack = state.stacks.entry(thread).or_default();
            let path = match stack.last() {
                Some(parent) => format!("{}/{name}", parent.path),
                None => name.to_owned(),
            };
            stack.push(Frame {
                id: id.into_u64(),
                name,
                path,
                start,
                children: Duration::ZERO,
            });
        }
    }

    fn on_exit(&self, id: &Id, _context: Context<'_, S>) {
        self.synchronize();
        let end = Instant::now();
        if let Ok(mut state) = self.state.lock() {
            let thread = std::thread::current().id();
            let Some(stack) = state.stacks.get_mut(&thread) else {
                state.error = Some("trace span exited without entering".to_owned());
                return;
            };
            let Some(frame) = stack.pop() else {
                state.error = Some("trace span stack is empty".to_owned());
                return;
            };
            if frame.id != id.into_u64() {
                state.error = Some("trace spans exited out of order".to_owned());
                return;
            }
            let elapsed = end.duration_since(frame.start);
            if let Some(parent) = stack.last_mut() {
                parent.children += elapsed;
            }
            let own = elapsed.saturating_sub(frame.children);
            let timing = state.totals.entry(frame.path.clone()).or_default();
            timing.calls += 1;
            timing.total += elapsed;
            timing.own += own;
            let thread_id = state.thread_ids[&thread];
            state.events.push(json!({
                "name": frame.name, "cat": "modurl::performance", "ph": "X",
                "ts": frame.start.duration_since(self.origin).as_secs_f64() * 1e6,
                "dur": elapsed.as_secs_f64() * 1e6, "pid": 1, "tid": thread_id,
                "args": { "path": frame.path, "self_us": own.as_secs_f64() * 1e6 },
            }));
        }
    }
}

pub(super) fn record(
    path: &Path,
    device: &Device,
    synchronize: bool,
    workload: impl FnOnce() -> Result<(), Box<dyn Error>>,
) -> Result<f64, Box<dyn Error>> {
    let timings = Timings {
        state: Arc::new(Mutex::new(State::default())),
        origin: Instant::now(),
        sync_device: synchronize.then(|| device.clone()),
    };
    let subscriber =
        tracing_subscriber::registry().with(timings.clone().with_filter(filter_fn(|metadata| {
            metadata.target() == "modurl::performance"
        })));
    let start = Instant::now();
    let result = tracing::subscriber::with_default(subscriber, || {
        tracing::info_span!(target: "modurl::performance", "benchmark.measured").in_scope(|| {
            workload()?;
            device.sync()?;
            Ok::<(), Box<dyn Error>>(())
        })
    });
    let elapsed = start.elapsed().as_secs_f64();
    timings.save(path)?;
    result?;
    Ok(elapsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_timings_exclude_children_from_self_time_and_ignore_other_targets() {
        let timings = Timings {
            state: Arc::new(Mutex::new(State::default())),
            origin: Instant::now(),
            sync_device: None,
        };
        let subscriber = tracing_subscriber::registry().with(timings.clone().with_filter(
            filter_fn(|metadata| metadata.target() == "modurl::performance"),
        ));
        tracing::subscriber::with_default(subscriber, || {
            tracing::info_span!(target: "modurl::performance", "parent").in_scope(|| {
                tracing::info_span!(target: "other", "ignored").in_scope(|| {});
                for _ in 0..2 {
                    tracing::info_span!(target: "modurl::performance", "child").in_scope(|| {});
                }
            });
        });
        let state = timings.state.lock().unwrap();
        let parent = &state.totals["parent"];
        let child = &state.totals["parent/child"];
        assert_eq!(child.calls, 2);
        assert_eq!(parent.total, parent.own + child.total);
        assert_eq!(state.events.len(), 3);
        assert_eq!(state.totals.len(), 2);
        assert!(state.error.is_none());
    }
}
