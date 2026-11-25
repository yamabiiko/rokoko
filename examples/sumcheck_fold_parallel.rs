//! Shrinking-chain parallel folding experiment; does not modify the prover.
//! cargo run --release --features parallel --example sumcheck_fold_parallel --
//!   --log-pairs 16 --workers 1,2,4,8,16 --samples 3
//!
//! Three differently sized ordinary tables are folded to lengths 4, 2, and 1.
//! All variants use the same inputs, challenges, and sparse boundaries. Each
//! variant has one discarded warmup followed by `samples` complete chains.
//! Library input cloning, input generation, validation, and pool creation are
//! outside timing. `parallel_allocated` includes buffer initialization and drop;
//! reused variants retain two destination buffers across rounds and samples.
//! Pool dispatch, round bookkeeping, zero-tail writes, and serial tails are timed.
//! This measures folding only: no transcript, sweep, mapped views, or proof.

use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use rokoko::{
    common::{
        config::{DEGREE, MOD_Q},
        hash::HashWrapper,
        init_common,
        ring_arithmetic::{Representation, RingElement},
    },
    protocol::sumcheck_utils::{common::SumcheckBaseData, linear::LinearSumcheck},
};
use std::{hint::black_box, time::Instant};

fn zero() -> RingElement {
    RingElement::zero(Representation::IncompleteNTT)
}

struct Inputs {
    tables: Vec<Vec<RingElement>>,
    nonzero: Vec<usize>,
    challenges: Vec<RingElement>,
}

impl Inputs {
    fn library_tables(&self) -> Vec<LinearSumcheck> {
        self.tables
            .iter()
            .zip(&self.nonzero)
            .map(|(table, &end)| LinearSumcheck::from_data_with_non_zero_end(table.clone(), end))
            .collect()
    }
}

struct Buffers {
    first: Vec<Vec<RingElement>>,
    second: Vec<Vec<RingElement>>,
}

impl Buffers {
    fn new(input: &Inputs) -> Self {
        Self {
            first: input
                .tables
                .iter()
                .map(|v| vec![zero(); v.len() / 2])
                .collect(),
            second: input
                .tables
                .iter()
                .map(|v| vec![zero(); v.len() / 4])
                .collect(),
        }
    }
}

// `offset` is measured in output elements. Source and destination never alias.
fn fold_chunk(
    source: &[RingElement],
    dest: &mut [RingElement],
    offset: usize,
    active_pairs: usize,
    challenge: &RingElement,
) {
    let count = active_pairs.saturating_sub(offset).min(dest.len());
    for (i, out) in dest[..count].iter_mut().enumerate() {
        let at = 2 * (offset + i);
        *out -= (&source[at + 1], &source[at]);
        *out *= challenge;
        *out += &source[at];
    }
    for out in &mut dest[count..] {
        out.set_zero();
    }
}

fn fold_round(
    sources: &[&[RingElement]],
    dest: &mut [&mut [RingElement]],
    active: &[usize],
    challenge: &RingElement,
    pool: Option<&ThreadPool>,
    grain: usize,
) {
    let work: usize = active.iter().sum();
    if let Some(pool) = pool.filter(|_| work >= 2 * grain) {
        // One pool across tables and chunks, never a nested OS-thread team.
        pool.install(|| {
            dest.par_iter_mut()
                .zip(sources.par_iter())
                .zip(active.par_iter())
                .for_each(|((dest, source), &active)| {
                    dest.par_chunks_mut(grain)
                        .enumerate()
                        .for_each(|(i, chunk)| {
                            fold_chunk(source, chunk, i * grain, active, challenge);
                        });
                });
        });
    } else {
        for ((source, dest), &active) in sources.iter().zip(dest).zip(active) {
            fold_chunk(source, dest, 0, active, challenge);
        }
    }
}

fn chain(
    input: &Inputs,
    buffers: &mut Buffers,
    pool: Option<&ThreadPool>,
    grain: usize,
    mut inspect: impl FnMut(usize, &[&mut [RingElement]]),
) {
    let mut lengths: Vec<_> = input.tables.iter().map(Vec::len).collect();
    let mut active = input.nonzero.clone();
    for (round, challenge) in input.challenges.iter().enumerate() {
        let (source, dest) = if round == 0 {
            (&input.tables, &mut buffers.first)
        } else if round % 2 == 1 {
            (&buffers.first, &mut buffers.second)
        } else {
            (&buffers.second, &mut buffers.first)
        };
        let sources: Vec<_> = source.iter().zip(&lengths).map(|(v, &n)| &v[..n]).collect();
        for (n, end) in lengths.iter_mut().zip(&mut active) {
            *n /= 2;
            *end = end.div_ceil(2).min(*n);
        }
        let mut outputs: Vec<_> = dest
            .iter_mut()
            .zip(&lengths)
            .map(|(v, &n)| &mut v[..n])
            .collect();
        fold_round(&sources, &mut outputs, &active, challenge, pool, grain);
        inspect(round, &outputs);
        black_box(&outputs);
    }
}

fn validate(input: &Inputs, pool: Option<&ThreadPool>, grain: usize) {
    let mut reference = input.library_tables();
    let mut buffers = Buffers::new(input);
    // Poison all destination storage: sparse zero tails must be written correctly,
    // including when buffers contain old, larger rounds from a previous chain.
    let poison = RingElement::random(Representation::IncompleteNTT);
    for table in buffers.first.iter_mut().chain(&mut buffers.second) {
        table.fill(poison.clone());
    }
    for _ in 0..2 {
        chain(input, &mut buffers, pool, grain, |round, outputs| {
            for (table, output) in reference.iter_mut().zip(outputs) {
                table.partial_evaluate(&input.challenges[round]);
                assert_eq!(
                    table.data.as_slice(),
                    &**output,
                    "fold mismatch in round {round}"
                );
            }
        });
        reference = input.library_tables();
    }
}

fn print_samples(name: &str, workers: usize, samples: usize, mut run: impl FnMut() -> f64) -> f64 {
    black_box(run()); // one warmup chain
    let mut total = 0.0;
    for sample in 1..=samples {
        let ms = run();
        total += ms;
        println!("sample,{name},{workers},{sample},{ms:.6}");
    }
    total / samples as f64
}

fn fixture(log_pairs: u32, active_percent: usize, zero_tail: usize) -> Inputs {
    let n = 1usize << (log_pairs + 1);
    let lengths = [n, n / 2, n / 4];
    let nonzero: Vec<_> = lengths
        .iter()
        .map(|n| (n * active_percent / 100).saturating_sub(zero_tail))
        .collect();
    let tables = lengths
        .iter()
        .zip(&nonzero)
        .map(|(&len, &end)| {
            (0..len)
                .map(|i| {
                    if i < end {
                        RingElement::random(Representation::IncompleteNTT)
                    } else {
                        zero()
                    }
                })
                .collect()
        })
        .collect();
    let mut challenges = vec![zero(); (log_pairs - 1) as usize];
    HashWrapper::new().sample_ring_element_ntt_slots_same_vec_into(&mut challenges);
    Inputs {
        tables,
        nonzero,
        challenges,
    }
}

fn main() {
    let mut log_pairs = 14u32;
    let mut workers = vec![1usize, 2, 4, 8, 16];
    let mut samples = 3usize;
    let mut active_percent = 100usize;
    let mut zero_tail = 0usize;
    let mut grain = 2048usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().expect("each flag needs a value");
        match flag.as_str() {
            "--log-pairs" => log_pairs = value.parse().unwrap(),
            "--workers" => workers = value.split(',').map(|n| n.parse().unwrap()).collect(),
            "--samples" => samples = value.parse().unwrap(),
            "--active-percent" => active_percent = value.parse().unwrap(),
            "--zero-tail" => zero_tail = value.parse().unwrap(),
            "--grain" => grain = value.parse().unwrap(),
            _ => panic!("unknown flag {flag}"),
        }
    }
    assert!(
        (3..=19).contains(&log_pairs),
        "log-pairs must be 3..19; large cases need substantial RAM"
    );
    assert!(samples > 0 && grain > 0 && grain <= usize::MAX / 2 && active_percent <= 100);
    assert!(!workers.is_empty() && workers.iter().all(|&n| n > 0 && n <= 256));
    workers.sort_unstable();
    workers.dedup();
    init_common();
    incomplete_rexl::cpu_features::print_features();
    let input = fixture(log_pairs, active_percent, zero_tail);
    let input_bytes = input.tables.iter().map(Vec::len).sum::<usize>() * size_of::<RingElement>();
    println!("degree={DEGREE},q={MOD_Q},element_bytes={},log_pairs={log_pairs},rounds={},active_percent={active_percent},zero_tail={zero_tail},grain={grain},samples={samples}", size_of::<RingElement>(), input.challenges.len());
    println!(
        "input_mib={:.3},destination_capacity_mib={:.3},available_parallelism={:?}",
        input_bytes as f64 / 1048576.0,
        input_bytes as f64 * 0.75 / 1048576.0,
        std::thread::available_parallelism()
    );
    println!("sample,variant,workers,sample,chain_ms");
    validate(&input, None, grain);
    let library = print_samples("library_inplace", 1, samples, || {
        let mut tables = input.library_tables(); // reset excluded
        let start = Instant::now();
        for challenge in &input.challenges {
            for table in &mut tables {
                table.partial_evaluate(challenge);
            }
        }
        black_box(&tables);
        start.elapsed().as_secs_f64() * 1000.0
    });
    let mut buffers = Buffers::new(&input);
    let serial = print_samples("serial_reused", 1, samples, || {
        let start = Instant::now();
        chain(&input, &mut buffers, None, grain, |_, _| {});
        start.elapsed().as_secs_f64() * 1000.0
    });
    drop(buffers);
    println!("summary,variant,workers,mean_ms,speedup_vs_library,speedup_vs_buffered");
    println!(
        "summary,library_inplace,1,{library:.6},1.000,{:.3}",
        serial / library
    );
    println!(
        "summary,serial_reused,1,{serial:.6},{:.3},1.000",
        library / serial
    );
    for count in workers {
        let start = Instant::now();
        let pool = ThreadPoolBuilder::new().num_threads(count).build().unwrap();
        pool.broadcast(|_| ()); // ensure every persistent worker has started
        println!(
            "pool_startup_ms,{count},{:.6}",
            start.elapsed().as_secs_f64() * 1000.0
        );
        validate(&input, Some(&pool), grain);
        // Small non-dyadic partitions, odd sparse boundaries, and all-zero tables.
        // Force parallel dispatch here even when the measured case uses a large cutoff.
        for (percent, tail) in [(100, 0), (100, 3), (25, 1), (0, 0)] {
            validate(&fixture(5, percent, tail), Some(&pool), 3);
        }
        println!("correctness,{count},all_rounds_and_reused_buffers_passed");
        let mut buffers = Buffers::new(&input);
        let reused = print_samples("parallel_reused", count, samples, || {
            let start = Instant::now();
            chain(&input, &mut buffers, Some(&pool), grain, |_, _| {});
            start.elapsed().as_secs_f64() * 1000.0
        });
        drop(buffers);
        let allocated = print_samples("parallel_allocated", count, samples, || {
            let start = Instant::now();
            let mut buffers = Buffers::new(&input);
            chain(&input, &mut buffers, Some(&pool), grain, |_, _| {});
            drop(buffers);
            start.elapsed().as_secs_f64() * 1000.0
        });
        for (name, ms) in [
            ("parallel_reused", reused),
            ("parallel_allocated", allocated),
        ] {
            println!(
                "summary,{name},{count},{ms:.6},{:.3},{:.3}",
                library / ms,
                serial / ms
            );
        }
    }
}
