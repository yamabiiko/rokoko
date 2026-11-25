//! Fixed-size sumcheck scaling probe; no changes to the prover.
//! cargo run --release --example sumcheck_parallel -- --log-pairs 14 --workers 1,2,4,8
//! Measures two quadratic terms sharing one table, and folding all three tables.
//! Workers persist across samples; timing includes dispatch, barriers and reduction.
//! Run several sizes (e.g. log-pairs 3, 14, 16); all inputs are reused between
//! samples. Allocations, input generation, thread creation and correctness checks
//! are excluded. No affinity/NUMA tuning, sparse selectors, cubic terms or
//! previous-round shortcut are modeled. This is a kernel probe, not a prover speedup.
//! Sequential baselines include both the same kernel without dispatch and the
//! existing library flat sweep / in-place fold. The threaded fold needs half-sized
//! destination buffers in addition to the immutable inputs.
use rokoko::{
    common::{
        config::{DEGREE, MOD_Q},
        hash::HashWrapper,
        init_common,
        ring_arithmetic::{incomplete_ntt_dot_into, Representation, RingElement},
    },
    protocol::sumcheck_utils::{
        common::{HighOrderSumcheckData, SumcheckBaseData},
        elephant_cell::ElephantCell,
        flat::FlatRounds,
        linear::LinearSumcheck,
        polynomial::Polynomial,
        product::ProductSumcheck,
    },
};
use std::{
    hint::black_box,
    sync::{
        atomic::{AtomicU8, Ordering},
        Barrier, Mutex,
    },
    thread,
    time::Instant,
};

type Raw = [RingElement; 6];
type Inputs<'a> = [&'a [RingElement]; 3];
const TILE: usize = 32;
const SWEEP: u8 = 0;
const FOLD: u8 = 1;
const STOP: u8 = 2;

fn zero() -> RingElement {
    RingElement::zero(Representation::IncompleteNTT)
}
fn zeros() -> Raw {
    std::array::from_fn(|_| zero())
}

// Same pair layout, slope preparation and fused dots as the degree-two flat kernel.
// One worker handles all terms for its contiguous range, reusing the witness slopes.
fn sweep(input: &Inputs<'_>, slopes: &mut [Vec<RingElement>; 3]) -> Raw {
    let mut sums = zeros();
    for start in (0..input[0].len() / 2).step_by(TILE) {
        let count = TILE.min(input[0].len() / 2 - start);
        for (source, slope) in input.iter().zip(slopes.iter_mut()) {
            for k in 0..count {
                slope[k] -= (&source[2 * (start + k) + 1], &source[2 * (start + k)]);
            }
        }
        for term in 0..2 {
            let other = term + 1;
            incomplete_ntt_dot_into(
                &mut sums[3 * term],
                &input[0][2 * start..],
                2,
                &input[other][2 * start..],
                2,
                count,
            );
            incomplete_ntt_dot_into(
                &mut sums[3 * term + 1],
                &input[0][2 * start + 1..],
                2,
                &input[other][2 * start + 1..],
                2,
                count,
            );
            incomplete_ntt_dot_into(
                &mut sums[3 * term + 2],
                &slopes[0],
                1,
                &slopes[other],
                1,
                count,
            );
        }
    }
    sums
}

// Interpolate from sum@0, sum@1 and sum of slope products, then batch the terms.
fn polynomial(sums: &Raw, gamma: &[RingElement; 2]) -> Polynomial<RingElement> {
    let mut poly = Polynomial::new(3);
    let mut coefficient = zero();
    for term in 0..2 {
        let values = &sums[3 * term..3 * term + 3];
        for k in 0..3 {
            coefficient.set_from(&values[k]);
            if k == 1 {
                coefficient -= &values[0];
                coefficient -= &values[2];
            }
            coefficient *= &gamma[term];
            poly.coefficients[k] += &coefficient;
        }
    }
    poly
}

fn fold(input: &Inputs<'_>, output: &mut [&mut [RingElement]; 3], challenge: &RingElement) {
    for (source, dest) in input.iter().zip(output.iter_mut()) {
        for (pair, out) in source.chunks_exact(2).zip(dest.iter_mut()) {
            *out -= (&pair[1], &pair[0]);
            *out *= challenge;
            *out += &pair[0];
        }
    }
}

fn median_ms(mut f: impl FnMut(), samples: usize, iterations: usize) -> f64 {
    f(); // warm up kernels, worker startup and pages outside timing
    let mut times: Vec<_> = (0..samples)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..iterations {
                f();
            }
            start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

// Actual library evaluator: borrowed input buffers stay alive and immutable throughout.
// Neither this graph nor its Rc/RefCell state is passed to the workers.
fn library_baseline(
    input: &Inputs<'_>,
    gamma: &[RingElement; 2],
    samples: usize,
    iterations: usize,
) -> Polynomial<RingElement> {
    let tables: Vec<_> = input
        .iter()
        .map(|source| {
            // SAFETY: the caller owns the input for this entire function; no mutation,
            // folding or detached jobs can invalidate it, and the graph is dropped here.
            ElephantCell::new(unsafe { LinearSumcheck::from_borrowed(source, source.len(), None) })
        })
        .collect();
    let outputs: Vec<ElephantCell<dyn HighOrderSumcheckData<Element = RingElement>>> = vec![
        ElephantCell::new(ProductSumcheck::new(tables[0].clone(), tables[1].clone())),
        ElephantCell::new(ProductSumcheck::new(tables[0].clone(), tables[2].clone())),
    ];
    let flat = FlatRounds::new(&outputs, input[0].len().ilog2() as usize);
    let expected = flat.round_polynomial(&outputs, gamma, None);
    let ms = median_ms(
        || {
            black_box(flat.round_polynomial(&outputs, gamma, None));
        },
        samples,
        iterations,
    );
    println!("library flat sweep: {ms:.3} ms (first-round path, no previous-round shortcut)");
    expected
}

fn parallel(
    input: &Inputs<'_>,
    gamma: &[RingElement; 2],
    challenge: &RingElement,
    output: &mut [Vec<RingElement>; 3],
    workers: usize,
    samples: usize,
    iterations: usize,
    expected: &Polynomial<RingElement>,
) -> (f64, f64) {
    let start = Barrier::new(workers + 1);
    let done = Barrier::new(workers + 1);
    let command = AtomicU8::new(SWEEP);
    // One publication per worker per sweep, outside the numerical loop.
    // Coordinator reads only after the completion barrier: no contested hot accumulator.
    let results: Vec<_> = (0..workers).map(|_| Mutex::new(zeros())).collect();
    let (sweep_ms, fold_ms, before, after) = thread::scope(|scope| {
        let mut tails = output.each_mut().map(|v| v.as_mut_slice());
        let pairs = input[0].len() / 2;
        let mut offset = 0;
        for (id, result) in results.iter().enumerate() {
            let count = pairs / workers + usize::from(id < pairs % workers);
            let [(a, ta), (b, tb), (c, tc)] = tails.map(|tail| tail.split_at_mut(count));
            tails = [ta, tb, tc];
            let sources = input.map(|source| &source[2 * offset..2 * (offset + count)]);
            offset += count;
            let (start, done, command) = (&start, &done, &command);
            scope.spawn(move || {
                let mut slopes = std::array::from_fn(|_| vec![zero(); TILE]);
                let mut dest = [a, b, c];
                loop {
                    start.wait();
                    match command.load(Ordering::Acquire) {
                        SWEEP => *result.lock().unwrap() = sweep(&sources, &mut slopes),
                        FOLD => fold(&sources, &mut dest, challenge),
                        STOP => break,
                        _ => unreachable!(),
                    }
                    done.wait();
                }
            });
        }
        let dispatch = |phase| {
            command.store(phase, Ordering::Release);
            start.wait();
            done.wait();
        };
        let evaluate = || {
            dispatch(SWEEP);
            let mut total = zeros();
            for result in &results {
                for (sum, partial) in total.iter_mut().zip(result.lock().unwrap().iter()) {
                    *sum += partial;
                }
            }
            polynomial(&total, gamma)
        };
        let before = evaluate();
        let sweep_ms = median_ms(
            || {
                black_box(evaluate());
            },
            samples,
            iterations,
        );
        let fold_ms = median_ms(|| dispatch(FOLD), samples, iterations);
        let after = evaluate();
        command.store(STOP, Ordering::Release);
        start.wait();
        (sweep_ms, fold_ms, before, after)
    });
    // Shut down the team before asserting, so a failed check cannot leave workers
    // waiting on a barrier while the coordinator unwinds.
    assert_eq!(before.coefficients, expected.coefficients);
    assert_eq!(after.coefficients, expected.coefficients);
    (sweep_ms, fold_ms)
}

fn main() {
    let mut log_pairs = 14u32;
    let mut workers = vec![1usize, 2, 4, 8];
    let mut samples = 5usize;
    let mut iterations = 3usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().expect("each flag requires a value");
        match flag.as_str() {
            "--log-pairs" => log_pairs = value.parse().unwrap(),
            "--workers" => workers = value.split(',').map(|n| n.parse().unwrap()).collect(),
            "--samples" => samples = value.parse().unwrap(),
            "--iterations" => iterations = value.parse().unwrap(),
            _ => panic!("unknown flag {flag}; use --log-pairs, --workers, --samples, --iterations"),
        }
    }
    assert!(
        (3..=22).contains(&log_pairs),
        "--log-pairs must be between 3 and 22 (check RAM before large runs)"
    );
    assert!(samples > 0 && iterations > 0);
    let pairs = 1usize << log_pairs;
    assert!(!workers.is_empty() && workers.iter().all(|&n| n > 0 && n <= pairs));
    workers.sort_unstable();
    workers.dedup();
    init_common();
    incomplete_rexl::cpu_features::print_features();
    let inputs: [Vec<_>; 3] = std::array::from_fn(|_| {
        (0..2 * pairs)
            .map(|_| RingElement::random(Representation::IncompleteNTT))
            .collect()
    });
    let input = inputs.each_ref().map(|v| v.as_slice());
    let mut gamma = [zero(), zero()];
    let mut hash = HashWrapper::new();
    hash.sample_ring_element_vec_into(&mut gamma);
    let mut point = [zero()];
    hash.sample_ring_element_ntt_slots_same_vec_into(&mut point);
    let challenge = &point[0];
    let mut output = std::array::from_fn(|_| vec![zero(); pairs]);
    let mib = std::mem::size_of::<RingElement>() as f64 * pairs as f64 / (1024.0 * 1024.0);
    println!(
        "degree={DEGREE}, q={MOD_Q}, element_bytes={}, pairs=2^{log_pairs}",
        std::mem::size_of::<RingElement>()
    );
    println!("input={:.3} MiB, fold destination={:.3} MiB, tile={TILE}, samples={samples}, iterations={iterations}", 6.0 * mib, 3.0 * mib);
    println!("available_parallelism={:?}; worker count includes compute workers, coordinator is additional", thread::available_parallelism().map(usize::from));

    let expected = library_baseline(&input, &gamma, samples, iterations);
    let mut slopes = std::array::from_fn(|_| vec![zero(); TILE]);
    assert_eq!(
        polynomial(&sweep(&input, &mut slopes), &gamma).coefficients,
        expected.coefficients
    );
    let serial_sweep = median_ms(
        || {
            black_box(polynomial(&sweep(&input, &mut slopes), &gamma));
        },
        samples,
        iterations,
    );
    let serial_fold = median_ms(
        || {
            fold(
                &input,
                &mut output.each_mut().map(|v| v.as_mut_slice()),
                challenge,
            )
        },
        samples,
        iterations,
    );
    println!("direct sequential: sweep={serial_sweep:.3} ms, buffered fold={serial_fold:.3} ms");

    // Compare against the existing in-place fold, excluding input clones/reset from timing.
    let mut inplace_times = Vec::new();
    for _ in 0..samples + 1 {
        let mut tables = inputs
            .each_ref()
            .map(|source| LinearSumcheck::from_data(source.clone()));
        let start = Instant::now();
        for table in &mut tables {
            table.partial_evaluate(challenge);
        }
        inplace_times.push(start.elapsed().as_secs_f64() * 1000.0);
        for (table, reference) in tables.iter().zip(&output) {
            assert_eq!(&table.data, reference);
        }
    }
    inplace_times.remove(0);
    inplace_times.sort_by(f64::total_cmp);
    let inplace = inplace_times[inplace_times.len() / 2];
    println!("library in-place fold: {inplace:.3} ms (input reset excluded)");

    // Sumcheck identity: g(0)+g(1) is the original claim; g(r) is the folded claim.
    let claim = |tables: Inputs<'_>| {
        let mut sum = zero();
        for term in 0..2 {
            let mut value = zero();
            incomplete_ntt_dot_into(
                &mut value,
                tables[0],
                1,
                tables[term + 1],
                1,
                tables[0].len(),
            );
            value *= &gamma[term];
            sum += &value;
        }
        sum
    };
    let mut original_sum = expected.at_zero();
    original_sum += &expected.at_one();
    assert_eq!(original_sum, claim(input));
    assert_eq!(
        expected.at(challenge),
        claim(output.each_ref().map(|v| v.as_slice()))
    );
    println!("workers,sweep_ms,sweep_speedup,buffered_fold_ms,fold_speedup,fold_vs_inplace");
    for count in workers {
        let (sweep_ms, fold_ms) = parallel(
            &input,
            &gamma,
            challenge,
            &mut output,
            count,
            samples,
            iterations,
            &expected,
        );
        // Check every output element, outside timing, without retaining another output copy.
        let mut want = zero();
        for (source, dest) in input.iter().zip(&output) {
            for (pair, value) in source.chunks_exact(2).zip(dest) {
                want -= (&pair[1], &pair[0]);
                want *= challenge;
                want += &pair[0];
                assert_eq!(&want, value);
            }
        }
        println!(
            "{count},{sweep_ms:.3},{:.2},{fold_ms:.3},{:.2},{:.2}",
            serial_sweep / sweep_ms,
            serial_fold / fold_ms,
            inplace / fold_ms
        );
    }
}
