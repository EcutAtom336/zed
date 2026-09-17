//! Regression test for a parse buffer that outlives the command it was built for.
//!
//! Every `terminal::Terminal` constructs a `vte::ansi::Processor` (`terminal.rs`
//! builds it as the `output_processor` field), and `Processor`'s `Default` eagerly
//! reserves `SYNC_BUFFER_SIZE` for synchronized updates, whether or not the
//! terminal ever receives one. Once a one-shot command has finished,
//! `Terminal::release_pty_resources` shuts the PTY down but leaves that buffer
//! allocated, so every caller that keeps the terminal around afterwards — the
//! agent panel keeps each tool call's terminal so it can render the finished
//! command's output inline — retains that reservation for as long as it holds the
//! terminal. A session that runs commands through the agent therefore grows by
//! roughly `VTE_SYNC_BUFFER_SIZE` per command.
//!
//! This lives in `tests/` rather than next to the other terminal tests for two
//! reasons: it installs a counting global allocator, which must not perturb the
//! crate's unit tests, and it measures process-wide numbers, which need a test
//! binary that does not run other tests in parallel with it.

#![cfg(unix)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use collections::HashMap;
use gpui::{AppContext as _, Entity, TestAppContext};
use task::{Shell, SpawnInTerminal};
use terminal::{
    Terminal, TerminalBuilder, TerminalMode,
    terminal_settings::{AlternateScroll, CursorShape as SettingsCursorShape},
};
use util::paths::PathStyle;

/// `vte::ansi::SYNC_BUFFER_SIZE`: the buffer `Processor::new` reserves for
/// synchronized updates.
const VTE_SYNC_BUFFER_SIZE: usize = 0x20_0000;

/// Enough terminals that one reservation each dominates unrelated noise.
const TERMINALS: usize = 8;

/// What we tolerate for [`TERMINALS`] finished-but-still-referenced terminals.
/// Terminals that still hold their parse buffer account for `TERMINALS *
/// VTE_SYNC_BUFFER_SIZE` = 16 MiB; what a terminal legitimately keeps (its
/// emulator grid, entity slot, ...) is tens of kilobytes.
const RETAINED_BUDGET_BYTES: usize = 3 * 1024 * 1024;

struct CountingAllocator;

/// Bytes currently handed out by the global allocator: the same quantity
/// heaptrack reports as live heap.
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Live allocations of exactly [`VTE_SYNC_BUFFER_SIZE`] bytes, i.e. live
/// `vte::ansi::Processor` sync buffers. Reported for diagnosis only.
static LIVE_SYNC_BUFFERS: AtomicUsize = AtomicUsize::new(0);

fn record_alloc(size: usize) {
    LIVE_BYTES.fetch_add(size, Ordering::Relaxed);
    if size == VTE_SYNC_BUFFER_SIZE {
        LIVE_SYNC_BUFFERS.fetch_add(1, Ordering::Relaxed);
    }
}

fn record_dealloc(size: usize) {
    LIVE_BYTES.fetch_sub(size, Ordering::Relaxed);
    if size == VTE_SYNC_BUFFER_SIZE {
        LIVE_SYNC_BUFFERS.fetch_sub(1, Ordering::Relaxed);
    }
}

// Deliberately counting reservations rather than RSS. `Vec::with_capacity`
// reserves address space without touching the pages, so a retained parse buffer
// is invisible in RSS. The bug is the reservation, so measure reservations.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record_dealloc(layout.size());
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            record_alloc(new_size);
            record_dealloc(layout.size());
        }
        new_ptr
    }
}

#[global_allocator]
static COUNTING_ALLOCATOR: CountingAllocator = CountingAllocator;

fn live_bytes() -> usize {
    LIVE_BYTES.load(Ordering::Relaxed)
}

fn live_sync_buffers() -> usize {
    LIVE_SYNC_BUFFERS.load(Ordering::Relaxed)
}

/// Lets queued work run and any starting-up or winding-down PTY event loop
/// finish, so measurements are taken in the test app's steady state.
async fn settle(cx: &mut TestAppContext) {
    cx.run_until_parked();
    cx.background_executor
        .timer(Duration::from_millis(100))
        .await;
    cx.run_until_parked();
}

/// Runs a short one-shot command to completion in a task terminal, then shuts
/// the PTY down and returns the still-referenced terminal.
///
/// The shape mirrors an agent tool call: `SandboxedTerminalTool` spawns its
/// command through `AcpThread::create_terminal`, and
/// `acp_thread::terminal::Terminal` calls `release_pty_resources` once the
/// command exits. What the test adds is the part the agent panel does — holding
/// on to the terminal afterwards so the finished command's output can still be
/// rendered.
async fn finished_task_terminal(cx: &mut TestAppContext) -> Entity<Terminal> {
    let program = "sleep".to_string();
    let args = vec!["0.2".to_string()];

    let mode = TerminalMode::task(SpawnInTerminal {
        command: Some(program.clone()),
        args: args.clone(),
        ..Default::default()
    });

    let builder = cx
        .update(|cx| {
            TerminalBuilder::new(
                None,
                mode,
                Shell::WithArguments {
                    program,
                    args,
                    title_override: None,
                },
                HashMap::default(),
                SettingsCursorShape::default(),
                AlternateScroll::On,
                None,
                Vec::new(),
                Duration::ZERO,
                false,
                0,
                cx,
                Vec::new(),
                PathStyle::local(),
            )
        })
        .await
        .expect("failed to spawn the test terminal");

    let terminal = cx.new(|cx| builder.subscribe(cx));

    terminal
        .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
        .await
        .expect("the command should have run to completion");

    terminal.update(cx, |terminal, _| terminal.release_pty_resources());
    cx.run_until_parked();

    terminal
}

#[gpui::test]
async fn test_finished_task_terminal_releases_its_parse_buffer(cx: &mut TestAppContext) {
    cx.update(|cx| {
        let settings_store = settings::SettingsStore::test(cx);
        cx.set_global(settings_store);
        theme_settings::init(theme::LoadThemes::JustBase, cx);
    });
    cx.executor().allow_parking();

    // Warm up on a couple of terminals so this test's one-time costs (executors,
    // settings, the first PTY event loop and its thread) are outside the delta
    // measured below.
    for _ in 0..2 {
        drop(finished_task_terminal(cx).await);
    }
    settle(cx).await;
    let baseline_bytes = live_bytes();

    // Hold every finished terminal, the way the agent panel holds the terminals
    // of the commands it has run.
    let mut finished = Vec::with_capacity(TERMINALS);
    for _ in 0..TERMINALS {
        finished.push(finished_task_terminal(cx).await);
    }
    settle(cx).await;

    let retained_bytes = live_bytes().saturating_sub(baseline_bytes);
    assert!(
        retained_bytes <= RETAINED_BUDGET_BYTES,
        "holding {TERMINALS} finished task terminals retained {retained_bytes} bytes \
         (budget {RETAINED_BUDGET_BYTES}), ~{} bytes per terminal, with {} \
         {VTE_SYNC_BUFFER_SIZE}-byte parse buffers live. `Terminal::new` builds a \
         `vte::ansi::Processor`, which reserves {VTE_SYNC_BUFFER_SIZE} bytes for \
         synchronized updates when it is constructed, and `release_pty_resources` tears the PTY \
         down without releasing that reservation — so every command that has already finished \
         still costs every holder of its terminal.",
        retained_bytes / TERMINALS,
        live_sync_buffers(),
    );

    drop(finished);
}
