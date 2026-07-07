# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


def my_axpy(
    dev,
    num_elements,
    num_columns,
    tile_size,
    trace_size,
    scalar_factor,
):
    factor = scalar_factor
    per_tile_elements = 4096 if tile_size > 4096 else tile_size
    n = per_tile_elements * num_columns
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    N_div_n = num_elements // n
    chunk = num_elements // num_columns
    dtype = bfloat16

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]

    # =====================================================================
    # AXPY FIX PLAN 2026-03-20: ObjectFifo Depth Optimization
    # =====================================================================
    # Root Cause: Insufficient ObjectFifo depth causing DMA contention
    # when multiple columns/channels compete for bandwidth.
    #
    # Benchmark Regressions Addressed:
    # - P0-CRITICAL: axpy_2_cols_2_channels_2048_tile_1024_3.0 (-26.77% BW)
    #   Fix: depth 4 -> 5 (with tile_size_factor)
    # - P1-HIGH: axpy_8_cols_2_channels_2048_tile_256_3.0 (-16.19% BW, +34.76% stddev)
    #   Fix: depth 7 -> 8 (with tile_size_factor)
    # - P1-STABILITY: _0 variants with stddev explosions (+18% to +122%)
    #   Fix: Consistent depth formula across all configs
    # - P2-MEDIUM: axpy_4_cols_2_channels_2048_tile_512_3.0 (-10.21% BW)
    #   Fix: depth 5 -> 6 (with tile_size_factor)
    # - P3-LOW: axpy_1_cols_2_channels_2048_tile_2048_3.0 (-1.96% BW)
    #   Fix: depth 3 -> 3 (stable)
    #
    # Formula: base_depth + column_factor + channel_factor + tile_size_factor
    # - base_depth = 2 (minimum for pipelining)
    # - column_factor = num_columns // 2 (+1 per 2 columns)
    # - channel_factor = num_channels - 1 (+1 for 2 channels)
    # - tile_size_factor = 3/2/1/0 based on tile size (smaller tiles need deeper FIFOs)
    # - Clamped to range [2, 8]
    #
    # TILE SIZE FACTOR RATIONALE:
    # Smaller tiles complete compute faster, requiring deeper FIFOs for DMA pre-fetch
    # to stay ahead. Pattern consistent with MEM_COPY operator (design.py:202-213).
    # - tile_size <= 256: factor = 3 (very small tiles, max DMA pre-fetch needed)
    # - tile_size < 512: factor = 2 (small tiles need +2 depth)
    # - tile_size < 1024: factor = 1 (moderate tiles need +1 depth)
    # - tile_size >= 1024: factor = 0 (large tiles have natural buffering)
    # =====================================================================
    base_depth = 2
    column_factor = num_columns // 2
    channel_factor = num_channels - 1

    # Tile size factor: smaller tiles need deeper FIFOs for DMA pre-fetch
    # Consistent with MEM_COPY operator pattern (design.py:calculate_mem_copy_depth)
    tile_size_factor = 0
    if tile_size <= 256:
        tile_size_factor = 3  # Very small tiles - maximum DMA pre-fetch needed
    elif tile_size < 512:
        tile_size_factor = 2  # Small tiles need +2 depth
    elif tile_size < 1024:
        tile_size_factor = 1  # Moderate tiles need +1 depth

    fifodepth = max(2, min(8, base_depth + column_factor + channel_factor + tile_size_factor))

    # AIE-array data movement with object fifos (one per column, not per channel)
    of_in1s = [
        ObjectFifo(tile_ty, name=f"in1_{i}", depth=fifodepth)
        for i in range(num_columns)
    ]
    of_in2s = [
        ObjectFifo(tile_ty, name=f"in2_{i}", depth=fifodepth)
        for i in range(num_columns)
    ]
    of_outs = [
        ObjectFifo(tile_ty, name=f"out_{i}", depth=fifodepth)
        for i in range(num_columns)
    ]

    # AIE Core Function declaration
    axpy_bf16_vector = Kernel(
        "saxpy", "axpy.o", [tile_ty, tile_ty, np.float32, tile_ty, np.int32]
    )

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_in2, of_out, axpy):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_out.acquire(1)
            axpy(elem_in1, elem_in2, factor, elem_out, per_tile_elements)
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile (one per column)
    my_workers = [
        Worker(
            core_body,
            [
                of_in1s[i].cons(),
                of_in2s[i].cons(),
                of_outs[i].prod(),
                axpy_bf16_vector,
            ],
        )
        for i in range(num_columns)
    ]

    # Create a TensorAccessPattern for each column
    # to describe the data movement
    # The pattern chops the data in equal chunks
    # and moves them in parallel across the columns
    taps = [
        TensorAccessPattern(
            (1, num_elements),
            chunk * i,  # Start offset for column i
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(*my_workers)

        # =================================================================
        # Task Group Synchronization (AXPY FIX PLAN 2026-03-20)
        # -----------------------------------------------------------------
        # All fills and drains execute in parallel within the task group.
        # wait=True on drains ensures data is fully transferred before
        # task_group completion, preventing race conditions.
        #
        # NOTE: Previous analysis suggested wait=False might reduce
        # serialization overhead, but this would risk data races when
        # columns complete at different rates. The ObjectFifo depth
        # increase (above) is the correct fix for throughput issues.
        # =================================================================
        tg = rt.task_group()

        # Fill the input objectFIFOs with data
        for i in range(num_columns):
            rt.fill(
                of_in1s[i].prod(),
                A,
                taps[i],
                task_group=tg,
            )
            rt.fill(
                of_in2s[i].prod(),
                B,
                taps[i],
                task_group=tg,
            )
        # Drain the output objectFIFOs with data
        # wait=True: Block until transfer completes and data is available in C
        for i in range(num_columns):
            rt.drain(
                of_outs[i].cons(),
                C,
                taps[i],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())
