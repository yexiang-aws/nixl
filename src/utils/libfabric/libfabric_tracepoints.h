/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * NIXL Telemetry-based tracepoints for libfabric backend.
 *
 * Replaces LTTng-UST tracepoints with NIXL built-in telemetry system.
 * Events are emitted via nixlBackendEngine::addTelemetryEvent() and exported
 * through the configured telemetry exporter (shared memory buffer by default).
 *
 * Enable: set NIXL_TELEMETRY_ENABLE=1 and optionally NIXL_TELEMETRY_DIR=/path
 * Read:   use the telemetry_reader example or any compatible exporter consumer.
 *
 * Thread safety: g_telemetry_enabled (atomic) guards access to the callback.
 * Registration sets callback first, then enables. Teardown disables first,
 * then clears callback. This ensures no thread invokes a stale callback.
 */

#ifndef NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_TRACEPOINTS_H
#define NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_TRACEPOINTS_H

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <functional>

#define NIXL_TP_OP_WRITE 0
#define NIXL_TP_OP_READ 1

using nixl_telemetry_cb_t = std::function<void(const char *event_name, uint64_t value)>;

extern nixl_telemetry_cb_t g_nixl_libfabric_telemetry_cb;
extern std::atomic<bool> g_nixl_libfabric_telemetry_enabled;

/*
 * Bit-packing layouts for uint64_t event values.
 *
 * post_wr/rd/snd_begin: [dev:8 | rail:16 | xid:16 | reserved:24]
 *   - len is sent as a separate "_sz" event (full 64-bit) to avoid truncation
 * post_wr/rd/snd_end:   [dev:8 | rail:16 | xid:16 | retries:24]
 * local_comp:            [dev:8 | reserved:16 | rail:16 | op:8 | xid:16]
 * remote_wr_comp/recv:   [rail:16 | aidx:16 | xid:16 | len_lo:16]
 */

#define NIXL_TRACE_TRANSFER_BEGIN(dev, op, sz, nr, stripe, xid) \
    do { if (g_nixl_libfabric_telemetry_enabled.load(std::memory_order_acquire)) { \
        g_nixl_libfabric_telemetry_cb("xfer_begin", \
            ((uint64_t)(uint8_t)(dev) << 56) | ((uint64_t)(uint8_t)(op) << 48) | \
            ((uint64_t)(uint8_t)(nr) << 40) | ((uint64_t)(uint8_t)(stripe) << 32) | \
            ((uint64_t)(uint16_t)(xid) << 16)); \
        g_nixl_libfabric_telemetry_cb("xfer_begin_sz", (uint64_t)(sz)); \
    } } while(0)

#define NIXL_TRACE_TRANSFER_SUBMITTED(dev, cnt, xid) \
    do { if (g_nixl_libfabric_telemetry_enabled.load(std::memory_order_acquire)) { \
        g_nixl_libfabric_telemetry_cb("xfer_submitted", \
            ((uint64_t)(uint8_t)(dev) << 40) | ((uint64_t)(cnt) << 16) | (uint64_t)(uint16_t)(xid)); \
    } } while(0)

#define NIXL_TRACE_POST_WRITE_BEGIN(dev, rail, len, xid) \
    do { if (g_nixl_libfabric_telemetry_enabled.load(std::memory_order_acquire)) { \
        g_nixl_libfabric_telemetry_cb("post_wr_begin", \
            ((uint64_t)(uint8_t)(dev) << 56) | ((uint64_t)(uint16_t)(rail) << 40) | \
            ((uint64_t)(uint16_t)(xid) << 24)); \
        g_nixl_libfabric_telemetry_cb("post_wr_begin_sz", (uint64_t)(len)); \
    } } while(0)

#define NIXL_TRACE_POST_WRITE_END(dev, rail, len, retries, xid) \
    do { if (g_nixl_libfabric_telemetry_enabled.load(std::memory_order_acquire)) { \
        g_nixl_libfabric_telemetry_cb("post_wr_end", \
            ((uint64_t)(uint8_t)(dev) << 56) | ((uint64_t)(uint16_t)(rail) << 40) | \
            ((uint64_t)(uint16_t)(xid) << 24) | ((uint64_t)(retries) & 0xFFFFFF)); \
    } } while(0)

#define NIXL_TRACE_POST_READ_BEGIN(dev, rail, len, xid) \
    do { if (g_nixl_libfabric_telemetry_enabled.load(std::memory_order_acquire)) { \
        g_nixl_libfabric_telemetry_cb("post_rd_begin", \
            ((uint64_t)(uint8_t)(dev) << 56) | ((uint64_t)(uint16_t)(rail) << 40) | \
            ((uint64_t)(uint16_t)(xid) << 24)); \
        g_nixl_libfabric_telemetry_cb("post_rd_begin_sz", (uint64_t)(len)); \
    } } while(0)

#define NIXL_TRACE_POST_READ_END(dev, rail, len, retries, xid) \
    do { if (g_nixl_libfabric_telemetry_enabled.load(std::memory_order_acquire)) { \
        g_nixl_libfabric_telemetry_cb("post_rd_end", \
            ((uint64_t)(uint8_t)(dev) << 56) | ((uint64_t)(uint16_t)(rail) << 40) | \
            ((uint64_t)(uint16_t)(xid) << 24) | ((uint64_t)(retries) & 0xFFFFFF)); \
    } } while(0)

#define NIXL_TRACE_POST_SEND_BEGIN(dev, rail, len, xid) \
    do { if (g_nixl_libfabric_telemetry_enabled.load(std::memory_order_acquire)) { \
        g_nixl_libfabric_telemetry_cb("post_snd_begin", \
            ((uint64_t)(uint8_t)(dev) << 56) | ((uint64_t)(uint16_t)(rail) << 40) | \
            ((uint64_t)(uint16_t)(xid) << 24)); \
        g_nixl_libfabric_telemetry_cb("post_snd_begin_sz", (uint64_t)(len)); \
    } } while(0)

#define NIXL_TRACE_POST_SEND_END(dev, rail, len, retries, xid) \
    do { if (g_nixl_libfabric_telemetry_enabled.load(std::memory_order_acquire)) { \
        g_nixl_libfabric_telemetry_cb("post_snd_end", \
            ((uint64_t)(uint8_t)(dev) << 56) | ((uint64_t)(uint16_t)(rail) << 40) | \
            ((uint64_t)(uint16_t)(xid) << 24) | ((uint64_t)(retries) & 0xFFFFFF)); \
    } } while(0)

#define NIXL_TRACE_LOCAL_TRANSFER_COMPLETION(dev, rail, op, xid) \
    do { if (g_nixl_libfabric_telemetry_enabled.load(std::memory_order_acquire)) { \
        g_nixl_libfabric_telemetry_cb("local_comp", \
            ((uint64_t)(uint8_t)(dev) << 40) | ((uint64_t)(uint16_t)(rail) << 24) | \
            ((uint64_t)(uint8_t)(op) << 16) | (uint64_t)(uint16_t)(xid)); \
    } } while(0)

#define NIXL_TRACE_REMOTE_WRITE_COMPLETION(rail, aidx, xid, len) \
    do { if (g_nixl_libfabric_telemetry_enabled.load(std::memory_order_acquire)) { \
        g_nixl_libfabric_telemetry_cb("remote_wr_comp", \
            ((uint64_t)(uint16_t)(rail) << 48) | ((uint64_t)(uint16_t)(aidx) << 32) | \
            ((uint64_t)(uint16_t)(xid) << 16) | ((uint64_t)(len) & 0xFFFF)); \
    } } while(0)

#define NIXL_TRACE_RECV_COMPLETION(rail, aidx, xid, len) \
    do { if (g_nixl_libfabric_telemetry_enabled.load(std::memory_order_acquire)) { \
        g_nixl_libfabric_telemetry_cb("recv_comp", \
            ((uint64_t)(uint16_t)(rail) << 48) | ((uint64_t)(uint16_t)(aidx) << 32) | \
            ((uint64_t)(uint16_t)(xid) << 16) | ((uint64_t)(len) & 0xFFFF)); \
    } } while(0)

#endif
