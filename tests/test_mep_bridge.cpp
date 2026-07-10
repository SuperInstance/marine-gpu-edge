/**
 * test_mep_bridge.cpp — CPU-only unit tests for the MEP protocol bridge.
 *
 * These tests exercise the serialisation, framing, and scheduler logic in
 * src/mep_bridge.cpp without requiring a CUDA toolkit or a live network.
 * They use a local socketpair() for send/recv round-trips so the real
 * send_all()/recv_all() wrappers are exercised.
 *
 * Compile and run standalone with plain g++:
 *   g++ -std=c++17 -I include tests/test_mep_bridge.cpp src/mep_bridge.cpp -o tests/test_mep_bridge
 *   ./tests/test_mep_bridge
 *
 * Author: Forgemaster ⚒️
 * License: MIT
 */

#include "mep_bridge.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include <sys/socket.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Tiny test harness
// ---------------------------------------------------------------------------

static int g_tests_run   = 0;
static int g_tests_passed = 0;

#define RUN_TEST(fn)                                                        \
    do {                                                                    \
        g_tests_run++;                                                      \
        std::printf("  %-50s ", #fn);                                       \
        if ((fn)()) {                                                       \
            g_tests_passed++;                                               \
            std::printf("PASS\n");                                          \
        } else {                                                            \
            std::printf("FAIL\n");                                          \
        }                                                                   \
    } while (0)

#define REQUIRE(cond)                                                       \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "\n    assertion failed: %s (line %d)\n",  \
                         #cond, __LINE__);                                  \
            return false;                                                   \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

class SocketPair {
public:
    int send_fd = -1;
    int recv_fd = -1;

    explicit SocketPair() {
        int sv[2] = {-1, -1};
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0) {
            send_fd = sv[0];
            recv_fd = sv[1];
        }
    }

    ~SocketPair() {
        if (send_fd >= 0) ::close(send_fd);
        if (recv_fd >= 0) ::close(recv_fd);
    }

    bool ok() const { return send_fd >= 0 && recv_fd >= 0; }

private:
    SocketPair(const SocketPair&) = delete;
    SocketPair& operator=(const SocketPair&) = delete;
};

static MEPDeviceInfo make_device_info(const char* hostname,
                                      const char* gpu_name,
                                      uint32_t sm_version,
                                      uint64_t total_mem,
                                      uint64_t free_mem,
                                      float temperature,
                                      float power_usage,
                                      float power_limit,
                                      uint8_t is_jetson) {
    MEPDeviceInfo info = {};
    std::strncpy(info.hostname, hostname, sizeof(info.hostname) - 1);
    std::strncpy(info.gpu_name, gpu_name, sizeof(info.gpu_name) - 1);
    info.sm_version   = sm_version;
    info.total_memory = total_mem;
    info.free_memory  = free_mem;
    info.temperature  = temperature;
    info.power_usage  = power_usage;
    info.power_limit  = power_limit;
    info.is_jetson    = is_jetson;
    info.max_threads  = 1024;
    info.clock_rate   = 1000000;
    info.num_sms      = 16;
    info.compute_mode = 0;
    return info;
}

static MEPTaskSubmit make_task(uint32_t grid_x, uint32_t block_x,
                               uint32_t input_size, uint32_t output_size,
                               uint8_t precision, uint8_t flags,
                               float priority, float deadline,
                               uint32_t shared_mem = 0) {
    MEPTaskSubmit task = {};
    task.task_id    = 1;
    task.kernel_id  = 1;
    task.grid_x     = grid_x;
    task.grid_y     = 1;
    task.grid_z     = 1;
    task.block_x    = block_x;
    task.block_y    = 1;
    task.block_z    = 1;
    task.shared_mem = shared_mem;
    task.input_addr = 0;
    task.output_addr = 0;
    task.input_size = input_size;
    task.output_size = output_size;
    task.priority   = priority;
    task.deadline   = deadline;
    task.precision  = precision;
    task.flags      = flags;
    return task;
}

// ---------------------------------------------------------------------------
// Header tests
// ---------------------------------------------------------------------------

static bool test_header_size_is_16_bytes() {
    REQUIRE(sizeof(MEPHeader) == 16);
    return true;
}

static bool test_header_serialisation_roundtrip() {
    MEPHeader hdr = {};
    hdr.magic   = MEP_MAGIC;
    hdr.version = MEP_VERSION;
    hdr.type    = MEP_DEVICE_INFO;
    hdr.length  = 1024;
    hdr.seq     = 0xDEADBEEFu;

    unsigned char bytes[16];
    std::memcpy(bytes, &hdr, sizeof(hdr));

    MEPHeader decoded = {};
    std::memcpy(&decoded, bytes, sizeof(decoded));

    REQUIRE(decoded.magic   == MEP_MAGIC);
    REQUIRE(decoded.version == MEP_VERSION);
    REQUIRE(decoded.type    == MEP_DEVICE_INFO);
    REQUIRE(decoded.length  == 1024);
    REQUIRE(decoded.seq     == 0xDEADBEEFu);
    return true;
}

static bool test_header_bad_magic_is_rejected() {
    SocketPair sp;
    REQUIRE(sp.ok());

    MEPHeader hdr = {};
    hdr.magic   = 0xBADBADBAu; // wrong
    hdr.version = MEP_VERSION;
    hdr.type    = MEP_PING;
    hdr.length  = 0;
    hdr.seq     = 0;

    REQUIRE(::send(sp.send_fd, &hdr, sizeof(hdr), MSG_NOSIGNAL) == (ssize_t)sizeof(hdr));

    MEPBridge bridge;
    MEPHeader rx_hdr = {};
    char buf[64];
    REQUIRE(!bridge.recv_msg(sp.recv_fd, rx_hdr, buf, sizeof(buf)));
    return true;
}

static bool test_header_bad_version_is_rejected() {
    SocketPair sp;
    REQUIRE(sp.ok());

    MEPHeader hdr = {};
    hdr.magic   = MEP_MAGIC;
    hdr.version = 99; // unsupported
    hdr.type    = MEP_PING;
    hdr.length  = 0;
    hdr.seq     = 0;

    REQUIRE(::send(sp.send_fd, &hdr, sizeof(hdr), MSG_NOSIGNAL) == (ssize_t)sizeof(hdr));

    MEPBridge bridge;
    MEPHeader rx_hdr = {};
    char buf[64];
    REQUIRE(!bridge.recv_msg(sp.recv_fd, rx_hdr, buf, sizeof(buf)));
    return true;
}

// ---------------------------------------------------------------------------
// Send / receive framing tests (via local socketpair)
// ---------------------------------------------------------------------------

static bool test_send_recv_payload_roundtrip() {
    SocketPair sp;
    REQUIRE(sp.ok());

    const char payload[] = "Hello, MEP!";
    const uint32_t plen = static_cast<uint32_t>(std::strlen(payload));

    MEPBridge sender;
    REQUIRE(sender.send_msg(sp.send_fd, MEP_SENSOR_BATCH, payload, plen));

    MEPBridge receiver;
    MEPHeader hdr = {};
    char buf[64] = {};
    REQUIRE(receiver.recv_msg(sp.recv_fd, hdr, buf, sizeof(buf)));

    REQUIRE(hdr.magic   == MEP_MAGIC);
    REQUIRE(hdr.version == MEP_VERSION);
    REQUIRE(hdr.type    == MEP_SENSOR_BATCH);
    REQUIRE(hdr.length  == plen);
    REQUIRE(hdr.seq     == 0); // first message from fresh bridge
    REQUIRE(std::memcmp(buf, payload, plen) == 0);
    return true;
}

static bool test_send_recv_empty_payload() {
    SocketPair sp;
    REQUIRE(sp.ok());

    MEPBridge sender;
    REQUIRE(sender.send_msg(sp.send_fd, MEP_PING, nullptr, 0));

    MEPBridge receiver;
    MEPHeader hdr = {};
    char buf[8] = {};
    REQUIRE(receiver.recv_msg(sp.recv_fd, hdr, buf, sizeof(buf)));

    REQUIRE(hdr.type   == MEP_PING);
    REQUIRE(hdr.length == 0);
    return true;
}

static bool test_oversized_payload_is_truncated_not_undefined() {
    SocketPair sp;
    REQUIRE(sp.ok());

    const char payload[] = "this payload is longer than the receive buffer";
    const uint32_t plen = static_cast<uint32_t>(std::strlen(payload));

    MEPBridge sender;
    REQUIRE(sender.send_msg(sp.send_fd, MEP_ALERT, payload, plen));

    MEPBridge receiver;
    MEPHeader hdr = {};
    char buf[16] = {}; // smaller than payload
    REQUIRE(receiver.recv_msg(sp.recv_fd, hdr, buf, sizeof(buf)));

    REQUIRE(hdr.length == plen);
    // recv_msg() uses std::min(length, bufsize) and returns true on a short
    // read rather than failing.  Confirm the available bytes were received
    // safely and no crash occurred.
    REQUIRE(std::memcmp(buf, payload, sizeof(buf)) == 0);
    return true;
}

// ---------------------------------------------------------------------------
// Constraint scheduler tests
// ---------------------------------------------------------------------------

static bool test_scheduler_register_bounds() {
    ConstraintScheduler sched;
    REQUIRE(sched.num_nodes == 0);

    MEPDeviceInfo info = make_device_info(
        "node", "GPU", 0x0890,
        8ULL * 1024 * 1024 * 1024,
        6ULL * 1024 * 1024 * 1024,
        45.0f, 35.0f, 115.0f, 0);

    REQUIRE(sched.register_node(info, 10));
    REQUIRE(sched.register_node(info, 11));
    REQUIRE(sched.register_node(info, 12));
    REQUIRE(sched.register_node(info, 13));
    REQUIRE(!sched.register_node(info, 14)); // table full
    REQUIRE(sched.num_nodes == ConstraintScheduler::MAX_NODES);
    return true;
}

static bool test_scheduler_inactive_node_is_ineligible() {
    ConstraintScheduler sched;
    MEPDeviceInfo info = make_device_info(
        "node", "GPU", 0x0890,
        8ULL * 1024 * 1024 * 1024,
        6ULL * 1024 * 1024 * 1024,
        45.0f, 35.0f, 115.0f, 0);

    REQUIRE(sched.register_node(info, 10));
    sched.nodes[0].active = false;

    MEPTaskSubmit task = make_task(64, 64, 1024, 1024, PREC_FP32, 0, 0.5f, 0.0f);
    REQUIRE(sched.score_node(0, task) == -1.0f);
    REQUIRE(sched.select_node(task) == -1);
    return true;
}

static bool test_scheduler_rejects_thermal_throttle() {
    ConstraintScheduler sched;
    // Hot Jetson: temperature + estimated rise exceeds 85 °C.
    MEPDeviceInfo jetson = make_device_info(
        "jetson", "Orin", 0x0870,
        8ULL * 1024 * 1024 * 1024,
        4ULL * 1024 * 1024 * 1024,
        82.0f, 10.0f, 15.0f, 1);

    REQUIRE(sched.register_node(jetson, 10));

    MEPTaskSubmit task = make_task(256, 256, 1024, 1024, PREC_FP16, 0, 0.5f, 0.1f);
    REQUIRE(sched.score_node(0, task) == -1.0f);
    REQUIRE(sched.select_node(task) == -1);
    return true;
}

static bool test_scheduler_rejects_insufficient_memory() {
    ConstraintScheduler sched;
    MEPDeviceInfo node = make_device_info(
        "desktop", "RTX", 0x0890,
        8ULL * 1024 * 1024 * 1024,
        1ULL * 1024 * 1024 * 1024,
        45.0f, 35.0f, 115.0f, 0);

    REQUIRE(sched.register_node(node, 10));

    // Needs more memory than free_memory
    MEPTaskSubmit task = make_task(64, 64, 2ULL * 1024 * 1024 * 1024, 0, PREC_FP32, 0, 0.5f, 0.0f);
    REQUIRE(sched.score_node(0, task) == -1.0f);
    REQUIRE(sched.select_node(task) == -1);
    return true;
}

static bool test_scheduler_prefers_jetson_for_fp16() {
    ConstraintScheduler sched;

    MEPDeviceInfo desktop = make_device_info(
        "eileen", "RTX 4050", 0x0890,
        8ULL * 1024 * 1024 * 1024,
        6ULL * 1024 * 1024 * 1024,
        45.0f, 35.0f, 115.0f, 0);

    MEPDeviceInfo jetson = make_device_info(
        "jetsonclaw1", "Orin Nano", 0x0870,
        8ULL * 1024 * 1024 * 1024,
        5ULL * 1024 * 1024 * 1024,
        52.0f, 10.0f, 15.0f, 1);

    REQUIRE(sched.register_node(desktop, 10));
    REQUIRE(sched.register_node(jetson, 11));

    // Identical load/latency so the FP16 bonus decides the winner.
    sched.nodes[0].current_load = 0.0f;
    sched.nodes[1].current_load = 0.0f;
    sched.nodes[0].latency_ms   = 0.1f;
    sched.nodes[1].latency_ms   = 0.1f;

    MEPTaskSubmit task = make_task(128, 128, 1024 * 1024, 1024 * 1024,
                                   PREC_FP16, 0, 0.8f, 0.1f);
    int best = sched.select_node(task);
    REQUIRE(best == 1); // Jetson wins for FP16
    return true;
}

static bool test_scheduler_prefers_workstation_for_ptx_kernel() {
    ConstraintScheduler sched;

    MEPDeviceInfo desktop = make_device_info(
        "eileen", "RTX 4050", 0x0890,
        8ULL * 1024 * 1024 * 1024,
        6ULL * 1024 * 1024 * 1024,
        45.0f, 35.0f, 115.0f, 0);

    // Hot Jetson so its thermal-headroom score does not swamp the modest
    // workstation bonus for PTX kernels (flags bit 2).
    MEPDeviceInfo jetson = make_device_info(
        "jetsonclaw1", "Orin Nano", 0x0870,
        8ULL * 1024 * 1024 * 1024,
        5ULL * 1024 * 1024 * 1024,
        82.0f, 10.0f, 15.0f, 1);

    REQUIRE(sched.register_node(desktop, 10));
    REQUIRE(sched.register_node(jetson, 11));

    sched.nodes[0].current_load = 0.0f;
    sched.nodes[1].current_load = 0.0f;
    sched.nodes[0].latency_ms   = 0.1f;
    sched.nodes[1].latency_ms   = 0.1f;

    // flags bit 2 set => PTX kernel
    MEPTaskSubmit task = make_task(128, 128, 1024 * 1024, 1024 * 1024,
                                   PREC_FP32, 0x04, 0.5f, 0.0f);
    int best = sched.select_node(task);
    REQUIRE(best == 0); // workstation wins for PTX
    return true;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main() {
    std::printf("=== MEP Bridge CPU-only unit tests ===\n\n");

    RUN_TEST(test_header_size_is_16_bytes);
    RUN_TEST(test_header_serialisation_roundtrip);
    RUN_TEST(test_header_bad_magic_is_rejected);
    RUN_TEST(test_header_bad_version_is_rejected);
    RUN_TEST(test_send_recv_payload_roundtrip);
    RUN_TEST(test_send_recv_empty_payload);
    RUN_TEST(test_oversized_payload_is_truncated_not_undefined);
    RUN_TEST(test_scheduler_register_bounds);
    RUN_TEST(test_scheduler_inactive_node_is_ineligible);
    RUN_TEST(test_scheduler_rejects_thermal_throttle);
    RUN_TEST(test_scheduler_rejects_insufficient_memory);
    RUN_TEST(test_scheduler_prefers_jetson_for_fp16);
    RUN_TEST(test_scheduler_prefers_workstation_for_ptx_kernel);

    std::printf("\n%2d / %2d tests passed\n", g_tests_passed, g_tests_run);
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
