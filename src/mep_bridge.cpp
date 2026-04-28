/**
 * mep_bridge.cpp — Marine Edge Protocol implementations
 *
 * Author: Forgemaster ⚒️
 * License: MIT
 */

#include "mep_bridge.h"

#include <cstdlib>
#include <algorithm>

// ============================================================
// HELPERS — reliable send / recv over TCP
// ============================================================

ssize_t MEPBridge::send_all(int fd, const void* buf, size_t len) {
    const char* ptr  = static_cast<const char*>(buf);
    size_t      sent = 0;
    while (sent < len) {
        ssize_t n = ::send(fd, ptr + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return (ssize_t)sent; // connection closed
        sent += (size_t)n;
    }
    return (ssize_t)sent;
}

ssize_t MEPBridge::recv_all(int fd, void* buf, size_t len) {
    char*  ptr     = static_cast<char*>(buf);
    size_t recvd   = 0;
    while (recvd < len) {
        ssize_t n = ::recv(fd, ptr + recvd, len - recvd, MSG_WAITALL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return (ssize_t)recvd; // connection closed
        recvd += (size_t)n;
    }
    return (ssize_t)recvd;
}

// ============================================================
// CONSTRAINT SCHEDULER
// ============================================================

float ConstraintScheduler::score_node(int idx, const MEPTaskSubmit& task) const {
    const ComputeNode& node = nodes[idx];
    if (!node.active) return -1.0f;

    float score = 0.0f;

    // Thermal headroom (critical on Jetson)
    if (node.info.is_jetson) {
        score += node.thermal_headroom * 2.0f;

        // Rough temperature estimate for this task's workload
        float est_rise = 5.0f * (float)task.grid_x * task.block_x / 65536.0f;
        if (node.info.temperature + est_rise > 85.0f) {
            return -1.0f; // would thermal-throttle
        }
    } else {
        score += 10.0f; // desktop has more thermal budget
    }

    // Memory fit
    uint64_t mem_needed = (uint64_t)task.input_size + task.output_size + task.shared_mem;
    if (mem_needed > node.info.free_memory) return -1.0f;
    score += 20.0f * (1.0f - (float)mem_needed / (float)(node.info.total_memory | 1));

    // Load balancing
    score += 15.0f * (1.0f - node.current_load);

    // Deadline-aware latency bonus
    if (task.deadline > 0.0f) {
        score += 10.0f / (1.0f + node.latency_ms);
        score += task.priority * 5.0f;
    }

    // Jetson has excellent FP16 tensor-core throughput
    if (task.precision == (uint8_t)PREC_FP16 && node.info.is_jetson) {
        score += 8.0f;
    }

    // PTX forward-compatibility: sm_87 PTX runs on sm_89, not vice versa.
    // Prefer the workstation for PTX kernels so we never ship sm_89 PTX to Jetson.
    if (task.flags & 0x04) {
        if (!node.info.is_jetson) score += 3.0f;
    }

    return score;
}

int ConstraintScheduler::select_node(const MEPTaskSubmit& task) const {
    int   best       = -1;
    float best_score = -1.0f;

    for (int i = 0; i < num_nodes; i++) {
        float s = score_node(i, task);
        if (s > best_score) {
            best_score = s;
            best       = i;
        }
    }
    return best;
}

bool ConstraintScheduler::register_node(const MEPDeviceInfo& info, int fd) {
    if (num_nodes >= MAX_NODES) return false;

    ComputeNode& node = nodes[num_nodes++];
    node.info             = info;
    node.socket_fd        = fd;
    node.active           = true;
    node.current_load     = 0.0f;
    node.latency_ms       = 0.1f;
    node.thermal_headroom = 85.0f - info.temperature;
    node.power_headroom   = info.power_limit - info.power_usage;
    return true;
}

// ============================================================
// MEP BRIDGE
// ============================================================

bool MEPBridge::start_server(int port) {
    listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket");
        return false;
    }

    int opt = 1;
    if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt(SO_REUSEADDR)");
        ::close(listen_fd); listen_fd = -1;
        return false;
    }

    struct sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons((uint16_t)port);

    if (::bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        ::close(listen_fd); listen_fd = -1;
        return false;
    }

    if (::listen(listen_fd, 3) < 0) {
        perror("listen");
        ::close(listen_fd); listen_fd = -1;
        return false;
    }

    fprintf(stdout, "MEP Bridge listening on port %d\n", port);
    return true;
}

bool MEPBridge::connect_to(const char* host, int port) {
    node_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (node_fd < 0) {
        perror("socket");
        return false;
    }

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)port);

    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        fprintf(stderr, "Invalid address: %s\n", host);
        ::close(node_fd); node_fd = -1;
        return false;
    }

    if (::connect(node_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect");
        ::close(node_fd); node_fd = -1;
        return false;
    }

    fprintf(stdout, "Connected to %s:%d\n", host, port);
    return true;
}

bool MEPBridge::send_msg(int fd, MEPType type, const void* payload, uint32_t len) {
    MEPHeader hdr = {};
    hdr.magic   = MEP_MAGIC;
    hdr.version = MEP_VERSION;
    hdr.type    = (uint16_t)type;
    hdr.length  = len;
    hdr.seq     = next_seq++;

    if (send_all(fd, &hdr, sizeof(hdr)) != sizeof(hdr)) {
        perror("send_msg: header");
        return false;
    }

    if (len > 0 && payload != nullptr) {
        if (send_all(fd, payload, len) != (ssize_t)len) {
            perror("send_msg: payload");
            return false;
        }
    }

    return true;
}

bool MEPBridge::recv_msg(int fd, MEPHeader& hdr, void* buf, uint32_t bufsize) {
    ssize_t n = recv_all(fd, &hdr, sizeof(hdr));
    if (n != sizeof(hdr)) {
        if (n == 0) return false; // clean close
        perror("recv_msg: header");
        return false;
    }

    if (hdr.magic != MEP_MAGIC) {
        fprintf(stderr, "MEP: bad magic 0x%08X (expected 0x%08X)\n",
                hdr.magic, MEP_MAGIC);
        return false;
    }

    if (hdr.version != MEP_VERSION) {
        fprintf(stderr, "MEP: unsupported version %u\n", hdr.version);
        return false;
    }

    if (hdr.length > 0) {
        if (buf == nullptr || bufsize == 0) {
            fprintf(stderr, "MEP: no buffer provided for %u-byte payload\n", hdr.length);
            return false;
        }
        uint32_t to_read = std::min(hdr.length, bufsize);
        ssize_t  r       = recv_all(fd, buf, to_read);
        if (r != (ssize_t)to_read) {
            perror("recv_msg: payload");
            return false;
        }
    }

    return true;
}

void MEPBridge::close_all() {
    if (listen_fd >= 0) { ::close(listen_fd); listen_fd = -1; }
    if (node_fd   >= 0) { ::close(node_fd);   node_fd   = -1; }
}

// ============================================================
// STANDALONE TEST
// ============================================================

#ifdef MEP_STANDALONE_TEST

#include <thread>
#include <chrono>

static void server_thread_func() {
    MEPBridge server;
    if (!server.start_server()) return;

    fprintf(stdout, "Server waiting for connection...\n");

    struct sockaddr_in caddr = {};
    socklen_t clen = sizeof(caddr);
    int cfd = ::accept(server.listen_fd, (struct sockaddr*)&caddr, &clen);
    if (cfd < 0) { perror("accept"); return; }

    fprintf(stdout, "Client connected!\n");

    // Send our device info
    MEPDeviceInfo info = {};
    strncpy(info.hostname, "eileen-rtx4050",     sizeof(info.hostname) - 1);
    strncpy(info.gpu_name, "NVIDIA RTX 4050 Ada", sizeof(info.gpu_name) - 1);
    info.sm_version   = 0x0890;
    info.total_memory = 8ULL * 1024 * 1024 * 1024;
    info.free_memory  = 6ULL * 1024 * 1024 * 1024;
    info.temperature  = 45.0f;
    info.power_usage  = 35.0f;
    info.power_limit  = 115.0f;
    info.is_jetson    = 0;

    server.send_msg(cfd, MEP_DEVICE_INFO, &info, sizeof(info));
    fprintf(stdout, "Sent device info\n");

    MEPHeader hdr;
    char buf[1024];
    while (server.recv_msg(cfd, hdr, buf, sizeof(buf))) {
        fprintf(stdout, "RX type=0x%04X len=%u seq=%u\n",
                hdr.type, hdr.length, hdr.seq);

        if (hdr.type == MEP_PING) {
            server.send_msg(cfd, MEP_PONG, nullptr, 0);
            fprintf(stdout, "Sent PONG\n");

        } else if (hdr.type == MEP_DEVICE_INFO) {
            const MEPDeviceInfo* remote = reinterpret_cast<const MEPDeviceInfo*>(buf);
            fprintf(stdout, "Remote: %s / %s / SM %x / %llu MB / %.1f°C / %s\n",
                    remote->hostname, remote->gpu_name,
                    remote->sm_version,
                    (unsigned long long)(remote->total_memory / (1024*1024)),
                    remote->temperature,
                    remote->is_jetson ? "JETSON" : "DESKTOP");

            server.scheduler.register_node(*remote, cfd);

            // Test scheduler
            MEPTaskSubmit task = {};
            task.task_id    = 1;
            task.grid_x     = 256;
            task.block_x    = 256;
            task.input_size = 1024 * 1024;
            task.precision  = (uint8_t)PREC_FP16;
            task.priority   = 0.8f;
            task.deadline   = 0.1f;

            int best = server.scheduler.select_node(task);
            fprintf(stdout, "FP16 task scheduled to node %d\n", best);
        }
    }

    ::close(cfd);
}

static void client_thread_func() {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    MEPBridge client;
    if (!client.connect_to("127.0.0.1")) return;

    MEPHeader hdr;
    char buf[1024];

    // Receive server device info
    if (client.recv_msg(client.node_fd, hdr, buf, sizeof(buf))) {
        if (hdr.type == MEP_DEVICE_INFO) {
            const MEPDeviceInfo* info = reinterpret_cast<const MEPDeviceInfo*>(buf);
            fprintf(stdout, "Server: %s / %s / SM %x\n",
                    info->hostname, info->gpu_name, info->sm_version);
            client.scheduler.register_node(*info, client.node_fd);
        }
    }

    // Announce ourselves as Jetson
    MEPDeviceInfo jinfo = {};
    strncpy(jinfo.hostname, "jetsonclaw1-orin",    sizeof(jinfo.hostname) - 1);
    strncpy(jinfo.gpu_name, "Jetson Orin Nano 8GB", sizeof(jinfo.gpu_name) - 1);
    jinfo.sm_version   = 0x0870;
    jinfo.total_memory = 8ULL * 1024 * 1024 * 1024;
    jinfo.free_memory  = 5ULL * 1024 * 1024 * 1024;
    jinfo.temperature  = 52.0f;
    jinfo.power_usage  = 10.0f;
    jinfo.power_limit  = 15.0f;
    jinfo.is_jetson    = 1;

    client.send_msg(client.node_fd, MEP_PING, nullptr, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    client.send_msg(client.node_fd, MEP_DEVICE_INFO, &jinfo, sizeof(jinfo));
    fprintf(stdout, "Sent Jetson device info\n");
}

int main() {
    fprintf(stdout, "=== MEP Bridge Test ===\n");
    fprintf(stdout, "Marine Edge Protocol + constraint-aware scheduling\n\n");

    std::thread srv(server_thread_func);
    std::thread cli(client_thread_func);

    srv.join();
    cli.join();

    return 0;
}

#endif // MEP_STANDALONE_TEST
