/**
 * mep_bridge.h — Marine Edge Protocol: types, class declarations
 *
 * Include from both C++ and CUDA translation units.
 * Implementations are in mep_bridge.cpp.
 *
 * Author: Forgemaster ⚒️
 * License: MIT
 */

#pragma once

#include "marine_types.h"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>

// ============================================================
// CONSTRAINT-AWARE SCHEDULER
// ============================================================

struct ComputeNode {
    MEPDeviceInfo info;
    float         current_load;      // 0.0 – 1.0
    float         thermal_headroom;  // degrees until thermal throttle
    float         power_headroom;    // watts until power limit
    float         latency_ms;        // measured round-trip latency
    int           socket_fd;
    bool          active;
};

class ConstraintScheduler {
public:
    static const int MAX_NODES = 4;

    ComputeNode nodes[MAX_NODES];
    int         num_nodes = 0;

    /** Score a node for a given task. Higher = better fit. -1 = ineligible. */
    float score_node(int node_idx, const MEPTaskSubmit& task) const;

    /** Return index of best node, or -1 if none available. */
    int select_node(const MEPTaskSubmit& task) const;

    /** Register a newly-connected compute node. Returns false if table is full. */
    bool register_node(const MEPDeviceInfo& info, int fd);
};

// ============================================================
// MEP BRIDGE — client / server
// ============================================================

class MEPBridge {
public:
    int                listen_fd = -1;
    int                node_fd   = -1;
    ConstraintScheduler scheduler;
    uint32_t           next_seq  = 0;

    /** Start listening as a server (workstation mode). */
    bool start_server(int port = MEP_PORT);

    /** Connect to a remote node (edge mode). */
    bool connect_to(const char* host, int port = MEP_PORT);

    /**
     * Send a MEP message.
     * @param payload  May be nullptr when len == 0.
     * @return true on success; false and prints errno on error.
     */
    bool send_msg(int fd, MEPType type, const void* payload, uint32_t len);

    /**
     * Receive a MEP message.
     * @param buf      Caller-allocated receive buffer.
     * @param bufsize  Buffer capacity in bytes.
     * @return true on success; false on connection close or protocol error.
     */
    bool recv_msg(int fd, MEPHeader& hdr, void* buf, uint32_t bufsize);

    void close_all();

    ~MEPBridge() { close_all(); }

private:
    /** recv() wrapper that retries on EINTR and handles short reads. */
    static ssize_t recv_all(int fd, void* buf, size_t len);

    /** send() wrapper that retries on EINTR and handles short writes. */
    static ssize_t send_all(int fd, const void* buf, size_t len);
};
