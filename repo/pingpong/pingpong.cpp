#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <assert.h>
#include <limits.h>
#include <random>

#if defined(USE_CALIPER)
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#endif

#ifndef USE_CALIPER
#define CALI_CXX_MARK_FUNCTION
#define CALI_MARK_BEGIN(...)
#define CALI_MARK_END(...)
#endif

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif

#if defined(USE_CUDA)
#include <cuda_runtime.h>
inline void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#endif

// ---- Operation selector (default: PingPong) ----
enum class OpKind { All, PingPong, Alltoall, Reduce, Allreduce };

const char *get_hostname_for_rank(int rank, char all_hostnames[][1024],
                                  int size)
{
    if (rank >= 0 && rank < size)
        return all_hostnames[rank];
    else
        return "INVALID_RANK";
}

int extract_node_number(const char *hostname)
{
    int len = strlen(hostname);
    int num = 0;
    int factor = 1;
    for (int i = len - 1; i >= 0; --i)
    {
        if (hostname[i] >= '0' && hostname[i] <= '9')
        {
            num += (hostname[i] - '0') * factor;
            factor *= 10;
        }
        else
        {
            break;
        }
    }
    return num;
}

// Fill buf[0..len-1] with a repeating random pattern of length 16.
void fill_with_random_pattern(char* buf, size_t len)
{
    if (!buf || len == 0)
        return;

    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 25); // 'a'..'z'

    char pattern[16];
    for (int i = 0; i < 16; ++i) {
        pattern[i] = static_cast<char>('a' + dist(gen));
    }

    for (size_t i = 0; i < len; ++i) {
        buf[i] = pattern[i % 16];
    }
}

struct RankPair {
    int src;
    int dst;
};

static std::vector<RankPair>
build_pingpong_pairs(const std::string& region_label,
                     int size,
                     int sys_cores_per_socket,
                     int sys_cores_per_node,
                     int max_pairs)
{
    std::vector<RankPair> pairs;

    auto add_pair = [&](int s, int d) {
        if (s < 0 || d < 0 || s >= size || d >= size)
            return;
        pairs.push_back({s, d});
    };

    if (region_label == "Same Node Same Socket") {
        int rps = sys_cores_per_socket;
        if (rps < 2) return pairs;

        int s0 = 0;
        int s1 = rps / 4;
        int s2 = rps / 2;
        int s3 = rps - 2;

        add_pair(s0, s0 + 1);
        add_pair(s1, s1 + 1);
        add_pair(s2, s2 + 1);
        add_pair(s3, s3 + 1);

        return pairs;
    }

    if (region_label == "Same Node Different Socket") {
        int rps = sys_cores_per_socket;
        int delta = rps;

        if (2 * delta > size)
            return pairs; // not enough ranks

        int srcs[4] = { 0, delta / 4, delta / 2, delta - 1 };

        for (int s : srcs)
            add_pair(s, s + delta);

        return pairs;
    }

    int nodes_in_comm = 0;
    {
        std::stringstream ss(region_label);
        ss >> nodes_in_comm; // stops at "nodes"
    }

    if (nodes_in_comm >= 2) {
        int rpn   = sys_cores_per_node;
        int delta = (nodes_in_comm / 2) * rpn;

        // We spread srcs across the first node:
        int srcs[4] = { 0, rpn / 4, rpn / 2, rpn - 1 };

        for (int s : srcs)
            add_pair(s, s + delta);
    }

    if (max_pairs > 0 && (int)pairs.size() > max_pairs){
        pairs.resize(max_pairs);
    }

    return pairs;
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char my_hostname[1024];
    gethostname(my_hostname, 1023);
    my_hostname[1023] = '\0';
    char all_hostnames[size][1024];
    MPI_Gather(my_hostname, 1024, MPI_CHAR, all_hostnames, 1024, MPI_CHAR, 0,
               MPI_COMM_WORLD);

#if defined(USE_CALIPER)
    std::vector<std::string> all_comm_pairs;
    static std::map<int, cali::ConfigManager> mgr;
    MPI_Comm adiak_comm = MPI_COMM_WORLD;
    adiak::init(&adiak_comm);
    adiak::collect_all();
    CALI_CXX_MARK_FUNCTION;
#endif

    int PING_PONG_LIMIT = 10;
    const int WINDOW_SIZE = 1; // OSU-style window exchange; no SINGLE/MULTIPLE modes
    int msg_size = 1;
    int n_nodes = 1;
    int sys_cores_per_socket = 1;
    int sys_cores_per_node = 1;
    std::string metadata;
    const char *warmup_region = "warmup";
    const char *warmup_region_aa = "warmup_aa";
    const char *warmup_region_red = "warmup_red";
    const char *warmup_region_ar = "warmup_ar";
    int pingpong_num_pairs = 1;

    // ---- default to PingPong ----
    OpKind op = OpKind::PingPong;

    int opt;
    const char *usage =
        "Usage: %s [-h] [-i n-iterations] [-p rank1,rank2] [-m msg_sz] "
        "[-n n_nodes] [-s sys_cores_per_socket] [-c sys_cores_per_node] [-b metadata] "
        "[-O pingpong|alltoall|reduce|allreduce|all]\n"
        "Default: -O pingpong\n";

    while ((opt = getopt(argc, argv, "hi:p:m:n:s:c:b:O:")) != -1)
    {
        switch (opt)
        {
            case 'h':
                printf(usage, argv[0]);
                MPI_Finalize();
                return 0;
            case 'i':
                PING_PONG_LIMIT = atoi(optarg);
                break;
            case 'p':
                // kept for compatibility; parse partners here if desired
                break;
            case 'm':
                msg_size = atoi(optarg);
                break;
            case 'n':
                n_nodes = atoi(optarg);
                break;
            case 's':
                sys_cores_per_socket = atoi(optarg);
                break;
            case 'c':
                sys_cores_per_node = atoi(optarg);
                break;
            case 'b':
                metadata = optarg;
                break;
            case 'O':
            {
                std::string s = optarg ? std::string(optarg) : std::string();
                if (s == "pingpong")      op = OpKind::PingPong;
                else if (s == "alltoall") op = OpKind::Alltoall;
                else if (s == "reduce")   op = OpKind::Reduce;
                else if (s == "allreduce")op = OpKind::Allreduce;
                else if (s == "all")      op = OpKind::All;
                else {
                    if (rank == 0)
                        fprintf(stderr, "Unknown -O value '%s'. Expected pingpong|alltoall|reduce|allreduce|all\n", s.c_str());
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                break;
            }
            default:
                if (rank == 0) printf(usage, argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (rank == 0)
    {
        printf("Configuration:\n");
        printf("PING_PONG_LIMIT: %d\n", PING_PONG_LIMIT);
        printf("Message size: %d bytes\n", msg_size);
        printf("Cores per socket: %d\n", sys_cores_per_socket);
        printf("Cores per node: %d\n", sys_cores_per_node);
        printf("Nodes: %d\n", n_nodes);
        printf("World size: %d\n", size);
        printf("Mode (-O): %s\n",
               op == OpKind::PingPong ? "pingpong" :
               op == OpKind::Alltoall ? "alltoall" :
               op == OpKind::Reduce   ? "reduce" :
               op == OpKind::Allreduce? "allreduce" : "all");

#if defined(USE_CALIPER)
        std::stringstream rankmap;
        rankmap << "{";
        for (int i = 0; i < size; ++i)
        {
            rankmap << "\"" << i << "\": \"" << all_hostnames[i] << "\"";
            if (i < size - 1)
                rankmap << ", ";
        }
        rankmap << "}";
        adiak::value("rank_node_map", rankmap.str());
        adiak::value("iterations", PING_PONG_LIMIT);
        adiak::value("pingpong_num_pairs", pingpong_num_pairs);
#endif
    }

#if defined(USE_CALIPER)
    cali_id_t src_rank_attr = cali_create_attribute("src_rank", CALI_TYPE_INT,
                              CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t dest_rank_attr = cali_create_attribute("dest_rank", CALI_TYPE_INT,
                               CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t src_node_attr = cali_create_attribute("src_node", CALI_TYPE_INT,
                              CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t dest_node_attr = cali_create_attribute("dest_node", CALI_TYPE_INT,
                               CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t message_size_attr = cali_create_attribute("message_size_bytes",
                                  CALI_TYPE_INT, CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t comm_phase_attr = cali_create_attribute("comm_phase", CALI_TYPE_STRING,
                                CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t aa_avg_time_sec_attr = cali_create_attribute("aa_avg_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t aa_max_time_sec_attr = cali_create_attribute("aa_max_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t aa_min_time_sec_attr = cali_create_attribute("aa_min_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t pp_avg_time_sec_attr = cali_create_attribute("pp_avg_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t pp_max_time_sec_attr = cali_create_attribute("pp_max_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t pp_min_time_sec_attr = cali_create_attribute("pp_min_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t red_avg_time_sec_attr = cali_create_attribute("red_avg_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t red_max_time_sec_attr = cali_create_attribute("red_max_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t red_min_time_sec_attr = cali_create_attribute("red_min_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t ar_avg_time_sec_attr  = cali_create_attribute("ar_avg_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t ar_max_time_sec_attr  = cali_create_attribute("ar_max_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t ar_min_time_sec_attr  = cali_create_attribute("ar_min_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);

    const char *src_dest_attributes = R"json(
        {
            "name": "pingpong_attributes",
            "type": "boolean",
            "category": "metric",
            "description": "Collect pingpong attributes",
            "query":
            [
            {
                "level": "local",
                "select":
                [
                {"expr": "any(max#src_rank)", "as": "src_rank"},
                {"expr": "any(max#dest_rank)", "as": "dest_rank"},
                {"expr": "any(max#src_node)", "as": "src_node"},
                {"expr": "any(max#dest_node)", "as": "dest_node"},
                {"expr": "any(max#message_size_bytes)", "as": "message_size_bytes"},
                {"expr": "any(max#aa_avg_time_sec)", "as" : "aa_avg_s"},
                {"expr": "any(max#aa_max_time_sec)", "as" : "aa_max_s"},
                {"expr": "any(max#aa_min_time_sec)", "as" : "aa_min_s"},
                {"expr": "any(max#pp_avg_time_sec)", "as" : "pp_avg_s"},
                {"expr": "any(max#pp_max_time_sec)", "as" : "pp_max_s"},
                {"expr": "any(max#pp_min_time_sec)", "as" : "pp_min_s"},
                {"expr": "any(max#red_avg_time_sec)", "as" : "red_avg_s"},
                {"expr": "any(max#red_max_time_sec)", "as" : "red_max_s"},
                {"expr": "any(max#red_min_time_sec)", "as" : "red_min_s"},
                {"expr": "any(max#ar_avg_time_sec)", "as" : "ar_avg_s"},
                {"expr": "any(max#ar_max_time_sec)", "as" : "ar_max_s"},
                {"expr": "any(max#ar_min_time_sec)", "as" : "ar_min_s"}
                ],
                "group by": ["comm_phase"],
            },
            {
                "level": "cross",
                "select":
                [
                {"expr": "any(any#max#src_rank)", "as": "src_rank"},
                {"expr": "any(any#max#dest_rank)", "as": "dest_rank"},
                {"expr": "any(any#max#src_node)", "as": "src_node"},
                {"expr": "any(any#max#dest_node)", "as": "dest_node"},
                {"expr": "any(any#max#message_size_bytes)", "as": "message_size_bytes"},
                {"expr": "any(any#max#aa_avg_time_sec)", "as" : "aa_avg_s"},
                {"expr": "any(any#max#aa_max_time_sec)", "as" : "aa_max_s"},
                {"expr": "any(any#max#aa_min_time_sec)", "as" : "aa_min_s"},
                {"expr": "any(any#max#pp_avg_time_sec)", "as" : "pp_avg_s"},
                {"expr": "any(any#max#pp_max_time_sec)", "as" : "pp_max_s"},
                {"expr": "any(any#max#pp_min_time_sec)", "as" : "pp_min_s"},
                {"expr": "any(any#max#red_avg_time_sec)", "as" : "red_avg_s"},
                {"expr": "any(any#max#red_max_time_sec)", "as" : "red_max_s"},
                {"expr": "any(any#max#red_min_time_sec)", "as" : "red_min_s"},
                {"expr": "any(any#max#ar_avg_time_sec)", "as" : "ar_avg_s"},
                {"expr": "any(any#max#ar_max_time_sec)", "as" : "ar_max_s"},
                {"expr": "any(any#max#ar_min_time_sec)", "as" : "ar_min_s"}
                ],
                "group by": ["comm_phase"],
            }
            ]
        }
        )json";
#endif

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&now_time), "%Y%m%d_%H%M%S");

    int P = sys_cores_per_node * n_nodes;
    std::vector<int> partners;
    std::map<int, std::string> region_names;

    int current_nodes = n_nodes;
    int current_p = P;

    while (current_p > 2)
    {
        int partner_rank = current_p - 1;
        std::string label;
        if (current_nodes >= 2)
            label = std::to_string(current_nodes) + " nodes";
        else
            label = "Same Node Different Socket";

        if (partner_rank > 0 && partner_rank < size)
        {
            partners.push_back(partner_rank);
            region_names[partner_rank] = label;
        }
        current_nodes = current_nodes / 2;
        current_p = current_nodes * sys_cores_per_node;
    }

    // Always add same node same socket as 0 <-> 1
    if (1 < size)
    {
        partners.push_back(1);
        region_names[1] = "Same Node Same Socket";
    }

    for (int message = msg_size; message <= pow(msg_size, 6); message *= 8)
    {
#if defined(USE_CALIPER)
        std::string profile = "spot(output=" + std::to_string(message) + "_" +
                              timestamp.str() +
                              ".cali, profile.mpi),metadata(file=" + metadata +
                              "),metadata(file=/etc/node_info.json,keys=\"host.os\")";

        cali_set_int(message_size_attr, message);

        mgr[message].add_option_spec(src_dest_attributes);
        mgr[message].set_default_parameter("pingpong_attributes", "true");
        adiak::value("message_size", message);
        mgr[message].add(profile.c_str());
        mgr[message].start();
#endif

        // ===================== PINGPONG =====================
        if (op == OpKind::PingPong || op == OpKind::All)
        {
            for (int partner_rank : partners)
            {
                std::string region_label = region_names[partner_rank];

                // Build up to N pairs for this region
                std::vector<RankPair> pairs =
                    build_pingpong_pairs(region_label,
                                         size,
                                         sys_cores_per_socket,
                                         sys_cores_per_node, pingpong_num_pairs);

                if (pairs.empty()) {
                    if (rank == 0)
                        printf("Skipping region %s: no valid pingpong pairs\n",
                               region_label.c_str());
                    continue;
                }

                // Map each rank to its partner (or MPI_PROC_NULL if not in any pair)
                std::vector<int> my_partner(size, MPI_PROC_NULL);
                for (auto &p : pairs) {
                    my_partner[p.src] = p.dst;
                    my_partner[p.dst] = p.src;
                }

                int partner = my_partner[rank];
                if (partner == MPI_PROC_NULL) {
                    continue;
                }

                if (rank == 0)
                {
                    printf("\n--- Testing %s (PINGPONG) with %zu pairs ---\n",
                           region_label.c_str(), pairs.size());
                    for (auto &p : pairs) {
                        printf("  pair %d (%s) <-> %d (%s)\n",
                               p.src, all_hostnames[p.src],
                               p.dst, all_hostnames[p.dst]);
                    }

#if defined(USE_CALIPER)
                    // Use the first pair as representative for Caliper attributes
                    cali_set_int(src_rank_attr, pairs[0].src);
                    cali_set_int(dest_rank_attr, pairs[0].dst);
                    cali_set_int(src_node_attr,
                                 extract_node_number(all_hostnames[pairs[0].src]));
                    cali_set_int(dest_node_attr,
                                 extract_node_number(all_hostnames[pairs[0].dst]));
#endif
                }

                double total_time = 0.0;
                int warmup = 1;

                // OSU-style directional tags for this pair:
                // lower rank receives TAG_A and sends TAG_B; higher rank receives TAG_B and sends TAG_A
                const int TAG_A = 10;
                const int TAG_B = 100;
                const int my_recv_tag = (rank < partner) ? TAG_A : TAG_B;
                const int my_send_tag = (rank < partner) ? TAG_B : TAG_A;

                // ---------- buffer allocation ----------
#if defined(USE_HIP)
                char *send_buf;
                char *recv_buf;

                hipError_t err1 = hipMalloc((void**)&send_buf, message);
                hipError_t err2 = hipMalloc((void**)&recv_buf, message);

                if (err1 != hipSuccess || err2 != hipSuccess) {
                    fprintf(stderr, "HIP malloc failed: %s %s\n",
                            hipGetErrorString(err1),
                            hipGetErrorString(err2));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                {
                    char* h_rand = (char*)malloc(message);
                    fill_with_random_pattern(h_rand, (size_t)message);
                    hipError_t cuerr1 =
                        hipMemcpy(send_buf, h_rand, message, hipMemcpyHostToDevice);
                    assert(cuerr1 == hipSuccess);
                    free(h_rand);
                }
                hipError_t cuerr2 = hipMemset(recv_buf, 0, message);
                assert(cuerr2 == hipSuccess);

#elif defined(USE_CUDA)
                int dev_count = 0;
                cuda_check(cudaGetDeviceCount(&dev_count));
                cuda_check(cudaSetDevice(rank % (dev_count > 0 ? dev_count : 1)));

                char *d_send = nullptr;
                char *d_recv = nullptr;
                cuda_check(cudaMalloc((void**)&d_send, message));
                cuda_check(cudaMalloc((void**)&d_recv, message));

                char *h_send = nullptr, *h_recv = nullptr;
                cuda_check(cudaMallocHost((void**)&h_send, message));
                cuda_check(cudaMallocHost((void**)&h_recv, message));

                fill_with_random_pattern(h_send, (size_t)message);
                memset(h_recv, 0, message);

                cuda_check(cudaMemcpy(d_send, h_send, message, cudaMemcpyHostToDevice));
                cuda_check(cudaMemset(d_recv, 0, message));

#else
                char *send_flat = (char*)malloc((size_t)WINDOW_SIZE * (size_t)message);
                char *recv_flat = (char*)malloc((size_t)WINDOW_SIZE * (size_t)message);

                std::vector<char*> s_buf(WINDOW_SIZE);
                std::vector<char*> r_buf(WINDOW_SIZE);

                for (int j = 0; j < WINDOW_SIZE; ++j) {
                    s_buf[j] = send_flat + (size_t)j * (size_t)message;
                    r_buf[j] = recv_flat + (size_t)j * (size_t)message;
                    fill_with_random_pattern(s_buf[j], (size_t)message);
                    memset(r_buf[j], 0, (size_t)message);
                }

                std::vector<MPI_Request> send_request(WINDOW_SIZE);
                std::vector<MPI_Request> recv_request(WINDOW_SIZE);
#endif

                // ---------- warmup ----------
#if defined(USE_CALIPER)
                CALI_MARK_BEGIN(warmup_region);
#endif
                for (int i = 0; i < warmup; i++)
                {
#if defined(USE_HIP)
                    if (rank < partner) {
                        MPI_Send(send_buf, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD);
                        MPI_Recv(recv_buf, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                    } else {
                        MPI_Recv(recv_buf, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                        MPI_Send(send_buf, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD);
                    }
#elif defined(USE_CUDA)
                    if (rank < partner) {
                        cuda_check(cudaMemcpy(h_send, d_send, message, cudaMemcpyDeviceToHost));
                        MPI_Send(h_send, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD);
                        MPI_Recv(h_recv, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                        cuda_check(cudaMemcpy(d_recv, h_recv, message, cudaMemcpyHostToDevice));
                    } else {
                        MPI_Recv(h_recv, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                        cuda_check(cudaMemcpy(d_recv, h_recv, message, cudaMemcpyHostToDevice));
                        cuda_check(cudaMemcpy(h_send, d_send, message, cudaMemcpyDeviceToHost));
                        MPI_Send(h_send, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD);
                    }
#else
                    for (int j = 0; j < WINDOW_SIZE; ++j) {
                        MPI_Irecv(r_buf[j], message, MPI_CHAR, partner, my_recv_tag,
                                  MPI_COMM_WORLD, &recv_request[j]);
                    }
                    for (int j = 0; j < WINDOW_SIZE; ++j) {
                        MPI_Isend(s_buf[j], message, MPI_CHAR, partner, my_send_tag,
                                  MPI_COMM_WORLD, &send_request[j]);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    MPI_Waitall(WINDOW_SIZE, send_request.data(), MPI_STATUSES_IGNORE);
                    MPI_Waitall(WINDOW_SIZE, recv_request.data(), MPI_STATUSES_IGNORE);
#endif
                }

#if defined(USE_CALIPER)
                CALI_MARK_END(warmup_region);
                CALI_MARK_BEGIN(region_label.c_str());
#endif

                // ---------- timed ping-pong ----------
                double min_rtt = std::numeric_limits<double>::infinity();
                double max_rtt = 0.0;
                int iters = 0;

                bool i_am_timing_rank = (rank < partner);

                MPI_Barrier(MPI_COMM_WORLD);

                for (int i = 0; i < PING_PONG_LIMIT; i++)
                {
                    double start = 0.0, end = 0.0;

                    if (i_am_timing_rank)
                        start = MPI_Wtime();

#if defined(USE_HIP)
                    if (rank < partner) {
                        MPI_Send(send_buf, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD);
                        MPI_Recv(recv_buf, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                    } else {
                        MPI_Recv(recv_buf, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                        MPI_Send(send_buf, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD);
                    }
#elif defined(USE_CUDA)
                    if (rank < partner) {
                        cuda_check(cudaMemcpy(h_send, d_send, message, cudaMemcpyDeviceToHost));
                        MPI_Send(h_send, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD);
                        MPI_Recv(h_recv, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                        cuda_check(cudaMemcpy(d_recv, h_recv, message, cudaMemcpyHostToDevice));
                    } else {
                        MPI_Recv(h_recv, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                        cuda_check(cudaMemcpy(d_recv, h_recv, message, cudaMemcpyHostToDevice));
                        cuda_check(cudaMemcpy(h_send, d_send, message, cudaMemcpyDeviceToHost));
                        MPI_Send(h_send, message, MPI_CHAR, partner, 0, MPI_COMM_WORLD);
                    }
#else
                    for (int j = 0; j < WINDOW_SIZE; ++j) {
                        MPI_Irecv(r_buf[j], message, MPI_CHAR, partner, my_recv_tag,
                                  MPI_COMM_WORLD, &recv_request[j]);
                    }
                    for (int j = 0; j < WINDOW_SIZE; ++j) {
                        MPI_Isend(s_buf[j], message, MPI_CHAR, partner, my_send_tag,
                                  MPI_COMM_WORLD, &send_request[j]);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    MPI_Waitall(WINDOW_SIZE, send_request.data(), MPI_STATUSES_IGNORE);
                    MPI_Waitall(WINDOW_SIZE, recv_request.data(), MPI_STATUSES_IGNORE);
#endif

                    if (i_am_timing_rank) {
                        end = MPI_Wtime();
                        double rtt = end - start;
                        total_time += rtt;
                        if (rtt < min_rtt) min_rtt = rtt;
                        if (rtt > max_rtt) max_rtt = rtt;
                        ++iters;
                    }
                }

                if (i_am_timing_rank)
                {
                    double avg_rtt = (iters > 0) ? (total_time / iters) : 0.0;

#if defined(USE_CALIPER)
                    cali_set_string(comm_phase_attr, "pingpong");
                    cali_set_double(pp_avg_time_sec_attr, avg_rtt);
                    cali_set_double(pp_max_time_sec_attr, max_rtt);
                    cali_set_double(pp_min_time_sec_attr, min_rtt);
#endif

                    printf("PINGPONG %s pair (%d,%d): avg=%g s, min=%g s, max=%g s\n",
                           region_label.c_str(), rank, partner,
                           avg_rtt, min_rtt, max_rtt);
                }

#if defined(USE_CALIPER)
                CALI_MARK_END(region_label.c_str());
#endif

                // ---------- cleanup ----------
#if defined(USE_HIP)
                hipFree(send_buf);
                hipFree(recv_buf);
#elif defined(USE_CUDA)
                cuda_check(cudaFreeHost(h_send));
                cuda_check(cudaFreeHost(h_recv));
                cuda_check(cudaFree(d_send));
                cuda_check(cudaFree(d_recv));
#else
                free(send_flat);
                free(recv_flat);
#endif
            } // end for (partner_rank : partners)
        }     // end if (PingPong || All)

        MPI_Barrier(MPI_COMM_WORLD);

        // ===================== ALLTOALL =====================
        if (op == OpKind::Alltoall || op == OpKind::All)
        {
            for (int partner_rank : partners)
            {
                std::string region_label = region_names[partner_rank];

                size_t aa_bytes_per_rank = static_cast<size_t>(message);
                size_t aa_total_bytes = aa_bytes_per_rank * static_cast<size_t>(size);

                int warmup = 1;
                double alltoall_total_time = 0.0;

#if defined(USE_HIP)
                char *aa_send_dev;
                char *aa_recv_dev;

                hipError_t a_err1 = hipMalloc((void**)&aa_send_dev, aa_total_bytes);
                hipError_t a_err2 = hipMalloc((void**)&aa_recv_dev, aa_total_bytes);

                if (a_err1 != hipSuccess || a_err2 != hipSuccess) {
                    fprintf(stderr, "HIP malloc failed: %s %s\n", hipGetErrorString(a_err1), hipGetErrorString(a_err2));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                char *aa_send_host = (char*) malloc(aa_total_bytes);
                char *aa_recv_host = (char*) malloc(aa_total_bytes);
                fill_with_random_pattern(aa_send_host, aa_total_bytes);
                memset(aa_recv_host, 0, aa_total_bytes);

                hipError_t a_cuerr1 = hipMemcpy(aa_send_dev, aa_send_host, aa_total_bytes, hipMemcpyHostToDevice);
                assert(a_cuerr1 == hipSuccess);
                hipError_t a_cuerr2 = hipMemset(aa_recv_dev, 0, aa_total_bytes);
                assert(a_cuerr2 == hipSuccess);

#elif defined(USE_CUDA)
                int a_dev_count = 0;
                cuda_check(cudaGetDeviceCount(&a_dev_count));
                cuda_check(cudaSetDevice(rank % (a_dev_count > 0 ? a_dev_count : 1)));

                char *ad_send = nullptr;
                char *ad_recv = nullptr;
                cuda_check(cudaMalloc((void**)&ad_send, aa_total_bytes));
                cuda_check(cudaMalloc((void**)&ad_recv, aa_total_bytes));

                char *ah_send = nullptr;
                char *ah_recv = nullptr;
                cuda_check(cudaMallocHost((void**)&ah_send, aa_total_bytes));
                cuda_check(cudaMallocHost((void**)&ah_recv, aa_total_bytes));

                fill_with_random_pattern(ah_send, aa_total_bytes);
                memset(ah_recv, 0, aa_total_bytes);

                cuda_check(cudaMemcpy(ad_send, ah_send, aa_total_bytes, cudaMemcpyHostToDevice));
                cuda_check(cudaMemset(ad_recv, 0, aa_total_bytes));

#else
                char *aa_send = (char*) malloc(aa_total_bytes);
                char *aa_recv = (char*) malloc(aa_total_bytes);
                fill_with_random_pattern(aa_send, aa_total_bytes);
                memset(aa_recv,  0, aa_total_bytes);
#endif

#if defined(USE_CALIPER)
                CALI_MARK_BEGIN(warmup_region_aa);
#endif

                for (int i = 0; i < warmup; i++)
                {
#if defined(USE_HIP)
                    hipMemcpy(aa_send_host, aa_send_dev, aa_total_bytes, hipMemcpyDeviceToHost);
                    hipDeviceSynchronize();
                    MPI_Alltoall(aa_send_host, (int)aa_bytes_per_rank, MPI_CHAR, aa_recv_host, (int)aa_bytes_per_rank, MPI_CHAR, MPI_COMM_WORLD);
                    hipMemcpy(aa_recv_dev, aa_recv_host, aa_total_bytes, hipMemcpyHostToDevice);
                    hipDeviceSynchronize();
#elif defined(USE_CUDA)
                    cuda_check(cudaMemcpy(ah_send, ad_send, aa_total_bytes, cudaMemcpyDeviceToHost));
                    MPI_Alltoall(ah_send, (int)aa_bytes_per_rank, MPI_CHAR, ah_recv, (int)aa_bytes_per_rank, MPI_CHAR, MPI_COMM_WORLD);
                    cuda_check(cudaMemcpy(ad_recv, ah_recv, aa_total_bytes, cudaMemcpyHostToDevice));
#else
                    MPI_Alltoall(aa_send, (int)aa_bytes_per_rank, MPI_CHAR, aa_recv, (int)aa_bytes_per_rank, MPI_CHAR, MPI_COMM_WORLD);
#endif
                }

#if defined(USE_CALIPER)
                CALI_MARK_END(warmup_region_aa);
                CALI_MARK_BEGIN(region_label.c_str());
#endif

                double min_rtt = std::numeric_limits<double>::infinity();
                double max_rtt = 0.0;
                int iters = 0;

                for (int it = 0; it < PING_PONG_LIMIT; ++it)
                {
                    MPI_Barrier(MPI_COMM_WORLD);

                    double t0 = MPI_Wtime();

#if defined(USE_HIP)
                    hipMemcpy(aa_send_host, aa_send_dev, aa_total_bytes, hipMemcpyDeviceToHost);
                    MPI_Alltoall(aa_send_host, (int)aa_bytes_per_rank, MPI_CHAR, aa_recv_host, (int)aa_bytes_per_rank, MPI_CHAR, MPI_COMM_WORLD);
                    hipMemcpy(aa_recv_dev, aa_recv_host, aa_total_bytes, hipMemcpyHostToDevice);
                    hipDeviceSynchronize();
#elif defined(USE_CUDA)
                    cuda_check(cudaMemcpy(ah_send, ad_send, aa_total_bytes, cudaMemcpyDeviceToHost));
                    MPI_Alltoall(ah_send, (int)aa_bytes_per_rank, MPI_CHAR, ah_recv, (int)aa_bytes_per_rank, MPI_CHAR, MPI_COMM_WORLD);
                    cuda_check(cudaMemcpy(ad_recv, ah_recv, aa_total_bytes, cudaMemcpyHostToDevice));
#else
                    MPI_Alltoall(aa_send, (int)aa_bytes_per_rank, MPI_CHAR, aa_recv, (int)aa_bytes_per_rank, MPI_CHAR, MPI_COMM_WORLD);
#endif

                    double t1 = MPI_Wtime();
                    double dt = t1 - t0;

                    double iter_max = 0.0;
                    MPI_Reduce(&dt, &iter_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                    if (rank == 0)
                    {
                        alltoall_total_time += iter_max;
                        if(dt < min_rtt) min_rtt = dt;
                        if(dt > max_rtt) max_rtt = dt;
                        ++iters;

                        double avg_rtt = (iters > 0) ? (alltoall_total_time / iters) : 0.0;
#if defined(USE_CALIPER)
                        cali_set_string(comm_phase_attr, "alltoall");
                        cali_set_double(aa_avg_time_sec_attr, avg_rtt);
                        cali_set_double(aa_max_time_sec_attr, max_rtt);
                        cali_set_double(aa_min_time_sec_attr, min_rtt);
#endif
                    }
                }
#if defined(USE_CALIPER)
                CALI_MARK_END(region_label.c_str());
#endif

#if defined(USE_HIP)
                free(aa_send_host);
                free(aa_recv_host);
                hipFree(aa_send_dev);
                hipFree(aa_recv_dev);
#elif defined(USE_CUDA)
                cuda_check(cudaFreeHost(ah_send));
                cuda_check(cudaFreeHost(ah_recv));
                cuda_check(cudaFree(ad_send));
                cuda_check(cudaFree(ad_recv));
#else
                free(aa_send);
                free(aa_recv);
#endif
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // ===================== REDUCE =====================
        if (op == OpKind::Reduce || op == OpKind::All)
        {
            for (int partner_rank : partners)
            {
                std::string region_label = region_names[partner_rank];

                size_t red_count = static_cast<size_t>(message);
                if (red_count > INT_MAX) {
                    if (rank == 0) fprintf(stderr, "Reduce count too large\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                int warmup = 1;
                double red_total_time = 0.0;

#if defined(USE_CUDA)
                char *rd_d_send=nullptr;
                char *rd_d_recv=nullptr;

                cuda_check(cudaMalloc((void**)&rd_d_send, red_count));
                cuda_check(cudaMalloc((void**)&rd_d_recv, red_count));

                char *rd_h_send=nullptr;
                char *rd_h_recv=nullptr;
                cuda_check(cudaMallocHost((void**)&rd_h_send, red_count));
                cuda_check(cudaMallocHost((void**)&rd_h_recv, red_count));

                fill_with_random_pattern(rd_h_send, red_count);
                memset(rd_h_recv, 0,   red_count);

                cuda_check(cudaMemcpy(rd_d_send, rd_h_send, red_count, cudaMemcpyHostToDevice));
                cuda_check(cudaMemset(rd_d_recv, 0,   red_count));

#elif defined(USE_HIP)
                char *rd_d_send=nullptr;
                char *rd_d_recv=nullptr;

                hipError_t a_err1 = hipMalloc((void**)&rd_d_send, red_count);
                hipError_t a_err2 = hipMalloc((void**)&rd_d_recv, red_count);

                if (a_err1 != hipSuccess || a_err2 != hipSuccess) {
                    fprintf(stderr, "HIP malloc failed: %s %s\n", hipGetErrorString(a_err1), hipGetErrorString(a_err2));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                char *rd_h_send = (char*)malloc(red_count);
                char *rd_h_recv = (char*)malloc(red_count);
                fill_with_random_pattern(rd_h_send, red_count);
                memset(rd_h_recv, 0,   red_count);

                hipError_t a_cuerr1 = hipMemcpy(rd_d_send, rd_h_send, red_count, hipMemcpyHostToDevice);
                assert(a_cuerr1 == hipSuccess);
                hipError_t a_cuerr2 = hipMemset(rd_d_recv, 0, red_count);
                assert(a_cuerr2 == hipSuccess);
#else
                char *rd_send = (char*)malloc(red_count);
                char *rd_recv = (char*)malloc(red_count);
                fill_with_random_pattern(rd_send, red_count);
                memset(rd_recv, 0,   red_count);
#endif

#if defined(USE_CALIPER)
                CALI_MARK_BEGIN(warmup_region_red);
#endif
                for (int i = 0; i < warmup; i++)
                {
#if defined(USE_CUDA)
                    cuda_check(cudaMemcpy(rd_h_send, rd_d_send, red_count, cudaMemcpyDeviceToHost));
                    MPI_Reduce(rd_h_send, rd_h_recv, (int)red_count, MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);
                    cuda_check(cudaMemcpy(rd_d_recv, rd_h_recv, red_count, cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                    hipMemcpy(rd_h_send, rd_d_send, red_count, hipMemcpyDeviceToHost);
                    MPI_Reduce(rd_h_send, rd_h_recv, (int)red_count, MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);
                    hipMemcpy(rd_d_recv, rd_h_recv, red_count, hipMemcpyHostToDevice);
#else
                    MPI_Reduce(rd_send, rd_recv, (int)red_count, MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
                }

#if defined(USE_CALIPER)
                CALI_MARK_END(warmup_region_red);
                CALI_MARK_BEGIN(region_label.c_str());
#endif

                double min_rtt = std::numeric_limits<double>::infinity();
                double max_rtt = 0.0;
                int iters = 0;

                for(int i = 0; i < PING_PONG_LIMIT; i++)
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    double t0 = MPI_Wtime();
#if defined(USE_CUDA)
                    cuda_check(cudaMemcpy(rd_h_send, rd_d_send, red_count, cudaMemcpyDeviceToHost));
                    MPI_Reduce(rd_h_send, rd_h_recv, (int)red_count, MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);
                    cuda_check(cudaMemcpy(rd_d_recv, rd_h_recv, red_count, cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                    hipMemcpy(rd_h_send, rd_d_send, red_count, hipMemcpyDeviceToHost);
                    MPI_Reduce(rd_h_send, rd_h_recv, (int)red_count, MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);
                    hipMemcpy(rd_d_recv, rd_h_recv, red_count, hipMemcpyHostToDevice);
#else
                    MPI_Reduce(rd_send, rd_recv, (int)red_count, MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
                    double dt = MPI_Wtime() - t0;
                    double iter_max = 0.0;
                    MPI_Reduce(&dt, &iter_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

                    if(rank == 0)
                    {
                        red_total_time += iter_max;
                        if(dt < min_rtt) min_rtt = dt;
                        if(dt > max_rtt) max_rtt = dt;
                        ++iters;

                        double avg_rtt = (iters > 0) ? (red_total_time / iters) : 0.0;
#if defined(USE_CALIPER)
                        cali_set_string(comm_phase_attr, "reduce");
                        cali_set_double(red_avg_time_sec_attr, avg_rtt);
                        cali_set_double(red_max_time_sec_attr, max_rtt);
                        cali_set_double(red_min_time_sec_attr, min_rtt);
#endif
                    }
                }
#if defined(USE_CALIPER)
                CALI_MARK_END(region_label.c_str());
#endif

#if defined(USE_HIP)
                free(rd_h_send);
                free(rd_h_recv);
                hipFree(rd_d_send);
                hipFree(rd_d_recv);
#elif defined(USE_CUDA)
                cuda_check(cudaFreeHost(rd_h_send));
                cuda_check(cudaFreeHost(rd_h_recv));
                cuda_check(cudaFree(rd_d_send));
                cuda_check(cudaFree(rd_d_recv));
#else
                free(rd_send);
                free(rd_recv);
#endif
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // ===================== ALLREDUCE =====================
        if (op == OpKind::Allreduce || op == OpKind::All)
        {
            for (int partner_rank : partners)
            {
                std::string region_label = region_names[partner_rank];

                size_t ar_count = static_cast<size_t>(message);
                if (ar_count > INT_MAX) {
                    if (rank == 0) fprintf(stderr, "Allreduce count too large\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                int warmup = 1;
                double ar_total_time = 0.0;

#if defined(USE_CUDA)
                char *ar_d_send=nullptr;
                char *ar_d_recv=nullptr;
                cuda_check(cudaMalloc((void**)&ar_d_send, ar_count));
                cuda_check(cudaMalloc((void**)&ar_d_recv, ar_count));

                char *ar_h_send=nullptr;
                char *ar_h_recv=nullptr;
                cuda_check(cudaMallocHost((void**)&ar_h_send, ar_count));
                cuda_check(cudaMallocHost((void**)&ar_h_recv, ar_count));

                fill_with_random_pattern(ar_h_send, ar_count);
                memset(ar_h_recv, 0,   ar_count);

                cuda_check(cudaMemcpy(ar_d_send, ar_h_send, ar_count, cudaMemcpyHostToDevice));
                cuda_check(cudaMemset(ar_d_recv, 0,   ar_count));

#elif defined(USE_HIP)
                char *ar_d_send=nullptr;
                char *ar_d_recv=nullptr;

                hipError_t a_err1 = hipMalloc((void**)&ar_d_send, ar_count);
                hipError_t a_err2 = hipMalloc((void**)&ar_d_recv, ar_count);

                if (a_err1 != hipSuccess || a_err2 != hipSuccess) {
                    fprintf(stderr, "HIP malloc failed: %s %s\n", hipGetErrorString(a_err1), hipGetErrorString(a_err2));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                char *ar_h_send = (char*)malloc(ar_count);
                char *ar_h_recv = (char*)malloc(ar_count);
                fill_with_random_pattern(ar_h_send, ar_count);
                memset(ar_h_recv, 0,   ar_count);

                hipError_t a_cuerr1 = hipMemcpy(ar_d_send, ar_h_send, ar_count, hipMemcpyHostToDevice);
                assert(a_cuerr1 == hipSuccess);
                hipError_t a_cuerr2 = hipMemset(ar_d_recv, 0, ar_count);
                assert(a_cuerr2 == hipSuccess);
#else
                char *ar_send = (char*)malloc(ar_count);
                char *ar_recv = (char*)malloc(ar_count);
                fill_with_random_pattern(ar_send, ar_count);
                memset(ar_recv, 0,   ar_count);
#endif

#if defined(USE_CALIPER)
                CALI_MARK_BEGIN(warmup_region_ar);
#endif
                for (int i = 0; i < warmup; i++)
                {
#if defined(USE_CUDA)
                    cuda_check(cudaMemcpy(ar_h_send, ar_d_send, ar_count, cudaMemcpyDeviceToHost));
                    MPI_Allreduce(ar_h_send, ar_h_recv, (int)ar_count, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);
                    cuda_check(cudaMemcpy(ar_d_recv, ar_h_recv, ar_count, cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                    hipMemcpy(ar_h_send, ar_d_send, ar_count, hipMemcpyDeviceToHost);
                    MPI_Allreduce(ar_h_send, ar_h_recv, (int)ar_count, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);
                    hipMemcpy(ar_d_recv, ar_h_recv, ar_count, hipMemcpyHostToDevice);
#else
                    MPI_Allreduce(ar_send, ar_recv, (int)ar_count, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);
#endif
                }
#if defined(USE_CALIPER)
                CALI_MARK_END(warmup_region_ar);
                CALI_MARK_BEGIN(region_label.c_str());
#endif
                double min_rtt = std::numeric_limits<double>::infinity();
                double max_rtt = 0.0;
                int iters = 0;

                for(int i = 0; i < PING_PONG_LIMIT; i++)
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    double t0 = MPI_Wtime();
#if defined(USE_CUDA)
                    cuda_check(cudaMemcpy(ar_h_send, ar_d_send, ar_count, cudaMemcpyDeviceToHost));
                    MPI_Allreduce(ar_h_send, ar_h_recv, (int)ar_count, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);
                    cuda_check(cudaMemcpy(ar_d_recv, ar_h_recv, ar_count, cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                    hipMemcpy(ar_h_send, ar_d_send, ar_count, hipMemcpyDeviceToHost);
                    MPI_Allreduce(ar_h_send, ar_h_recv, (int)ar_count, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);
                    hipMemcpy(ar_d_recv, ar_h_recv, ar_count, hipMemcpyHostToDevice);
#else
                    MPI_Allreduce(ar_send, ar_recv, (int)ar_count, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);
#endif
                    double dt = MPI_Wtime() - t0;
                    double iter_max = 0.0;
                    MPI_Allreduce(&dt, &iter_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

                    if(rank == 0)
                    {
                        ar_total_time += iter_max;
                        if(dt < min_rtt) min_rtt = dt;
                        if(dt > max_rtt) max_rtt = dt;
                        ++iters;

                        double avg_rtt = (iters > 0) ? (ar_total_time / iters) : 0.0;
#if defined(USE_CALIPER)
                        cali_set_string(comm_phase_attr, "allreduce");
                        cali_set_double(ar_avg_time_sec_attr, avg_rtt);
                        cali_set_double(ar_max_time_sec_attr, max_rtt);
                        cali_set_double(ar_min_time_sec_attr, min_rtt);
#endif
                    }
                }
#if defined(USE_CALIPER)
                CALI_MARK_END(region_label.c_str());
#endif

#if defined(USE_HIP)
                free(ar_h_send);
                free(ar_h_recv);
                hipFree(ar_d_send);
                hipFree(ar_d_recv);
#elif defined(USE_CUDA)
                cuda_check(cudaFreeHost(ar_h_send));
                cuda_check(cudaFreeHost(ar_h_recv));
                cuda_check(cudaFree(ar_d_send));
                cuda_check(cudaFree(ar_d_recv));
#else
                free(ar_send);
                free(ar_recv);
#endif
            }
        }

#if defined(USE_CALIPER)
        mgr[message].stop();
#endif
    }

#if defined(USE_CALIPER)
    if (rank == 0 && !all_comm_pairs.empty())
    {
        adiak::value("all_comm_pairs", all_comm_pairs);
    }
    for (auto &m : mgr)
    {
        m.second.flush();
    }
#endif

    MPI_Finalize();
    return 0;
}
