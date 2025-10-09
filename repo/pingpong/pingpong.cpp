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

#if defined(USE_CALIPER)
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#endif

#ifndef USE_CALIPER
#define CALI_CXX_MARK_FUNCTION
#define CALI_MARK_BEGIN
#define CALI_MARK_END
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

const char *get_hostname_for_rank(int rank, char all_hostnames[][1024],
                                  int size)
{
    if (rank >= 0 && rank < size)
    {
        return all_hostnames[rank];
    }
    else
    {
        return "INVALID_RANK";
    }
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
    int msg_size = 1;
    int n_nodes = 1;
    int sys_cores_per_socket = 1;
    int sys_cores_per_node = 1;
    std::string metadata;
    const char *warmup_region = "warmup";

    int opt;
    const char *usage =
        "Usage: %s [-h] [-i n-iterations] [-p rank1,rank2] [-m msg_sz] "
        "[-n n_nodes] [-s sys_cores_per_socket] [-c sys_cores_per_node] [-b metadata]\n";

    while ((opt = getopt(argc, argv, "hi:p:m:n:s:c:b:")) != -1)
    {
        switch (opt)
        {
            case 'h':
                printf(usage, argv[0]);
                return 0;
            case 'i':
                PING_PONG_LIMIT = atoi(optarg);
                break;
            case 'p':
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
            default:
                printf(usage, argv[0]);
                exit(EXIT_FAILURE);
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

#if defined(USE_CALIPER)
        std::stringstream rankmap;
        rankmap << "{";
        for (int i = 0; i < size; ++i)
        {
            rankmap << "\"" << i << "\": \"" << all_hostnames[i] << "\"";
            if (i < size - 1)
            {
                rankmap << ", ";
            }
        }
        rankmap << "}";
        adiak::value("rank_node_map", rankmap.str());
        adiak::value("iterations", PING_PONG_LIMIT);
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
                {"expr": "any(max#message_size_bytes)", "as": "message_size_bytes"}
                ]
            },
            {
                "level": "cross",
                "select":
                [
                {"expr": "any(any#max#src_rank)", "as": "src_rank"},
                {"expr": "any(any#max#dest_rank)", "as": "dest_rank"},
                {"expr": "any(any#max#src_node)", "as": "src_node"},
                {"expr": "any(any#max#dest_node)", "as": "dest_node"},
                {"expr": "any(any#max#message_size_bytes)", "as": "message_size_bytes"}
                ]
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
        {
            label = std::to_string(current_nodes) + " nodes";
        }
        else
        {
            label = "Same Node Different Socket";
        }

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

        for (int partner_rank : partners)
        {
            std::string region_label = region_names[partner_rank];

            if (rank != 0 && rank != partner_rank)
            {
                continue;
            }

            if (rank == 0)
            {
                printf("\n--- Testing %s between ranks 0 (%s) and %d (%s) ---\n",
                       region_label.c_str(),
                       all_hostnames[0], partner_rank, all_hostnames[partner_rank]);

#if defined(USE_CALIPER)
                std::string comm_pair = "0(" + std::string(all_hostnames[0]) + ")<->" +
                                        std::to_string(partner_rank) + "(" + std::string(all_hostnames[partner_rank]) +
                                        ")";
                all_comm_pairs.push_back(comm_pair);

                cali_set_int(src_rank_attr, 0);
                cali_set_int(dest_rank_attr, partner_rank);
                cali_set_int(src_node_attr, extract_node_number(all_hostnames[0]));
                cali_set_int(dest_node_attr, extract_node_number(all_hostnames[partner_rank]));
#endif
            }

            double total_time = 0.0;
            int warmup = 1;

#if defined(USE_HIP)
            char *send_buf;
            char *recv_buf;

            hipError_t err1 = hipMalloc((void**)&send_buf, message);
            hipError_t err2 = hipMalloc((void**)&recv_buf, message);

            if (err1 != hipSuccess || err2 != hipSuccess) {
                fprintf(stderr, "HIP malloc failed: %s %s\n", hipGetErrorString(err1), hipGetErrorString(err2));
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            hipError_t cuerr1 = hipMemset(send_buf, 'a', message);
            assert(cuerr1 == hipSuccess);
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
            cuda_check(cudaMemset(d_send, 'a', message));
            cuda_check(cudaMemset(d_recv, 0, message));

            char *h_send = nullptr, *h_recv = nullptr;
            cuda_check(cudaMallocHost((void**)&h_send, message));
            cuda_check(cudaMallocHost((void**)&h_recv, message));
            memset(h_send, 'a', message);
            memset(h_recv, 0, message);
#else
            char *send_buf = (char *)malloc(message);
            char *recv_buf = (char *)malloc(message);
            memset(send_buf, 'a', message);
            memset(recv_buf, 0, message);
#endif

#if defined(USE_CALIPER)
            CALI_MARK_BEGIN(warmup_region);
#endif

            for (int i = 0; i < warmup; i++)
            {
                if (rank == 0)
                {
#if defined(USE_HIP)
                    MPI_Send(send_buf, message, MPI_CHAR, partner_rank, 0, MPI_COMM_WORLD);
                    MPI_Recv(recv_buf, message, MPI_CHAR, partner_rank, 0, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
#elif defined(USE_CUDA)
                    cuda_check(cudaMemcpy(h_send, d_send, message, cudaMemcpyDeviceToHost));
                    MPI_Send(h_send, message, MPI_CHAR, partner_rank, 0, MPI_COMM_WORLD);
                    MPI_Recv(h_recv, message, MPI_CHAR, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cuda_check(cudaMemcpy(d_recv, h_recv, message, cudaMemcpyHostToDevice));
#else
                    MPI_Send(send_buf, message, MPI_CHAR, partner_rank, 0, MPI_COMM_WORLD);
                    MPI_Recv(recv_buf, message, MPI_CHAR, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
                }
                else if (rank == partner_rank)
                {
#if defined(USE_HIP)
                    MPI_Recv(recv_buf, message, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(send_buf, message, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
#elif defined(USE_CUDA)
                    MPI_Recv(h_recv, message, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cuda_check(cudaMemcpy(d_recv, h_recv, message, cudaMemcpyHostToDevice));
                    cuda_check(cudaMemcpy(h_send, d_send, message, cudaMemcpyDeviceToHost));
                    MPI_Send(h_send, message, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
#else
                    MPI_Recv(recv_buf, message, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(send_buf, message, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
#endif
                }
            }

#if defined(USE_CALIPER)
            CALI_MARK_END(warmup_region);
            CALI_MARK_BEGIN(region_label.c_str());
#endif

            for (int i = 0; i < PING_PONG_LIMIT; i++)
            {
                if (rank == 0)
                {
#if defined(USE_CUDA)
                    double start = MPI_Wtime();
                    cuda_check(cudaMemcpy(h_send, d_send, message, cudaMemcpyDeviceToHost));
                    MPI_Send(h_send, message, MPI_CHAR, partner_rank, 0, MPI_COMM_WORLD);
                    MPI_Recv(h_recv, message, MPI_CHAR, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cuda_check(cudaMemcpy(d_recv, h_recv, message, cudaMemcpyHostToDevice));
                    double end = MPI_Wtime();
#else
                    double start = MPI_Wtime();
                    MPI_Send(send_buf, message, MPI_CHAR, partner_rank, 0, MPI_COMM_WORLD);
                    MPI_Recv(recv_buf, message, MPI_CHAR, partner_rank, 0, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    double end = MPI_Wtime();
#endif
                    double rtt = end - start;
                    total_time += rtt;
                }
                else if (rank == partner_rank)
                {
#if defined(USE_CUDA)
                    MPI_Recv(h_recv, message, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    cuda_check(cudaMemcpy(d_recv, h_recv, message, cudaMemcpyHostToDevice));
                    cuda_check(cudaMemcpy(h_send, d_send, message, cudaMemcpyDeviceToHost));
                    MPI_Send(h_send, message, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
#else
                    MPI_Recv(recv_buf, message, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(send_buf, message, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
#endif
                }
            }

#if defined(USE_CALIPER)
            CALI_MARK_END(region_label.c_str());
#endif

#if defined(USE_HIP)
            hipFree(send_buf);
            hipFree(recv_buf);
#elif defined(USE_CUDA)
            cuda_check(cudaFreeHost(h_send));
            cuda_check(cudaFreeHost(h_recv));
            cuda_check(cudaFree(d_send));
            cuda_check(cudaFree(d_recv));
#else
            free(send_buf);
            free(recv_buf);
#endif
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
