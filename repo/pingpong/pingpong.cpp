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

#if defined(USE_CALIPER)
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#endif

#ifndef USE_CALIPER
#define CALI_CXX_MARK_FUNCTION
#define CALI_MARK_BEGIN(x)
#define CALI_MARK_END(x)
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

// ------------ helpers ------------
static int extract_node_number(const char *hostname) {
    int len = (int)strlen(hostname);
    int num = 0, factor = 1;
    for (int i = len - 1; i >= 0; --i) {
        if (hostname[i] >= '0' && hostname[i] <= '9') {
            num += (hostname[i] - '0') * factor;
            factor *= 10;
        } else break;
    }
    return num;
}

enum class OpKind { PingPong, Alltoall, Reduce, Allreduce };

static OpKind parse_opkind(const char* s) {
    if (!s) return OpKind::PingPong;
    std::string m(s);
    if (m == "pingpong")  return OpKind::PingPong;
    if (m == "alltoall")  return OpKind::Alltoall;
    if (m == "reduce")    return OpKind::Reduce;
    if (m == "allreduce") return OpKind::Allreduce;
    fprintf(stderr, "Unknown -O option '%s' (use pingpong|alltoall|reduce|allreduce)\n", s);
    MPI_Abort(MPI_COMM_WORLD, 2);
    return OpKind::PingPong;
}

// ------------ main ------------
int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // gather hostnames to rank 0 for printing/adiak
    char my_hostname[1024];
    gethostname(my_hostname, 1023);
    my_hostname[1023] = '\0';
    std::vector<char> all_hostnames;
    if (rank == 0) all_hostnames.resize(1024 * size, 0);
    MPI_Gather(my_hostname, 1024, MPI_CHAR,
               rank==0? all_hostnames.data():nullptr, 1024, MPI_CHAR, 0, MPI_COMM_WORLD);
    auto host = [&](int r){ return rank==0? &all_hostnames[r*1024] : my_hostname; };

#if defined(USE_CALIPER)
    static std::map<int, cali::ConfigManager> mgr;
    std::vector<std::string> all_comm_pairs;
    MPI_Comm adiak_comm = MPI_COMM_WORLD;
    adiak::init(&adiak_comm);
    adiak::collect_all();
    CALI_CXX_MARK_FUNCTION;
#endif

    // params
    int PING_PONG_LIMIT = 10;
    int msg_size = 1;
    int n_nodes = 1;
    int sys_cores_per_socket = 1;
    int sys_cores_per_node = 1;
    std::string metadata;
    OpKind op = OpKind::PingPong;

    const char *warmup_region    = "warmup";
    const char *warmup_region_aa = "warmup_aa";
    const char *warmup_region_red= "warmup_red";
    const char *warmup_region_ar = "warmup_ar";

    const char *usage =
        "Usage: %s [-h] [-i iters] [-m msg_sz] [-n n_nodes] [-s cores/socket] [-c cores/node] [-b metadata] "
        "[-O pingpong|alltoall|reduce|allreduce]\n";

    int opt;
    while ((opt = getopt(argc, argv, "hi:m:n:s:c:b:O:")) != -1) {
        switch (opt) {
            case 'h': if (rank==0) printf(usage, argv[0]); MPI_Finalize(); return 0;
            case 'i': PING_PONG_LIMIT = atoi(optarg); break;
            case 'm': msg_size = atoi(optarg); break;
            case 'n': n_nodes = atoi(optarg); break;
            case 's': sys_cores_per_socket = atoi(optarg); break;
            case 'c': sys_cores_per_node = atoi(optarg); break;
            case 'b': metadata = optarg; break;
            case 'O': op = parse_opkind(optarg); break;
            default:  if (rank==0) printf(usage, argv[0]); MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (rank == 0) {
        printf("Configuration:\n");
        printf("Operation: %s\n",
               op==OpKind::PingPong ? "pingpong" :
               op==OpKind::Alltoall ? "alltoall" :
               op==OpKind::Reduce   ? "reduce"   : "allreduce");
        printf("PING_PONG_LIMIT: %d\n", PING_PONG_LIMIT);
        printf("Message size: %d bytes\n", msg_size);
        printf("Cores per socket: %d\n", sys_cores_per_socket);
        printf("Cores per node: %d\n", sys_cores_per_node);
        printf("Nodes: %d\n", n_nodes);
        printf("World size: %d\n", size);

#if defined(USE_CALIPER)
        // rank->host map
        std::stringstream rankmap;
        rankmap << "{";
        for (int i = 0; i < size; ++i) {
            rankmap << "\"" << i << "\": \"" << (&all_hostnames[i*1024])[0] << "\"";
            if (i < size - 1) rankmap << ", ";
        }
        rankmap << "}";
        adiak::value("rank_node_map", rankmap.str());
        adiak::value("iterations", PING_PONG_LIMIT);
#endif
    }

#if defined(USE_CALIPER)
    // attributes
    cali_id_t src_rank_attr = cali_create_attribute("src_rank", CALI_TYPE_INT, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t dest_rank_attr= cali_create_attribute("dest_rank",CALI_TYPE_INT, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t src_node_attr = cali_create_attribute("src_node", CALI_TYPE_INT, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t dest_node_attr= cali_create_attribute("dest_node",CALI_TYPE_INT, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t message_size_attr = cali_create_attribute("message_size_bytes", CALI_TYPE_INT, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t comm_phase_attr   = cali_create_attribute("comm_phase",        CALI_TYPE_STRING, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t aa_avg_time_sec_attr = cali_create_attribute("aa_avg_time_sec", CALI_TYPE_DOUBLE, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t aa_max_time_sec_attr = cali_create_attribute("aa_max_time_sec", CALI_TYPE_DOUBLE, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t aa_min_time_sec_attr = cali_create_attribute("aa_min_time_sec", CALI_TYPE_DOUBLE, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t pp_avg_time_sec_attr = cali_create_attribute("pp_avg_time_sec", CALI_TYPE_DOUBLE, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t pp_max_time_sec_attr = cali_create_attribute("pp_max_time_sec", CALI_TYPE_DOUBLE, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t pp_min_time_sec_attr = cali_create_attribute("pp_min_time_sec", CALI_TYPE_DOUBLE, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t red_avg_time_sec_attr= cali_create_attribute("red_avg_time_sec",CALI_TYPE_DOUBLE, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t red_max_time_sec_attr= cali_create_attribute("red_max_time_sec",CALI_TYPE_DOUBLE, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t red_min_time_sec_attr= cali_create_attribute("red_min_time_sec",CALI_TYPE_DOUBLE, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t ar_avg_time_sec_attr = cali_create_attribute("ar_avg_time_sec", CALI_TYPE_DOUBLE, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t ar_max_time_sec_attr = cali_create_attribute("ar_max_time_sec", CALI_TYPE_DOUBLE, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);
    cali_id_t ar_min_time_sec_attr = cali_create_attribute("ar_min_time_sec", CALI_TYPE_DOUBLE, CALI_ATTR_ASVALUE|CALI_ATTR_AGGREGATABLE);

    const char *src_dest_attributes = R"json(
        {
            "name": "pingpong_attributes",
            "type": "boolean",
            "category": "metric",
            "description": "Collect pingpong attributes",
            "query": [
                {
                    "level": "local",
                    "select": [
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
                    "group by": ["comm_phase"]
                },
                {
                    "level": "cross",
                    "select": [
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
                    "group by": ["comm_phase"]
                }
            ]
        }
    )json";
#endif

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&now_time), "%Y%m%d_%H%M%S");

    // Build partner list for pingpong (0 <-> last rank of each bucket)
    int P = sys_cores_per_node * n_nodes;
    std::vector<int> partners;
    std::map<int,std::string> region_names;
    int current_nodes = n_nodes;
    int current_p = P;
    while (current_p > 2) {
        int partner_rank = current_p - 1;
        std::string label = (current_nodes >= 2) ? std::to_string(current_nodes) + " nodes"
                                                 : "Same Node Different Socket";
        if (partner_rank > 0 && partner_rank < size) {
            partners.push_back(partner_rank);
            region_names[partner_rank] = label;
        }
        current_nodes /= 2;
        current_p = current_nodes * sys_cores_per_node;
    }
    if (size > 1) {
        partners.push_back(1);
        region_names[1] = "Same Node Same Socket";
    }

    // sweep message sizes
    for (int message = msg_size; message <= (int)std::pow((double)msg_size, 6.0); message *= 8) {
#if defined(USE_CALIPER)
        std::string profile = "spot(output=" + std::to_string(message) + "_" + timestamp.str()
                            + ".cali, profile.mpi)"
                            + (metadata.empty() ? "" : (",metadata(file=" + metadata + ")"))
                            + ",metadata(file=/etc/node_info.json,keys=\"host.os\")";
        mgr[message].add_option_spec(src_dest_attributes);
        mgr[message].set_default_parameter("pingpong_attributes", "true");
        cali_set_int(message_size_attr, message);
        adiak::value("message_size", message);
        mgr[message].add(profile.c_str());
        mgr[message].start();
#endif

        // ---------------- PINGPONG (selected) ----------------
        if (op == OpKind::PingPong) {
            for (int partner_rank : partners) {
                if (rank != 0 && rank != partner_rank) continue;

                std::string region_label = region_names[partner_rank];
                if (rank == 0) {
                    printf("\n--- PINGPONG %s: 0 (%s) <-> %d (%s), msg=%d ---\n",
                           region_label.c_str(),
                           &all_hostnames[0],
                           partner_rank, &all_hostnames[partner_rank*1024], message);
#if defined(USE_CALIPER)
                    std::string comm_pair = "0(" + std::string(&all_hostnames[0]) + ")<->" +
                                            std::to_string(partner_rank) + "(" + std::string(&all_hostnames[partner_rank*1024]) + ")";
                    all_comm_pairs.push_back(comm_pair);
                    cali_set_int(src_rank_attr, 0);
                    cali_set_int(dest_rank_attr, partner_rank);
                    cali_set_int(src_node_attr, extract_node_number(&all_hostnames[0]));
                    cali_set_int(dest_node_attr, extract_node_number(&all_hostnames[partner_rank*1024]));
#endif
                }

                int warmup = 1;
                double total_time = 0.0;
                double min_rtt = std::numeric_limits<double>::infinity();
                double max_rtt = 0.0;
                int iters = 0;

#if defined(USE_HIP)
                char *send_buf; char *recv_buf;
                hipMalloc((void**)&send_buf, message); hipMalloc((void**)&recv_buf, message);
                hipMemset(send_buf,'a',message); hipMemset(recv_buf,0,message);
#elif defined(USE_CUDA)
                int dev_count=0; cuda_check(cudaGetDeviceCount(&dev_count));
                cuda_check(cudaSetDevice(rank % (dev_count>0?dev_count:1)));
                char *d_send=nullptr,*d_recv=nullptr; cuda_check(cudaMalloc((void**)&d_send,message));
                cuda_check(cudaMalloc((void**)&d_recv,message));
                cuda_check(cudaMemset(d_send,'a',message)); cuda_check(cudaMemset(d_recv,0,message));
                char *h_send=nullptr,*h_recv=nullptr;
                cuda_check(cudaMallocHost((void**)&h_send,message));
                cuda_check(cudaMallocHost((void**)&h_recv,message));
                memset(h_send,'a',message); memset(h_recv,0,message);
#else
                char *send_buf=(char*)malloc(message), *recv_buf=(char*)malloc(message);
                memset(send_buf,'a',message); memset(recv_buf,0,message);
#endif

                CALI_MARK_BEGIN(warmup_region);
                for (int i=0;i<warmup;i++) {
                    if (rank==0) {
#if defined(USE_CUDA)
                        cuda_check(cudaMemcpy(h_send,d_send,message,cudaMemcpyDeviceToHost));
                        MPI_Send(h_send,message,MPI_CHAR,partner_rank,0,MPI_COMM_WORLD);
                        MPI_Recv(h_recv,message,MPI_CHAR,partner_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                        cuda_check(cudaMemcpy(d_recv,h_recv,message,cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                        MPI_Send(send_buf,message,MPI_CHAR,partner_rank,0,MPI_COMM_WORLD);
                        MPI_Recv(recv_buf,message,MPI_CHAR,partner_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
#else
                        MPI_Send(send_buf,message,MPI_CHAR,partner_rank,0,MPI_COMM_WORLD);
                        MPI_Recv(recv_buf,message,MPI_CHAR,partner_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
#endif
                    } else {
#if defined(USE_CUDA)
                        MPI_Recv(h_recv,message,MPI_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                        cuda_check(cudaMemcpy(d_recv,h_recv,message,cudaMemcpyHostToDevice));
                        cuda_check(cudaMemcpy(h_send,d_send,message,cudaMemcpyDeviceToHost));
                        MPI_Send(h_send,message,MPI_CHAR,0,0,MPI_COMM_WORLD);
#else
                        MPI_Recv(recv_buf,message,MPI_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                        MPI_Send(send_buf,message,MPI_CHAR,0,0,MPI_COMM_WORLD);
#endif
                    }
                }
                CALI_MARK_END(warmup_region);

                CALI_MARK_BEGIN(region_label.c_str());
                for (int i=0;i<PING_PONG_LIMIT;i++) {
                    if (rank==0) {
                        double t0 = MPI_Wtime();
#if defined(USE_CUDA)
                        cuda_check(cudaMemcpy(h_send,d_send,message,cudaMemcpyDeviceToHost));
                        MPI_Send(h_send,message,MPI_CHAR,partner_rank,0,MPI_COMM_WORLD);
                        MPI_Recv(h_recv,message,MPI_CHAR,partner_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                        cuda_check(cudaMemcpy(d_recv,h_recv,message,cudaMemcpyHostToDevice));
#else
                        MPI_Send(send_buf,message,MPI_CHAR,partner_rank,0,MPI_COMM_WORLD);
                        MPI_Recv(recv_buf,message,MPI_CHAR,partner_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
#endif
                        double dt = MPI_Wtime() - t0;
                        total_time += dt;
                        if (dt<min_rtt) min_rtt=dt;
                        if (dt>max_rtt) max_rtt=dt;
                        ++iters;
                    } else {
#if defined(USE_CUDA)
                        MPI_Recv(h_recv,message,MPI_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                        cuda_check(cudaMemcpy(d_recv,h_recv,message,cudaMemcpyHostToDevice));
                        cuda_check(cudaMemcpy(h_send,d_send,message,cudaMemcpyDeviceToHost));
                        MPI_Send(h_send,message,MPI_CHAR,0,0,MPI_COMM_WORLD);
#else
                        MPI_Recv(recv_buf,message,MPI_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                        MPI_Send(send_buf,message,MPI_CHAR,0,0,MPI_COMM_WORLD);
#endif
                    }
                }
                CALI_MARK_END(region_label.c_str());

                if (rank==0) {
                    double avg_rtt = iters ? total_time / iters : 0.0;
#if defined(USE_CALIPER)
                    cali_set_string(comm_phase_attr, "pingpong");
                    cali_set_double(pp_avg_time_sec_attr, avg_rtt);
                    cali_set_double(pp_max_time_sec_attr, max_rtt);
                    cali_set_double(pp_min_time_sec_attr, min_rtt);
#endif
                    double avg_latency_us = (avg_rtt*1e6)/2.0;
                    double min_latency_us = (min_rtt*1e6)/2.0;
                    double max_latency_us = (max_rtt*1e6)/2.0;
                    printf("[PINGPONG] msg=%d  RTT avg=%.6f s min=%.6f s max=%.6f s | latency(one-way) avg=%.1f us min=%.1f us max=%.1f us\n",
                           message, avg_rtt, min_rtt, max_rtt, avg_latency_us, min_latency_us, max_latency_us);
                }

#if defined(USE_HIP)
                hipFree(send_buf); hipFree(recv_buf);
#elif defined(USE_CUDA)
                cuda_check(cudaFreeHost(h_send)); cuda_check(cudaFreeHost(h_recv));
                cuda_check(cudaFree(d_send));     cuda_check(cudaFree(d_recv));
#else
                free(send_buf); free(recv_buf);
#endif
                MPI_Barrier(MPI_COMM_WORLD);
            } // partners
        } // end pingpong

        // ---------------- ALLTOALL (selected) ----------------
        if (op == OpKind::Alltoall) {
            size_t count_per_rank = (size_t)message;
            size_t total_bytes = count_per_rank * (size_t)size;
            int warmup = 1;
            double total=0.0, min_t=std::numeric_limits<double>::infinity(), max_t=0.0;
            int iters=0;

#if defined(USE_HIP)
            char *d_send,*d_recv; hipMalloc((void**)&d_send,total_bytes); hipMalloc((void**)&d_recv,total_bytes);
            hipMemset(d_send,'a',total_bytes); hipMemset(d_recv,0,total_bytes);
            char *h_send=(char*)malloc(total_bytes), *h_recv=(char*)malloc(total_bytes);
            memset(h_send,'a',total_bytes); memset(h_recv,0,total_bytes);
#elif defined(USE_CUDA)
            int dev_count=0; cuda_check(cudaGetDeviceCount(&dev_count));
            cuda_check(cudaSetDevice(rank % (dev_count>0?dev_count:1)));
            char *d_send=nullptr,*d_recv=nullptr; cuda_check(cudaMalloc((void**)&d_send,total_bytes));
            cuda_check(cudaMalloc((void**)&d_recv,total_bytes));
            cuda_check(cudaMemset(d_send,'a',total_bytes)); cuda_check(cudaMemset(d_recv,0,total_bytes));
            char *h_send=nullptr,*h_recv=nullptr; cuda_check(cudaMallocHost((void**)&h_send,total_bytes));
            cuda_check(cudaMallocHost((void**)&h_recv,total_bytes));
            memset(h_send,'a',total_bytes); memset(h_recv,0,total_bytes);
#else
            char *sendbuf=(char*)malloc(total_bytes), *recvbuf=(char*)malloc(total_bytes);
            memset(sendbuf,'b',total_bytes); memset(recvbuf,0,total_bytes);
#endif

            CALI_MARK_BEGIN(warmup_region_aa);
            for (int i=0;i<warmup;i++) {
#if defined(USE_CUDA)
                cuda_check(cudaMemcpy(h_send,d_send,total_bytes,cudaMemcpyDeviceToHost));
                MPI_Alltoall(h_send,(int)count_per_rank,MPI_CHAR,h_recv,(int)count_per_rank,MPI_CHAR,MPI_COMM_WORLD);
                cuda_check(cudaMemcpy(d_recv,h_recv,total_bytes,cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                hipMemcpy(h_send,d_send,total_bytes,hipMemcpyDeviceToHost);
                MPI_Alltoall(h_send,(int)count_per_rank,MPI_CHAR,h_recv,(int)count_per_rank,MPI_CHAR,MPI_COMM_WORLD);
                hipMemcpy(d_recv,h_recv,total_bytes,hipMemcpyHostToDevice);
#else
                MPI_Alltoall(sendbuf,(int)count_per_rank,MPI_CHAR,recvbuf,(int)count_per_rank,MPI_CHAR,MPI_COMM_WORLD);
#endif
            }
            CALI_MARK_END(warmup_region_aa);

            CALI_MARK_BEGIN("Alltoall");
            for (int it=0; it<PING_PONG_LIMIT; ++it) {
                MPI_Barrier(MPI_COMM_WORLD);
                double t0 = MPI_Wtime();
#if defined(USE_CUDA)
                cuda_check(cudaMemcpy(h_send,d_send,total_bytes,cudaMemcpyDeviceToHost));
                MPI_Alltoall(h_send,(int)count_per_rank,MPI_CHAR,h_recv,(int)count_per_rank,MPI_CHAR,MPI_COMM_WORLD);
                cuda_check(cudaMemcpy(d_recv,h_recv,total_bytes,cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                hipMemcpy(h_send,d_send,total_bytes,hipMemcpyDeviceToHost);
                MPI_Alltoall(h_send,(int)count_per_rank,MPI_CHAR,h_recv,(int)count_per_rank,MPI_CHAR,MPI_COMM_WORLD);
                hipMemcpy(d_recv,h_recv,total_bytes,hipMemcpyHostToDevice);
#else
                MPI_Alltoall(sendbuf,(int)count_per_rank,MPI_CHAR,recvbuf,(int)count_per_rank,MPI_CHAR,MPI_COMM_WORLD);
#endif
                double dt = MPI_Wtime()-t0;
                double mx=0; MPI_Reduce(&dt,&mx,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
                if (rank==0) {
                    total+=mx; if (dt<min_t) min_t=dt; if (dt>max_t) max_t=dt; ++iters;
                }
            }
            CALI_MARK_END("Alltoall");

            if (rank==0) {
                double avg = iters? total/iters:0.0;
#if defined(USE_CALIPER)
                cali_set_string(comm_phase_attr,"alltoall");
                cali_set_double(aa_avg_time_sec_attr,avg);
                cali_set_double(aa_max_time_sec_attr,max_t);
                cali_set_double(aa_min_time_sec_attr,min_t);
#endif
                printf("[ALLTOALL] msg/rank=%d  avg=%.6f s  min=%.6f s  max=%.6f s\n", message, avg, min_t, max_t);
            }

#if defined(USE_HIP)
            free(h_send); free(h_recv); hipFree(d_send); hipFree(d_recv);
#elif defined(USE_CUDA)
            cuda_check(cudaFreeHost(h_send)); cuda_check(cudaFreeHost(h_recv));
            cuda_check(cudaFree(d_send)); cuda_check(cudaFree(d_recv));
#else
            free(sendbuf); free(recvbuf);
#endif
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // ---------------- REDUCE (selected) ----------------
        if (op == OpKind::Reduce) {
            size_t count = (size_t)message;
            if (count > INT_MAX) { if(rank==0) fprintf(stderr,"Reduce count too large\n"); MPI_Abort(MPI_COMM_WORLD,3); }
            int warmup=1; double total=0, min_t=std::numeric_limits<double>::infinity(), max_t=0; int iters=0;

#if defined(USE_CUDA)
            char *d_send=nullptr,*d_recv=nullptr; cuda_check(cudaMalloc((void**)&d_send,count)); cuda_check(cudaMalloc((void**)&d_recv,count));
            cuda_check(cudaMemset(d_send,'r',count)); cuda_check(cudaMemset(d_recv,0,count));
            char *h_send=nullptr,*h_recv=nullptr; cuda_check(cudaMallocHost((void**)&h_send,count)); cuda_check(cudaMallocHost((void**)&h_recv,count));
            memset(h_send,'r',count); memset(h_recv,0,count);
#elif defined(USE_HIP)
            char *d_send=nullptr,*d_recv=nullptr; hipMalloc((void**)&d_send,count); hipMalloc((void**)&d_recv,count);
            hipMemset(d_send,'r',count); hipMemset(d_recv,0,count);
            char *h_send=(char*)malloc(count), *h_recv=(char*)malloc(count);
            memset(h_send,'r',count); memset(h_recv,0,count);
#else
            char *sendbuf=(char*)malloc(count), *recvbuf=(char*)malloc(count);
            memset(sendbuf,'r',count); memset(recvbuf,0,count);
#endif

            CALI_MARK_BEGIN(warmup_region_red);
            for (int i=0;i<warmup;i++) {
#if defined(USE_CUDA)
                cuda_check(cudaMemcpy(h_send,d_send,count,cudaMemcpyDeviceToHost));
                MPI_Reduce(h_send,h_recv,(int)count,MPI_CHAR,MPI_SUM,0,MPI_COMM_WORLD);
                cuda_check(cudaMemcpy(d_recv,h_recv,count,cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                hipMemcpy(h_send,d_send,count,hipMemcpyDeviceToHost);
                MPI_Reduce(h_send,h_recv,(int)count,MPI_CHAR,MPI_SUM,0,MPI_COMM_WORLD);
                hipMemcpy(d_recv,h_recv,count,hipMemcpyHostToDevice);
#else
                MPI_Reduce(sendbuf,recvbuf,(int)count,MPI_CHAR,MPI_SUM,0,MPI_COMM_WORLD);
#endif
            }
            CALI_MARK_END(warmup_region_red);

            CALI_MARK_BEGIN("Reduce");
            for (int i=0;i<PING_PONG_LIMIT;i++) {
                MPI_Barrier(MPI_COMM_WORLD);
                double t0 = MPI_Wtime();
#if defined(USE_CUDA)
                cuda_check(cudaMemcpy(h_send,d_send,count,cudaMemcpyDeviceToHost));
                MPI_Reduce(h_send,h_recv,(int)count,MPI_CHAR,MPI_SUM,0,MPI_COMM_WORLD);
                cuda_check(cudaMemcpy(d_recv,h_recv,count,cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                hipMemcpy(h_send,d_send,count,hipMemcpyDeviceToHost);
                MPI_Reduce(h_send,h_recv,(int)count,MPI_CHAR,MPI_SUM,0,MPI_COMM_WORLD);
                hipMemcpy(d_recv,h_recv,count,hipMemcpyHostToDevice);
#else
                MPI_Reduce(sendbuf,recvbuf,(int)count,MPI_CHAR,MPI_SUM,0,MPI_COMM_WORLD);
#endif
                double dt = MPI_Wtime()-t0;
                double mx=0; MPI_Reduce(&dt,&mx,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
                if (rank==0) { total+=mx; if (dt<min_t) min_t=dt; if (dt>max_t) max_t=dt; ++iters; }
            }
            CALI_MARK_END("Reduce");

            if (rank==0) {
                double avg = iters? total/iters:0.0;
#if defined(USE_CALIPER)
                cali_set_string(comm_phase_attr,"reduce");
                cali_set_double(red_avg_time_sec_attr,avg);
                cali_set_double(red_max_time_sec_attr,max_t);
                cali_set_double(red_min_time_sec_attr,min_t);
#endif
                printf("[REDUCE] msg=%d  avg=%.6f s  min=%.6f s  max=%.6f s\n", message, avg, min_t, max_t);
            }

#if defined(USE_CUDA)
            cuda_check(cudaFreeHost(h_send)); cuda_check(cudaFreeHost(h_recv));
            cuda_check(cudaFree(d_send)); cuda_check(cudaFree(d_recv));
#elif defined(USE_HIP)
            free(h_send); free(h_recv); hipFree(d_send); hipFree(d_recv);
#else
            free(sendbuf); free(recvbuf);
#endif
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // ---------------- ALLREDUCE (selected) ----------------
        if (op == OpKind::Allreduce) {
            size_t count = (size_t)message;
            if (count > INT_MAX) { if(rank==0) fprintf(stderr,"Allreduce count too large\n"); MPI_Abort(MPI_COMM_WORLD,4); }
            int warmup=1; double total=0, min_t=std::numeric_limits<double>::infinity(), max_t=0; int iters=0;

#if defined(USE_CUDA)
            char *d_send=nullptr,*d_recv=nullptr; cuda_check(cudaMalloc((void**)&d_send,count)); cuda_check(cudaMalloc((void**)&d_recv,count));
            cuda_check(cudaMemset(d_send,'a',count)); cuda_check(cudaMemset(d_recv,0,count));
            char *h_send=nullptr,*h_recv=nullptr; cuda_check(cudaMallocHost((void**)&h_send,count)); cuda_check(cudaMallocHost((void**)&h_recv,count));
            memset(h_send,'a',count); memset(h_recv,0,count);
#elif defined(USE_HIP)
            char *d_send=nullptr,*d_recv=nullptr; hipMalloc((void**)&d_send,count); hipMalloc((void**)&d_recv,count);
            hipMemset(d_send,'a',count); hipMemset(d_recv,0,count);
            char *h_send=(char*)malloc(count), *h_recv=(char*)malloc(count);
            memset(h_send,'a',count); memset(h_recv,0,count);
#else
            char *sendbuf=(char*)malloc(count), *recvbuf=(char*)malloc(count);
            memset(sendbuf,'a',count); memset(recvbuf,0,count);
#endif

            CALI_MARK_BEGIN(warmup_region_ar);
            for (int i=0;i<warmup;i++) {
#if defined(USE_CUDA)
                cuda_check(cudaMemcpy(h_send,d_send,count,cudaMemcpyDeviceToHost));
                MPI_Allreduce(h_send,h_recv,(int)count,MPI_CHAR,MPI_SUM,MPI_COMM_WORLD);
                cuda_check(cudaMemcpy(d_recv,h_recv,count,cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                hipMemcpy(h_send,d_send,count,hipMemcpyDeviceToHost);
                MPI_Allreduce(h_send,h_recv,(int)count,MPI_CHAR,MPI_SUM,MPI_COMM_WORLD);
                hipMemcpy(d_recv,h_recv,count,hipMemcpyHostToDevice);
#else
                MPI_Allreduce(sendbuf,recvbuf,(int)count,MPI_CHAR,MPI_SUM,MPI_COMM_WORLD);
#endif
            }
            CALI_MARK_END(warmup_region_ar);

            CALI_MARK_BEGIN("Allreduce");
            for (int i=0;i<PING_PONG_LIMIT;i++) {
                MPI_Barrier(MPI_COMM_WORLD);
                double t0 = MPI_Wtime();
#if defined(USE_CUDA)
                cuda_check(cudaMemcpy(h_send,d_send,count,cudaMemcpyDeviceToHost));
                MPI_Allreduce(h_send,h_recv,(int)count,MPI_CHAR,MPI_SUM,MPI_COMM_WORLD);
                cuda_check(cudaMemcpy(d_recv,h_recv,count,cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                hipMemcpy(h_send,d_send,count,hipMemcpyDeviceToHost);
                MPI_Allreduce(h_send,h_recv,(int)count,MPI_CHAR,MPI_SUM,MPI_COMM_WORLD);
                hipMemcpy(d_recv,h_recv,count,hipMemcpyHostToDevice);
#else
                MPI_Allreduce(sendbuf,recvbuf,(int)count,MPI_CHAR,MPI_SUM,MPI_COMM_WORLD);
#endif
                double dt = MPI_Wtime()-t0;
                double mx=0; MPI_Allreduce(&dt,&mx,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
                if (rank==0) { total+=mx; if (dt<min_t) min_t=dt; if (dt>max_t) max_t=dt; ++iters; }
            }
            CALI_MARK_END("Allreduce");

            if (rank==0) {
                double avg = iters? total/iters:0.0;
#if defined(USE_CALIPER)
                cali_set_string(comm_phase_attr,"allreduce");
                cali_set_double(ar_avg_time_sec_attr,avg);
                cali_set_double(ar_max_time_sec_attr,max_t);
                cali_set_double(ar_min_time_sec_attr,min_t);
#endif
                printf("[ALLREDUCE] msg=%d  avg=%.6f s  min=%.6f s  max=%.6f s\n", message, avg, min_t, max_t);
            }

#if defined(USE_CUDA)
            cuda_check(cudaFreeHost(h_send)); cuda_check(cudaFreeHost(h_recv));
            cuda_check(cudaFree(d_send)); cuda_check(cudaFree(d_recv));
#elif defined(USE_HIP)
            free(h_send); free(h_recv); hipFree(d_send); hipFree(d_recv);
#else
            free(sendbuf); free(recvbuf);
#endif
            MPI_Barrier(MPI_COMM_WORLD);
        }

#if defined(USE_CALIPER)
        mgr[message].stop();
#endif
    } // message sweep

#if defined(USE_CALIPER)
    if (rank == 0 && !all_comm_pairs.empty()) {
        adiak::value("all_comm_pairs", all_comm_pairs);
    }
    for (auto &m : mgr) m.second.flush();
#endif

    MPI_Finalize();
    return 0;
}
