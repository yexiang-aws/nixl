/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "worker/nixl/nixl_worker.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include "utils/utils.h"
#include <unistd.h>
#include <utility>
#include <sys/time.h>
#include <sys/stat.h>
#include <utils/serdes/serdes.h>
#include <omp.h>

#define ROUND_UP(value, granularity) \
    ((((value) + (granularity) - 1) / (granularity)) * (granularity))

#define CHECK_NIXL_ERROR(result, message)                                                       \
    do {                                                                                        \
        if (0 != result) {                                                                      \
            std::cerr << "NIXL: " << message << " (Error code: " << result << ")" << std::endl; \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while (0)

#if HAVE_CUDA
#define HANDLE_VRAM_SEGMENT(_seg_type) _seg_type = VRAM_SEG;
#else
#define HANDLE_VRAM_SEGMENT(_seg_type)                                        \
    std::cerr << "VRAM segment type not supported without CUDA" << std::endl; \
    std::exit(EXIT_FAILURE);
#endif

#define GET_SEG_TYPE(is_initiator)                                                          \
    ({                                                                                      \
        std::string _seg_type_str = ((is_initiator) ? xferBenchConfig::initiator_seg_type : \
                                                      xferBenchConfig::target_seg_type);    \
        nixl_mem_t _seg_type;                                                               \
        if (0 == _seg_type_str.compare("DRAM")) {                                           \
            _seg_type = DRAM_SEG;                                                           \
        } else if (0 == _seg_type_str.compare("VRAM")) {                                    \
            HANDLE_VRAM_SEGMENT(_seg_type);                                                 \
        } else {                                                                            \
            std::cerr << "Invalid segment type: " << _seg_type_str << std::endl;            \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
        _seg_type;                                                                          \
    })

xferBenchNixlWorker::xferBenchNixlWorker(int *argc, char ***argv, std::vector<std::string> devices)
    : xferBenchWorker(argc, argv) {
    seg_type = GET_SEG_TYPE(isInitiator());

    int rank;
    std::string backend_name;
    nixl_b_params_t backend_params;
    bool enable_pt = xferBenchConfig::enable_pt;
    nixl_thread_sync_t sync_mode = xferBenchConfig::num_threads > 1 ?
        nixl_thread_sync_t::NIXL_THREAD_SYNC_RW :
        nixl_thread_sync_t::NIXL_THREAD_SYNC_DEFAULT;
    char hostname[256];
    nixl_mem_list_t mems;
    std::vector<nixl_backend_t> plugins;

    rank = rt->getRank();

    nixlAgentConfig dev_meta(enable_pt, false, 0, sync_mode);

    agent = new nixlAgent(name, dev_meta);

    agent->getAvailPlugins(plugins);

    if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX) ||
        0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX_MO) ||
        0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_LIBFABRIC) ||
        0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_GPUNETIO) ||
        0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_MOONCAKE) ||
        xferBenchConfig::isStorageBackend()) {
        backend_name = xferBenchConfig::backend;
    } else {
        std::cerr << "Unsupported NIXLBench backend: " << xferBenchConfig::backend << std::endl;
        exit(EXIT_FAILURE);
    }

    agent->getPluginParams(backend_name, mems, backend_params);

    if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX) ||
        0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX_MO)) {
        backend_params["num_threads"] = std::to_string(xferBenchConfig::progress_threads);

        // No need to set device_list if all is specified
        // fallback to backend preference
        if (devices[0] != "all" && devices.size() >= 1) {
            if (isInitiator()) {
                backend_params["device_list"] = devices[rank];
                if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX_MO)) {
                    backend_params["num_ucx_engines"] = xferBenchConfig::num_initiator_dev;
                }
            } else {
                backend_params["device_list"] = devices[rank - xferBenchConfig::num_initiator_dev];
                if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX_MO)) {
                    backend_params["num_ucx_engines"] = xferBenchConfig::num_target_dev;
                }
            }
        }

        if (gethostname(hostname, 256)) {
            std::cerr << "Failed to get hostname" << std::endl;
            exit(EXIT_FAILURE);
        }

        backend_params["num_workers"] = std::to_string(xferBenchConfig::num_threads + 1);

        std::cout << "Init nixl worker, dev "
                  << (("all" == devices[0]) ? "all" : backend_params["device_list"]) << " rank "
                  << rank << ", type " << name << ", hostname " << hostname << std::endl;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_LIBFABRIC)) {
        if (gethostname(hostname, 256)) {
            std::cerr << "Failed to get hostname" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::cout << "Init nixl worker, dev " << (("all" == devices[0]) ? "all" : devices[rank])
                  << " rank " << rank << ", type " << name << ", hostname " << hostname
                  << std::endl;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_GDS)) {
        // Using default param values for GDS backend
        std::cout << "GDS backend" << std::endl;
        backend_params["batch_pool_size"] = std::to_string(xferBenchConfig::gds_batch_pool_size);
        backend_params["batch_limit"] = std::to_string(xferBenchConfig::gds_batch_limit);
        std::cout << "GDS batch pool size: " << xferBenchConfig::gds_batch_pool_size << std::endl;
        std::cout << "GDS batch limit: " << xferBenchConfig::gds_batch_limit << std::endl;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_GDS_MT)) {
        std::cout << "GDS_MT backend" << std::endl;
        backend_params["thread_count"] = std::to_string(xferBenchConfig::gds_mt_num_threads);
        std::cout << "GDS MT Num threads: " << xferBenchConfig::gds_mt_num_threads << std::endl;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_POSIX)) {
        // Set API type parameter for POSIX backend
        if (xferBenchConfig::posix_api_type == XFERBENCH_POSIX_API_AIO) {
            backend_params["use_aio"] = true;
            backend_params["use_uring"] = false;
        } else if (xferBenchConfig::posix_api_type == XFERBENCH_POSIX_API_URING) {
            backend_params["use_aio"] = false;
            backend_params["use_uring"] = true;
        }
        std::cout << "POSIX backend with API type: " << xferBenchConfig::posix_api_type
                  << std::endl;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_GPUNETIO)) {
        std::cout << "GPUNETIO backend, network device " << devices[0] << " GPU device "
                  << xferBenchConfig::gpunetio_device_list << std::endl;
        backend_params["network_devices"] = devices[0];
        backend_params["gpu_devices"] = xferBenchConfig::gpunetio_device_list;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_MOONCAKE)) {
        std::cout << "Mooncake backend" << std::endl;
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_HF3FS)) {
        // Using default param values for HF3FS backend
        std::cout << "HF3FS backend iopool_size " << xferBenchConfig::hf3fs_iopool_size
                  << std::endl;
        backend_params["iopool_size"] = std::to_string(xferBenchConfig::hf3fs_iopool_size);
    } else if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_OBJ)) {
        // Using default param values for OBJ backend
        backend_params["access_key"] = xferBenchConfig::obj_access_key;
        backend_params["secret_key"] = xferBenchConfig::obj_secret_key;
        backend_params["session_token"] = xferBenchConfig::obj_session_token;
        backend_params["bucket"] = xferBenchConfig::obj_bucket_name;
        backend_params["scheme"] = xferBenchConfig::obj_scheme;
        backend_params["region"] = xferBenchConfig::obj_region;
        backend_params["use_virtual_addressing"] =
            xferBenchConfig::obj_use_virtual_addressing ? "true" : "false";
        backend_params["req_checksum"] = xferBenchConfig::obj_req_checksum;

        if (xferBenchConfig::obj_ca_bundle != "") {
            backend_params["ca_bundle"] = xferBenchConfig::obj_ca_bundle;
        }

        if (xferBenchConfig::obj_endpoint_override != "") {
            backend_params["endpoint_override"] = xferBenchConfig::obj_endpoint_override;
        }

        std::cout << "OBJ backend" << std::endl;
    } else {
        std::cerr << "Unsupported NIXLBench backend: " << xferBenchConfig::backend << std::endl;
        exit(EXIT_FAILURE);
    }

    agent->createBackend(backend_name, backend_params, backend_engine);
}

xferBenchNixlWorker::~xferBenchNixlWorker() {
    if (agent) {
        delete agent;
        agent = nullptr;
    }
}

// Convert vector of xferBenchIOV to nixl_reg_dlist_t
static void
iovListToNixlRegDlist(const std::vector<xferBenchIOV> &iov_list, nixl_reg_dlist_t &dlist) {
    nixlBlobDesc desc;
    for (const auto &iov : iov_list) {
        desc.addr = iov.addr;
        desc.len = iov.len;
        desc.devId = iov.devId;
        desc.metaInfo = iov.metaInfo;
        dlist.addDesc(desc);
    }
}

// Convert nixl_xfer_dlist_t to vector of xferBenchIOV
static std::vector<xferBenchIOV>
nixlXferDlistToIOVList(const nixl_xfer_dlist_t &dlist) {
    std::vector<xferBenchIOV> iov_list;
    for (const auto &desc : dlist) {
        iov_list.emplace_back(desc.addr, desc.len, desc.devId);
    }
    return iov_list;
}

// Convert vector of xferBenchIOV to nixl_xfer_dlist_t
static void
iovListToNixlXferDlist(const std::vector<xferBenchIOV> &iov_list, nixl_xfer_dlist_t &dlist) {
    nixlBasicDesc desc;
    for (const auto &iov : iov_list) {
        desc.addr = iov.addr;
        desc.len = iov.len;
        desc.devId = iov.devId;
        dlist.addDesc(desc);
    }
}


enum class AllocationType { POSIX_MEMALIGN, CALLOC, MALLOC };

static bool
allocateXferMemory(size_t buffer_size,
                   void **addr,
                   std::optional<AllocationType> allocation_type = std::nullopt,
                   std::optional<size_t> num = 1) {

    if (!addr) {
        std::cerr << "Invalid address" << std::endl;
        return false;
    }
    if (buffer_size == 0) {
        std::cerr << "Invalid buffer size" << std::endl;
        return false;
    }
    AllocationType type = allocation_type.value_or(AllocationType::MALLOC);

    if (type == AllocationType::POSIX_MEMALIGN) {
        if (xferBenchConfig::page_size == 0) {
            std::cerr << "Error: Invalid page size returned by sysconf" << std::endl;
            return false;
        }
        int rc = posix_memalign(addr, xferBenchConfig::page_size, buffer_size);
        if (rc != 0 || !*addr) {
            std::cerr << "Failed to allocate " << buffer_size
                      << " bytes of page-aligned DRAM memory" << std::endl;
            return false;
        }
        memset(*addr, 0, buffer_size);
    } else if (type == AllocationType::CALLOC) {
        *addr = calloc(num.value_or(1), buffer_size);
        if (!*addr) {
            std::cerr << "Failed to allocate " << buffer_size << " bytes of DRAM memory"
                      << std::endl;
            return false;
        }
    } else if (type == AllocationType::MALLOC) {
        *addr = malloc(buffer_size);
        if (!*addr) {
            std::cerr << "Failed to allocate " << buffer_size << " bytes of DRAM memory"
                      << std::endl;
            return false;
        }
    } else {
        std::cerr << "Invalid allocation type" << std::endl;
        return false;
    }
    return true;
}

std::optional<xferBenchIOV>
xferBenchNixlWorker::initBasicDescDram(size_t buffer_size, int mem_dev_id) {
    void *addr;

    AllocationType type = AllocationType::CALLOC;
    if (xferBenchConfig::storage_enable_direct) {
        type = AllocationType::POSIX_MEMALIGN;
    }

    if (!allocateXferMemory(buffer_size, &addr, type)) {
        std::cerr << "Failed to allocate " << buffer_size << " bytes of DRAM memory" << std::endl;
        return std::nullopt;
    }
    if (isInitiator()) {
        memset(addr, XFERBENCH_INITIATOR_BUFFER_ELEMENT, buffer_size);
    } else if (isTarget()) {
        memset(addr, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size);
    }

    // TODO: Does device id need to be set for DRAM?
    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, mem_dev_id);
}

#if HAVE_CUDA
static std::optional<xferBenchIOV>
getVramDescCuda(int devid, size_t buffer_size, uint8_t memset_value) {
    void *addr;
    CHECK_CUDA_ERROR(cudaMalloc(&addr, buffer_size), "Failed to allocate CUDA buffer");
    CHECK_CUDA_ERROR(cudaMemset(addr, memset_value, buffer_size), "Failed to set device memory");

    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, devid);
}

static std::optional<xferBenchIOV>
getVramDescCudaVmm(int devid, size_t buffer_size, uint8_t memset_value) {
#if HAVE_CUDA_FABRIC
    CUdeviceptr addr = 0;
    CUmemAllocationProp prop = {};
    CUmemAccessDesc access = {};

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.allocFlags.gpuDirectRDMACapable = 1;
    prop.location.id = devid;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    // Get the allocation granularity
    size_t granularity = 0;
    CHECK_CUDA_DRIVER_ERROR(
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "Failed to get allocation granularity");
    std::cout << "Granularity: " << granularity << std::endl;

    size_t padded_size = ROUND_UP(buffer_size, granularity);
    CUmemGenericAllocationHandle handle;
    CHECK_CUDA_DRIVER_ERROR(cuMemCreate(&handle, padded_size, &prop, 0),
                            "Failed to create allocation");

    // Reserve the memory address
    CHECK_CUDA_DRIVER_ERROR(cuMemAddressReserve(&addr, padded_size, granularity, 0, 0),
                            "Failed to reserve address");

    // Map the memory
    CHECK_CUDA_DRIVER_ERROR(cuMemMap(addr, padded_size, 0, handle, 0), "Failed to map memory");

    std::cout << "Address: " << std::hex << std::showbase << addr << " Buffer size: " << std::dec
              << buffer_size << " Padded size: " << std::dec << padded_size << std::endl;

    // Set the memory access rights
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = devid;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDA_DRIVER_ERROR(cuMemSetAccess(addr, buffer_size, &access, 1), "Failed to set access");

    // Set memory content based on role
    CHECK_CUDA_DRIVER_ERROR(cuMemsetD8(addr, memset_value, buffer_size),
                            "Failed to set VMM device memory");

    return std::optional<xferBenchIOV>(
        std::in_place, (uintptr_t)addr, buffer_size, devid, padded_size, handle);

#else
    std::cerr << "CUDA_FABRIC is not supported" << std::endl;
    return std::nullopt;
#endif /* HAVE_CUDA_FABRIC */
}

static std::optional<xferBenchIOV>
getVramDesc(int devid, size_t buffer_size, bool isInit) {
    CHECK_CUDA_ERROR(cudaSetDevice(devid), "Failed to set device");
    uint8_t memset_value =
        isInit ? XFERBENCH_INITIATOR_BUFFER_ELEMENT : XFERBENCH_TARGET_BUFFER_ELEMENT;

    if (xferBenchConfig::enable_vmm) {
        return getVramDescCudaVmm(devid, buffer_size, memset_value);
    } else {
        return getVramDescCuda(devid, buffer_size, memset_value);
    }
}

std::optional<xferBenchIOV>
xferBenchNixlWorker::initBasicDescVram(size_t buffer_size, int mem_dev_id) {
    if (IS_PAIRWISE_AND_SG()) {
        int devid = rt->getRank();

        if (isTarget()) {
            devid -= xferBenchConfig::num_initiator_dev;
        }

        if (devid != mem_dev_id) {
            return std::nullopt;
        }
    }

    return getVramDesc(mem_dev_id, buffer_size, isInitiator());
}
#endif /* HAVE_CUDA */

static std::vector<xferFileState>
createFileFds(std::string name, int num_files) {
    std::vector<xferFileState> fds;
    int flags = O_RDWR | O_CREAT;

    if (!xferBenchConfig::isStorageBackend()) {
        std::cerr << "Unknown storage backend: " << xferBenchConfig::backend << std::endl;
        exit(EXIT_FAILURE);
    }

    if (xferBenchConfig::storage_enable_direct) {
        flags |= O_DIRECT;
    }

    const std::string file_path = xferBenchConfig::filepath != "" ?
        xferBenchConfig::filepath :
        std::filesystem::current_path().string();
    std::string file_backend = xferBenchConfig::backend;
    std::transform(file_backend.begin(), file_backend.end(), file_backend.begin(), ::tolower);
    const std::string file_name_prefix = "/nixlbench_" + file_backend + "_test_file_";

    for (int i = 0; i < num_files; i++) {
        std::string file_name = file_path + file_name_prefix + name + "_" + std::to_string(i);
        std::cout << "Creating file: " << file_name << std::endl;

        uint64_t file_size = 0;
        if (XFERBENCH_OP_READ == xferBenchConfig::op_type) {
            struct stat st;
            if (::stat(file_name.c_str(), &st) == 0) {
                std::cout << "File " << file_name << " exists, size: " << st.st_size << std::endl;
                file_size = st.st_size;
            } else {
                std::cout << "File " << file_name << " does not exist, will be created."
                          << std::endl;
            }
        }

        int fd = open(file_name.c_str(), flags, 0744);
        if (fd < 0) {
            std::cerr << "Failed to open file: " << file_name << " with error: " << strerror(errno)
                      << std::endl;
            for (int j = 0; j < i; j++) {
                close(fds[j].fd);
            }
            return {};
        }
        fds.emplace_back(xferFileState{fd, file_size, 0});
    }
    return fds;
}

std::optional<xferBenchIOV>
xferBenchNixlWorker::initBasicDescFile(size_t buffer_size, xferFileState &fstate, int mem_dev_id) {
    int fd = fstate.fd;
    uint64_t start_offset = fstate.offset;
    uint64_t end_offset = fstate.offset + buffer_size;
    auto ret = std::optional<xferBenchIOV>(std::in_place, fstate.offset, buffer_size, fd);

    fstate.offset = end_offset;

    // If in READ mode, only write if the region is not already present in the file
    if (XFERBENCH_OP_READ == xferBenchConfig::op_type && end_offset <= fstate.file_size) {
        return ret;
    }

    // Fill up with data
    void *buf;
    AllocationType type = AllocationType::MALLOC;

    if (xferBenchConfig::storage_enable_direct) {
        type = AllocationType::POSIX_MEMALIGN;
    }

    if (!allocateXferMemory(buffer_size, &buf, type) || !buf) {
        std::cerr << "Failed to allocate " << buffer_size << " bytes of memory" << std::endl;
        return std::nullopt;
    }

    // File is always initialized with XFERBENCH_TARGET_BUFFER_ELEMENT
    memset(buf, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size);

    size_t offset = start_offset;
    while (buffer_size > 0) {
        ssize_t rc = pwrite(fd, buf, buffer_size, offset);
        if (rc < 0) {
            std::cerr << "Failed to write to file: " << fd << " with error: " << strerror(errno)
                      << std::endl;
            return std::nullopt;
        }

        buffer_size -= rc;
        offset += rc;
    }

    free(buf);

    if (end_offset > fstate.file_size) fstate.file_size = end_offset;

    return ret;
}

std::optional<xferBenchIOV>
xferBenchNixlWorker::initBasicDescObj(size_t buffer_size, int mem_dev_id, std::string name) {
    return std::optional<xferBenchIOV>(std::in_place, 0, buffer_size, mem_dev_id, name);
}

void
xferBenchNixlWorker::cleanupBasicDescDram(xferBenchIOV &iov) {
    free((void *)iov.addr);
}

#if HAVE_CUDA
void
xferBenchNixlWorker::cleanupBasicDescVram(xferBenchIOV &iov) {
    CHECK_CUDA_ERROR(cudaSetDevice(iov.devId), "Failed to set device");

    if (xferBenchConfig::enable_vmm) {
        CHECK_CUDA_DRIVER_ERROR(cuMemUnmap(iov.addr, iov.len), "Failed to unmap memory");
        CHECK_CUDA_DRIVER_ERROR(cuMemRelease(iov.handle), "Failed to release memory");
        CHECK_CUDA_DRIVER_ERROR(cuMemAddressFree(iov.addr, iov.padded_size),
                                "Failed to free reserved address");
    } else {
        CHECK_CUDA_ERROR(cudaFree((void *)iov.addr), "Failed to deallocate CUDA buffer");
    }
}
#endif /* HAVE_CUDA */

void
xferBenchNixlWorker::cleanupBasicDescFile(xferBenchIOV &iov) {
    close(iov.devId);
}

void
xferBenchNixlWorker::cleanupBasicDescObj(xferBenchIOV &iov) {
    if (!xferBenchUtils::rmObjS3(iov.metaInfo)) {
        std::cerr << "Failed to remove S3 object: " << iov.metaInfo << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::vector<std::vector<xferBenchIOV>>
xferBenchNixlWorker::allocateMemory(int num_threads) {
    std::vector<std::vector<xferBenchIOV>> iov_lists;
    size_t i, buffer_size, num_devices = 0;
    nixl_opt_args_t opt_args;

    if (isInitiator()) {
        num_devices = xferBenchConfig::num_initiator_dev;
    } else if (isTarget()) {
        num_devices = xferBenchConfig::num_target_dev;
    }
    buffer_size = xferBenchConfig::total_buffer_size / (num_devices * num_threads);

    if (xferBenchConfig::storage_enable_direct) {
        if (xferBenchConfig::page_size == 0) {
            std::cerr << "Error: Invalid page size returned by sysconf" << std::endl;
            exit(EXIT_FAILURE);
        }
        buffer_size =
            ((buffer_size + xferBenchConfig::page_size - 1) / xferBenchConfig::page_size) *
            xferBenchConfig::page_size;
    }

    opt_args.backends.push_back(backend_engine);

    if (xferBenchConfig::backend == XFERBENCH_BACKEND_OBJ) {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        uint64_t timestamp = tv.tv_sec * 1000000ULL + tv.tv_usec;

        for (int list_idx = 0; list_idx < num_threads; list_idx++) {
            std::vector<xferBenchIOV> iov_list;
            for (i = 0; i < num_devices; i++) {
                std::optional<xferBenchIOV> basic_desc;
                std::string unique_name = "nixlbench_obj" + std::to_string(list_idx) + "_" +
                    std::to_string(i) + "_" + std::to_string(timestamp);

                if (xferBenchConfig::op_type == XFERBENCH_OP_READ) {
                    if (!xferBenchUtils::putObjS3(buffer_size, unique_name)) {
                        std::cerr << "Failed to put S3 object: " << unique_name << std::endl;
                        continue;
                    }
                }

                basic_desc = initBasicDescObj(buffer_size, i, unique_name);
                if (basic_desc) {
                    std::cout << "Creating obj: " << unique_name << std::endl;
                    iov_list.push_back(basic_desc.value());
                }
            }
            nixl_reg_dlist_t desc_list(OBJ_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->registerMem(desc_list, &opt_args), "registerMem failed");
            remote_iovs.push_back(iov_list);
        }
    } else if (xferBenchConfig::isStorageBackend()) {
        int num_buffers = num_threads * num_devices;
        int num_files = xferBenchConfig::num_files;
        int remainder_buffers = num_buffers % num_files;

        if (num_files > num_buffers) {
            std::cerr << "Error: number of buffers (" << num_buffers
                      << ") needs to be bigger or equal to the number of files (" << num_files
                      << "). Try adjusting num_files." << std::endl;
            exit(EXIT_FAILURE);
        }

        if (remainder_buffers != 0) {
            std::cerr << "Error: number of buffers (" << num_buffers
                      << ") needs to be divisible by the number of files (" << num_files
                      << "). Try adjusting num_files." << std::endl;
            exit(EXIT_FAILURE);
        }

        remote_fds = createFileFds(getName(), num_files);
        if (remote_fds.empty()) {
            std::cerr << "Failed to create " << xferBenchConfig::backend << " file" << std::endl;
            exit(EXIT_FAILURE);
        }

        int file_idx = 0;
        for (int list_idx = 0; list_idx < num_threads; list_idx++) {
            std::vector<xferBenchIOV> iov_list;
            for (i = 0; i < num_devices; i++) {
                std::optional<xferBenchIOV> basic_desc;
                basic_desc = initBasicDescFile(buffer_size, remote_fds[file_idx], i);
                if (basic_desc) {
                    iov_list.push_back(basic_desc.value());
                }
                file_idx += 1;
                if (file_idx >= num_files) file_idx = 0;
            }
            nixl_reg_dlist_t desc_list(FILE_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->registerMem(desc_list, &opt_args), "registerMem failed");
            remote_iovs.push_back(iov_list);
        }
    }

    for (int list_idx = 0; list_idx < num_threads; list_idx++) {
        std::vector<xferBenchIOV> iov_list;
        for (i = 0; i < num_devices; i++) {
            std::optional<xferBenchIOV> basic_desc;

            switch (seg_type) {
            case DRAM_SEG:
                basic_desc = initBasicDescDram(buffer_size, i);
                break;
#if HAVE_CUDA
            case VRAM_SEG:
                basic_desc = initBasicDescVram(buffer_size, i);
                break;
#endif
            default:
                std::cerr << "Unsupported mem type: " << seg_type << std::endl;
                exit(EXIT_FAILURE);
            }

            if (basic_desc) {
                if (!remote_iovs.empty()) {
                    basic_desc.value().metaInfo = remote_iovs[list_idx][i].metaInfo;
                }
                iov_list.push_back(basic_desc.value());
            }
        }

        nixl_reg_dlist_t desc_list(seg_type);
        iovListToNixlRegDlist(iov_list, desc_list);
        CHECK_NIXL_ERROR(agent->registerMem(desc_list, &opt_args), "registerMem failed");
        iov_lists.push_back(iov_list);
    }

    return iov_lists;
}

void
xferBenchNixlWorker::deallocateMemory(std::vector<std::vector<xferBenchIOV>> &iov_lists) {
    nixl_opt_args_t opt_args;

    opt_args.backends.push_back(backend_engine);
    for (auto &iov_list : iov_lists) {
        for (auto &iov : iov_list) {
            switch (seg_type) {
            case DRAM_SEG:
                cleanupBasicDescDram(iov);
                break;
#if HAVE_CUDA
            case VRAM_SEG:
                cleanupBasicDescVram(iov);
                break;
#endif
            default:
                std::cerr << "Unsupported mem type: " << seg_type << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        nixl_reg_dlist_t desc_list(seg_type);
        iovListToNixlRegDlist(iov_list, desc_list);
        CHECK_NIXL_ERROR(agent->deregisterMem(desc_list, &opt_args), "deregisterMem failed");
    }

    if (xferBenchConfig::backend == XFERBENCH_BACKEND_OBJ) {
        for (auto &iov_list : remote_iovs) {
            for (auto &iov : iov_list) {
                cleanupBasicDescObj(iov);
            }
            nixl_reg_dlist_t desc_list(OBJ_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->deregisterMem(desc_list, &opt_args), "deregisterMem failed");
        }
    } else if (xferBenchConfig::isStorageBackend()) {
        for (auto &iov_list : remote_iovs) {
            for (auto &iov : iov_list) {
                cleanupBasicDescFile(iov);
            }
            nixl_reg_dlist_t desc_list(FILE_SEG);
            iovListToNixlRegDlist(iov_list, desc_list);
            CHECK_NIXL_ERROR(agent->deregisterMem(desc_list, &opt_args), "deregisterMem failed");
        }
    }
}

int
xferBenchNixlWorker::exchangeMetadata() {
    int meta_sz, ret = 0;

    // Skip metadata exchange for storage backends or when ETCD is not available
    if (xferBenchConfig::isStorageBackend()) {
        return 0;
    }

    if (isTarget()) {
        std::string local_metadata;
        const char *buffer;
        int destrank;

        agent->getLocalMD(local_metadata);

        buffer = local_metadata.data();
        meta_sz = local_metadata.size();

        if (IS_PAIRWISE_AND_SG()) {
            destrank = rt->getRank() - xferBenchConfig::num_target_dev;
            // XXX: Fix up the rank, depends on processes distributed on hosts
            // assumes placement is adjacent ranks to same node
        } else {
            destrank = 0;
        }
        rt->sendInt(&meta_sz, destrank);
        rt->sendChar((char *)buffer, meta_sz, destrank);
    } else if (isInitiator()) {
        std::string remote_agent;
        int srcrank;

        if (IS_PAIRWISE_AND_SG()) {
            srcrank = rt->getRank() + xferBenchConfig::num_initiator_dev;
            // XXX: Fix up the rank, depends on processes distributed on hosts
            // assumes placement is adjacent ranks to same node
        } else {
            srcrank = 1;
        }

        ret = rt->recvInt(&meta_sz, srcrank);
        if (ret < 0) {
            std::cerr << "NIXL: failed to receive metadata size" << std::endl;
            return ret;
        }

        std::string remote_metadata(meta_sz, '\0');
        ret = rt->recvChar(remote_metadata.data(), meta_sz, srcrank);
        if (ret < 0) {
            std::cerr << "NIXL: failed to receive metadata" << std::endl;
            return ret;
        }

        nixl_status_t status = agent->loadRemoteMD(remote_metadata, remote_agent);
        if (status != NIXL_SUCCESS) {
            std::cerr << "NIXL: loadRemoteMD failed: " << nixlEnumStrings::statusStr(status)
                      << std::endl;
            return -1;
        }
    }

    return ret;
}

std::vector<std::vector<xferBenchIOV>>
xferBenchNixlWorker::exchangeIOV(const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                                 size_t block_size) {
    std::vector<std::vector<xferBenchIOV>> res;
    int desc_str_sz;

    if (xferBenchConfig::isStorageBackend()) {
        size_t fd_idx = 0;
        uint64_t file_offset = 0;
        for (auto &iov_list : local_iovs) {
            std::vector<xferBenchIOV> remote_iov_list;
            for (auto &iov : iov_list) {
                if (XFERBENCH_BACKEND_OBJ == xferBenchConfig::backend) {
                    std::optional<xferBenchIOV> basic_desc;
                    basic_desc = initBasicDescObj(iov.len, iov.devId, iov.metaInfo);
                    if (basic_desc) {
                        remote_iov_list.push_back(basic_desc.value());
                    }
                } else {
                    xferBenchIOV iov_remote(iov);
                    iov_remote.addr = file_offset;
                    iov_remote.len = block_size;
                    iov_remote.devId = remote_fds[fd_idx].fd;
                    remote_iov_list.push_back(iov_remote);
                    fd_idx++;
                    if (fd_idx >= remote_fds.size()) {
                        file_offset += block_size;
                        fd_idx = 0;
                    }
                }
            }
            res.push_back(remote_iov_list);
        }
    } else {
        for (const auto &local_iov : local_iovs) {
            nixlSerDes ser_des;
            nixl_xfer_dlist_t local_desc(seg_type);

            iovListToNixlXferDlist(local_iov, local_desc);

            if (isTarget()) {
                int destrank;
                if (IS_PAIRWISE_AND_SG()) {
                    destrank = rt->getRank() - xferBenchConfig::num_target_dev;
                    // XXX: Fix up the rank, depends on processes distributed on hosts
                    // assumes placement is adjacent ranks to same node
                } else {
                    destrank = 0;
                }

                local_desc.serialize(&ser_des);
                std::string desc_str = ser_des.exportStr();
                desc_str_sz = desc_str.size();
                rt->sendInt(&desc_str_sz, destrank);
                rt->sendChar(desc_str.data(), desc_str.size(), destrank);
            } else if (isInitiator()) {
                int srcrank;
                if (IS_PAIRWISE_AND_SG()) {
                    srcrank = rt->getRank() + xferBenchConfig::num_initiator_dev;
                    // XXX: Fix up the rank, depends on processes distributed on hosts
                    // assumes placement is adjacent ranks to same node
                } else {
                    srcrank = 1;
                }

                if (rt->recvInt(&desc_str_sz, srcrank) != 0) {
                    std::cerr << "NIXL: failed to receive metadata size" << std::endl;
                    std::exit(EXIT_FAILURE);
                }

                std::string desc_str;
                desc_str.resize(desc_str_sz, '\0');
                if (rt->recvChar(desc_str.data(), desc_str.size(), srcrank) != 0) {
                    std::cerr << "NIXL: failed to receive metadata" << std::endl;
                    std::exit(EXIT_FAILURE);
                }

                ser_des.importStr(desc_str);

                nixl_xfer_dlist_t remote_desc(&ser_des);
                res.emplace_back(nixlXferDlistToIOVList(remote_desc));
            }
        }
    }
    // Ensure all processes have completed the exchange with a barrier/sync
    synchronize();
    return res;
}

static int
execTransfer(nixlAgent *agent,
             const std::vector<std::vector<xferBenchIOV>> &local_iovs,
             const std::vector<std::vector<xferBenchIOV>> &remote_iovs,
             const nixl_xfer_op_t op,
             const int num_iter,
             const int num_threads,
             xferBenchStats &stats) {
    int ret = 0;
    stats.clear();

    xferBenchTimer total_timer;
#pragma omp parallel num_threads(num_threads)
    {
        xferBenchStats thread_stats;
        thread_stats.reserve(num_iter);
        xferBenchTimer timer;
        const int tid = omp_get_thread_num();
        const auto &local_iov = local_iovs[tid];
        const auto &remote_iov = remote_iovs[tid];

        // TODO: fetch local_desc and remote_desc directly from config
        nixl_xfer_dlist_t local_desc(GET_SEG_TYPE(true));
        nixl_xfer_dlist_t remote_desc(GET_SEG_TYPE(false));

        if (XFERBENCH_BACKEND_OBJ == xferBenchConfig::backend) {
            remote_desc = nixl_xfer_dlist_t(OBJ_SEG);
        } else if (xferBenchConfig::isStorageBackend()) {
            remote_desc = nixl_xfer_dlist_t(FILE_SEG);
        }

        iovListToNixlXferDlist(local_iov, local_desc);
        iovListToNixlXferDlist(remote_iov, remote_desc);

        nixl_opt_args_t params;
        nixl_b_params_t b_params;
        nixlXferReqH *req;
        nixl_status_t rc;
        std::string target;

        if (xferBenchConfig::isStorageBackend()) {
            target = "initiator";
        } else {
            params.notifMsg = "0xBEEF";
            params.hasNotif = true;
            target = "target";
        }

        CHECK_NIXL_ERROR(agent->createXferReq(op, local_desc, remote_desc, target, req, &params),
                         "createXferReq failed");

        const nixlTime::us_t prepare_duration = timer.lap();
        thread_stats.prepare_duration.add(prepare_duration);

        for (int i = 0; i < num_iter; i++) {
            rc = agent->postXferReq(req);
            const nixlTime::us_t post_duration = timer.lap();
            thread_stats.post_duration.add(post_duration);
            while (NIXL_IN_PROG == rc) {
                /* XXX agent isn't const because the getXferStatus() is not const  */
                rc = agent->getXferStatus(req);
            }

            if (NIXL_SUCCESS != rc) {
                std::cout << "NIXL Xfer failed with status: " << nixlEnumStrings::statusStr(rc)
                          << std::endl;
                ret = -1;
                break;
            }

            const nixlTime::us_t transfer_duration = timer.lap();
            thread_stats.transfer_duration.add(transfer_duration);
        }

        rc = agent->releaseXferReq(req);
        if (NIXL_SUCCESS != rc) {
            std::cout << "NIXL releaseXferReq failed" << std::endl;
            ret = -1;
        }

#pragma omp critical
        { stats.add(thread_stats); }
    }
    const nixlTime::us_t total_duration = total_timer.lap();
    stats.total_duration.add(total_duration);
    return ret;
}

std::variant<xferBenchStats, int>
xferBenchNixlWorker::transfer(size_t block_size,
                              const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                              const std::vector<std::vector<xferBenchIOV>> &remote_iovs) {
    int num_iter = xferBenchConfig::num_iter / xferBenchConfig::num_threads;
    int skip = xferBenchConfig::warmup_iter / xferBenchConfig::num_threads;
    xferBenchStats stats;
    int ret = 0;
    nixl_xfer_op_t xfer_op = XFERBENCH_OP_READ == xferBenchConfig::op_type ? NIXL_READ : NIXL_WRITE;
    // int completion_flag = 1;

    // Reduce skip by 10x for large block sizes
    if (block_size > LARGE_BLOCK_SIZE) {
        skip /= xferBenchConfig::large_blk_iter_ftr;
        num_iter /= xferBenchConfig::large_blk_iter_ftr;
    }

    if (skip > 0) {
        ret = execTransfer(
            agent, local_iovs, remote_iovs, xfer_op, skip, xferBenchConfig::num_threads, stats);
        if (ret < 0) {
            return std::variant<xferBenchStats, int>(ret);
        }
    }

    // Synchronize to ensure all processes have completed the warmup (iter and polling)
    synchronize();

    stats.clear();

    ret = execTransfer(
        agent, local_iovs, remote_iovs, xfer_op, num_iter, xferBenchConfig::num_threads, stats);
    if (ret < 0) {
        return std::variant<xferBenchStats, int>(ret);
    }

    synchronize();
    return std::variant<xferBenchStats, int>(stats);
}

void
xferBenchNixlWorker::poll(size_t block_size) {
    nixl_notifs_t notifs;
    nixl_status_t status;
    int skip = 0, num_iter = 0, total_iter = 0;

    skip = xferBenchConfig::warmup_iter;
    num_iter = xferBenchConfig::num_iter;
    // Reduce skip by 10x for large block sizes
    if (block_size > LARGE_BLOCK_SIZE) {
        skip /= xferBenchConfig::large_blk_iter_ftr;
        num_iter /= xferBenchConfig::large_blk_iter_ftr;
    }
    total_iter = skip + num_iter;

    /* Ensure warmup is done*/
    do {
        status = agent->getNotifs(notifs);
    } while (status == NIXL_SUCCESS && skip != int(notifs["initiator"].size()));
    synchronize();

    /* Polling for actual iterations*/
    do {
        status = agent->getNotifs(notifs);
    } while (status == NIXL_SUCCESS && total_iter != int(notifs["initiator"].size()));
    synchronize();
}

int
xferBenchNixlWorker::synchronizeStart() {
    // For storage backends without ETCD, no synchronization needed
    if (xferBenchConfig::isStorageBackend() && xferBenchConfig::etcd_endpoints.empty()) {
        std::cout << "Single instance storage backend - no synchronization needed" << std::endl;
        return 0;
    }

    if (IS_PAIRWISE_AND_SG()) {
        std::cout << "Waiting for all processes to start... (expecting " << rt->getSize()
                  << " total: " << xferBenchConfig::num_initiator_dev << " initiators and "
                  << xferBenchConfig::num_target_dev << " targets)" << std::endl;
    } else {
        std::cout << "Waiting for all processes to start... (expecting " << rt->getSize()
                  << " total" << std::endl;
    }
    if (rt) {
        int ret = rt->barrier("start_barrier");
        if (ret != 0) {
            std::cerr << "Failed to synchronize at start barrier" << std::endl;
            return -1;
        }
        std::cout << "All processes are ready to proceed" << std::endl;
        return 0;
    }
    return -1;
}
