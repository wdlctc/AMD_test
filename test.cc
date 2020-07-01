#include <stdio.h>
#include <string>
#include <iostream>
#include <unordered_map>
#include <rccl.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "tools/check.h"
#include "mpi_context.h"

const int test_times = 100;
const size_t maxerrcount = 10;
const int nDev = 1;

const int size = 16 * 1024 * 1024;

hipStream_t *s;
ncclComm_t *comm = new ncclComm_t();

float *gradients;
float *sendbuff  = nullptr;;
float *recvbuff  = nullptr;;

void gradients_init(float* &gradients, const size_t &size){
    for (int i = 0; i < size; i++){
        gradients[i] = i;
    }
}

void gradients_check(float* &gradients, const size_t &size, int times){
    std::cout << "After all reduce" << std::endl;
    for (int i = 0; i < size; i++)
    {
	    float target_value = i * times;
        if (std::fabs(gradients[i] - target_value) > 0.0001 * target_value)
        {
            std::cout <<  "test_host fail " <<gradients[i] << " != " << target_value << "\n" ;
            return;
        }
    }

    std::cout << "all reduce success!" << std::endl;
}

void initialization(MPIContext &mpi_context)
{
    //initializing MPI
    MPICHECK(MPI_Init(NULL, NULL));

    mpi_context.mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &mpi_context.local_comm);
    
    MPICHECK(MPI_Comm_rank(mpi_context.mpi_comm, &mpi_context.myRank));
    MPICHECK(MPI_Comm_size(mpi_context.mpi_comm, &mpi_context.nRanks));
    MPICHECK(MPI_Comm_rank(mpi_context.local_comm, &mpi_context.localRank));
    MPICHECK(MPI_Comm_size(mpi_context.local_comm, &mpi_context.nlocalRank));

    MPI_Comm_split(MPI_COMM_WORLD, mpi_context.localRank, mpi_context.myRank,
                   &mpi_context.cross_comm);

    HIPCHECK(hipSetDevice(mpi_context.localRank));

    gradients = new float[size];
    gradients_init(gradients, size);
    
    HIPCHECK(hipMalloc(&sendbuff, size * sizeof(float)));
    HIPCHECK(hipMemset(sendbuff, 0, size * sizeof(float)));
    HIPCHECK(hipMalloc(&recvbuff, size * sizeof(float)));
    HIPCHECK(hipMemset(recvbuff, 0, size * sizeof(float)));

    s = (hipStream_t *)malloc(sizeof(hipStream_t) * nDev);
    HIPCHECK(hipStreamCreate(s));

    ncclUniqueId id;
    if (mpi_context.localRank == 0){
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, mpi_context.local_comm));

    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclCommInitRank(comm, mpi_context.nlocalRank, id, mpi_context.localRank));
    NCCLCHECK(ncclGroupEnd());
}

void finalization()
{
    //finalizing MPI
    MPICHECK(MPI_Finalize());
    ncclCommDestroy(*comm);
    HIPCHECK(hipFree(sendbuff));
    HIPCHECK(hipFree(recvbuff));
}


void test_device(const MPIContext mpi_context)
{
    // warming up
    HIPCHECK(hipMemcpy(sendbuff, gradients, size * sizeof(float), hipMemcpyHostToDevice));
    NCCLCHECK(ncclReduce((const void *)sendbuff, (void *)recvbuff, 0, ncclFloat, ncclSum, 0, *comm, *s));
    HIPCHECK(hipStreamSynchronize(*s));

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto start_time = std::chrono::high_resolution_clock::now();
    for(int itr = 0; itr < test_times; itr++)
    {
        // ncclReduce to local root
        NCCLCHECK(ncclReduce((const void *)sendbuff, (void *)recvbuff, size, ncclFloat, ncclSum, 0, *comm, *s));

        // mpiallreduce between remote servers
        if(mpi_context.localRank == 0) {
            HIPCHECK(hipStreamSynchronize(*s));
            HIPCHECK(hipMemcpyAsync(gradients, recvbuff, size * sizeof(float), hipMemcpyDeviceToHost, *s));
            MPI_Allreduce(gradients, gradients, size, MPI_FLOAT, MPI_SUM, mpi_context.cross_comm);
            HIPCHECK(hipMemcpyAsync(recvbuff, gradients, size * sizeof(float), hipMemcpyHostToDevice, *s));
            MPICHECK(MPI_Barrier(mpi_context.cross_comm));
        }

        // ncclBroadcast on local ranks
        NCCLCHECK(ncclBroadcast((const void *)recvbuff, (void *)recvbuff, size, ncclFloat, 0, *comm, *s));
    }
    HIPCHECK(hipStreamSynchronize(*s));

    std::chrono::duration<double> elapsed_seconds = (std::chrono::high_resolution_clock::now() - start_time) / test_times;
	std::cout << "test_device_nccl, gradient size: " << std::to_string(size) << ", elapsed time: " << elapsed_seconds.count() << "s, Throughput: " << std::to_string(size*4 / elapsed_seconds.count() / 1024 / 1024 / 1024) << "GB/s\n";

    HIPCHECK(hipMemcpy(gradients, recvbuff, sizeof(float) * size, hipMemcpyDeviceToHost));

    gradients_check(gradients, size, mpi_context.nRanks);

}

int main()
{
    initialization(mpi_context);

    std::cout << "=======================================================================" << std::endl;
    std::cout << "test_device nccl" << std::endl;
    test_device(mpi_context);

    finalization();

    return 0;
}

