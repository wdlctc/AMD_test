#ifndef CHECK_H_
#define CHECK_H_

#include <mpi.h>
#include <rccl.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#define MPICHECK(cmd)                                \
    do                                               \
    {                                                \
        int e = cmd;                                 \
        if (e != MPI_SUCCESS)                        \
        {                                            \
            printf("Failed: MPI error %s:%d '%d'\n", \
                   __FILE__, __LINE__, e);           \
            exit(EXIT_FAILURE);                      \
        }                                            \
    } while (0)

#define HIPCHECK(cmd)                                        \
    do                                                        \
    {                                                         \
        hipError_t e = cmd;                                   \
        if (e != hipSuccess)                                  \
        {                                                     \
            printf("Failed: Hip error %s:%d '%s'\n",         \
                   __FILE__, __LINE__, hipGetErrorString(e)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

#define NCCLCHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        ncclResult_t r = cmd;                                  \
        if (r != ncclSuccess)                                  \
        {                                                      \
            printf("Failed, NCCL error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)
#endif // CHECK_H_
