#ifndef MPI_CONTEXT
#define MPI_CONTEXT

#include <mpi.h>
#include <vector>

struct MPIContext {

    // Private MPI communicator for Horovod to ensure no collisions with other
    // threads using MPI.
    MPI_Comm mpi_comm;

    // Node-local communicator.
    MPI_Comm local_comm;

    // Node-local communicator.
    MPI_Comm cross_comm;

    int myRank, nRanks, localRank, nlocalRank;
    std::vector<int> localRanks;

};

MPIContext mpi_context;

#endif // SUPER_SCALAR_H_
