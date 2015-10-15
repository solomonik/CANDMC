#ifndef __PMPI_H__
#define __PMPI_H__

#include "mpi.h"

#ifdef PMPI
#define MPI_Bcast(...)                                            \
  do { CTF_Timer __t("MPI_Bcast");                                \
              __t.start();                                        \
    PMPI_Bcast(__VA_ARGS__);                                      \
              __t.stop(); } while (0)
#define MPI_Reduce(...)                                           \
  do { CTF_Timer __t("MPI_Reduce");                               \
              __t.start();                                        \
    PMPI_Reduce(__VA_ARGS__);                                     \
              __t.stop(); }while (0)
#define MPI_Wait(...)                                             \
  do { CTF_Timer __t("MPI_Wait");                                 \
              __t.start();                                        \
    PMPI_Wait(__VA_ARGS__);                                       \
              __t.stop(); } while (0)
#define MPI_Send(...)                                             \
  do { CTF_Timer __t("MPI_Send");                                 \
              __t.start();                                        \
    PMPI_Send(__VA_ARGS__);                                       \
              __t.stop(); } while (0)
#define MPI_Recv(...)                                             \
  do { CTF_Timer __t("MPI_Recv");                                 \
              __t.start();                                        \
    PMPI_Recv(__VA_ARGS__);                                       \
              __t.stop(); } while (0)
#define MPI_Sendrecv(...)                                         \
  do { CTF_Timer __t("MPI_Sendrecv");                             \
              __t.start();                                        \
    PMPI_Sendrecv(__VA_ARGS__);                                   \
              __t.stop(); } while (0)
#define MPI_Allreduce(...)                                        \
  do { CTF_Timer __t("MPI_Allreduce");                            \
              __t.start();                                        \
    PMPI_Allreduce(__VA_ARGS__);                                  \
              __t.stop(); } while (0)
#define MPI_Allgather(...)                                        \
  do { CTF_Timer __t("MPI_Allgather");                            \
              __t.start();                                        \
    PMPI_Allgather(__VA_ARGS__);                                  \
              __t.stop(); } while (0)
#define MPI_Scatter(...)                                          \
  do { CTF_Timer __t("MPI_Scatter");                              \
              __t.start();                                        \
    PMPI_Scatter(__VA_ARGS__);                                    \
              __t.stop(); } while (0)
#define MPI_Alltoall(...)                                         \
  do { CTF_Timer __t("MPI_Alltoall");                             \
              __t.start();                                        \
    PMPI_Alltoall(__VA_ARGS__);                                   \
              __t.stop(); } while (0)
#define MPI_Alltoallv(...)                                        \
  do { CTF_Timer __t("MPI_Alltoallv");                            \
              __t.start();                                        \
    PMPI_Alltoallv(__VA_ARGS__);                                  \
              __t.stop(); } while (0)
#define MPI_Gatherv(...)                                          \
  do { CTF_Timer __t("MPI_Gatherv");                              \
              __t.start();                                        \
    PMPI_Gatherv(__VA_ARGS__);                                    \
              __t.stop(); } while (0)
#define MPI_Scatterv(...)                                         \
  do { CTF_Timer __t("MPI_Scatterv");                             \
              __t.start();                                        \
   PMPI_Scatterv(__VA_ARGS__);                                    \
              __t.stop(); } while (0)
#define MPI_Waitall(...)                                          \
  do { CTF_Timer __t("MPI_Waitall");                              \
              __t.start();                                        \
    PMPI_Waitall(__VA_ARGS__);                                    \
              __t.stop(); } while (0)
#define MPI_Barrier(...)                                          \
  do { CTF_Timer __t("MPI_Barrier");                              \
              __t.start();                                        \
    PMPI_Barrier(__VA_ARGS__);                                    \
              __t.stop(); } while (0)
#endif

#endif
