#include <cstdlib>
using std::exit;
using std::strtol;
using std::rand;
using std::srand;
// Macros: EXIT_SUCCESS, EXIT_FAILURE, NULL

#include<algorithm>
using std::generate_n;

// C++11 features not yet available in Intel compilers:
// #include <random>
// using std::uniform_real_distribution;
// using std::mt19937;

// #include<functional>
// using std::bind;   

#include <iostream>
using std::cerr;
using std::endl;
 
#include <sstream>
using std::ostringstream;   

#include <vector>
using std::vector;

#include <string>
using std::string;
// C++11 feature not yet available in Intel compilers:
// using std::stoi;
int stoi (string s)
{ return static_cast<int>( strtol (s.c_str(), NULL, 10) ); }

#include "mpi.h"

extern "C"
{
  void blacs_pinfo_(int*, int*);
  void blacs_get_(int const*, int const*, int*);
  void blacs_gridinit_(int*, char const*, int const*, int const*);
  void blacs_gridinfo_(int const*, int*, int*, int*, int*);
  void blacs_barrier_(int const*, char const*);
  void blacs_gridexit_(int const*);
  void dgamx2d_(int const*, char const*, char const*, int const*, int const*, double*, int const*, int*, int*, int const*, int const*, int const*);

  void pdtran_ (int const*, int const*, double const*, double const*, int const*, int const*, int const*, double const*, double*, int const*, int const*, int const*); 

  void descinit_ (int*, int const*, int const*, int const*, int const*, int const*, int const*, int const*, int const*, int*);
  int numroc_ (int const*, int const*, int const*, int const*, int const*);
  void pdsytrd_ (char const*, int const*, double*, int const*, int const*, int const*, double*, double*, double*, double*, int const*, int*);
  
#ifdef ELPA
void elpa1_mp_tridiag_real_(int const*, double*, int const*, int const*, MPI_Comm const*, MPI_Comm const*, double*, double*, double*);
#endif  
}

void die(bool speak, string s)
{
  if (speak) cerr << s << endl;
  MPI_Finalize();
  exit(EXIT_FAILURE);
}

int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init (&argc, &argv);
  
  // Get MPI info
  int mpi_pid, mpi_np;
  MPI_Comm_rank (MPI_COMM_WORLD, &mpi_pid);
  MPI_Comm_size (MPI_COMM_WORLD, &mpi_np);

  // Get BLACS info
  int blacs_pid, blacs_np;
  blacs_pinfo_ (&blacs_pid, &blacs_np);
  if (blacs_np != mpi_np)
    die(mpi_pid == 0, static_cast<ostringstream&>(ostringstream().seekp(0) << "BLACS only recognizes " << blacs_np << " of the available " << mpi_np << " MPI processes; something's wrong.").str());

  // Parse command line arguments (mpiexec args already stripped away)
  vector<string> args (argv, argv + argc);
  
  // Ensure usage
  if (argc <= 4)
    die(mpi_pid == 0, static_cast<ostringstream&>(ostringstream().seekp(0) << "Usage: " << argv[0] << " <matrix_dim> <n_proc_rows> <n_proc_cols> <blocksize>").str());

  int n = ::stoi(args[1]);
  if (n <= 0)
    die(mpi_pid == 0, "Matrix dimension must be positive.");

  int npr = ::stoi(args[2]);
  int npc = ::stoi(args[3]);
  if (npr <= 0 || npc <= 0)
    die(mpi_pid == 0, "Processor grid dimensions must be positive.");

  int bsz = ::stoi(args[4]);
  if (bsz <= 0)
    die(mpi_pid == 0, "Block dimension must be positive.");

  if (mpi_np < npr*npc)
    die(mpi_pid == 0, static_cast<ostringstream&>(ostringstream().seekp(0) << "Need more than " << mpi_np << " BLACS processes to construct a " << npr << "-by-" << npc << " BLACS grid.").str()); 

  // Initialize BLACS grid and context
  int context;
  int imone = -1, izero = 0;
  blacs_get_ (&imone, &izero, &context);
  blacs_gridinit_ (&context, "R", &npr, &npc);
  int pr, pc;
  int blacs_npr, blacs_npc;
  blacs_gridinfo_ (&context, &blacs_npr, &blacs_npc, &pr, &pc);

  // Split the Cblacs grid as a subcommunicator:
  MPI_Comm grid_comm;
  MPI_Comm_split(
    MPI_COMM_WORLD,
    blacs_npr == -1 || blacs_npc == -1 || pr == -1 || pc == -1 ? MPI_UNDEFINED : 1,
    blacs_pid,
    &grid_comm); 
  
 // cerr << "MPI PID: " << mpi_pid << ", BLACS PID: " << blacs_pid << ", (npr,npc,pr,pc) = (" << blacs_npr << "," << blacs_npc << "," << pr << "," << pc << ").";

  if (grid_comm != MPI_COMM_NULL)
  { 
    int info;
    int izero = 0, ione = 1;
    
    // Initialize matrix descriptor
    int desca[9];
    int m_loc = numroc_ (&n, &bsz, &pr, &izero, &npr);
    int n_loc = numroc_ (&n, &bsz, &pc, &izero, &npc);
    descinit_ (desca, &n, &n, &bsz, &bsz, &izero, &izero, &context, &m_loc, &info);
     
    // Generate A = rand(n,n); A = A + A^T;
    vector<double> A (m_loc*n_loc);
    double done = 1.;
    double a = -1., b = 1.;
    srand(mpi_pid);
    generate_n(
      A.begin(), A.size(),
      [=]{return a+(b-a)*rand();}); 
   //   bind(uniform_real_distribution<double>(a, b),
   //        mt19937(mpi_pid)));
    pdtran_ (&n, &n, &done, A.data(), &ione, &ione, desca, &done, A.data(), &ione, &ione, desca);

    vector<double> ScaLAPACK_A = A;

    // NB: no matrix descriptors for d, e, and tau (see 
    // http://netlib.org/scalapack/slug/node80.html )
    vector<double> ScaLAPACK_d(n_loc), ScaLAPACK_e(n_loc), ScaLAPACK_tau(n_loc);

    // Perform work query
    vector<double> work(1);
    pdsytrd_ ("L", &n, ScaLAPACK_A.data(), &ione, &ione, desca, ScaLAPACK_d.data(), ScaLAPACK_e.data(), ScaLAPACK_tau.data(), work.data(), &imone, &info);
    int lwork = static_cast<int>( work.front() );
    work.resize(lwork);

    // Call ScaLAPACK:
    MPI_Barrier(grid_comm); // BLACS version: blacs_barrier_(&context, "A");
    double T = MPI_Wtime();

    pdsytrd_ ("L", &n, ScaLAPACK_A.data(), &ione, &ione, desca, ScaLAPACK_d.data(), ScaLAPACK_e.data(), ScaLAPACK_tau.data(), work.data(), &lwork, &info);

    MPI_Barrier(grid_comm); // BLACS version: blacs_barrier_(&context, "A");
    T = MPI_Wtime() - T;

    double T_max;
    int mpi_grid_pid;
    MPI_Comm_rank(grid_comm, &mpi_grid_pid);
    MPI_Reduce(static_cast<void const*>(&T), static_cast<void*>(&T_max), 1, MPI_DOUBLE, MPI_MAX, 0, grid_comm);
    // BLACS version:
    // dgamx2d_(&context, "A", " ", &ione, &ione, &T, &ione, static_cast<int*>(NULL), static_cast<int*>(NULL), &imone, &izero, &izero); 
    // if (pr == 0 && pc == 0)
    if (mpi_grid_pid == 0)
      cerr << "pdsytrd (ScaLAPACK) = " << T_max << " (max MPI_Wtime() over all processes in the BLACS grid)" << endl;

#ifdef ELPA
    MPI_Comm comm_rows, comm_cols;
    MPI_Comm_split(grid_comm, pc, pr, &comm_rows); 
    MPI_Comm_split(grid_comm, pr, pc, &comm_cols); 
    vector<double> ELPA_A = A, ELPA_d(n), ELPA_e(n), ELPA_tau(n);
    
    MPI_Barrier(grid_comm);
    T = MPI_Wtime();
    
    elpa1_mp_tridiag_real_(&n, ELPA_A.data(), &n, &bsz, &comm_rows, &comm_cols, ELPA_d.data(), ELPA_e.data(), ELPA_tau.data());
    
    MPI_Barrier(grid_comm); 
    T = MPI_Wtime() - T;       
    MPI_Reduce(static_cast<void const*>(&T), static_cast<void*>(&T_max), 1, MPI_DOUBLE, MPI_MAX, 0, grid_comm);
    if (mpi_grid_pid == 0)
      cerr << "tridiag_real (ELPA1) = " << T_max << " (max MPI_Wtime() over all processes in the grid)" << endl;  
    MPI_Comm_free(&comm_rows);
    MPI_Comm_free(&comm_cols);
#endif

    MPI_Comm_free(&grid_comm);
    blacs_gridexit_(&context);
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}
