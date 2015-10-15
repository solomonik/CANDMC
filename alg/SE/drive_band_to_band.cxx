#include <cstdlib>
// Functions:
using std::exit;
// Macros: EXIT_SUCCESS

#include<algorithm>
// Functions:
using std::min;

#include <iostream>
// Objects:
using std::cout;

#include <sstream>
// Types:
using std::ostringstream;   

#include <vector>
// Types:
using std::vector;

#include <string>
// Types:
using std::string;     

#include <cmath>
// Functions:
using std::sqrt;

#include <utility>
// Types:
using std::pair;

#include "band_util.hpp"
// Functions:
using band_util::mat_gen;
using band_util::die;

#include "mpi.h"
// Macros: MPI_COMM_WORLD, MPI_DOUBLE, MPI_UNDEFINED, MPI_MAX, MPI_IN_PLACE, MPI_COMM_NULL
// Types: MPI_Comm
// Functions: MPI_Init, MPI_Comm_rank, MPI_Comm_size, MPI_Comm_split, MPI_Barrier, MPI_Wtime, MPI_Reduce, MPI_Comm_free, MPI_Finalize     

#if defined(bench_me)
// #include "mpi.h"
// Macros: MPI_DOUBLE, MPI_MAX
// Functions: MPI_Barrier, MPI_Wtime, MPI_Reduce

#elif defined(test_me)

#ifndef USE_SCALAPACK
#error "B2B test requires ScaLAPACK (make sure to compile with -DUSE_SCALAPACK)"
#endif

// #include "band_util.hpp"
// Functions:
using band_util::blacs_pinfo_wrap;
using band_util::blacs_get_wrap;
using band_util::blacs_gridinit_wrap;
using band_util::blacs_gridmap_wrap;
using band_util::blacs_gridinfo_wrap;
using band_util::blacs_gridexit_wrap;
using band_util::is_hermitian;
using band_util::semibandwidths;
using band_util::max_eval_diff;  

#include <ios>
// Functions:
using std::boolalpha; 

#endif

#if defined(CPP11)

// #include <string>
// Functions:
using std::stoi;

#include <array>
// Types:
using std::array;    

#else

// #include <cstdlib>
// Functions:
using std::strtol;

#include <cstddef>
// Macros: NULL

#define nullptr NULL

#define stoi(x) my_stoi(x)
namespace 
{
  // TODO: why does GCC give an error when I try to overload stoi in an anonymous namespace? 
  int my_stoi (string s)
  { return static_cast<int>( strtol (s.c_str(), NULL, 10) ); }  
}

#endif

#include "../shared/timer.h"
// Macros:
// TODO

void band_to_band(double*, int, int, int, int, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm);  

int main(int argc, char* argv[])
{
  // TODO omp_set_num_threads(32);   

  // Initialize MPI
  MPI_Init (&argc, &argv);

  // Get MPI info
  int p, np;
  MPI_Comm_rank (MPI_COMM_WORLD, &p);
  MPI_Comm_size (MPI_COMM_WORLD, &np);

  // Ensure square number of processes:
  int sqrt_np = static_cast<int>(sqrt(np));
  if (sqrt_np * sqrt_np < np)
    die(p == 0, "Requires square number of processes.");   

  // Parse command line arguments (mpiexec args already stripped away)
  vector<string> args (argv, argv + argc);

  // default arguments:
  int n = 128;
  int block_size = 32;
  int b_initial = 96;
  int b_terminal = 64; 

  if (argc < 5)
  {  if (p == 0)
    cout << "Usage: " << argv[0] << " <matrix_dim> <block_size> <initial_bandwidth> <terminal_bandwidth>\nUsing default arguments " << n << ", " << block_size << ", " << b_initial << ", " << b_terminal << ".\n";
  }
  else
  {
    n = stoi(args[1]);
    if (n <= 0)
      die(p == 0, "Matrix dimension must be positive");

    block_size = stoi(args[2]);
    if (n % block_size != 0)
      die(p == 0, "Matrix dimension must be a multiple of the block size");

    b_initial = stoi(args[3]);
    if (b_initial <= 0)
      die(p == 0, "Initial semibandwidth must be positive");

    // "b_initial == n" really means "== n-1"
    b_initial = min(n, b_initial);   

    if (b_initial % block_size != 0)
      die(p == 0, "Initial semibandwidth must be a multiple of the block size");

    b_terminal = stoi(args[4]);
    if (b_terminal <= 0)
      die(p == 0, "Terminal semibandwidth must be positive");

    if (b_terminal >= b_initial)
      die(p == 0, "Terminal semibandwidth must be strictly smaller than initial bandwidth");

    if (b_terminal % block_size != 0)
      die(p == 0, "Terminal semibandiwidth must be a multiple of the block size");
  }

  int N = n / block_size; // exact division
  int B = b_initial / block_size; // exact division

  int pr = p % sqrt_np;
  int pc = p / sqrt_np;

  MPI_Comm comm_grid, comm_rows, comm_cols, comm_diag;
  MPI_Comm_split(MPI_COMM_WORLD, 0, pr + pc*sqrt_np, &comm_grid); 
  MPI_Comm_split(MPI_COMM_WORLD, pr, pc, &comm_rows); 
  MPI_Comm_split(MPI_COMM_WORLD, pc, pr, &comm_cols);
  MPI_Comm_split(MPI_COMM_WORLD, pr == pc ? 1 : MPI_UNDEFINED, pr /* == pc, when this argument matters */, &comm_diag);

  int MA_local = N / sqrt_np + (pr < (N % sqrt_np) ? 1 : 0); 
  int NA_local = N / sqrt_np + (pc < (N % sqrt_np) ? 1 : 0); 

  // Generate symmetric (TODO: Hermitian) band matrix A.
  vector<double> A (MA_local*block_size*NA_local*block_size);
  mat_gen (A.data(), N, block_size, B, pr, pc, sqrt_np, comm_grid); 

#if defined(bench_me)

#if defined(CPP11)   
  array<double,1> times;
#else
  double times[1];
#endif

  TAU_PROFILE_TIMER(timer, "main", "int (int, char**)", TAU_USER);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_SET_NODE(p);
  TAU_PROFILE_SET_CONTEXT(0);

  // Time CANDMC
  vector<double> A_copy = A;   
  MPI_Barrier(MPI_COMM_WORLD);
  double T = MPI_Wtime();    
  band_to_band(A_copy.data(), n, b_initial, b_terminal, block_size, comm_grid, comm_rows, comm_cols, comm_diag); 
  MPI_Barrier(MPI_COMM_WORLD); 
  times[0] = MPI_Wtime() - T;

  TAU_PROFILE_STOP(timer);

  // TODO Time ScaLAPACK, etc.
  if (p != 0)
  {
#if defined(CPP11) 
    MPI_Reduce (
      static_cast<void*>(times.data()),
      nullptr, // ignored
      1, 
      MPI_DOUBLE, 
      MPI_MAX, 
      0, 
      MPI_COMM_WORLD);   
#else 
    MPI_Reduce (
      static_cast<void*>(times),
      nullptr, // ignored
      1, 
      MPI_DOUBLE, 
      MPI_MAX, 
      0, 
      MPI_COMM_WORLD);  
#endif
  }
  else
  {
#if defined(CPP11)    
    MPI_Reduce(
      MPI_IN_PLACE,
      static_cast<void*>(times.data()), 
      1,
      MPI_DOUBLE, 
      MPI_MAX, 
      0, 
      MPI_COMM_WORLD);
#else
    MPI_Reduce(
      MPI_IN_PLACE,
      static_cast<void*>(times),
      1,
      MPI_DOUBLE, 
      MPI_MAX, 
      0, 
      MPI_COMM_WORLD);
#endif

    cout << "Reduction of an " << n << "-dimensional matrix (block size " << block_size << ") from semibandwidth " << b_initial << " to " << b_terminal << " took " << times[0] << " seconds.\n";  
  }

#elif defined(test_me)

  bool is_sym;
  pair<int,int> bws;
  vector<double> A_copy = A;   

  // Check input matrix:
  is_sym = is_hermitian(A_copy.data(), N, block_size, pr, pc, sqrt_np, comm_grid);   
  bws = semibandwidths(A_copy.data(), N, block_size, pr, pc, sqrt_np, comm_grid);
  if (p == 0) cout << "Input matrix: symmetric == " << boolalpha << is_sym << ", bandwidths == (" << bws.first << ", " << bws.second << ").\n";  
  
  // TODO DEBUG
  // MPI_Barrier(comm_grid);
  // string Astr = band_util::matrix_to_string(A_copy.data(), N, block_size, pr, pc, sqrt_np, comm_grid); 
  // if (pr == 0 && pc == 0) cout << "A: " << Astr << "\n";    

  band_to_band(A_copy.data(), n, b_initial, b_terminal, block_size, comm_grid, comm_rows, comm_cols, comm_diag); 

  // Check output matrix:
  is_sym = is_hermitian(A_copy.data(), N, block_size, pr, pc, sqrt_np, comm_grid);
  bws = semibandwidths(A_copy.data(), N, block_size, pr, pc, sqrt_np, comm_grid);
  if (p == 0) cout << "Output matrix: symmetric == " << boolalpha << is_sym << ", bandwidths == (" << bws.first << ", " << bws.second << ").\n";  
     
  // TODO DEBUG
  // MPI_Barrier(comm_grid);
  // Astr = band_util::matrix_to_string(A_copy.data(), N, block_size, pr, pc, sqrt_np, comm_grid); 
  // if (pr == 0 && pc == 0) cout << "A: " << Astr << "\n";
  
  // Compare eigenvalues (using ScaLAPACK)

  // Get BLACS info
  int blacs_p, blacs_np;
  blacs_pinfo_wrap (&blacs_p, &blacs_np);
  if (blacs_np != np)
    die(p == 0, static_cast<ostringstream&>(ostringstream().seekp(0) << "BLACS only recognizes " << blacs_np << " of the available " << np << " MPI processes; something's wrong.").str());

  // Initialize BLACS grid and context that matches comm_grid
  vector<int> blacs_perm (np);
  blacs_perm[pr+pc*sqrt_np] = blacs_p;
  MPI_Allgather(
    MPI_IN_PLACE,
    MPI_UNDEFINED,     // ignored
    MPI_DATATYPE_NULL, // ignored
    static_cast<void*>(blacs_perm.data()),
    1,
    MPI_INT,
    comm_grid);
  int blacs_context;
  blacs_get_wrap (-1, 0, &blacs_context);
  blacs_gridmap_wrap (&blacs_context, blacs_perm.data(), sqrt_np, sqrt_np, sqrt_np);   

  // Verify new BLACS context
  int blacs_npr, blacs_npc, blacs_pr, blacs_pc;
  blacs_gridinfo_wrap (blacs_context, &blacs_npr, &blacs_npc, &blacs_pr, &blacs_pc);
  if (blacs_npr != sqrt_np || blacs_npc != sqrt_np || blacs_pr != pr || blacs_pc != pc)
    die(true, "BLACS context does not match comm_grid");   

  pair<double,double> err = max_eval_diff(A_copy.data(), A.data(), N, block_size, MA_local, blacs_context);

  if (p == 0) cout << "Maximum absolute (and corresponding relative) eigenvalue difference vs. PDSYEV: " << err.first << " (" << err.second << ").\n";   

  blacs_gridexit_wrap (blacs_context);   

#endif

  MPI_Comm_free(&comm_grid);
  MPI_Comm_free(&comm_rows);
  MPI_Comm_free(&comm_cols);
  if (comm_diag != MPI_COMM_NULL)
    MPI_Comm_free(&comm_diag);

  MPI_Finalize();

  return EXIT_SUCCESS; 
}
