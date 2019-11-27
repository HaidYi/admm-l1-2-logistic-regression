# admm-l1-2-logistic-regression

## Description
This repo is the final project of COMP633 at UNC-Chapel Hill. We used **A**lternating **D**irection **M**ethod of **M**ultipliers (ADMM) optimization methods to solve the L-1/L-2 regularized binary logistic regression. In this repo, we provide efficient squential and distributed implementations of ADMM powered by GNU Scientific Library (GSL) and Message Passing Interface (MPI) such that the program can handle millions of samples within couple of minutes. 

## Dependencies
* GCC
* GNU Scientific Library (GSL)
* Message Passing Interface (MPI)

## Usage
### Compile
We use gcc/icc to compile the whole project. The standard Makefile has been provided. You may need to change the Makefile slightly to accommodate your local *include* path and *library* path. The program has been compiled successfully in MacOS and Linux environment.

### Runner
* `logit`: Sequential Runner
* `mpi_logit`: Distributed Runner

```{bash}
usage: logit/mpi_logit [-A Feature Matrix] [-b Response Vector] [-e Relative Tolerance]
                       [-E Absolute Telerance] [-r Regularization Type] 
                       [-t Maximum Iteration] [-o Output File] [-p Print Progress]
arguments:
  -A:       The path to feature matrix,  e.g. data/input.dat;
  -b:       The path to response vevtor, e.g. data/output.dat;
  -e:       The relative precision tolerance in ADMM algorithm;
  -E:       The absolute precision tolerance in ADMM algorithm;
  -r:       Regularization type to use: only support L-1 and L-2, e.g. -r l2;
  -t:       The maximum number of iterations;
  -o:       The path to the result file, e.g. data/solution.dat
  -p:       Whether to print the optimization progress.
```
Note: Matrix and vector uses the standard Matrix Market File Format, i.e. [__Matrix Market Exchange Formats__](https://math.nist.gov/MatrixMarket/index.html).

### Distributed Runner
```bash
mpirun -np {\# of cores} ./mpi_logit [Arguments]
```
Note: For the distributed runner, we follow the data-parallel principle. Every process reads the part of feature matrix and response vector (partitioned by sample). Therefore, the argument `-A` and `b` should be the prefix path of data file, e.g.  we use `-A data/A` when `A1.dat--A{N}.dat` are located in `data`.

## Performance


## Reference
Boyd, Stephen, et al. "Distributed optimization and statistical learning via the alternating direction method of multipliers." Foundations and Trends® in Machine learning 3.1 (2011): 1-122.