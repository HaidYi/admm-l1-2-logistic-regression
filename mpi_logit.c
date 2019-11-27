/* COMP633 Parallel Computing @ UNC Chapel Hill, 2019 
 *
 * PA2:
 * Solve a distributed regularized logistic regression problem, i.e.,
 *    
 *   minimize  sum( log(1 + exp(-b_i*(a_i'x + x_0)) ) ) + 0.5 * lambda * l(x).
 * 
 * where l(x) = ||x||_1,   l1 regularization.
 *       l(x) = ||x||_2^2, l2 regularization.
 * 
 * The implementation uses MPI (Message Passing Interface) for distributed communication 
 * and the GNU Scientific Library (GSL) for math and solver.
 * 
 * @author: Haidong Yi,   haidyi@cs.unc.edu
 *          Minzhi Jiang, minzhi@live.unc.edu
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <mpi.h>
#include "header/mmio.h"
#include "header/solver.h"


void soft_threshold(gsl_vector *v, double k);
double objective(gsl_vector *x, void *param, gsl_vector *z, double lambda, char *type);

int main(int argc, char **argv)
{
    int opt;
    char type[3];
    /* set default configs */
    static int MAX_ITER  = 1000;
    static double RELTOL = 1e-3;
    static double ABSTOL = 1e-5;
    int isPrint = 0;
    char dirA[80], dirb[80], dir_soln[80];

    /* command line parser */
    while ((opt = getopt(argc, argv, "A:b:e:E:r:t:o:p")) != -1) {
        switch (opt)
        {
        case 'A':
            if (strlen(optarg) > 80) {
                fprintf(stderr, "file path -%s is too long\n", optarg);
            }    
            strcpy(dirA, optarg);
            break;
        case 'b':
            if (strlen(optarg) > 80) {
                fprintf(stderr, "file path %s is too long, please shorten the path name, or use relative path, existing...\n", optarg);
            }
            strcpy(dirb, optarg);
            break;
        case 'e':
            if ((RELTOL = atof(optarg)) == 0) {
                fprintf(stderr, "Option -%c's argument %s is invalid\n", optopt, optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'E':
            if ((ABSTOL = atof(optarg)) == 0) {
                fprintf(stderr, "Option -%c's argument %s is invalid\n", optopt, optarg);
                return EXIT_FAILURE;
            }
            break;
        case 't':
            if ((MAX_ITER = atoi(optarg)) == 0) {
                fprintf(stderr, "Option -%c's argument %s is invalid\n", optopt, optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'r':
            if (strcmp(optarg, "l1") == 0 || strcmp(optarg, "l2") == 0)
                strcpy(type, optarg);
            else {
                fprintf(stderr, "Option -%c only supports argument %s or %s\n", optopt, "l1", "l2");
                return EXIT_FAILURE;
            }     
            break;
        case 'p':
            isPrint = 1;
            break;
        case 'o':
            if (strlen(optarg) > 80)
                fprintf(stderr, "file path -%s is too long\n", optarg);
            strcpy(dir_soln, optarg);
            break;
        case '?':
            if (optopt == 'A'|| optopt == 'b' || optopt == 'r' || optopt == 'e' || optopt == 't' || optopt == 'E')
                fprintf(stderr, "Option -%c needs an argument\n", optopt);
            else
                fprintf(stderr, "Unknown option %c. \n", optopt);
            return EXIT_FAILURE;
            break;
        default:
            abort();
        }
    }
    
    int rank;
    int size=4;

    MPI_Init(&argc, &argv);               // Initialize the MPI execution environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Determine current running process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Total number of processes
    double N = (double) size;             // Number of subsystems/slaves for ADMM

    /* Read the data in very subsystem slaver */
    FILE *f;
    int m, n;
    int row, col;
    double entry;

    sprintf(dirA, "%s%d.dat", dirA, rank + 1);
    f = fopen(dirA, "r");
    if (f == NULL) {
        printf("ERROR: %s does not exist, exiting.\n", dirA);
        exit(EXIT_FAILURE);
    }
    printf("process [%d] is reading %s\n", rank, dirA);
    mm_read_mtx_array_size(f, &m, &n);

    gsl_matrix *A = gsl_matrix_calloc(m, n);
    for (int i = 0; i < m*n; i++) {
        row = i % m;
        col = floor(i/m);
        fscanf(f, "%lf", &entry);
        gsl_matrix_set(A, row, col, entry);
    }
    fclose(f);

    /* Reading response vector b */
    sprintf(dirb, "%s%d.dat", dirb, rank + 1);

    f = fopen(dirb, "r");
    if (f == NULL) {
        printf("ERROR: %s does not exist, exiting.\n", dirb);
        exit(EXIT_FAILURE);
    }
    printf("process [%d] is reading %s\n", rank, dirb);
    mm_read_mtx_array_size(f, &m, &n);
    gsl_vector *b = gsl_vector_calloc(m);
    for (int i = 0; i < m; i++) {
        fscanf(f, "%lf", &entry);
        gsl_vector_set(b, i, entry);
    }
    fclose(f);

    m = A->size1;
    n = A->size2;

    /* These are all variable definitions using by ADMM algorithm */
    double rho = 1.0;  // augmented Lagrangian parameter
    double lambda = 1.0;  // regularized parameter 
    
    gsl_vector *x      = gsl_vector_calloc(n+1);  // primal variable
    gsl_vector *z      = gsl_vector_calloc(n+1);  // primal variable (splited from x)
    gsl_vector *u      = gsl_vector_calloc(n+1);  // scaled dual variable i.e. u = (1/rho)y  
    gsl_vector *r      = gsl_vector_calloc(n+1);  // residual variable i.e. r = x - z
    gsl_vector *zprev  = gsl_vector_calloc(n+1);  // used fro save previous iter z
    gsl_vector *zdiff  = gsl_vector_calloc(n+1);  // difference between consecutive z, i.e. zdiff = z^{(i+1)} - z^{(i)}
    gsl_vector *w      = gsl_vector_calloc(n+1);  // used for aggregate x + z in different process

    double send[3]; // aggregate three scalars, i.e. [||r||^2, ||x||^2, ||u||^2]
    double recv[3]; // reveive aggregated three scalars.

    double nxstack  = 0;
    double nustack  = 0;
    double prires   = 0;
    double dualres  = 0;
    double eps_pri  = 0;
    double eps_dual = 0;

    // /* Main ADMM solver loop */
    int iter = 0;
    if (isPrint) {
        printf("Config: #proc: %d\tReg: %s\tMax_Iter: %6d\tRELTOL: %.6f\tABSTOL: %.6f\n", rank+1, type, MAX_ITER, RELTOL, ABSTOL);
        printf("%3s %10s %10s %10s %10s %10s\n", "#", "r norm", "eps_pri", "s norm", "eps_dual", "objective");
    }
    /* init model parameters */
    struct Param param = {A, b, u, z, rho};
    gsl_vector_set_all(x, 0.);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    double time_x = 0;
    while(iter < MAX_ITER) {    
        /* x-update: via gsl multi-dimensional optimiation*/
        
        double x_start = MPI_Wtime();
        x = update_x(&param, x);  // warm start with previous x
        double x_end = MPI_Wtime();
        time_x += (x_end - x_start);

        /* prepare for the Message passing: compute the global
           sum of x and z, then update z. 
        */
        gsl_vector_memcpy(w, x);
        gsl_vector_add(w, u);   // w = x + u

        /* z-update: Aggregate x + u uing MPI ALLreduce*/
        gsl_vector_memcpy(zprev, z);
        MPI_Allreduce(w->data, z->data, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        gsl_vector_scale(z, 1/N);
        gsl_vector_view z1_n = gsl_vector_subvector(z, 1, n);

        if (strcmp(type, "l1") == 0) {
            /* l1 reg */
            soft_threshold(&z1_n.vector, 0.5*lambda/(N*rho)); // l1: z1 = S_{lambda/(2 * rho)} (x1 + u1)
        } else if (strcmp(type, "l2") == 0) {
            /* l2 reg */
            gsl_vector_scale(&z1_n.vector, N*rho/(N*rho + lambda));
        }

        /* u-update: u = u + x - z */
        gsl_vector_add(u, x);
        gsl_vector_sub(u, z);

        /* Compute residual: r = x - z */
        gsl_vector_memcpy(r, x);
        gsl_vector_sub(r, z);

        gsl_blas_ddot(r, r, &send[0]);
        gsl_blas_ddot(x, x, &send[1]);
        gsl_blas_ddot(u, u, &send[2]);

        MPI_Allreduce(send, recv, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        prires  = sqrt(recv[0]);  // sqrt(sum ||r_i||_2^2)
        nxstack = sqrt(recv[1]);  // sqrt(sum ||x_i||_2^2)
        nustack = sqrt(recv[2]);  // sqrt(sum ||u_i||_2^2)
  
        /* Termination checks */

        /* dual residual */
        gsl_vector_memcpy(zdiff, z);
        gsl_vector_sub(zdiff, zprev);
        dualres =  sqrt(N) * rho * gsl_blas_dnrm2(zdiff); /* ||s^k||_2^2 = N rho^2 ||z - zprev||_2^2 */

        /* compute primal and dual feasibility tolerances */
        eps_pri  = sqrt(n*N)*ABSTOL + RELTOL * fmax(nxstack, sqrt(N) * gsl_blas_dnrm2(z));
        eps_dual = sqrt(n*N)*ABSTOL + RELTOL * rho * nustack;

        /* check the object values */
        if (rank == 0 && isPrint) {
            printf("%3d %10.4f %10.4f %10.4f %10.4f %10.4f\n", iter, 
                prires, eps_pri, dualres, eps_dual, objective(z, &param, z, lambda, type));
        }
        
        if (prires <= eps_pri && dualres <= eps_dual) {
            break;
        }

        iter++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    MPI_Finalize();  /* Close the MPI execuation */


    /* write the solutions using the master process */
    if (rank == 0) {
        printf("Total execuation time is %.8f/iter, time_x: %.8f.\n", (end - start)/iter, time_x/iter);
        f = fopen(dir_soln, "w");
        printf("Writing solutions to %s\n", dir_soln);
        gsl_vector_fprintf(f, z, "%lf");
        fclose(f);
        printf("Done\n");
    }


    /* Clear memory */
    gsl_matrix_free(A);
    gsl_vector_free(b);
    gsl_vector_free(x);
    gsl_vector_free(u);
    gsl_vector_free(z);
    gsl_vector_free(r);
    gsl_vector_free(zprev);
    gsl_vector_free(zdiff);
    gsl_vector_free(w);

    return 0;
}

/*
 * Implement a soft threshold element-wise function, i.e.
 *
 *            { v - k  if  v > k
 *   S_k(v) = { 0      if |v|<= k
 *            { v + k  if  v < -k
 * 
 * @param v: input vector
 * @param k: threshold paramter
*/
void soft_threshold(gsl_vector *v, double k) {
    double vi;
    for (int i = 0; i < v->size; i++) {
        vi = gsl_vector_get(v, i);
        if (vi > k)       { gsl_vector_set(v, i, vi - k); }
        else if (vi < -k) { gsl_vector_set(v, i, vi + k); }
        else              { gsl_vector_set(v, i, 0); }
    }
}

/*
 * Implement the computation the objective function, i.e.
 * 
 *   sum( log(1 + exp(-b_i*(a_i'x + x_0)) ) ) + 0.5 * lambda * l(x).  
 *
 * where l(x) = ||x||_1,   l1 regularization.
 *       l(x) = ||x||_2^2, l2 regularization.
 * 
*/
double objective(gsl_vector *x, void *param, gsl_vector *z, double lambda, char *type) {
    gsl_vector *y = ((struct Param *)param)->y;
    gsl_matrix *X = ((struct Param *)param)->X;

    int n = X->size1;
    int p = x->size;
    gsl_vector *X_beta = gsl_vector_calloc(n);
    double beta_0 = gsl_vector_get(x, 0);
    gsl_vector_const_view beta = gsl_vector_const_subvector(x, 1, p-1);

    gsl_blas_dgemv(CblasNoTrans, 1, X, &beta.vector, 0, X_beta);
    gsl_vector_add_constant(X_beta, beta_0);
    gsl_vector_mul(X_beta, y);  // 
    gsl_vector_scale(X_beta, -1); //

    double obj = 0.;

    gsl_vector_view z_1 = gsl_vector_subvector(z, 1, p-1);
    if (strcmp(type, "l1") == 0) {
        double reg_nrm = gsl_blas_dasum(&z_1.vector);
        obj += 0.5 * lambda * reg_nrm;
    } else if (strcmp(type, "l2") == 0) {
        double reg_nrm = gsl_blas_dnrm2(&z_1.vector);
        obj += 0.5 * lambda * pow(reg_nrm, 2);
    }
    
    for (int i = 0; i < X_beta->size; i++) {
        double vi;
        vi = gsl_vector_get(X_beta, i);
        vi = log(1 + exp(vi));
        obj += vi;
    } /* for i */
    
    gsl_vector_free(X_beta);

    return obj;
}

