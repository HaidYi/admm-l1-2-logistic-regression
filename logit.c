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
 * The implementation uses GNU Scientific Library (GSL) for math and solver.
 * 
 * @author: Haidong Yi,   haidyi@cs.unc.edu
 *          Minzhi Jiang, minzhi@live.unc.edu
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "header/mmio.h"
#include "header/solver.h"


void soft_threshold(gsl_vector *v, double k);
double objective(gsl_vector *x, void *param, gsl_vector *z, double lambda, char *type);

int main(int argc, char const *argv[])
{
    /* code */
    char *type = "l2";
    FILE *f;
    int m, n;
	int row, col;
	double entry;

    char s[20]="data/input.dat";
	printf("reading %s\n", s);

    f = fopen(s, "r");
	if (f == NULL) {
		printf("ERROR: %s does not exist, exiting.\n", s);
		exit(EXIT_FAILURE);
	}
    mm_read_mtx_array_size(f, &m, &n);

    gsl_matrix *A = gsl_matrix_calloc(m, n);
	for (int i = 0; i < m*n; i++) {
		row = i % m;
		col = floor(i/m);
		fscanf(f, "%lf", &entry);
		gsl_matrix_set(A, row, col, entry);
	}
	fclose(f);

    char s1[20] = "data/output.dat";
	printf("reading %s\n", s1);

    f = fopen(s1, "r");
	if (f == NULL) {
		printf("ERROR: %s does not exist, exiting.\n", s);
		exit(EXIT_FAILURE);
	}
	mm_read_mtx_array_size(f, &m, &n);
	gsl_vector *b = gsl_vector_calloc(m);
	for (int i = 0; i < m; i++) {
		fscanf(f, "%lf", &entry);
		gsl_vector_set(b, i, entry);
	}
	fclose(f);

    m = A->size1;
	n = A->size2;

    double rho = 1.0;
    double lambda = 1.0;
    
	gsl_vector *x      = gsl_vector_calloc(n+1);
	gsl_vector *u      = gsl_vector_calloc(n+1);
	gsl_vector *z      = gsl_vector_calloc(n+1);
	gsl_vector *r      = gsl_vector_calloc(n+1);
	gsl_vector *zprev  = gsl_vector_calloc(n+1);
	gsl_vector *zdiff  = gsl_vector_calloc(n+1);

    // double nxstack  = 0;
	// double nystack  = 0;
	double prires   = 0;
	double dualres  = 0;
	double eps_pri  = 0;
	double eps_dual = 0;

    const int MAX_ITER  = 50;
	const double RELTOL = 1e-3;
	const double ABSTOL = 1e-5;
	/* Main ADMM solver loop */
    int iter = 0;
    printf("%3s %10s %10s %10s %10s %10s\n", "#", "r norm", "eps_pri", "s norm", "eps_dual", "objective");

    /* init model parameters */
    struct Param param = {A, b, u, z, rho};
    gsl_vector_set_all(x, 0.);

    while(iter < MAX_ITER) {
        
        /* x-update: */
        x = update_x(&param, x);  // warm start with previous x

        gsl_vector_memcpy(zprev, z);
        /* z-update: */
        gsl_vector_memcpy(z, x);
        gsl_vector_add(z, u);
        gsl_vector_view z1_n = gsl_vector_subvector(z, 1, n);

        if (strcmp(type, "l1") == 0) {
            /* l1 reg */
            soft_threshold(&z1_n.vector, 0.5*lambda/rho); // l1: z1 = S_{lambda/(2 * rho)} (x1 + u1)
        } else if (strcmp(type, "l2") == 0) {
            /* l2 reg */
            gsl_vector_scale(&z1_n.vector, rho/(rho + lambda));
        }

        /* u-update: u = u + x - z */
        gsl_vector_add(u, x);
        gsl_vector_sub(u, z);
        
        /* Termination checks */

        /* Compute residual: r = x - z */
        gsl_vector_memcpy(r, x);
        gsl_vector_sub(r, z);
        prires = gsl_blas_dnrm2(r); /* ||r^k||_2^2 = ||x - z||_2^2 */

        /* dual residual */
        gsl_vector_memcpy(zdiff, z);
        gsl_vector_sub(zdiff, zprev);
        dualres =  rho * gsl_blas_dnrm2(zdiff); /* ||s^k||_2^2 = N rho^2 ||z - zprev||_2^2 */

        /* compute primal and dual feasibility tolerances */
        eps_pri  = sqrt(n)*ABSTOL + RELTOL * fmax(gsl_blas_dnrm2(x), gsl_blas_dnrm2(z));
        eps_dual = sqrt(n)*ABSTOL + RELTOL * rho * gsl_blas_dnrm2(u);

        printf("%3d %10.4f %10.4f %10.4f %10.4f %10.4f\n", iter, 
                prires, eps_pri, dualres, eps_dual, objective(z, &param, z, lambda, type));
        
        if (prires <= eps_pri && dualres <= eps_dual) {
            break;
        }

        iter++;
    }

    /* write the solutions */
    f = fopen("data/result.dat", "w");
	gsl_vector_fprintf(f, z, "%lf");
	fclose(f);

	/* Clear memory */
	gsl_matrix_free(A);
	gsl_vector_free(b);
	gsl_vector_free(x);
	gsl_vector_free(u);
	gsl_vector_free(z);
	gsl_vector_free(r);
	gsl_vector_free(zprev);
	gsl_vector_free(zdiff);

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
        double reg_nrm2_square;
        gsl_blas_ddot(&z_1.vector, &z_1.vector, &reg_nrm2_square);
        obj += 0.5 * lambda * reg_nrm2_square;
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

/* test result l_2
0.257460
0.828513
0.058432
-0.726962
1.181708
*/

/* test result l_1
0.445411
1.054801
0.000000
0.000000
3.051893
*/

