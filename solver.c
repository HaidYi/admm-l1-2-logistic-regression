#include "header/solver.h"

double 
logit_f(const gsl_vector *x, void *param) {
    gsl_vector *y = ((struct Param *)param)->y;
    gsl_matrix *X = ((struct Param *)param)->X;
    gsl_vector *u = ((struct Param *)param)->u;
    gsl_vector *z = ((struct Param *)param)->z;
    double rho = ((struct Param *)param)->rho;

    int n = X->size1;
    int p = x->size;
    gsl_vector *X_beta = gsl_vector_calloc(n);
    double beta_0 = gsl_vector_get(x, 0);
    gsl_vector_const_view beta = gsl_vector_const_subvector(x, 1, p-1);

    gsl_blas_dgemv(CblasNoTrans, 1, X, &beta.vector, 0, X_beta);
    gsl_vector_add_constant(X_beta, beta_0);
    gsl_vector_mul(X_beta, y);  // 
    gsl_vector_scale(X_beta, -1); //

    gsl_vector* x_z_u = gsl_vector_calloc(p);
    gsl_vector_memcpy(x_z_u, x);
    gsl_vector_sub(x_z_u, z);
    gsl_vector_add(x_z_u, u); // x_z_u = x - z + u
    double reg_nrm2 = gsl_blas_dnrm2(x_z_u); // ||x - z + u||_2

    double obj = 0.;
    for (int i = 0; i < X_beta->size; i++) {
        double vi;
        vi = gsl_vector_get(X_beta, i);
        vi = log(1 + exp(vi));
        obj += vi;
    } /* for i */
    
    obj += 0.5 * rho * (reg_nrm2 * reg_nrm2);
    
    gsl_vector_free(X_beta);
    gsl_vector_free(x_z_u);
    return obj;
}

void
logit_df(const gsl_vector *x, void *param,
         gsl_vector *df) {
    gsl_vector *y = ((struct Param *)param)->y;
    gsl_matrix *X = ((struct Param *)param)->X;
    gsl_vector *u = ((struct Param *)param)->u;
    gsl_vector *z = ((struct Param *)param)->z;
    double rho = ((struct Param *)param)->rho;

    gsl_vector_memcpy(df, x); 
    gsl_vector_sub(df, z);
    gsl_vector_add(df, u);      // x - z + u
    gsl_vector_scale(df, -rho);
    
    int n = X->size1;
    int p = x->size;
    gsl_vector *tmp = gsl_vector_calloc(n);
    double beta_0 = gsl_vector_get(x, 0);
    gsl_vector_const_view beta = gsl_vector_const_subvector(x, 1, p-1);

    gsl_blas_dgemv(CblasNoTrans, 1, X, &beta.vector, 0, tmp); // x_beta = x*\beta
    gsl_vector_add_constant(tmp, beta_0); // X\beta + \beta_0
    gsl_vector_mul(tmp, y);  // y(X\beta + \beta_0)
    gsl_vector_scale(tmp, -1); // -y(X\beta + \beta_0)

    for (int i = 0; i < tmp->size; i++) {  // 1 - prob
        double vi;
        vi = gsl_vector_get(tmp, i);
        vi = 1. - 1./(1. + exp(vi));
        gsl_vector_set(tmp, i, vi);
    } /* for i */

    gsl_vector_mul(tmp, y);
    double grad_beta_0 = 0.;
    for (int i = 0; i < tmp->size; i++) {
        grad_beta_0 += gsl_vector_get(tmp, i);
    }

    gsl_vector *gradVal = gsl_vector_calloc(p);
    gsl_vector_view grad_beta = gsl_vector_subvector(gradVal, 1, p-1);
    gsl_blas_dgemv(CblasTrans, 1, X, tmp, 0, &grad_beta.vector);
    gsl_vector_set(gradVal, 0, grad_beta_0);

    gsl_vector_add(df, gradVal);
    gsl_vector_scale(df, -1);
    
    /* clear memory */
    gsl_vector_free(tmp);
    gsl_vector_free(gradVal);
}

void
logit_fdf(const gsl_vector *x, void *param,
          double *obj, gsl_vector *df) {
    *obj = logit_f(x, param);
    logit_df(x, param, df);
}

/* solve the x update
 *   minimize log( exp( -y(A \cdot x_i + b) ) )  + (rho/2)||x_i - z^k + u^k||_2^2
 * via conjugate gradient (CG) method to optimize.
 */
gsl_vector*
update_x(struct Param *param,  gsl_vector *x0) {

    const int MAX_ITER = 50;
    const double tol = 1e-5;
    double ini_step = 0.01; 

    int p = x0->size;  // add 1 for interception.
    
    if (x0 == NULL) {
        x0 = gsl_vector_calloc(p);
        gsl_vector_set_all(x0, 0.);
    } /* default start point*/

    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *solver;

    gsl_multimin_function_fdf logistic;
    logistic.n = p;
    logistic.f = &logit_f;
    logistic.df = &logit_df;
    logistic.fdf = &logit_fdf;
    logistic.params = param;

    T = gsl_multimin_fdfminimizer_conjugate_fr;
    solver = gsl_multimin_fdfminimizer_alloc(T, p);
    
    gsl_multimin_fdfminimizer_set(solver, &logistic, x0, ini_step, tol);
    
    int iter = 0;
    int status;
    do
    {
        iter++;
        status = gsl_multimin_fdfminimizer_iterate(solver);
        if (status) break;

        status = gsl_multimin_test_gradient(solver->gradient, 1e-3);

    } while (status == GSL_CONTINUE && iter < MAX_ITER);
    
    /* update x with x^* */
    gsl_vector_memcpy(x0, solver->x);

    /* clear memory */
    gsl_multimin_fdfminimizer_free(solver);

    return x0;
}

// printf("%5d grad: %.5f %.5f %.5f %.5f %.5f\n", iter,
// gsl_vector_get(solver->gradient, 0),
// gsl_vector_get(solver->gradient, 1),
// gsl_vector_get(solver->gradient, 2),
// gsl_vector_get(solver->gradient, 3),
// gsl_vector_get(solver->gradient, 4));

// if (status == GSL_SUCCESS)
//     printf("Minimum found at:\n");
// printf("%5d %2.5f %2.5f %2.5f %2.5f %2.5f %10.5f\n", iter,
// gsl_vector_get(solver->x, 0),
// gsl_vector_get(solver->x, 1),
// gsl_vector_get(solver->x, 2),
// gsl_vector_get(solver->x, 3),
// gsl_vector_get(solver->x, 4),
// solver->f);