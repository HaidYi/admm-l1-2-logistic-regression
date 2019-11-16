/*
 *  Definition of L2 logistic regression solver using GSL Multi-dimension Optimizer
 * 
 *  see https://www.gnu.org/software/gsl/doc/html/multimin.html
 * 
 * 
*/

#ifndef SOLVER_H
#define SOLVER_H

#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multimin.h>


struct Param
{
    gsl_matrix *X;  // data matrix (w/o augmentation)
    gsl_vector *y;  // label vector, y \in {1, -1}
    gsl_vector *u;  // dual variable
    gsl_vector *z;  // primal varialbe
    double rho;     // augmented Lagrangian parameter
};

double logit_f(const gsl_vector* , void*);
void logit_df(const gsl_vector*, void*, gsl_vector*);
void logit_fdf(const gsl_vector*, void*, double*, gsl_vector*);
gsl_vector* update_x(struct Param*,  gsl_vector*);

#endif