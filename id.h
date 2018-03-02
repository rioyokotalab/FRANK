#include "gsl_wrapper.h"

/* computes the approximate low rank SVD of rank k of matrix M using BBt version */
void randomized_low_rank_svd1(gsl_matrix *M, int k, gsl_matrix **U, gsl_matrix **S, gsl_matrix **V){
    int i,m,n;
    m = M->size1; n = M->size2;

    *U = gsl_matrix_calloc(m,k);
    *S = gsl_matrix_calloc(k,k);
    *V = gsl_matrix_calloc(n,k);

    // build random matrix
    gsl_matrix *RN = gsl_matrix_calloc(n, k); // calloc sets all elements to zero
    initialize_random_matrix(RN);

    // multiply to get matrix of random samples Y
    gsl_matrix *Y = gsl_matrix_alloc(m,k);
    matrix_matrix_mult(M, RN, Y);

    // build Q from Y
    gsl_matrix *Q = gsl_matrix_alloc(m,k);
    QR_factorization_getQ(Y, Q);


    // build the matrix B B^T = Q^T M M^T Q
    gsl_matrix *B = gsl_matrix_alloc(k,n);
    matrix_transpose_matrix_mult(Q,M,B);

    gsl_matrix *Bt = gsl_matrix_alloc(n,k);
    matrix_transpose_matrix_mult(M,Q,Bt);

    gsl_matrix *BBt = gsl_matrix_alloc(k,k);
    matrix_matrix_mult(B,Bt,BBt);


    // compute eigendecomposition of BBt
    gsl_vector *evals = gsl_vector_alloc(k);
    gsl_matrix *Uhat = gsl_matrix_alloc(k, k);
    compute_evals_and_evecs_of_symm_matrix(BBt, evals, Uhat);


    // compute singular values and matrix Sigma
    gsl_vector *singvals = gsl_vector_alloc(k);
    for(i=0; i<k; i++){
        gsl_vector_set(singvals,i,sqrt(gsl_vector_get(evals,i)));
    }
    build_diagonal_matrix(singvals, k, *S);


    // compute U = Q*Uhat mxk * kxk = mxk
    matrix_matrix_mult(Q,Uhat,*U);


    // compute nxk V
    // V = B^T Uhat * Sigma^{-1}
    gsl_matrix *Sinv = gsl_matrix_alloc(k,k);
    gsl_matrix *UhatSinv = gsl_matrix_alloc(k,k);
    invert_diagonal_matrix(Sinv,*S);
    matrix_matrix_mult(Uhat,Sinv,UhatSinv);
    matrix_matrix_mult(Bt,UhatSinv,*V);

    // clean up
    gsl_matrix_free(RN);
    gsl_matrix_free(Y);
    gsl_matrix_free(Q);
    gsl_matrix_free(B);
    gsl_matrix_free(Bt);
    gsl_matrix_free(Sinv);
    gsl_matrix_free(UhatSinv);
}


/* computes the approximate low rank SVD of rank k of matrix M using QR method */
void randomized_low_rank_svd2(gsl_matrix *M, int k, gsl_matrix **U, gsl_matrix **S, gsl_matrix **V){
    int m,n;
    m = M->size1; n = M->size2;

    // setup mats
    *U = gsl_matrix_calloc(m,k);
    *S = gsl_matrix_calloc(k,k);
    *V = gsl_matrix_calloc(n,k);

    // build random matrix
    gsl_matrix *RN = gsl_matrix_calloc(n,k); // calloc sets all elements to zero
    //RN = matrix_load_from_file("data/R.mtx");
    initialize_random_matrix(RN);

    // multiply to get matrix of random samples Y
    gsl_matrix *Y = gsl_matrix_alloc(m,k);
    matrix_matrix_mult(M, RN, Y);

    // build Q from Y
    gsl_matrix *Q = gsl_matrix_alloc(m,k);
    QR_factorization_getQ(Y, Q);

    // form Bt = Mt*Q : nxm * mxk = nxk
    gsl_matrix *Bt = gsl_matrix_alloc(n,k);
    matrix_transpose_matrix_mult(M,Q,Bt);

    gsl_matrix *Qhat = gsl_matrix_calloc(n,k);
    gsl_matrix *Rhat = gsl_matrix_calloc(k,k);
    compute_QR_compact_factorization(Bt,Qhat,Rhat);

    // compute SVD of Rhat (kxk)
    gsl_matrix *Uhat = gsl_matrix_alloc(k,k);
    gsl_vector *Sigmahat = gsl_vector_alloc(k);
    gsl_matrix *Vhat = gsl_matrix_alloc(k,k);
    gsl_vector *svd_work_vec = gsl_vector_alloc(k);
    gsl_matrix_memcpy(Uhat, Rhat);
    gsl_linalg_SV_decomp (Uhat, Vhat, Sigmahat, svd_work_vec);

    // record singular values
    build_diagonal_matrix(Sigmahat, k, *S);

    // U = Q*Vhat
    matrix_matrix_mult(Q,Vhat,*U);

    // V = Qhat*Uhat
    matrix_matrix_mult(Qhat,Uhat,*V);

    // free stuff
    gsl_matrix_free(RN);
    gsl_matrix_free(Y);
    gsl_matrix_free(Q);
    gsl_matrix_free(Rhat);
    gsl_matrix_free(Qhat);
    gsl_matrix_free(Uhat);
    gsl_matrix_free(Vhat);
    gsl_matrix_free(Bt);
}


/* computes the approximate low rank SVD of rank k of matrix M using QR method and
(M M^T)^q M R sampling */
void randomized_low_rank_svd3(gsl_matrix *M, int k, int q, int s, gsl_matrix **U, gsl_matrix **S, gsl_matrix **V){
    int j,m,n;
    m = M->size1; n = M->size2;

    // setup mats
    *U = gsl_matrix_calloc(m,k);
    *S = gsl_matrix_calloc(k,k);
    *V = gsl_matrix_calloc(n,k);

    // build random matrix
    gsl_matrix *RN = gsl_matrix_calloc(n,k); // calloc sets all elements to zero
    //RN = matrix_load_from_file("data/R.mtx");
    initialize_random_matrix(RN);

    // multiply to get matrix of random samples Y
    gsl_matrix *Y = gsl_matrix_alloc(m,k);
    matrix_matrix_mult(M, RN, Y);


    // now build up (M M^T)^q R
    gsl_matrix *Z = gsl_matrix_alloc(n,k);
    gsl_matrix *Yorth = gsl_matrix_alloc(m,k);
    gsl_matrix *Zorth = gsl_matrix_alloc(n,k);
    for(j=1; j<q; j++){

        if((2*j-2) % s == 0){
            QR_factorization_getQ(Y, Yorth);
            matrix_transpose_matrix_mult(M,Yorth,Z);
        }
        else{
            matrix_transpose_matrix_mult(M,Y,Z);
        }


        if((2*j-1) % s == 0){
            QR_factorization_getQ(Z, Zorth);
            matrix_matrix_mult(M,Zorth,Y);
        }
        else{
            matrix_matrix_mult(M,Z,Y);
        }
    }


    // build Q from Y
    gsl_matrix *Q = gsl_matrix_alloc(m,k);
    QR_factorization_getQ(Y, Q);

    // form Bt = Mt*Q : nxm * mxk = nxk
    gsl_matrix *Bt = gsl_matrix_alloc(n,k);
    matrix_transpose_matrix_mult(M,Q,Bt);

    gsl_matrix *Qhat = gsl_matrix_calloc(n,k);
    gsl_matrix *Rhat = gsl_matrix_calloc(k,k);
    compute_QR_compact_factorization(Bt,Qhat,Rhat);

    // compute SVD of Rhat (kxk)
    gsl_matrix *Uhat = gsl_matrix_alloc(k,k);
    gsl_vector *Sigmahat = gsl_vector_alloc(k);
    gsl_matrix *Vhat = gsl_matrix_alloc(k,k);
    gsl_vector *svd_work_vec = gsl_vector_alloc(k);
    gsl_matrix_memcpy(Uhat, Rhat);
    gsl_linalg_SV_decomp (Uhat, Vhat, Sigmahat, svd_work_vec);

    // record singular values
    build_diagonal_matrix(Sigmahat, k, *S);

    // U = Q*Vhat
    matrix_matrix_mult(Q,Vhat,*U);

    // V = Qhat*Uhat
    matrix_matrix_mult(Qhat,Uhat,*V);

    // free stuff
    gsl_matrix_free(RN);
    gsl_matrix_free(Y);
    gsl_matrix_free(Q);
    gsl_matrix_free(Rhat);
    gsl_matrix_free(Qhat);
    gsl_matrix_free(Uhat);
    gsl_matrix_free(Vhat);
    gsl_matrix_free(Bt);
    gsl_matrix_free(Yorth);
    gsl_matrix_free(Zorth);
    gsl_matrix_free(Z);
}



/* computes the approximate low rank SVD of rank k of matrix M using QR version
automatically estimates the rank needed */
void randomized_low_rank_svd2_autorank1(gsl_matrix *M, double frac_of_max_rank, double TOL, gsl_matrix**U, gsl_matrix **S, gsl_matrix **V){
    int m,n,k;
    gsl_matrix *Q;
    m = M->size1; n = M->size2;

    estimate_rank_and_buildQ(M,frac_of_max_rank,TOL,&Q,&k);

    // setup U, S, and V
    *U = gsl_matrix_calloc(m,k);
    *S = gsl_matrix_calloc(k,k);
    *V = gsl_matrix_calloc(n,k);

    // form Bt = Mt*Q : nxm * mxk = nxk
    gsl_matrix *Bt = gsl_matrix_alloc(n,k);
    matrix_transpose_matrix_mult(M,Q,Bt);

    gsl_matrix *Qhat = gsl_matrix_calloc(n,k);
    gsl_matrix *Rhat = gsl_matrix_calloc(k,k);
    compute_QR_compact_factorization(Bt,Qhat,Rhat);

    // compute SVD of Rhat (kxk)
    gsl_matrix *Uhat = gsl_matrix_alloc(k,k);
    gsl_vector *Sigmahat = gsl_vector_alloc(k);
    gsl_matrix *Vhat = gsl_matrix_alloc(k,k);
    gsl_vector *svd_work_vec = gsl_vector_alloc(k);
    gsl_matrix_memcpy(Uhat, Rhat);
    gsl_linalg_SV_decomp (Uhat, Vhat, Sigmahat, svd_work_vec);

    // record singular values
    build_diagonal_matrix(Sigmahat, k, *S);

    // U = Q*Vhat
    matrix_matrix_mult(Q,Vhat,*U);

    // V = Qhat*Uhat
    matrix_matrix_mult(Qhat,Uhat,*V);

    // free stuff
    gsl_matrix_free(Q);
    gsl_matrix_free(Rhat);
    gsl_matrix_free(Qhat);
    gsl_matrix_free(Uhat);
    gsl_matrix_free(Vhat);
    gsl_matrix_free(Bt);
}


/* computes the approximate low rank SVD of rank k of matrix M using QR version
automatically estimates the rank needed */
void randomized_low_rank_svd2_autorank2(gsl_matrix *M, int kblocksize, double TOL, gsl_matrix **U, gsl_matrix **S, gsl_matrix **V){
    int m,n,k;
    gsl_matrix *Y,*Q;
    m = M->size1; n = M->size2;

    estimate_rank_and_buildQ2(M, kblocksize, TOL, &Y, &Q, &k);

    // setup U, S, and V
    *U = gsl_matrix_calloc(m,k);
    *S = gsl_matrix_calloc(k,k);
    *V = gsl_matrix_calloc(n,k);

    // form Bt = Mt*Q : nxm * mxk = nxk
    gsl_matrix *Bt = gsl_matrix_alloc(n,k);
    matrix_transpose_matrix_mult(M,Q,Bt);

    gsl_matrix *Qhat = gsl_matrix_calloc(n,k);
    gsl_matrix *Rhat = gsl_matrix_calloc(k,k);
    compute_QR_compact_factorization(Bt,Qhat,Rhat);

    // compute SVD of Rhat (kxk)
    gsl_matrix *Uhat = gsl_matrix_alloc(k,k);
    gsl_vector *Sigmahat = gsl_vector_alloc(k);
    gsl_matrix *Vhat = gsl_matrix_alloc(k,k);
    gsl_vector *svd_work_vec = gsl_vector_alloc(k);
    gsl_matrix_memcpy(Uhat, Rhat);
    gsl_linalg_SV_decomp (Uhat, Vhat, Sigmahat, svd_work_vec);

    // record singular values
    build_diagonal_matrix(Sigmahat, k, *S);

    // U = Q*Vhat
    matrix_matrix_mult(Q,Vhat,*U);

    // V = Qhat*Uhat
    matrix_matrix_mult(Qhat,Uhat,*V);

    // free stuff
    gsl_matrix_free(Q);
    gsl_matrix_free(Rhat);
    gsl_matrix_free(Qhat);
    gsl_matrix_free(Uhat);
    gsl_matrix_free(Vhat);
    gsl_matrix_free(Bt);
}


/* computes the approximate low rank SVD of rank k of matrix M using QR version
via (M M^T)^q M R, automatically estimates the rank needed */
void randomized_low_rank_svd3_autorank2(gsl_matrix *M, int kblocksize, double TOL, int q, int s, gsl_matrix **U, gsl_matrix **S, gsl_matrix **V){
    int m,n,k;
    gsl_matrix *Y,*Q;
    m = M->size1; n = M->size2;

    estimate_rank_and_buildQ2(M, kblocksize, TOL, &Y, &Q, &k);
    // setup mats
    *U = gsl_matrix_calloc(m,k);
    *S = gsl_matrix_calloc(k,k);
    *V = gsl_matrix_calloc(n,k);

    // now build up (M M^T)^q R
    gsl_matrix *Z = gsl_matrix_alloc(n,k);
    gsl_matrix *Yorth = gsl_matrix_alloc(m,k);
    gsl_matrix *Zorth = gsl_matrix_alloc(n,k);
    for(int j=1; j<q; j++){
        if((2*j-2) % s == 0){
            QR_factorization_getQ(Y, Yorth);
            matrix_transpose_matrix_mult(M,Yorth,Z);
        }
        else{
            matrix_transpose_matrix_mult(M,Y,Z);
        }


        if((2*j-1) % s == 0){
            QR_factorization_getQ(Z, Zorth);
            matrix_matrix_mult(M,Zorth,Y);
        }
        else{
            matrix_matrix_mult(M,Z,Y);
        }
    }


    // build Q from Y
    //gsl_matrix *Q = gsl_matrix_alloc(m,k);
    QR_factorization_getQ(Y, Q);

    // form Bt = Mt*Q : nxm * mxk = nxk
    gsl_matrix *Bt = gsl_matrix_alloc(n,k);
    matrix_transpose_matrix_mult(M,Q,Bt);

    gsl_matrix *Qhat = gsl_matrix_calloc(n,k);
    gsl_matrix *Rhat = gsl_matrix_calloc(k,k);
    compute_QR_compact_factorization(Bt,Qhat,Rhat);

    // compute SVD of Rhat (kxk)
    gsl_matrix *Uhat = gsl_matrix_alloc(k,k);
    gsl_vector *Sigmahat = gsl_vector_alloc(k);
    gsl_matrix *Vhat = gsl_matrix_alloc(k,k);
    gsl_vector *svd_work_vec = gsl_vector_alloc(k);
    gsl_matrix_memcpy(Uhat, Rhat);
    gsl_linalg_SV_decomp (Uhat, Vhat, Sigmahat, svd_work_vec);

    // record singular values
    build_diagonal_matrix(Sigmahat, k, *S);

    // U = Q*Vhat
    matrix_matrix_mult(Q,Vhat,*U);

    // V = Qhat*Uhat
    matrix_matrix_mult(Qhat,Uhat,*V);

    // free stuff
    gsl_matrix_free(Y);
    gsl_matrix_free(Q);
    gsl_matrix_free(Rhat);
    gsl_matrix_free(Qhat);
    gsl_matrix_free(Uhat);
    gsl_matrix_free(Vhat);
    gsl_matrix_free(Bt);
    gsl_matrix_free(Yorth);
    gsl_matrix_free(Zorth);
    gsl_matrix_free(Z);
}
