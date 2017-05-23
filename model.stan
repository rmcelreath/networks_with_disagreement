data{
    int N;
    int N_id;
    int N_dyad;
    int g_id[N];
    int r_id[N];
    int rater_id[N];
    int dyad_id[N];
    int y[N];
    int dyad_g[N];
    int dyad_r[N];
    int dyad_table[N_dyad,2];
    real x[N_dyad]; // covariate
}
transformed data{
    int Yobs[N_dyad,2,2]; // [dyad , rater k in {1,2} , q in {1,2}->{k->,->k} ]
    // q=1 corresponds to rater being giver
    // q=2 corresponds to rater being receiver
    for ( i in 1:N ) {
        int q;
        int k;
        q = 2;
        k = 2;
        if ( rater_id[i]==g_id[i] ) q = 1;
        if ( rater_id[i]==dyad_table[dyad_id[i],1] ) k = 1;
        Yobs[ dyad_id[i] , k , q ] = y[i];
    }
}
parameters{
    real beta;
    real H[2];
    real<lower=0> m;
    real bx;

    // g and r
    matrix[2,N_id] z_id; // g and r for each individual
    vector<lower=0>[2] sigma_id;
    cholesky_factor_corr[2] L_Rho_id;

    // dyads
    matrix[2,N_dyad] z_dyad; // Dij and Dji for each dyad 
    real<lower=0> sigma_dyad; // variance same for both directions
    cholesky_factor_corr[2] L_Rho_dyad;

    // question nomination rates
    matrix[2,N_id] zh;
    vector<lower=0>[2] sigma_h;
    cholesky_factor_corr[2] L_Rho_h;
}
transformed parameters{
    matrix[N_id,2] v_id;
    matrix[N_id,2] h;
    matrix[N_dyad,2] d;
    real logPuy[N_dyad,2,2,2]; // P(u_ijk|y_ij) : indices [dyad,k,q,y_ij={1,0}]
    real logPy[N_dyad,2]; // [ P(y_ij) , P(y_ji) ]

    // transform varying effects to scale
    v_id = (diag_pre_multiply( sigma_id , L_Rho_id ) * z_id)';
    h = (diag_pre_multiply( sigma_h , L_Rho_h ) * zh)';
    d = (diag_pre_multiply( rep_vector(sigma_dyad,2) , L_Rho_dyad ) * z_dyad)';

    // compute logPuy for each observed nomination
    for ( i in 1:N ) {
        int q;
        int k;
        real p_true;
        q = 2;
        k = 2;
        if ( rater_id[i]==g_id[i] ) q = 1;
        if ( rater_id[i]==dyad_table[dyad_id[i],1] ) k = 1;

        logPy[dyad_id[i],dyad_g[i]] = log_inv_logit( beta + 
                        v_id[g_id[i],1] +
                        v_id[r_id[i],2] +
                        d[dyad_id[i],dyad_g[i]] + 
                        bx*x[dyad_id[i]] );
        
        if ( y[i]==1 ) {
            logPuy[dyad_id[i],k,q,1] = log_inv_logit( H[q] + h[rater_id[i],q] + m );
            logPuy[dyad_id[i],k,q,2] = log_inv_logit( H[q] + h[rater_id[i],q] );
        }
        if ( y[i]==0 ) {
            logPuy[dyad_id[i],k,q,1] = log1m_inv_logit( H[q] + h[rater_id[i],q] + m );
            logPuy[dyad_id[i],k,q,2] = log1m_inv_logit( H[q] + h[rater_id[i],q] );
        }
    }//i

}
model{
    // unobserved variables
    beta ~ normal(0,5);
    H ~ normal(0,5);
    m ~ normal(1,1) T[0,];
    bx ~ normal(0,1);

    to_vector(z_id) ~ normal(0,1);
    to_vector(zh) ~ normal(0,1);
    to_vector(z_dyad) ~ normal(0,1);
    sigma_id ~ exponential(1);
    sigma_h ~ exponential(1);
    sigma_dyad ~ exponential(2);
    L_Rho_id ~ lkj_corr_cholesky(4);
    L_Rho_h ~ lkj_corr_cholesky(4);
    L_Rho_dyad ~ lkj_corr_cholesky(4);

    // observed variables
    for ( did in 1:N_dyad ) {
        int dir;
        for ( k in 1:2 )
            for ( q in 1:2 ) {
                if ( k==1 && q==1 ) dir = 1;
                if ( k==1 && q==2 ) dir = 2;
                if ( k==2 && q==1 ) dir = 2;
                if ( k==2 && q==2 ) dir = 1;
                target += log_mix( exp( logPy[ did , dir ] ) ,
                            logPuy[ did , k , q , 1 ] ,
                            logPuy[ did , k , q , 2 ] );
            }//q
    }//did
    
}//model
generated quantities{
    matrix[N_dyad,2] y_true;
    // let y_ij indicate true tie i->j
    // let u_ijk indicate nomination of tie i->j by k
    // to compute P(y_ih==1|u_ij) need:
    // P(y_ij=1|u_ij) = P(u_ij|y_ij=1)P(y_ij=1) / P(u_ij)
    // P(u_ij|y_ij=1) = Product over relevant {k,q}: inv_logit( H_q + h_{kq} + m(1) )
    // relevant {k,q} sets are:
    // y_ij : Yobs[d,k,q] : {k,q}={ {1,1} , {2,2} } 
    // y_ji : Yobs[d,k,q] : {k,q}={ {1,2} , {2,1} } 
    // P(y_ij=1) = inv_logit( beta + g + r + d )
    // P(u_ij) = P(y_ij)P(u_iji,u_ijj|y_ij=1) + (1-P(y_ij))P(u_iji,u_ijj|y_ij=0)
    // P(u_iji,u_ijj|y_ij) = P(u_iji|y_ij)P(u_ijj|y_ij)
    for ( did in 1:N_dyad ) {
        real logPu; // P(u_ij) = P(y_ij)P(u_iji,u_ijj|y_ij=1) + (1-P(y_ij))P(u_iji,u_ijj|y_ij=0)
        for ( dir in 1:2 ) { // 1:ij or 2:ji
            if ( dir==1 ) {
                logPu = log_mix( exp( logPy[ did , dir ] ) ,
                            logPuy[did,1,1,1] + logPuy[did,2,2,1] ,  // y_ij=1
                            logPuy[did,1,1,2] + logPuy[did,2,2,2] ); // y_ij=0
                y_true[did,dir] = exp( logPuy[did,1,1,1] + logPuy[did,2,2,1] + logPy[did,dir] - logPu );
            }//ij
            if ( dir==2 ) {
                logPu = log_mix( exp( logPy[ did , dir ] ) ,
                            logPuy[did,1,2,1] + logPuy[did,2,1,1] ,  // y_ji=1
                            logPuy[did,1,2,2] + logPuy[did,2,1,2] ); // y_ji=0
                y_true[did,dir] = exp( logPuy[did,1,2,1] + logPuy[did,2,1,1] + logPy[did,dir] - logPu );
            }//ji
            // can simulate binary outcomes, but yields same info
            //y_true[did,dir] = bernoulli_rng( y_true[did,dir] );
        }//dir
    }//did
}
