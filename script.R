# validation of inference from network nominations

library(rethinking)

N_id <- 30
N_dyads <- choose(N_id,2)
dyad_ij <- t(combn(1:N_id,2))
beta <- (-2)
m <- 5 # boost to reporting prob for true help event
H <- (-5) # baseline reporting rate

# add a kin covariate to predict ties
x <- rbern(N_dyads,0.1)
bx <- 1

# g and r
# individual giving and receiving offsets
vi <- rmvnorm2( N_id , Mu=c(0,0) , sigma=c(0.5,0.5) , Rho=matrix(c(1,0.2,0.2,1),2,2) )
g <- vi[,1]
r <- vi[,2]

# dij, dji
# dyad offsets
# make majority zero
Dij <- rmvnorm2( N_dyads , Mu=c(0,0) , sigma=c(1,1) , Rho=matrix(c(1,0.9,0.9,1),2,2) )
#Dij <- t(sapply( 1:nrow(Dij) , function(i) Dij[i,] * rbern(1,0.1) ))

# nomination rate offsets for individuals
h <- rmvnorm2( N_id , Mu=c(0,0) , sigma=c(0.5,0.5) , Rho=matrix(c(1,0.7,0.7,1),2,2) )

# generate true helping events from d,g,r

Y_true <- matrix(NA,nrow=N_dyads,ncol=2) # cols are ij,ji helping
for ( i in 1:N_dyads ) {
    for ( j in 1:2 )
        Y_true[i,j] <- rbern( 1 , inv_logit( 
            beta + 
            g[ dyad_ij[i,j] ] + 
            r[ dyad_ij[i,3-j] ] +
            Dij[ i , j ] +
            bx*x[i] ) )
}#i

# now generate nominations from true events

N <- N_dyads*4 # each individual in each dyad stating: (1) helped other (2) was helped
Y_obs <- rep(NA,N)
Y_true_vec <- rep(NA,N)
dyad_id <- rep( 1:N_dyads , each=4 )
rater <- rep( c(1,1,2,2) , times=N_dyads )
giver <- rep(c(1,2,1,2),times=N_dyads)
receiver <- rep(c(2,1,2,1),times=N_dyads)
giver_id <- rep(NA,N)
receiver_id <- rep(NA,N)
rater_id <- rep(NA,N)
for ( i in 1:N ) {
    giver_id[i] <- dyad_ij[ dyad_id[i] , giver[i] ]
    receiver_id[i] <- dyad_ij[ dyad_id[i] , receiver[i] ]
    rater_id[i] <- dyad_ij[ dyad_id[i] , rater[i] ]
    ytrue <- Y_true[ dyad_id[i] , giver[i] ]
    Y_true_vec[i] <- ytrue
    p_obs <- ytrue*inv_logit( H + h[rater_id[i],giver[i]] + m ) + (1-ytrue)*inv_logit( H + h[rater_id[i],giver[i]] )
    Y_obs[i] <- rbern( 1 , p_obs )
}#i

table(Y_true_vec,Y_obs)

dat_list <- list(
    N = N,
    N_id = N_id,
    N_dyad = N_dyads,
    g_id = giver_id,
    r_id = receiver_id,
    rater_id = rater_id,
    dyad_id = dyad_id,
    y = Y_obs,
    dyad_g = giver,
    dyad_r = receiver,
    dyad_table = dyad_ij,
    x = x
)

par_list <- c("H","beta","m","sigma_id","v_id","sigma_dyad","d","sigma_h","h","y_true","bx")

m1 <- stan( file="model.stan" , data=dat_list , pars=par_list , chains=1 , cores=1  , iter=1000 , control=list(adapt_delta=0.95) )

pars <- c("beta","H","m","sigma_id","sigma_dyad","sigma_h","bx")
precis(m1,2,pars=pars)
tracerplot(m1,pars=pars)

post <- extract.samples(m1)
g_est <- apply(post$v_id[,,1],2,mean)
r_est <- apply(post$v_id[,,2],2,mean)
h_est <- apply(post$h,2:3,mean)
d_est <- apply(post$d,2:3,mean)
y_true_est <- apply(post$y_true,2:3,mean)

blank(ex=2)
par(mfrow=c(3,3))
plot(g,g_est)
plot(r,r_est)
plot(g_est,r_est)
plot(h[,1],h_est[,1])
plot(h[,2],h_est[,2])
plot(h_est)
plot(Dij[,1],d_est[,1])
plot(Dij[,2],d_est[,2])
plot(d_est)

blank()
dcol1 <- col.alpha("blue",0.2)
dcol2 <- col.alpha("black",0.2)
plot( Y_true + rnorm(length(Y_true),0,0.01) , y_true_est , xlab="true state of tie" , ylab="posterior mean prob of tie" , col=dcol2 )


