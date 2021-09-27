# libs
library(ppclust)
library(cluster)
library(fclust)
library(ggplot2)
library(factoextra)
# library(EMD)

# functions

dists = function (v1, v2, p){
    sum(abs(v1-v2)^p)^(1/p);
}


WLI = function (Xca, U, H, f) 
{
    
    n = nrow(Xca)
    Xca = as.matrix(Xca)
    U = as.matrix(U)
    H = as.matrix(H)
    k = ncol(U)

    wli = list();
    
    D = matrix(0, nrow = n, ncol = k)

    vi.vj = c();

    for (i in 1:n) {
        for (c in 1:k) {
            D[i, c] = dists(Xca[i, ], H[c, ], f)^2
        }
    }

    for (c1 in 1:(k - 1)) {
        for (c2 in (c1 + 1):k) {
            vi.vj = c(vi.vj, dists(H[c1, ], H[c2, ], f)^2);
        }
    }    

    wln = sum(apply((U^2) * D, 2, sum)/apply(U, 2, sum));
    wli$comp = wln;

    wld = 1/2 * (min(vi.vj) + median(vi.vj));
    wli$sep = wld;

    wli = wli$comp/(2 * wli$sep);
    wli;
}   

PBMF = function (Xca, U, H, m, f) 
{
    
    n = nrow(Xca)
    Xca = as.matrix(Xca)
    U = as.matrix(U)
    H = as.matrix(H)
    k = ncol(U)

    pbmf = list();

    maximo = c();

    vXca = colMeans(Xca); #vXca = apply(Xca, 2, mean)

    D = matrix(0, nrow = n, ncol = k)

    e1.vXca = 0;

    for (i in 1:n) {
        for (c in 1:k) {
            D[i, c] = dists(Xca[i, ], H[c, ], f)   
        }
        e1.vXca = e1.vXca + dists(Xca[i, ], vXca, f)
    }

    for (c1 in 1:(k - 1)) {
        for (c2 in (c1 + 1):k) {
            maximo = c(maximo, dists(H[c1, ], H[c2, ], f));
        }
    }    

    jm = sum((U^m) * D);

    dc = max(maximo);

    pbmf = ((1/k) * (e1.vXca/jm) * dc)^2;

    pbmf;

    
}

K = function (Xca, U, H, f) 
{
    
    n = nrow(Xca)
    Xca = as.matrix(Xca)
    U = as.matrix(U)
    H = as.matrix(H)
    k = ncol(U)

    kindex = list();

    vi_vXca = c();

    vXca = colMeans(Xca); #v_Xca = apply(Xca, 2, mean)
    vH = colMeans(H); #v_H = apply(H, 2, mean)

    D = matrix(0, nrow = n, ncol = k)

    for (c in 1:k) {
        vi_vXca[c] = dists(H[c, ], vXca, f)^2;
        for (i in 1:n) {
            D[i, c] = dists(Xca[i, ], H[c, ], f)^2;  
        }
    }

    dists = 10^10 * sum(H^2)

    for (c1 in 1:(k - 1)) {
        for (c2 in (c1 + 1):k) {
            if (dists(H[c1, ], H[c2, ], f)^2 < dists) 
                dists = dists(H[c1, ], H[c2, ], f)^2;
        }
    }

    kindex$comp = sum((U^2) * D);

    kindex = (kindex$comp + (sum(vi_vXca)/k))/dists;

}

best_k_min <- function(scores, lim){
    menor = 0
    for (indice in 1:(lim-1)) {
        if (scores[indice] == min(scores)){
             menor = indice+1
        }
    }
    menor
}

best_k_max <- function(scores, lim){
    maior = 0
    for (indice in 1:(lim-1)) {
        if (scores[indice] == max(scores)){
             maior = indice+1
        }
    }
    maior
}

choose_index <- function(data, imfs.sel, limite = 20, file_save='score_indexs.csv'){
	# create daframe 
	X <- data.frame(data, imfs.sel)

	# silhueta fuzzy -  melhor quando assume o valor máximo
	scores_sf = c()
	scores_xb = c()
	scores_mpc = c()
	scores_pc = c()
	scores_wli = c()
	scores_pbmf = c()
	scores_k = c()
	scores_pe = c()
    scores_si <- c()

	for (indice in 1:(limite-1)) {
    
    # run FCM
    res.fcm <- fcm(X, centers=indice+1);
    
    # SF
    scores_sf[indice] <- SIL.F(X, res.fcm$u, alpha=1)
    
    # XB
    scores_xb[indice] <- XB(res.fcm$x, res.fcm$u, res.fcm$v)
    
    # MPF
    scores_mpc[indice] <- MPC(res.fcm$u)
    
    # PC
    # scores_pc[indice] <- PC(res.fcm$u)
    
    # WLI
    scores_wli[indice] <- WLI(X, res.fcm$u, res.fcm$v, 2)
    
    # PBMF
    scores_pbmf[indice] <- PBMF(X, res.fcm$u, res.fcm$v, 2, 2)
    
    # K
    scores_k[indice] <- K(X, res.fcm$u, res.fcm$v, 2)
    
    # PE
    # scores_pe[indice] <- PE(res.fcm$u)
    ss <- silhouette(res.fcm$cluster, dist(X))
    scores_si[indice] <- mean(ss[, 3])
    
    }


	# ================ PLOT ================ #

	#pdf('indices.pdf')
	par(mfrow=c(3,3))
	title.sf <- paste('SF = ',best_k_max(scores_sf, limite))
	plot(c(2:limite), scores_sf,
	       type="c", pch = 19, frame = FALSE, 
	       xlab="Number of clusters K - max ",
	       ylab="Scores SF", main=title.sf)
	text(2:limite, scores_sf, 2:limite)


	title.xb <- paste('XB = ',best_k_min(scores_xb, limite))
	plot(c(2:limite), scores_xb,
	       type="c", pch = 19, frame = FALSE, 
	       xlab="Number of clusters K - min",
	       ylab="Scores XB", main=title.xb)
	text(2:limite, scores_xb, 2:limite)


	title.mpc <- paste('MPC = ',best_k_max(scores_mpc, limite))
	plot(c(2:limite), scores_mpc,
	       type="c", pch = 19, frame = FALSE, 
	       xlab="Number of clusters K - max",
	       ylab="Scores MPC", main=title.mpc)
	text(2:limite, scores_mpc, 2:limite)


	title.wli <- paste('WLI = ',best_k_min(scores_wli, limite))
	plot(c(2:limite), scores_wli,
	       type="c", pch = 19, frame = FALSE, 
	       xlab="Number of clusters K - min",
	       ylab="Scores WLI", main=title.wli)
	text(2:limite, scores_wli, 2:limite)


	title.ppbmf <- paste('PBMF = ',best_k_max(scores_pbmf, limite))
	plot(c(2:limite), scores_pbmf,
	       type="c", pch = 19, frame = FALSE, 
	       xlab="Number of clusters K - max",
	       ylab="Scores PBMF", main=title.ppbmf)
	text(2:limite, scores_pbmf, 2:limite)


	title.k <- paste('K = ',best_k_min(scores_k, limite))
	plot(c(2:limite), scores_k,
	       type="c", pch = 19, frame = FALSE, 
	       xlab="Number of clusters K - min",
	       ylab="Scores K", main=title.k)
	text(2:limite, scores_k, 2:limite)


	title.si <- paste('SI = ',best_k_max(scores_si, limite))
    plot(c(2:limite), scores_si,
           type="c", pch = 19, frame = FALSE, 
           xlab="Number of clusters K - max",
           ylab="Scores K", main=title.si)
    text(2:limite, scores_si, 2:limite)

    # create table to storange scores validation index
    all_scores <- matrix(c(scores_si, scores_sf, scores_xb, scores_mpc, scores_wli, scores_pbmf, scores_k), ncol=7, byrow=TRUE)
    colnames(all_scores) <- c(title.si, title.sf, title.xb, title.mpc, title.wli, title.ppbmf, title.k)

	# # save scores_sf
	write.csv(all_scores, file=file_save)

}

