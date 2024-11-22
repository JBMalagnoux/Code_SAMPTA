%% CODE CSC DL ini NTF - Sensors space decomposition then inverse problem


clear variables;
close all;

addpath('ltfat/')
addpath('fct_tfnmf/')
addpath('data/')
addpath('fct_cdl/')
addpath('toolbox_signal')
addpath('toolbox_general')

ltfatstart


load('dataw80.mat')
SNR_motifs_average = zeros(k,1);
optimal_transport_Uk_average = zeros(k,1);
optimal_transport_U_average = 0;
DIS_atoms_spectrum_average = 0;
cpu_time_average=0;
lead_field = G;
Nubset=1;
for nub=1:Nubset
% Sensors Gaussian noising
snrlvl = 5;
XXsensors = Xsensors(:,:,nub);
sigma2 = var(XXsensors(:))*10^(-snrlvl/10); % add noise
XXsensors = XXsensors' + (sqrt(sigma2)*randn(size(XXsensors)))';
tic

% Gabor parameters
M = 256;
a = M/2;
g = gabwin({'tight', 'hann'}, a, M);
G = dgtreal(XXsensors, g, a, M);
[nf,nt,nq] = size(G);


% TF operator
op.analysis = @(x) dgtreal(x,g,a,M);
op.synthesis = @(x) idgtreal(x,g,a,M,T);


Gamma = op.analysis(XXsensors);
G = abs(Gamma);

k = 3;
options.creation = 1;
options.max_it = 1000;
options.T = T;
options.M = M;
options.a = a;
options.Xsensors = Xsensors(:,:,nub);

% Ini SVD
[U_svd,D_svd,S_svd] = svd(sum(G,3),"econ");
S_svd = sqrt(D_svd)*S_svd';
U_svd = U_svd*sqrt(D_svd);
W_init = abs(U_svd(:,1:k));
H_init = abs(S_svd(1:k,:))';
C_init = ones(nq,k);


%%% IS-NTF %%%

[W, H, C] = IS_NTF(G,W_init,H_init,C_init,options);

[alphak,atom] = get_alphak_by_Wiener_filter(W, H', Gamma, k, options, op);
Alphak = reshape(sum(alphak,3),[nf nt k]);


v_init = atom;
normV = sqrt(sum(abs(v_init).^2,1));
v_init = v_init./normV;
% u_init=C;
% normU = sqrt(sum(abs(u_init).^2,1));
% u_init = u_init./normU;
u_init =C_init;
z_tmp = H;
[~,nfr] = size(H');
%z_init = ifftreal(fftreal(H),T);
z_init = zeros(T,k);
for l=2:nfr
    z_init(a*(l-2)+1,:) = z_tmp(l,:);
end
M = length(v_init(:,1));

MaxLambda = MaxLambda_multi_cdl(XXsensors,z_init,u_init,v_init);
lambda = logspace(log10(MaxLambda)-0.6,log10(MaxLambda)-5,10);
%lambda = logspace(1,-4,15);
pos_z = 1;
pos_u = 0;
options.max_it = 1000;
options.max_it_cdl = 10;
options.max_it_csc = 10;
options.oracle = 1;

%%% CDl Multivariate rank-1

[xk, v, u, z2] = M1csc_dl(XXsensors,z_init,v_init, u_init,lambda,pos_z,pos_u,k,options);

x = sum(xk,3);
xtotk = reshape(sum(xk,2),[T k]);
V1=[zeros(91,1);V(:,1)/norm(V(:,1),'inf');[zeros(37,1)]]; 
V2=[zeros(31,1);0.5*V(:,2)/norm(V(:,2),'inf');[zeros(97,1)]];
V3=[zeros(115,1);V(:,3)/norm(V(:,3),'inf');[zeros(13,1)]];
%%% Results




%% Inverse problem for u and reconstruction of the sources


Nbit = 100;
L = norm(lead_field)^2;
Nblambda2 = 20;
A=lead_field'*u;
EnerA = zeros(20,1);
for h=1:N
    EnerA(h,1) = norm(A(h,:));
end
lambda2Max = 0.5*norm(A,'inf');
lambda2Min = 1e-5*lambda2Max;
lambda2Vect = logspace(log10(lambda2Max)-0.5,log10(lambda2Min),Nblambda2);
lambda2number = 0;

U_est = lead_field'*u;
U_est = max(U_est,0);


for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
    lambda2number = lambda2number + 1;
    lambdaOnL = lambda/L;
    z_est = U_est;
    for it=0:Nbit-1
        U_est_old = U_est;
        U_est = z_est + lead_field'*(u-lead_field*z_est)/L;
        U_est = U_est.*max(0,1-lambdaOnL./(abs(U_est)+eps));
        z_est = U_est + (it)/(it+5)*(U_est-U_est_old);
    end
 U_est = max(U_est,0);  
end

cpu_time=toc;
cpu_time_average = cpu_time_average + cpu_time/Nubset;
VV=zeros(M,K);
VV(:,1)=[zeros(91,1);V(:,1)/norm(V(:,1));[zeros(37,1)]]; 
VV(:,2)=[zeros(31,1);V(:,2)/norm(V(:,2));[zeros(97,1)]];
VV(:,3)=[zeros(115,1);V(:,3)/norm(V(:,3));[zeros(13,1)]];
%%

% 
% %%% results
% 
fh=figure(1);
fh.WindowState = 'maximized';
subplot(2,2,1)
plot(v(:,3)/norm(v(:,3)),'b','LineWidth',2)
hold on 
plot([zeros(31,1);V(:,2)/norm(V(:,2));[zeros(97,1)]],'r','LineWidth',1)


subplot(2,2,2)
plot(v(:,2)/norm(v(:,2)),'b','LineWidth',2)
hold on 
plot([zeros(91,1);V(:,1)/norm(V(:,1));[zeros(37,1)]],'r','LineWidth',1)


subplot(2,2,3)
plot(v(:,1)/norm(v(:,1)),'b','LineWidth',2)
hold on 
plot([zeros(115,1);V(:,3)/norm(V(:,3));[zeros(13,1)]],'r','LineWidth',1)

fig=gcf;
set(findall(fig,'-property','FontSize'),'FontSize',25)

fh=figure(2);
fh.WindowState = 'maximized';


subplot(2,2,1)
plot(z(:,2)/norm(z(:,2),'inf'),'b','LineWidth',2)
hold on
plot(z2(:,2)/norm(z2(:,2),'inf'),'r','LineWidth',1)

subplot(2,2,2)
plot(z(:,3)/norm(z(:,3),'inf'),'b','LineWidth',2)
hold on
plot(z2(:,1)/norm(z2(:,1),'inf'),'r','LineWidth',1)

subplot(2,2,3)
plot(z(:,1)/norm(z(:,1),'inf'),'b','LineWidth',2)
hold on
plot(z2(:,3)/norm(z2(:,3),'inf'),'r','LineWidth',1)

fig=gcf;
set(findall(fig,'-property','FontSize'),'FontSize',25)


fh=figure(3);
fh.WindowState = 'maximized';

subplot(2,2,1)
plot(lead_field*U(:,1)/norm(lead_field*U(:,1),'inf'))
hold on
plot(u(:,2)/norm(u(:,2),'inf'))

subplot(2,2,2)
plot(lead_field*U(:,2)/norm(lead_field*U(:,2),'inf'))
hold on
plot(u(:,1)/norm(u(:,1),'inf'))

subplot(2,2,3)
plot(lead_field*U(:,3)/norm(lead_field*U(:,3),'inf'))
hold on
plot(u(:,3)/norm(u(:,3),'inf'))


fig=gcf;
set(findall(fig,'-property','FontSize'),'FontSize',25)

fh=figure(5);
fh.WindowState = 'maximized';

stem(sum(U,2)/norm(sum(U,2),'inf'),'b','filled')
hold on
stem(sum(U_est,2)/norm(sum(U_est,2),'inf'),'r','filled')

fig=gcf;
set(findall(fig,'-property','FontSize'),'FontSize',25)


fh=figure(6);
fh.WindowState = 'maximized';


subplot(2,2,1)
stem(U(:,3)/norm(U(:,3),'inf'),'b','filled')
hold on
stem(U_est(:,3)/norm(U_est(:,3),'inf'),'r','filled')

subplot(2,2,2)
stem(U(:,2)/norm(U(:,2),'inf'),'b','filled')
hold on
stem(U_est(:,1)/norm(U_est(:,1),'inf'),'r','filled')

subplot(2,2,3)
stem(U(:,1)/norm(U(:,1),'inf'),'b','filled')
hold on
stem(U_est(:,2)/norm(U_est(:,2),'inf'),'r','filled')


fig=gcf;
set(findall(fig,'-property','FontSize'),'FontSize',25)


fh=figure(7);
fh.WindowState = 'maximized';


subplot(2,2,1)
plot(sum(X1(:,:,nub),2)/norm(sum(X1(:,:,nub),2),'inf'),'b','LineWidth',3)
hold on
plot(xtotk(:,3)/norm(xtotk(:,3),'inf'),'r','LineWidth',1)

subplot(2,2,2)
plot(sum(X2(:,:,nub),2)/norm(sum(X2(:,:,nub),2),'inf'),'b','LineWidth',3)
hold on
plot(xtotk(:,2)/norm(xtotk(:,2),'inf'),'r','LineWidth',1)


subplot(2,2,3)
plot(sum(X3(:,:,nub),2)/norm(sum(X3(:,:,nub),2),'inf'),'b','LineWidth',3)
hold on
plot(xtotk(:,1)/norm(xtotk(:,1),'inf'),'r','LineWidth',1)


fig=gcf;
set(findall(fig,'-property','FontSize'),'FontSize',25)


%%% Quant results


SNR_motifs = zeros(K,1);
SNR_sources = zeros(K,1);
optimal_transport_Uk = zeros(K,1);
corr_atoms = zeros(K,1);


SNR_motifs(1,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,2)),max(snr(sum(X2(:,:,nub),2),xtotk(:,2)),snr(sum(X3(:,:,nub),2),xtotk(:,2))));
SNR_motifs_average(1,1) = SNR_motifs_average(1,1)+SNR_motifs(1,1)/Nubset;
SNR_motifs(2,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,1)),max(snr(sum(X2(:,:,nub),2),xtotk(:,1)),snr(sum(X3(:,:,nub),2),xtotk(:,1))));
SNR_motifs_average(2,1) = SNR_motifs_average(2,1)+SNR_motifs(2,1)/Nubset;
SNR_motifs(3,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,3)),max(snr(sum(X2(:,:,nub),2),xtotk(:,3)),snr(sum(X3(:,:,nub),2),xtotk(:,3))));
SNR_motifs_average(3,1) = SNR_motifs_average(3,1)+SNR_motifs(3,1)/Nubset;
SNRtot = snr(sum(sum(X(:,:,nub),3),2),sum(sum(X(:,:,nub),3),2)-sum(xtotk,2));

Fv = abs(fftreal(v));
FV = abs(fftreal([V2 V1 V3]));
Fv = Fv./max(Fv);
FV = FV./max(FV);
FV = FV(FV>0);
Fv = Fv(FV>0);
Tmp = (FV)./(Fv) - log((FV)./(Fv) ) -1;
Tmp(isnan(Tmp)) = 0;
DIS_atoms_spectrum = sum(Tmp(:));
DIS_atoms_spectrum_average = DIS_atoms_spectrum_average + DIS_atoms_spectrum/Nubset;
for kk=1:K
    corr_atoms(kk)= max(max(max(xcorr(VV(:,1),v(:,kk)/norm(v(:,kk)))),max(xcorr(VV(:,2),v(:,kk)/norm(v(:,kk))))),max(xcorr(VV(:,3),v(:,kk)/norm(v(:,kk)))));
end
corr_atoms_average=sum(corr_atoms,1)/K;
SNR_atoms_average=10*log10(1/(2-2*corr_atoms_average));

flat = @(x)x(:);
Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
    flat(reshape(1:n0*n1,n0,n1) ), ...
    ones(n0*n1,1) );
Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
    flat(reshape(1:n0*n1,n0,n1)' ), ...
    ones(n0*n1,1) );
Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];




maxit = 1e4; tol = 1e-9;
otransp = @(C,p0,p1)reshape( perform_linprog( ...
    Sigma(length(p0),length(p1)), ...
    [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );

[a1,b1] = meshgrid(nn,nn);
Cost_opti_transp = abs(a1-b1).^2;

% opti transport U global

p0 = sum(U_est,2)/norm(sum(U_est,2),1);
p1 = sum(U,2)/norm(sum(U,2),1);

gamma = otransp(Cost_opti_transp,p0,p1);

fprintf('Number of non-zero: %d (N+N-1=%d)\n', full(sum(gamma(:)~=0)), N+N-1);
fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));

[I,J,gammaij] = find(gamma);
Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;

optimal_transport_U = Wasserstein_Distance_2;
optimal_transport_U_average = optimal_transport_U_average + optimal_transport_U/Nubset;
% opti transport Uk on each motif
index = zeros(k,1);
for kk=1:k
    WD2old = inf;
    for kkk=1:k
        p0 = U_est(:,kkk)/norm(U_est(:,kkk),1);
        p1 = U(:,kk)/norm(U(:,kk),1);
        gamma = otransp(Cost_opti_transp,p0,p1);
        [I,J,gammaij] = find(gamma);
        Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;
        if Wasserstein_Distance_2<WD2old
            WD2old = Wasserstein_Distance_2;
            index(kk) = kkk;
        end
    end
    optimal_transport_Uk(kk) = WD2old;
    optimal_transport_Uk_average(kk,1) = optimal_transport_Uk_average(kk,1) + optimal_transport_Uk(kk)/Nubset;
end

end
optimal_transport_Uk_average_average = sum(optimal_transport_Uk_average)/3;
SNR_motifs_average_average = sum(SNR_motifs_average)/3;
