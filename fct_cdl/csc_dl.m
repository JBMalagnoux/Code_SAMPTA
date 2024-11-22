function [xk, d, z, nrj]= csc_dl(y,z_init,d_init,lambda,pos_z,k,options,op)
%Input :
%       y is the original signal
%       z_init is the initialization of the sparse mapping
%       d_init is the initialization of the dictionnary
%       lambda is the hyper-parameter (l1-norm)
%       pos_z equal to 1 for positiv constraint
%       k the number of patterns researched by the algorithm
% Output :
%       xk is the reconstructed signals of the k patterns
%       d dictionnary
%       z sparse mapping
%       nrj is a structure wich contain the nrj threw the iteration and the
%       CPUtime associated
%
% This function is an alternative minimization between d and z to solve
% argmin_(z,d) 0.5||y-sum_k d_k*z_k||_2^2 + lambda sum_k |z_k|_1 using the
% function csc and cdl. Then it orders the patterns by decreasing energy.

myEps = getoptions(options, 'my_eps', 1e-8);
eps_stop = getoptions(options, 'eps_stop', 1e-8);
max_it = getoptions(options, 'max_it', 1000);
compute_nrj = getoptions(options, 'compute_nrj', 0);
options.myEps = myEps;

d = d_init;
z = z_init;
T = length(y);
Fd = fftreal(d,T,1);
Fz = fftreal(z,T,1);
F= Fd.*Fz;

if compute_nrj
    %nrj.cscdl_cost = zeros((max_it-1)*length(lambda)+1,1);
    nrj.cpu_time = zeros((max_it-1)*length(lambda)+1,1);
    nrj.snr_source1 = zeros((max_it-1)*length(lambda)+1,1);
    nrj.snr_source2 = zeros((max_it-1)*length(lambda)+1,1);
    %nrj.snr_source3 = zeros((max_it-1)*length(lambda)+1,1);
    %nrj.snr_source4 = zeros((max_it-1)*length(lambda)+1,1);
    %nrj.cscdl_cost(1) = Cost_function_CSC_classic(y,T,Fd,z,lambda(length(lambda)));
    nrj.snr_source1(1) = snr(ifftreal(F(:,1),T,1),options.source1);
    nrj.snr_source2(1) = snr(ifftreal(F(:,2),T,1),options.source2);
    %     nrj.snr_source3(1) = snr(ifftreal(F(:,4),T,1),options.sourcefa);
    %     nrj.snr_source4(1) = snr(ifftreal(F(:,5),T,1)+ifftreal(F(:,7),T,1),options.sourcereb);
    nrj.cpu_time(1) = 0;
    %nrj.crit_eval_source1 = zeros((max_it-1)*length(lambda)+1,1);
    %nrj.crit_eval_source2 = zeros((max_it-1)*length(lambda)+1,1);


    %F = Fd.*Fz;
    %xinter = ifftreal(F,T,1);
    %alphainter = abs(op.analysis(xinter)).^2;
    %[~,alphamatch] = alphamatch_sources(alphainter,options.anasources);
    %nrj.crit_eval_source1(1) = alpha_sources_dis(alphamatch(:,:,1),options.anasources(:,:,1));
    %nrj.crit_eval_source2(1) = alpha_sources_dis(alphamatch(:,:,2),options.anasources(:,:,2));
    l = 1;
    tic;
    tcost = 0;
else
    nrj = -1;
end

for it_warm=1:length(lambda)
    for it=2:max_it
        zold = z;
        dold = d;
        [z] = csc(y,z,d,lambda(it_warm),pos_z,options);
        [d] = cdl(y,z,d,options);
        Fd = fftreal(d,T,1);
        Fz = fftreal(z,T,1);
        F= Fd.*Fz;
        if compute_nrj
            tinter = toc;
            l = l+1;
            nrj.cscdl_cost((it_warm-1)*(max_it-1)+it) = Cost_function_CSC_classic(y,T,Fd,z,lambda(length(lambda)));
            x1 = ifftreal(F(:,1),T,1);
            x2 = ifftreal(F(:,2),T,1);
            nrj.snr_source1(l) = max(snr(x1,options.source1),snr(x1,options.source2));
            nrj.snr_source2(l) = max(snr(x2,options.source1),snr(x2,options.source2));
            
            if max(snr(x1,options.source1),snr(x1,options.source2))+max(snr(x2,options.source1),snr(x2,options.source2))>=max(nrj.snr_source1)+max(nrj.snr_source2)
                dsaved = d;
                zsaved = z;
                itsaved = it_warm;
                lambdasaved = lambda(it_warm);
                SNRmax = snr(x1,options.source1)+snr(x2,options.source2);
            end
            %             nrj.snr_source3((it_warm-1)*(max_it-1)+it) = snr(ifftreal(F(:,4),T,1),options.sourcefa);
            %             nrj.snr_source4((it_warm-1)*(max_it-1)+it) = snr(ifftreal(F(:,5),T,1)+ifftreal(F(:,7),T,1),options.sourcelab);
            %             F= Fd.*Fz;
            %             xinter = ifftreal(F,T,1);
            %             alphainter = abs(op.analysis(xinter)).^2;
            %             [~,alphamatch] = alphamatch_sources(alphainter,options.anasources);
            %             nrj.crit_eval_source1((it_warm-1)*(max_it-1)+it) = alpha_sources_dis(alphamatch(:,:,1),options.anasources(:,:,1));
            %             nrj.crit_eval_source2((it_warm-1)*(max_it-1)+it) = alpha_sources_dis(alphamatch(:,:,2),options.anasources(:,:,2));
            tcost = tcost + toc - tinter;
            nrj.cpu_time((it_warm-1)*(max_it-1)+it) = toc-tcost;

        end
        if ( 0.5 * norm(d - dold,'fro')/norm(d,'fro') + 0.5 * norm(z - zold,'fro')/norm(z,'fro') < eps_stop)
            if compute_nrj
                nrj.cscdl_cost((it_warm-1)*(max_it-1)+it+1:(max_it-1)*it_warm+1) = nrj.cscdl_cost((it_warm-1)*(max_it-1)+it)*ones(max_it-it,1);
                nrj.cpu_time((it_warm-1)*(max_it-1)+it+1:(max_it-1)*it_warm+1) = nrj.cpu_time((it_warm-1)*(max_it-1)+it)*ones(max_it-it,1);
                %                 nrj.crit_eval_source1((it_warm-1)*(max_it-1)+it+1:(max_it-1)*it_warm+1) = alpha_sources_dis(alphamatch(:,:,1),options.anasources(:,:,1))*ones(max_it-it,1);
                %                 nrj.crit_eval_source2((it_warm-1)*(max_it-1)+it+1:(max_it-1)*it_warm+1) = alpha_sources_dis(alphamatch(:,:,2),options.anasources(:,:,2))*ones(max_it-it,1);
                nrj.snr_source1((it_warm-1)*(max_it-1)+it+1:(max_it-1)*it_warm+1) = nrj.snr_source1((it_warm-1)*(max_it-1)+it)*ones(max_it-it,1);
                nrj.snr_source2((it_warm-1)*(max_it-1)+it+1:(max_it-1)*it_warm+1) = nrj.snr_source2((it_warm-1)*(max_it-1)+it)*ones(max_it-it,1);
                %                 nrj.snr_source3((it_warm-1)*(max_it-1)+it+1:(max_it-1)*it_warm+1) = nrj.snr_source3((it_warm-1)*(max_it-1)+it)*ones(max_it-it,1);
                %                 nrj.snr_source4((it_warm-1)*(max_it-1)+it+1:(max_it-1)*it_warm+1) = nrj.snr_source4((it_warm-1)*(max_it-1)+it)*ones(max_it-it,1);
            end
            break;
        end
    end
end
% d = dsaved;
% z = zsaved;
% fprintf('\nBetter lambda : %d (%d) Better SNR : %d \n',lambdasaved,itsaved,SNRmax);

Fz = fftreal(z,T,1);
Fd = fftreal(d,T,1);
% Order by decreasing energy
F= Fd.*Fz;


Tmp = zeros(T/2+1,k);
nrjF=zeros(k,1);

for kk=1:k
    nrjF(kk,1)=norm(F(:,kk),'fro');
end
for kk=1:k
    [~,INDEX]=max(nrjF);
    nrjF(INDEX,1)=0;
    Tmp(:,kk)=F(:,INDEX);
end
xk = ifftreal(Tmp,T,1);

end