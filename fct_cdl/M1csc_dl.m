function [xk, v, u, z]= M1csc_dl(y,z_init,v_init, u_init,lambda,pos_z,pos_u,k,options)
%Input :
%       y is the original signal
%       z_init is the initialization of the sparse mapping
%       v_init and u_init are respectively the initialization of the temporal and spacial dictionnary
%       lambda is the hyper-parameter (l1-norm)
%       pos_z equal to 1 for positiv constraint
%       k the number of patterns researched by the algorithm
% Output :
%       xk is the reconstructed signals of the k patterns
%       v temporal dictionnary
%       u spacial dictionnary
%       z sparse mapping
%
%
% This function is an alternative minimization between d and z to solve
% argmin_(z,u,v) 0.5||y-sum_k (vk uk^T_k)*z_k||_2^2 + lambda sum_k |z_k|_1 using the
% function csc and cdl. Then it orders the patterns by decreasing energy.

myEps = getoptions(options, 'my_eps', 1e-8);
eps_stop = getoptions(options, 'eps_stop', 1e-4);
max_it = getoptions(options, 'max_it', 1000);
options.myEps = myEps;
denoised_signal = options.Xsensors';

u = u_init;
v = v_init;
z = z_init;
T = length(y);
P = length(u) ;
Fz = fftreal(z,T,1);
d = pagemtimes(permute(v,[1 3 2]),reshape(u,[1,length(u),k]));
Fd = fftreal(d,T,1);
xtmp = sum(sum(ifftreal(permute(permute(Fd,[1 3 2]).*Fz,[1 3 2]),T,1),3),2);
snr_oracle = snr(sum(denoised_signal,2),xtmp);
u_oracle = u;
v_oracle = v;
z_oracle = z;
it_oracle = 0;
lambda_oracle = lambda(1);




for it_warm=1:length(lambda)
    fprintf("warmrestart it num : %d \n",it_warm)
    u = u_oracle;
    v = v_oracle;
    z = z_oracle;
    for it=2:max_it
        zold = z;
        vold = v;
        uold = u;
        [z] = Mcsc(y,z,u,v,lambda(it_warm),pos_z,options);
        [v,u] = M1cdl(y,z,u,v,pos_u,options);


        if ( 0.5 * norm(v - vold,'fro')/norm(v,'fro') + 0.5 * norm(u - uold,'fro')/norm(u,'fro') + 0.5 * norm(z - zold,'fro')/norm(z,'fro') < eps_stop)
            break;
        end
    end
    Fz = fftreal(z,T,1);
    d = pagemtimes(permute(v,[1 3 2]),reshape(u,[1,length(u),k]));
    Fd = fftreal(d,T,1);
    xtmp = sum(sum(ifftreal(permute(permute(Fd,[1 3 2]).*Fz,[1 3 2]),T,1),3),2);
    if snr(sum(denoised_signal,2),xtmp)>snr_oracle
        fprintf("upgrade\n")
        snr_oracle = snr(sum(denoised_signal,2),xtmp);
        it_oracle = it_warm;
        lambda_oracle = lambda(it_warm);
        u_oracle = u;
        v_oracle = v;
        z_oracle = z;
    else
        % fprintf("oracle found\n")
        % break
    end
end
if options.oracle
    u = u_oracle;
    v = v_oracle;
    z = z_oracle;
    fprintf("it oracle : %d \nlambda oracle : %f \n",it_oracle,lambda_oracle)
end
Fz = fftreal(z,T,1);
d = pagemtimes(permute(v,[1 3 2]),reshape(u,[1,length(u),k]));
Fd = fftreal(d,T,1);

% Order by decreasing energy
F= permute(permute(Fd,[1 3 2]).*Fz,[1 3 2]);


Tmp = zeros(T/2+1,P,k);
nrjF=zeros(k,1);

for kk=1:k
    nrjF(kk,1)=norm(F(:,:,kk),'fro');
end
for kk=1:k
    [~,INDEX]=max(nrjF);
    nrjF(INDEX,1)=0;
    Tmp(:,:,kk)=F(:,:,INDEX);
end
xk = ifftreal(Tmp,T,1);

end