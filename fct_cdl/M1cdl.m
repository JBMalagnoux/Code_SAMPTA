 function [v,u]= M1cdl(y,z,u_init,v_init,pos_u,options)
%Input :
%       y is the original signal
%       z is a sparse mapping
%       v_init and u_init are respectively the initialization of the temporal and spacial dictionnary
% Output :
%       xk is the reconstructed signals of the k patterns
%       v temporal dictionnary
%       u spacial dictionnary
%
% This function is agradient descent on d to solve the problem
% argmin_uk,vk 0.5||y-sum_k (vk uk^T_k)*z_k||_2^2 

myEps = getoptions(options, 'my_eps', 1e-8);
max_it = getoptions(options, 'max_it_cdl', 5);




T = length(y);
Fy = fftreal(y,T,1);
Fz = fftreal(z,T,1);
u = u_init;
v = v_init;
[M,k] = size(v);

L = norm(Fz(:),Inf)^2;
vv = v;
for it=1:max_it
    vold = v;
    dd = pagemtimes(permute(vv,[1 3 2]),reshape(u,[1,length(u),k]));
    Fdd = fftreal(dd,T,1);
    grad_dd = -permute(conj(Fz),[1 3 2]).*(Fy-permute(sum(permute(Fdd,[1 3 2]).*Fz,2),[1 3 2]));
    A = ifftreal(grad_dd,T,1);
    v = vv - 1/(L+eps) * permute(pagemtimes(A(1:M,:,:),permute(u,[1 3 2])),[1 3 2]);
    normv = sqrt(sum(abs(v).^2,1));
    normv(normv < 1) = 1;
    v = v./normv;
    vv = v + it/(it+5) * (v - vold);
end
uu=u;
for it=1:max_it
    uold = u;
    dd = pagemtimes(permute(v,[1 3 2]),reshape(uu,[1,length(uu),k]));
    Fdd = fftreal(dd,T,1);
    grad_dd = -permute(conj(Fz),[1 3 2]).*(Fy-permute(sum(permute(Fdd,[1 3 2]).*Fz,2),[1 3 2]));
    A = ifftreal(grad_dd,T,1);
    u = uu - 1/(L+eps) * permute(pagemtimes(permute(v,[3 1 2]),A(1:M,:,:)),[2 3 1]);
    normu = sqrt(sum(abs(u).^2,1));
    normu(normu < 1) = 1;
    u = u./normu;
    uu = u + it/(it+5) * (u - uold);
end
if pos_u
    u = max(u,0);
    normu = sqrt(sum(abs(u).^2,1));
    normu(normu < 1) = 1;
    u = u./normu;
end


end
