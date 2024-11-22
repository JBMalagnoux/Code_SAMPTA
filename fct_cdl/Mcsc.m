function [z]= Mcsc(y,z_init,u,v,lambda,pos_z,options)
%Input :
%       y is the original signal
%       z_init is the initialization of the sparse mapping
%       v and u are respectively the temporal and spacial
%       dictionnary such that D_k = v_k * u_k^T
%
%       lambda is the hyper-parameter (l1-norm)
%       pos_z equal to 1 for positiv constraint
% Output :
%       z sparse mapping
%
% This function is a proximal gradient descent to solve the problem
% argmin_z 0.5||y-sum_k d_k*z_k||_2^2 + lambda sum_k |z_k|_1

myEps = getoptions(options, 'my_eps', 1e-8);
max_it = getoptions(options, 'max_it_csc',20);

T = length(y);
Fy = fftreal(y,T,1);
d = pagemtimes(permute(v,[1 3 2]),permute(u,[3 1 2]));
Fd = fftreal(d,T,1);
z = z_init;
[ch,~] = size(u);
L = norm(Fd(:),Inf)^2*ch;


zz = z;
for it=0:max_it
    zold = z;
    Fzz = fftreal(zz,T,1);
    grad_zz = sum(permute(-conj(Fd).*(Fy-permute(sum(permute(Fd,[1 3 2]).*Fzz,2),[1 3 2])),[1 3 2]),3);
    z = zz - 1/(L+myEps) * ifftreal(grad_zz,T,1) ;
    z = z.*max(0,1-lambda./(L*abs(z)+myEps));
    zz = z + it/(it+5) * (z - zold);
end
if pos_z
    z = max(z,0);
end

end
