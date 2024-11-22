function [z]= csc(y,z_init,d,lambda,pos_z,options)
%Input :
%       y is the original signal
%       z_init is the initialization of the sparse mapping
%       d is the dictionnary
%       lambda is the hyper-parameter (l1-norm)
%       pos_z equal to 1 for positiv constraint
% Output :
%       z sparse mapping
%       nrj is a structure wich contain the nrj threw the iteration and the
%       CPUtime associated
%
% This function is a proximal gradient descent to solve the problem
% argmin_z 0.5||y-sum_k d_k*z_k||_2^2 + lambda sum_k |z_k|_1

myEps = getoptions(options, 'my_eps', 1e-8);
max_it = getoptions(options, 'max_it_csc', 10);

T = length(y);
Fy = fftreal(y,T,1);
Fd = fftreal(d,T,1);
z = z_init;

L = norm(Fd(:),Inf)^2;
zz = z;
for it=0:max_it
    zold = z;
    Fzz = fftreal(zz,T,1);
    grad_zz = -conj(Fd).*(Fy-sum(Fd.*Fzz,2));
    z = zz - 1/(L+myEps) * ifftreal(grad_zz,T,1) ;
    z = z.*max(0,1-lambda./(L*abs(z)+myEps));
    zz = z + it/(it+5) * (z - zold);
end
if pos_z
    z = max(z,0);
end

end
