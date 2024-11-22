function [d]= cdl(y,z,d_init,options)
%Input :
%       y is the original signal
%       z is a sparse mapping
%       d_init is the initialization of the dictionnary
%       M size of the dictionnary
% Output :
%       xk is the reconstructed signals of the k patterns
%       z sparse mapping
%       d dictionnary
%       nrj is a structure wich contain the nrj threw the iteration and the
%       CPUtime associated
%
% This function is agradient descent on d to solve the problem
% argmin_d 0.5||y-sum_k d_k*z_k||_2^2 

myEps = getoptions(options, 'my_eps', 1e-8);
max_it = getoptions(options, 'max_it_cdl', 5);




T = length(y);
Fy = fftreal(y,T,1);
Fz = fftreal(z,T,1);
d = d_init;
M = length(d(:,1));

L = norm(Fz(:),Inf)^2;
dd = d;
for it=0:max_it
    dold = d;
    Fdd = fftreal(dd,T,1);
    grad_dd = -conj(Fz).*(Fy-sum(Fdd.*Fz,2));
    A = ifftreal(grad_dd,T,1);
    d = dd - 1/(L+myEps) * A(1:M,:);
    normD = sqrt(sum(abs(d).^2,1));
    normD(normD < 1) = 1;
    d = d./normD;
    dd = d + it/(it+5) * (d - dold);
end


end
