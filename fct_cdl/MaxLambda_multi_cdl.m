function MaxLambda= MaxLambda_multi_cdl(y,z,u,v)
% Compute the maximum value of lambda  wich does not  treshold the entire
% signal

T = length(y);
Fy = fftreal(y,T,1);
d = pagemtimes(permute(v,[1 3 2]),permute(u,[3 1 2]));
Fd = fftreal(d,T,1);
[ch,~] = size(u);
L = norm(Fd(:),Inf)^2*ch;


zz = z;
Fzz = fftreal(zz,T,1);
grad_zz = sum(permute(-conj(Fd).*(Fy-permute(sum(permute(Fd,[1 3 2]).*Fzz,2),[1 3 2])),[1 3 2]),3);
z = zz - 1/(L+eps) * ifftreal(grad_zz,T,1) ;

MaxLambda = L*max(z(:));
end


