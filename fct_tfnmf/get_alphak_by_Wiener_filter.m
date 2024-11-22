function [alphak,atom]= get_alphak_by_Wiener_filter(W, H, Gamma, k, options, op)
% Input :
%       W is the dictionnary matrix
%       H is the activation matrix
%       Gamma contains the analysis coefficients of the signal
%       k number of patterns
% Output :
%       alphak contains the synthesis coefficients of the k-pattern
%
% This function extracts the synthesis coefficient of each reconstructed
% pattern from the synthesis coefficients of the reconstructed signal. It
% also orde;ars the different patterns by decreasing energy

myEps = getoptions(options, 'my_eps', 1e-8);
T = options.T;
M = options.M;
a = options.a;
nW = size(W,1);
nH = size(H,2);
P = size(Gamma,3);

alphak = zeros(nW,nH,P,k);
alpha2k = zeros(nW,nH,k);
xatom = zeros(T,k);
atom = zeros(M,k);
HH =H;
I = zeros(k,1);
Gamma2 = sum(Gamma,3);

for kk=1:k
    [~,I(kk,1)] = max(HH(kk,:));
    if I(kk,1)==1
        HH(kk,2:end) = 0;
    elseif I(kk,1)==nH
        HH(kk,1:nH-1) = 0;
    else
        HH(kk,[1:I(kk,1)-1,I(kk,1)+1:end]) = 0;
    end
end
VV = W*HH;
VV(VV==0) = 1;
V = W*H;
V(V==0) = 1;

% Wiener filterering
for kk=1:k
    for pp=1:P
        alphak(:,:,pp,kk) = ((W(:,kk)*H(kk,:)./(V))) .*Gamma(:,:,pp);
    end
end


% temporal atom reconstruction
for kk=1:k
    alpha2k(:,:,kk) = ((W(:,kk)*HH(kk,:)./(VV))) .*Gamma2;
    xatom(:,kk) = op.synthesis(alpha2k(:,:,kk));
    atom(:,kk) = xatom(a*(min(max(I(kk)-2,1),nH-4)):a*(min(max(I(kk)-2,1),nH-4))+M-1,kk);
end



end



