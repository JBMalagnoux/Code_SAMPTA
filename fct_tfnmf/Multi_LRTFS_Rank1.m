function [alpha,W,H,C] = Multi_LRTFS_Rank1(y,alpha_init,W_init,H_init,C_init,lambda,op,options)
%  LRSBL_type1 type 1 estimation for low rank time-frequency synthesis learning with one layer
%
%
% Inputs:
%	y: inputs signal
%	alpha_init: initialization of synthesis coeff.
%	W_init: initialization of W
%	H_init: initialization of H
%   lambda : prior's ponderation in the cost function
% authors: Cedric Fevotte and Matthieu Kowalski

myEps = getoptions(options, 'my_eps', 1e-8);
eps_stop = getoptions(options, 'eps_stop', 5e-2);
max_it = getoptions(options, 'max_it', 1000);
nbit_ista = getoptions(options, 'nbit_ista', 5);



alpha = alpha_init;
W = W_init;
H = H_init;
C = C_init;
[F,k] = size(W);
[T,~] = size(H);
[Q,~] = size(C);
WHC = zeros(F,T,Q);




for it_warm=1:length(lambda)
    fprintf("warmrestart it num : %d \n",it_warm)
    W = W_init;
    H = H_init;
    C = C_init;
    WHC = 0*WHC;
    for kk=1:k
        WHC = WHC + OuterProduct3vectors(W(:,kk),H(:,kk),C(:,kk))+myEps;
    end
    for it = 1:max_it
        Alphaold=alpha;


        %ISNMF

        WHCOLD = WHC ;
        [W, H, C] = IS_NTF(abs(alpha).^2,W_init,H_init,C_init,options);
        WHC = 0*WHC;
        for kk=1:k
            WHC = WHC + OuterProduct3vectors(W(:,kk),H(:,kk),C(:,kk))+myEps;
        end
        relative_err_WHC = (norm(WHC-WHCOLD,'fro')/(norm(WHC,'fro')+myEps))^2;
        if relative_err_WHC < eps_stop
            check_relative_err_WHC = 1;
        else
            check_relative_err_WHC = 0;
        end



        Lipschitz = 1;
        z = alpha;
        %FISTA
        for it_ista=0:nbit_ista
            alphaold = alpha;
            alpha = z +   op.analysis(y - op.synthesis(z))/Lipschitz;
            alpha = (WHC)./(WHC+lambda(it_warm)/Lipschitz) .* alpha;
            z = alpha + (it_ista)/(it_ista+5) * (alpha-alphaold);
            grad = -op.analysis(y - op.synthesis(alpha)) + lambda(it_warm) * alpha./(WHC+myEps);
            norm_grad =  norm(grad,'fro').^2/(F*T*Q);
            %func = 0.5*norm(y-op.synthesis(alpha)).^2 + lambda(it_warm)/2*norm(alpha(:)./(sqrt(V(:))+myEps))^2;
            err_rel = (norm(alpha-alphaold,'fro')/norm(alphaold,'fro'))^2;
            %fprintf('     in fista, it = %d -- func = %f -- ||Grad|| = %f -- err_rel = %f\n',it_ista,func,norm_grad,err_rel);
            if norm_grad < eps_stop || err_rel < eps_stop
                break;
            end
        end


        relative_err_alpha = (norm(alpha-Alphaold,'fro')/(norm(alpha,'fro')+myEps))^2;

        %fprintf('\n in AM, it = %d -- err_rel_alpha = %f -- err_rel_V = %f\n\n',it, relative_err_alpha, relative_err_V);

        if (relative_err_alpha < eps_stop && check_relative_err_WHC)
            break;
        end
    end

end

end



