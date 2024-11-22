 function [W, H, C]= IS_NTF(V,W_init,H_init,C_init,options)
% Input :
%       V is a tensor that contains all the Q spectrogram of your Q variate
%       data
%       W_init is the initialization of the frequency dictionnary matrix
%       H_init is the initialization of the temporal activation matrix
%       C_init is the initialization of the spatial dictionnary/activation matrix
% Output :
%       W is the frequency dictionnary matrix after running the algorithm
%       H is the temporal activation matrix after running the algorithm
%       C is the spatial dictionnary/activation matrix after running the algorithm
%
% This function approximates the matrix V by doing a NTF with Itakura-Saito
% divergence 


myEps = getoptions(options, 'my_eps', 1e-8);
eps_stop = getoptions(options, 'eps_stop', 1e-8);
max_it = getoptions(options, 'max_it', 1000);
%compute_nrj = getoptions(options, 'compute_nrj', 0);

W = W_init;
H = H_init;
C = C_init;
[F,k] = size(W);
[T,~] = size(H);
[Q,~] = size(C);
DenW = zeros(k,1);
NumW = zeros(F,k);
DenH = zeros(k,1);
NumH = zeros(T,k);
DenC = zeros(k,1);
NumC = zeros(Q,k);
WHC = zeros(F,T,Q);





for it=2:max_it
    % Mise à jour de W et H
    Wold = W;
    Hold = H;
    Cold = C;


    
    
    WHC = 0*WHC;
    for kk=1:k
        WHC = WHC + OuterProduct3vectors(W(:,kk),H(:,kk),C(:,kk))+myEps;
    end

    %%% Update C %%%

    R = V./(WHC + myEps);
    for kk=1:k
        NumC(:,kk) =  sum(sum(W(:,kk)*H(:,kk)'.*R));
    end
    for kk=1:k
        DenC(kk) =  sum(sum(W(:,kk)*H(:,kk)'));
    end
    C = C.*(   (NumC)  ./   (DenC'+ myEps )    );

    WHC = 0*WHC;
    for kk=1:k
        WHC = WHC + OuterProduct3vectors(W(:,kk),H(:,kk),C(:,kk));
    end


    %%% Update W %%%

    R = V./(WHC + myEps);
    R = permute(R,[2 3 1]);
    for kk=1:k
        NumW(:,kk) =  sum(sum(H(:,kk)*C(:,kk)'.*R));
    end
    for kk=1:k
        DenW(kk) =  sum(sum(H(:,kk)*C(:,kk)'));
    end
    W = W.*(   (NumW)  ./   (DenW'+ myEps )    );


    WHC = 0*WHC;
    for kk=1:k
        WHC = WHC + OuterProduct3vectors(W(:,kk),H(:,kk),C(:,kk))+myEps;
    end
    

    %%% Update H %%%

    R = V./(WHC + myEps);
    R = permute(R,[1 3 2]);
    for kk=1:k
        NumH(:,kk) =  sum(sum(W(:,kk)*C(:,kk)'.*R));
    end
    for kk=1:k
        DenH(kk) =  sum(sum(W(:,kk)*C(:,kk)'));
    end
    H = H.*(   (NumH)  ./   (DenH'+ myEps )    );

    


   



    if ( 0.5 * norm(W - Wold,'fro')/norm(W,'fro') + 0.5 * norm(H - Hold,'fro')/norm(H,'fro') +  0.5 * norm(C - Cold,'fro')/norm(C,'fro') < eps_stop)

        break;
    end
end


