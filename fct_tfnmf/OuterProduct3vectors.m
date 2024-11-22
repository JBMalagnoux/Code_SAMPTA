function [P]= OuterProduct3vectors(A,B,C)

%%%% This function give the outerproduct of the 3 vectors A, B and C.
%%%% If A size is I, B size is T, C size is Q you will get P size IxTxQ


Inter = A*B';
C = reshape(C,[1,1,length(C)]);
P = Inter .* C;

end




