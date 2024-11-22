%%%% data creation for toy inverse problom %%%%
%%%% The problem is composed of N sources obseved on P sensors %%%%
%%%% We assume there are K spatio-temporal atoms to learn %%%%
clear 

K = 3; % number of spatial-temporal atoms
P = 10; % number of sensors
N = 20; % number of sources
T = 10000; % temporal lengths of acquisitions
fs = 100; % frequency acquisition
Latom = 128; % temporal length of atoms
Sp = 0.2; % spatial sparsity
Tp = 0.005; % temporal sparsity
Nubset = 1; % number of z set in order to average the results

%%%% temporal atoms creation %%%

[psi, ~] = wavefun('morl');
v1 = psi(65:65+127)'; % atom 1
v2 = triang(Latom); % atom 2
v3 = ones(Latom,1); 
v3(65:128)= -1*v3(65:128); % atom 3
load('d4.mat'); % atome 4
v4 = d4/norm(d4,'inf');
V = [v1 v3 v4];

rng(421)

%%%% temporal mapping  (bernouilli-Gaussien) %%%%

Length_z =  T - Latom + 1;
z = zeros(Length_z, K, Nubset);
support_z = rand(Length_z, K, Nubset) < Tp;
activations_z = randn(Length_z, K, Nubset);
z(support_z==1) = abs(activations_z(support_z==1));
z(1:2*Latom,:,:)= zeros(2*Latom,K,Nubset);
z(Length_z-2*Latom+1:Length_z,:,:)= zeros(2*Latom,K,Nubset);

%%%% spatialisation  (bernouilli-Gaussien) %%%%

U = zeros(N,K);
support_U = rand(N, K) < Sp;
activations_U = randn(N, K);
U(support_U==1) = abs(activations_U(support_U==1));



%%%% Creation of the spatio-temporal dictionnary in the sources domain %%%%

D = zeros(N,Latom,K);
for k=1:K
    D(:,:,k)=U(:,k)*V(:,k)';
end




%%%% creation of the sources'signals %%%%


X = zeros(T,N,K,Nubset);
for nub=1:Nubset
    for k = 1:K
        for s=1:N
            X(:,s,k,nub) = conv(z(:,k,nub),D(s,:,k));
        end
    end
end
Xsources = permute(sum(X,3),[2 1 4 3]);

%%%% Matrix Sources spaces to sensors spaces %%%%

%G = abs(randn(N,P));
nn = 1:N;
[a,b] = meshgrid(nn,nn);
w=0.99;
cov = w.^(abs(a-b));
G = mvnrnd(0*nn,cov,P);
Xsensors = zeros(P,T,Nubset);
for nub=1:Nubset
    Xsensors(:,:,nub) = G*Xsources(:,:,nub);
end
XX = zeros(P,T,k,Nubset);
for nub=1:Nubset
    for kk=1:K
        XX(:,:,kk,nub) = G*X(:,:,kk,nub)';
    end
end
X1 = permute(XX(:,:,1,:),[2 1 4 3]);
X2 = permute(XX(:,:,2,:),[2 1 4 3]);
X3 = permute(XX(:,:,3,:),[2 1 4 3]);
%X4 = XX(:,:,4)';

plot(permute(sum(Xsensors,1),[2 3 1]))

