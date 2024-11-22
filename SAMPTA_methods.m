function [optimal_transport_Uk_average_average, SNR_motifs_average_average, SNR_atoms_average]= SAMPTA_methods(corr,iSNR,method)

if method == 1
    if corr == 0
        %% CODE CSC DL ini NTF - Inverse problem then sources space decomposition


        rng(42);
        load('dataw20.mat')
        SNR_motifs_average = zeros(k,1);
        optimal_transport_Uk_average = zeros(k,1);
        optimal_transport_U_average = 0;
        DIS_atoms_spectrum_average = 0;
        cpu_time_average=0;
        lead_field = G;
        Nubset=1;
        for nub=1:Nubset
            % Sensors Gaussian noising
            XXsensors = Xsensors(:,:,nub);
            snrlvl = iSNR;
            sigma2 = var(XXsensors(:))*10^(-snrlvl/10); % add noise
            XXsensors = XXsensors + sqrt(sigma2)*randn(size(XXsensors));

            % Gabor parameters
            M = 256;
            a = M/2;
            g = gabwin({'tight', 'hann'}, a, M);



            % TF operator
            op.analysis = @(x) dgtreal(x,g,a,M);
            op.synthesis = @(x) idgtreal(x,g,a,M,T);

            tic

            %%% Inverse problem



            Nbit = 300;
            L = norm(lead_field)^2;
            Nblambda2 = 15; %% lambda2 is for the l21 norm
            A=lead_field'*XXsensors;
            EnerA = zeros(20,1);
            for h=1:N
                EnerA(h,1) = norm(A(h,:));
            end
            lambda2Max = max(EnerA) ;
            lambda2Min = 1e-4*lambda2Max;
            lambda2Vect = logspace(log10(lambda2Max),log10(lambda2Min),Nblambda2);
            lambda2number = 0;
            mu = 0.001*lambda2Max; %% mu is for the l1 norm
            Xsources_est = lead_field'*XXsensors;

            % snr_oracle = snr(sum(Xsources,1)',sum(Xsources_est,1)');
            % for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
            %     lambda2number = lambda2number + 1;
            %     lambdaOnL = lambda/L;
            %     mu0nL = mu/L;
            %     z_est = Xsources_est;
            %     for it=0:Nbit-1
            %         Xsources_est_old = Xsources_est;
            %         Xsources_est = z_est + lead_field'*(XXsensors-lead_field*z_est)/L;
            %         Xsources_est = Xsources_est.*max(0,1-mu0nL./(abs(Xsources_est)+eps));
            %         Xsources_est = Xsources_est.*max(0,1-lambdaOnL./sqrt(sum(abs(Xsources_est).^2,2)+eps));
            %         z_est = Xsources_est + (it)/(it+5)*(Xsources_est-Xsources_est_old);
            %     end
            %     if snr(sum(Xsources,1)',sum(Xsources_est,1)')>snr_oracle
            %         snr_oracle = snr(sum(Xsources,1)',sum(Xsources_est,1)');
            %         it_oracle = it;
            %         lambda2_oracle = lambda2Vect(lambda2number);
            %         Xsources_est_oracle = Xsources_est;
            %     end
            % end
            % Xsources_est = Xsources_est_oracle';

            %%% inverse problem with TF dictionnary

            Gamma_est = op.analysis(Xsources_est');
            snr_oracle = snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2));
            for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
                lambda2number = lambda2number + 1;
                lambdaOnL = lambda/L;
                mu0nL = mu/L;
                z_est = Gamma_est;
                tau=1;
                for it=0:Nbit-1
                    Gamma_est_old = Gamma_est;
                    Gamma_est = z_est + op.analysis(transpose(lead_field'*(XXsensors-lead_field*op.synthesis(z_est)')/L));
                    Gamma_est = Gamma_est.*max(0,1-mu0nL./(abs(Gamma_est)+eps));
                    Gamma_est = Gamma_est.*max(0,1-lambdaOnL./sqrt(sum(abs(Gamma_est).^2,2)+eps));
                    tau0 = tau;
                    tau = (1+sqrt(1+4*tau^2))/2;
                    z_est = Gamma_est + (tau0-1)/(tau)*(Gamma_est-Gamma_est_old);
                    %z_est = Gamma_est + (it)/(it+5)*(Gamma_est-Gamma_est_old);
                end
                if snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2))>snr_oracle
                    snr_oracle = snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2));
                    it_oracle = it;
                    lambda2_oracle = lambda2Vect(lambda2number);
                    Gamma_est_oracle = Gamma_est;
                end
            end
            Xsources_est = op.synthesis(Gamma_est_oracle);


            G = dgtreal(Xsources_est, g, a, M);
            [nf,nt,nq] = size(G);

            Gamma = op.analysis(Xsources_est);
            G = abs(Gamma);

            k = 3;
            options.creation = 1;
            options.max_it = 1000;
            options.T = T;
            options.M = M;
            options.a = a;
            options.Xsensors = XXsensors;

            % Ini SVD
            [U_svd,D_svd,S_svd] = svd(sum(G,3),"econ");
            S_svd = sqrt(D_svd)*S_svd';
            U_svd = U_svd*sqrt(D_svd);
            W_init = abs(U_svd(:,1:k));
            H_init = abs(S_svd(1:k,:))';
            C_init = ones(nq,k);


            %%% IS-NTF %%%

            [W, H, C] = IS_NTF(G,W_init,H_init,C_init,options);

            [alphak,atom] = get_alphak_by_Wiener_filter(W, H', Gamma, k, options, op);
            Alphak = reshape(sum(alphak,3),[nf nt k]);
            Xk = zeros(T,k);
            for kk=1:k
                Xk(:,kk) = op.synthesis(Alphak(:,:,kk));
            end

            v_init = atom;
            normV = sqrt(sum(abs(v_init).^2,1));
            v_init = v_init./normV;
            % u_init=C;
            % normU = sqrt(sum(abs(u_init).^2,1));
            % u_init = u_init./normU;
            u_init =C_init;
            z_tmp = H;
            [~,nfr] = size(H');
            %z_init = ifftreal(fftreal(H),T);
            z_init = zeros(T,k);
            for l=2:nfr
                z_init(a*(l-2)+1,:) = z_tmp(l,:);
            end
            M = length(v_init(:,1));

            %maxLambda = 20;
            lambda = logspace(-1,-5,10);
            %lambda = 10;
            pos_z = 1;
            pos_u = 1;
            options.max_it = 1000;
            options.max_it_cdl = 5;
            options.max_it_csc = 15;
            options.oracle = 0;

            %%% CDl Multivariate rank-1

            [xk, v, u, z2] = M1csc_dl(Xsources_est,z_init,v_init, u_init,lambda,pos_z,pos_u,k,options);
            %u = max(u,0);
            cpu_time=toc;
            x = sum(xk,3);
            xtotk = reshape(sum(xk,2),[T k]);
            VV=zeros(M,K);
            VV(:,1)=[zeros(91,1);V(:,1)/norm(V(:,1));[zeros(37,1)]];
            VV(:,2)=[zeros(31,1);V(:,2)/norm(V(:,2));[zeros(97,1)]];
            VV(:,3)=[zeros(115,1);V(:,3)/norm(V(:,3));[zeros(13,1)]];
            V1=[zeros(91,1);V(:,1)/norm(V(:,1),'inf');[zeros(37,1)]];
            V2=[zeros(31,1);0.5*V(:,2)/norm(V(:,2),'inf');[zeros(97,1)]];
            V3=[zeros(115,1);V(:,3)/norm(V(:,3),'inf');[zeros(13,1)]];


            %%% Quant results

            SNR_motifs = zeros(k,1);

            optimal_transport_Uk = zeros(k,1);
            corr_atoms = zeros(K,1);

            SNR_motifs(1,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,2)),max(snr(sum(X(:,:,2,nub),2),Xk(:,2)),snr(sum(X(:,:,3,nub),2),Xk(:,2))));
            SNR_motifs_average(1,1) = SNR_motifs_average(1,1)+SNR_motifs(1,1)/Nubset;
            SNR_motifs(2,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,1)),max(snr(sum(X(:,:,2,nub),2),Xk(:,1)),snr(sum(X(:,:,3,nub),2),Xk(:,1))));
            SNR_motifs_average(2,1) = SNR_motifs_average(2,1)+SNR_motifs(2,1)/Nubset;
            SNR_motifs(3,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,3)),max(snr(sum(X(:,:,2,nub),2),Xk(:,3)),snr(sum(X(:,:,3,nub),2),Xk(:,3))));
            SNR_motifs_average(3,1) = SNR_motifs_average(3,1)+SNR_motifs(3,1)/Nubset;





            Fv = abs(fftreal(v));
            FV = abs(fftreal([V2 V1 V3]));
            Fv = Fv./max(Fv);
            FV = FV./max(FV);
            FV = FV(FV>0);
            Fv = Fv(FV>0);
            Tmp = (FV)./(Fv) - log((FV)./(Fv) ) -1;
            Tmp(isnan(Tmp)) = 0;

            for kk=1:K
                corr_atoms(kk)= max(max(max(xcorr(VV(:,1),v(:,kk)/norm(v(:,kk)))),max(xcorr(VV(:,2),v(:,kk)/norm(v(:,kk))))),max(xcorr(VV(:,3),v(:,kk)/norm(v(:,kk)))));
            end
            corr_atoms_average=sum(corr_atoms,1)/K;
            SNR_atoms_average=10*log10(1/(2-2*corr_atoms_average));


            flat = @(x)x(:);
            Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
                flat(reshape(1:n0*n1,n0,n1) ), ...
                ones(n0*n1,1) );
            Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
                flat(reshape(1:n0*n1,n0,n1)' ), ...
                ones(n0*n1,1) );
            Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];




            maxit = 1e4; tol = 1e-9;
            otransp = @(C,p0,p1)reshape( perform_linprog( ...
                Sigma(length(p0),length(p1)), ...
                [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );

            [a1,b1] = meshgrid(nn,nn);
            Cost_opti_transp = abs(a1-b1).^2;

            % opti transport U global

            p0 = sum(C,2)/norm(sum(C,2),1);
            p1 = sum(U,2)/norm(sum(U,2),1);

            gamma = otransp(Cost_opti_transp,p0,p1);

            fprintf('Number of non-zero: %d (N+N-1=%d)\n', full(sum(gamma(:)~=0)), N+N-1);
            fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));

            [I,J,gammaij] = find(gamma);
            Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;

            optimal_transport_U = Wasserstein_Distance_2;

            % opti transport Uk on each motif
            index = zeros(k,1);
            for kk=1:k
                WD2old = inf;
                for kkk=1:k
                    p0 = C(:,kkk)/norm(C(:,kkk),1);
                    p1 = U(:,kk)/norm(U(:,kk),1);
                    gamma = otransp(Cost_opti_transp,p0,p1);
                    [I,J,gammaij] = find(gamma);
                    Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;
                    if Wasserstein_Distance_2<WD2old
                        WD2old = Wasserstein_Distance_2;
                        index(kk) = kkk;
                    end
                end
                optimal_transport_Uk(kk) = WD2old;
                optimal_transport_Uk_average(kk,1) = optimal_transport_Uk_average(kk,1) + optimal_transport_Uk(kk)/Nubset;
            end


        end
        optimal_transport_Uk_average_average = sum(optimal_transport_Uk_average)/3;
        SNR_motifs_average_average = sum(SNR_motifs_average)/3;





    elseif corr == 0.2


        rng(42);
        load('dataw40.mat')
        SNR_motifs_average = zeros(k,1);
        optimal_transport_Uk_average = zeros(k,1);
        optimal_transport_U_average = 0;
        DIS_atoms_spectrum_average = 0;
        cpu_time_average=0;
        lead_field = G;
        Nubset=1;
        for nub=1:Nubset
            % Sensors Gaussian noising
            XXsensors = Xsensors(:,:,nub);
            snrlvl = iSNR;
            sigma2 = var(XXsensors(:))*10^(-snrlvl/10); % add noise
            XXsensors = XXsensors + sqrt(sigma2)*randn(size(XXsensors));

            % Gabor parameters
            M = 256;
            a = M/2;
            g = gabwin({'tight', 'hann'}, a, M);



            % TF operator
            op.analysis = @(x) dgtreal(x,g,a,M);
            op.synthesis = @(x) idgtreal(x,g,a,M,T);

            tic

            %%% Inverse problem



            Nbit = 300;
            L = norm(lead_field)^2;
            Nblambda2 = 15; %% lambda2 is for the l21 norm
            A=lead_field'*XXsensors;
            EnerA = zeros(20,1);
            for h=1:N
                EnerA(h,1) = norm(A(h,:));
            end
            lambda2Max = max(EnerA) ;
            lambda2Min = 1e-4*lambda2Max;
            lambda2Vect = logspace(log10(lambda2Max),log10(lambda2Min),Nblambda2);
            lambda2number = 0;
            mu = 0.001*lambda2Max; %% mu is for the l1 norm
            Xsources_est = lead_field'*XXsensors;

            % snr_oracle = snr(sum(Xsources,1)',sum(Xsources_est,1)');
            % for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
            %     lambda2number = lambda2number + 1;
            %     lambdaOnL = lambda/L;
            %     mu0nL = mu/L;
            %     z_est = Xsources_est;
            %     for it=0:Nbit-1
            %         Xsources_est_old = Xsources_est;
            %         Xsources_est = z_est + lead_field'*(XXsensors-lead_field*z_est)/L;
            %         Xsources_est = Xsources_est.*max(0,1-mu0nL./(abs(Xsources_est)+eps));
            %         Xsources_est = Xsources_est.*max(0,1-lambdaOnL./sqrt(sum(abs(Xsources_est).^2,2)+eps));
            %         z_est = Xsources_est + (it)/(it+5)*(Xsources_est-Xsources_est_old);
            %     end
            %     if snr(sum(Xsources,1)',sum(Xsources_est,1)')>snr_oracle
            %         snr_oracle = snr(sum(Xsources,1)',sum(Xsources_est,1)');
            %         it_oracle = it;
            %         lambda2_oracle = lambda2Vect(lambda2number);
            %         Xsources_est_oracle = Xsources_est;
            %     end
            % end
            % Xsources_est = Xsources_est_oracle';

            %%% inverse problem with TF dictionnary

            Gamma_est = op.analysis(Xsources_est');
            snr_oracle = snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2));
            for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
                lambda2number = lambda2number + 1;
                lambdaOnL = lambda/L;
                mu0nL = mu/L;
                z_est = Gamma_est;
                tau=1;
                for it=0:Nbit-1
                    Gamma_est_old = Gamma_est;
                    Gamma_est = z_est + op.analysis(transpose(lead_field'*(XXsensors-lead_field*op.synthesis(z_est)')/L));
                    Gamma_est = Gamma_est.*max(0,1-mu0nL./(abs(Gamma_est)+eps));
                    Gamma_est = Gamma_est.*max(0,1-lambdaOnL./sqrt(sum(abs(Gamma_est).^2,2)+eps));
                    tau0 = tau;
                    tau = (1+sqrt(1+4*tau^2))/2;
                    z_est = Gamma_est + (tau0-1)/(tau)*(Gamma_est-Gamma_est_old);
                    %z_est = Gamma_est + (it)/(it+5)*(Gamma_est-Gamma_est_old);
                end
                if snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2))>snr_oracle
                    snr_oracle = snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2));
                    it_oracle = it;
                    lambda2_oracle = lambda2Vect(lambda2number);
                    Gamma_est_oracle = Gamma_est;
                end
            end
            Xsources_est = op.synthesis(Gamma_est_oracle);


            G = dgtreal(Xsources_est, g, a, M);
            [nf,nt,nq] = size(G);

            Gamma = op.analysis(Xsources_est);
            G = abs(Gamma);

            k = 3;
            options.creation = 1;
            options.max_it = 1000;
            options.T = T;
            options.M = M;
            options.a = a;
            options.Xsensors = XXsensors;

            % Ini SVD
            [U_svd,D_svd,S_svd] = svd(sum(G,3),"econ");
            S_svd = sqrt(D_svd)*S_svd';
            U_svd = U_svd*sqrt(D_svd);
            W_init = abs(U_svd(:,1:k));
            H_init = abs(S_svd(1:k,:))';
            C_init = ones(nq,k);


            %%% IS-NTF %%%

            [W, H, C] = IS_NTF(G,W_init,H_init,C_init,options);

            [alphak,atom] = get_alphak_by_Wiener_filter(W, H', Gamma, k, options, op);
            Alphak = reshape(sum(alphak,3),[nf nt k]);
            Xk = zeros(T,k);
            for kk=1:k
                Xk(:,kk) = op.synthesis(Alphak(:,:,kk));
            end

            v_init = atom;
            normV = sqrt(sum(abs(v_init).^2,1));
            v_init = v_init./normV;
            % u_init=C;
            % normU = sqrt(sum(abs(u_init).^2,1));
            % u_init = u_init./normU;
            u_init =C_init;
            z_tmp = H;
            [~,nfr] = size(H');
            %z_init = ifftreal(fftreal(H),T);
            z_init = zeros(T,k);
            for l=2:nfr
                z_init(a*(l-2)+1,:) = z_tmp(l,:);
            end
            M = length(v_init(:,1));

            %maxLambda = 20;
            lambda = logspace(-1,-5,10);
            %lambda = 10;
            pos_z = 1;
            pos_u = 1;
            options.max_it = 1000;
            options.max_it_cdl = 5;
            options.max_it_csc = 15;
            options.oracle = 0;

            %%% CDl Multivariate rank-1

            [xk, v, u, z2] = M1csc_dl(Xsources_est,z_init,v_init, u_init,lambda,pos_z,pos_u,k,options);
            %u = max(u,0);
            cpu_time=toc;
            x = sum(xk,3);
            xtotk = reshape(sum(xk,2),[T k]);
            VV=zeros(M,K);
            VV(:,1)=[zeros(91,1);V(:,1)/norm(V(:,1));[zeros(37,1)]];
            VV(:,2)=[zeros(31,1);V(:,2)/norm(V(:,2));[zeros(97,1)]];
            VV(:,3)=[zeros(115,1);V(:,3)/norm(V(:,3));[zeros(13,1)]];
            V1=[zeros(91,1);V(:,1)/norm(V(:,1),'inf');[zeros(37,1)]];
            V2=[zeros(31,1);0.5*V(:,2)/norm(V(:,2),'inf');[zeros(97,1)]];
            V3=[zeros(115,1);V(:,3)/norm(V(:,3),'inf');[zeros(13,1)]];


            %%% Quant results

            SNR_motifs = zeros(k,1);

            optimal_transport_Uk = zeros(k,1);
            corr_atoms = zeros(K,1);

            SNR_motifs(1,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,2)),max(snr(sum(X(:,:,2,nub),2),Xk(:,2)),snr(sum(X(:,:,3,nub),2),Xk(:,2))));
            SNR_motifs_average(1,1) = SNR_motifs_average(1,1)+SNR_motifs(1,1)/Nubset;
            SNR_motifs(2,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,1)),max(snr(sum(X(:,:,2,nub),2),Xk(:,1)),snr(sum(X(:,:,3,nub),2),Xk(:,1))));
            SNR_motifs_average(2,1) = SNR_motifs_average(2,1)+SNR_motifs(2,1)/Nubset;
            SNR_motifs(3,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,3)),max(snr(sum(X(:,:,2,nub),2),Xk(:,3)),snr(sum(X(:,:,3,nub),2),Xk(:,3))));
            SNR_motifs_average(3,1) = SNR_motifs_average(3,1)+SNR_motifs(3,1)/Nubset;





            Fv = abs(fftreal(v));
            FV = abs(fftreal([V2 V1 V3]));
            Fv = Fv./max(Fv);
            FV = FV./max(FV);
            FV = FV(FV>0);
            Fv = Fv(FV>0);
            Tmp = (FV)./(Fv) - log((FV)./(Fv) ) -1;
            Tmp(isnan(Tmp)) = 0;

            for kk=1:K
                corr_atoms(kk)= max(max(max(xcorr(VV(:,1),v(:,kk)/norm(v(:,kk)))),max(xcorr(VV(:,2),v(:,kk)/norm(v(:,kk))))),max(xcorr(VV(:,3),v(:,kk)/norm(v(:,kk)))));
            end
            corr_atoms_average=sum(corr_atoms,1)/K;
            SNR_atoms_average=10*log10(1/(2-2*corr_atoms_average));


            flat = @(x)x(:);
            Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
                flat(reshape(1:n0*n1,n0,n1) ), ...
                ones(n0*n1,1) );
            Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
                flat(reshape(1:n0*n1,n0,n1)' ), ...
                ones(n0*n1,1) );
            Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];




            maxit = 1e4; tol = 1e-9;
            otransp = @(C,p0,p1)reshape( perform_linprog( ...
                Sigma(length(p0),length(p1)), ...
                [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );

            [a1,b1] = meshgrid(nn,nn);
            Cost_opti_transp = abs(a1-b1).^2;

            % opti transport U global

            p0 = sum(C,2)/norm(sum(C,2),1);
            p1 = sum(U,2)/norm(sum(U,2),1);

            gamma = otransp(Cost_opti_transp,p0,p1);

            fprintf('Number of non-zero: %d (N+N-1=%d)\n', full(sum(gamma(:)~=0)), N+N-1);
            fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));

            [I,J,gammaij] = find(gamma);
            Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;

            optimal_transport_U = Wasserstein_Distance_2;

            % opti transport Uk on each motif
            index = zeros(k,1);
            for kk=1:k
                WD2old = inf;
                for kkk=1:k
                    p0 = C(:,kkk)/norm(C(:,kkk),1);
                    p1 = U(:,kk)/norm(U(:,kk),1);
                    gamma = otransp(Cost_opti_transp,p0,p1);
                    [I,J,gammaij] = find(gamma);
                    Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;
                    if Wasserstein_Distance_2<WD2old
                        WD2old = Wasserstein_Distance_2;
                        index(kk) = kkk;
                    end
                end
                optimal_transport_Uk(kk) = WD2old;
                optimal_transport_Uk_average(kk,1) = optimal_transport_Uk_average(kk,1) + optimal_transport_Uk(kk)/Nubset;
            end


        end
        optimal_transport_Uk_average_average = sum(optimal_transport_Uk_average)/3;
        SNR_motifs_average_average = sum(SNR_motifs_average)/3;



    elseif corr == 0.4


        rng(42);
        load('dataw00.mat')
        SNR_motifs_average = zeros(k,1);
        optimal_transport_Uk_average = zeros(k,1);
        optimal_transport_U_average = 0;
        DIS_atoms_spectrum_average = 0;
        cpu_time_average=0;
        lead_field = G;
        Nubset=1;
        for nub=1:Nubset
            % Sensors Gaussian noising
            XXsensors = Xsensors(:,:,nub);
            snrlvl = iSNR;
            sigma2 = var(XXsensors(:))*10^(-snrlvl/10); % add noise
            XXsensors = XXsensors + sqrt(sigma2)*randn(size(XXsensors));

            % Gabor parameters
            M = 256;
            a = M/2;
            g = gabwin({'tight', 'hann'}, a, M);



            % TF operator
            op.analysis = @(x) dgtreal(x,g,a,M);
            op.synthesis = @(x) idgtreal(x,g,a,M,T);

            tic

            %%% Inverse problem



            Nbit = 300;
            L = norm(lead_field)^2;
            Nblambda2 = 15; %% lambda2 is for the l21 norm
            A=lead_field'*XXsensors;
            EnerA = zeros(20,1);
            for h=1:N
                EnerA(h,1) = norm(A(h,:));
            end
            lambda2Max = max(EnerA) ;
            lambda2Min = 1e-4*lambda2Max;
            lambda2Vect = logspace(log10(lambda2Max),log10(lambda2Min),Nblambda2);
            lambda2number = 0;
            mu = 0.001*lambda2Max; %% mu is for the l1 norm
            Xsources_est = lead_field'*XXsensors;

            % snr_oracle = snr(sum(Xsources,1)',sum(Xsources_est,1)');
            % for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
            %     lambda2number = lambda2number + 1;
            %     lambdaOnL = lambda/L;
            %     mu0nL = mu/L;
            %     z_est = Xsources_est;
            %     for it=0:Nbit-1
            %         Xsources_est_old = Xsources_est;
            %         Xsources_est = z_est + lead_field'*(XXsensors-lead_field*z_est)/L;
            %         Xsources_est = Xsources_est.*max(0,1-mu0nL./(abs(Xsources_est)+eps));
            %         Xsources_est = Xsources_est.*max(0,1-lambdaOnL./sqrt(sum(abs(Xsources_est).^2,2)+eps));
            %         z_est = Xsources_est + (it)/(it+5)*(Xsources_est-Xsources_est_old);
            %     end
            %     if snr(sum(Xsources,1)',sum(Xsources_est,1)')>snr_oracle
            %         snr_oracle = snr(sum(Xsources,1)',sum(Xsources_est,1)');
            %         it_oracle = it;
            %         lambda2_oracle = lambda2Vect(lambda2number);
            %         Xsources_est_oracle = Xsources_est;
            %     end
            % end
            % Xsources_est = Xsources_est_oracle';

            %%% inverse problem with TF dictionnary

            Gamma_est = op.analysis(Xsources_est');
            snr_oracle = snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2));
            for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
                lambda2number = lambda2number + 1;
                lambdaOnL = lambda/L;
                mu0nL = mu/L;
                z_est = Gamma_est;
                tau=1;
                for it=0:Nbit-1
                    Gamma_est_old = Gamma_est;
                    Gamma_est = z_est + op.analysis(transpose(lead_field'*(XXsensors-lead_field*op.synthesis(z_est)')/L));
                    Gamma_est = Gamma_est.*max(0,1-mu0nL./(abs(Gamma_est)+eps));
                    Gamma_est = Gamma_est.*max(0,1-lambdaOnL./sqrt(sum(abs(Gamma_est).^2,2)+eps));
                    tau0 = tau;
                    tau = (1+sqrt(1+4*tau^2))/2;
                    z_est = Gamma_est + (tau0-1)/(tau)*(Gamma_est-Gamma_est_old);
                    %z_est = Gamma_est + (it)/(it+5)*(Gamma_est-Gamma_est_old);
                end
                if snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2))>snr_oracle
                    snr_oracle = snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2));
                    it_oracle = it;
                    lambda2_oracle = lambda2Vect(lambda2number);
                    Gamma_est_oracle = Gamma_est;
                end
            end
            Xsources_est = op.synthesis(Gamma_est_oracle);


            G = dgtreal(Xsources_est, g, a, M);
            [nf,nt,nq] = size(G);

            Gamma = op.analysis(Xsources_est);
            G = abs(Gamma);

            k = 3;
            options.creation = 1;
            options.max_it = 1000;
            options.T = T;
            options.M = M;
            options.a = a;
            options.Xsensors = XXsensors;

            % Ini SVD
            [U_svd,D_svd,S_svd] = svd(sum(G,3),"econ");
            S_svd = sqrt(D_svd)*S_svd';
            U_svd = U_svd*sqrt(D_svd);
            W_init = abs(U_svd(:,1:k));
            H_init = abs(S_svd(1:k,:))';
            C_init = ones(nq,k);


            %%% IS-NTF %%%

            [W, H, C] = IS_NTF(G,W_init,H_init,C_init,options);

            [alphak,atom] = get_alphak_by_Wiener_filter(W, H', Gamma, k, options, op);
            Alphak = reshape(sum(alphak,3),[nf nt k]);
            Xk = zeros(T,k);
            for kk=1:k
                Xk(:,kk) = op.synthesis(Alphak(:,:,kk));
            end

            v_init = atom;
            normV = sqrt(sum(abs(v_init).^2,1));
            v_init = v_init./normV;
            % u_init=C;
            % normU = sqrt(sum(abs(u_init).^2,1));
            % u_init = u_init./normU;
            u_init =C_init;
            z_tmp = H;
            [~,nfr] = size(H');
            %z_init = ifftreal(fftreal(H),T);
            z_init = zeros(T,k);
            for l=2:nfr
                z_init(a*(l-2)+1,:) = z_tmp(l,:);
            end
            M = length(v_init(:,1));

            %maxLambda = 20;
            lambda = logspace(-1,-5,10);
            %lambda = 10;
            pos_z = 1;
            pos_u = 1;
            options.max_it = 1000;
            options.max_it_cdl = 5;
            options.max_it_csc = 15;
            options.oracle = 0;

            %%% CDl Multivariate rank-1

            [xk, v, u, z2] = M1csc_dl(Xsources_est,z_init,v_init, u_init,lambda,pos_z,pos_u,k,options);
            %u = max(u,0);
            cpu_time=toc;
            x = sum(xk,3);
            xtotk = reshape(sum(xk,2),[T k]);
            VV=zeros(M,K);
            VV(:,1)=[zeros(91,1);V(:,1)/norm(V(:,1));[zeros(37,1)]];
            VV(:,2)=[zeros(31,1);V(:,2)/norm(V(:,2));[zeros(97,1)]];
            VV(:,3)=[zeros(115,1);V(:,3)/norm(V(:,3));[zeros(13,1)]];
            V1=[zeros(91,1);V(:,1)/norm(V(:,1),'inf');[zeros(37,1)]];
            V2=[zeros(31,1);0.5*V(:,2)/norm(V(:,2),'inf');[zeros(97,1)]];
            V3=[zeros(115,1);V(:,3)/norm(V(:,3),'inf');[zeros(13,1)]];


            %%% Quant results

            SNR_motifs = zeros(k,1);

            optimal_transport_Uk = zeros(k,1);
            corr_atoms = zeros(K,1);

            SNR_motifs(1,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,2)),max(snr(sum(X(:,:,2,nub),2),Xk(:,2)),snr(sum(X(:,:,3,nub),2),Xk(:,2))));
            SNR_motifs_average(1,1) = SNR_motifs_average(1,1)+SNR_motifs(1,1)/Nubset;
            SNR_motifs(2,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,1)),max(snr(sum(X(:,:,2,nub),2),Xk(:,1)),snr(sum(X(:,:,3,nub),2),Xk(:,1))));
            SNR_motifs_average(2,1) = SNR_motifs_average(2,1)+SNR_motifs(2,1)/Nubset;
            SNR_motifs(3,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,3)),max(snr(sum(X(:,:,2,nub),2),Xk(:,3)),snr(sum(X(:,:,3,nub),2),Xk(:,3))));
            SNR_motifs_average(3,1) = SNR_motifs_average(3,1)+SNR_motifs(3,1)/Nubset;





            Fv = abs(fftreal(v));
            FV = abs(fftreal([V2 V1 V3]));
            Fv = Fv./max(Fv);
            FV = FV./max(FV);
            FV = FV(FV>0);
            Fv = Fv(FV>0);
            Tmp = (FV)./(Fv) - log((FV)./(Fv) ) -1;
            Tmp(isnan(Tmp)) = 0;

            for kk=1:K
                corr_atoms(kk)= max(max(max(xcorr(VV(:,1),v(:,kk)/norm(v(:,kk)))),max(xcorr(VV(:,2),v(:,kk)/norm(v(:,kk))))),max(xcorr(VV(:,3),v(:,kk)/norm(v(:,kk)))));
            end
            corr_atoms_average=sum(corr_atoms,1)/K;
            SNR_atoms_average=10*log10(1/(2-2*corr_atoms_average));


            flat = @(x)x(:);
            Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
                flat(reshape(1:n0*n1,n0,n1) ), ...
                ones(n0*n1,1) );
            Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
                flat(reshape(1:n0*n1,n0,n1)' ), ...
                ones(n0*n1,1) );
            Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];




            maxit = 1e4; tol = 1e-9;
            otransp = @(C,p0,p1)reshape( perform_linprog( ...
                Sigma(length(p0),length(p1)), ...
                [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );

            [a1,b1] = meshgrid(nn,nn);
            Cost_opti_transp = abs(a1-b1).^2;

            % opti transport U global

            p0 = sum(C,2)/norm(sum(C,2),1);
            p1 = sum(U,2)/norm(sum(U,2),1);

            gamma = otransp(Cost_opti_transp,p0,p1);

            fprintf('Number of non-zero: %d (N+N-1=%d)\n', full(sum(gamma(:)~=0)), N+N-1);
            fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));

            [I,J,gammaij] = find(gamma);
            Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;

            optimal_transport_U = Wasserstein_Distance_2;

            % opti transport Uk on each motif
            index = zeros(k,1);
            for kk=1:k
                WD2old = inf;
                for kkk=1:k
                    p0 = C(:,kkk)/norm(C(:,kkk),1);
                    p1 = U(:,kk)/norm(U(:,kk),1);
                    gamma = otransp(Cost_opti_transp,p0,p1);
                    [I,J,gammaij] = find(gamma);
                    Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;
                    if Wasserstein_Distance_2<WD2old
                        WD2old = Wasserstein_Distance_2;
                        index(kk) = kkk;
                    end
                end
                optimal_transport_Uk(kk) = WD2old;
                optimal_transport_Uk_average(kk,1) = optimal_transport_Uk_average(kk,1) + optimal_transport_Uk(kk)/Nubset;
            end


        end
        optimal_transport_Uk_average_average = sum(optimal_transport_Uk_average)/3;
        SNR_motifs_average_average = sum(SNR_motifs_average)/3;



    elseif corr == 0.6


        rng(42);
        load('dataw99.mat')
        SNR_motifs_average = zeros(k,1);
        optimal_transport_Uk_average = zeros(k,1);
        optimal_transport_U_average = 0;
        DIS_atoms_spectrum_average = 0;
        cpu_time_average=0;
        lead_field = G;
        Nubset=1;
        for nub=1:Nubset
            % Sensors Gaussian noising
            XXsensors = Xsensors(:,:,nub);
            snrlvl = iSNR;
            sigma2 = var(XXsensors(:))*10^(-snrlvl/10); % add noise
            XXsensors = XXsensors + sqrt(sigma2)*randn(size(XXsensors));

            % Gabor parameters
            M = 256;
            a = M/2;
            g = gabwin({'tight', 'hann'}, a, M);



            % TF operator
            op.analysis = @(x) dgtreal(x,g,a,M);
            op.synthesis = @(x) idgtreal(x,g,a,M,T);

            tic

            %%% Inverse problem



            Nbit = 300;
            L = norm(lead_field)^2;
            Nblambda2 = 15; %% lambda2 is for the l21 norm
            A=lead_field'*XXsensors;
            EnerA = zeros(20,1);
            for h=1:N
                EnerA(h,1) = norm(A(h,:));
            end
            lambda2Max = max(EnerA) ;
            lambda2Min = 1e-4*lambda2Max;
            lambda2Vect = logspace(log10(lambda2Max),log10(lambda2Min),Nblambda2);
            lambda2number = 0;
            mu = 0.001*lambda2Max; %% mu is for the l1 norm
            Xsources_est = lead_field'*XXsensors;

            % snr_oracle = snr(sum(Xsources,1)',sum(Xsources_est,1)');
            % for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
            %     lambda2number = lambda2number + 1;
            %     lambdaOnL = lambda/L;
            %     mu0nL = mu/L;
            %     z_est = Xsources_est;
            %     for it=0:Nbit-1
            %         Xsources_est_old = Xsources_est;
            %         Xsources_est = z_est + lead_field'*(XXsensors-lead_field*z_est)/L;
            %         Xsources_est = Xsources_est.*max(0,1-mu0nL./(abs(Xsources_est)+eps));
            %         Xsources_est = Xsources_est.*max(0,1-lambdaOnL./sqrt(sum(abs(Xsources_est).^2,2)+eps));
            %         z_est = Xsources_est + (it)/(it+5)*(Xsources_est-Xsources_est_old);
            %     end
            %     if snr(sum(Xsources,1)',sum(Xsources_est,1)')>snr_oracle
            %         snr_oracle = snr(sum(Xsources,1)',sum(Xsources_est,1)');
            %         it_oracle = it;
            %         lambda2_oracle = lambda2Vect(lambda2number);
            %         Xsources_est_oracle = Xsources_est;
            %     end
            % end
            % Xsources_est = Xsources_est_oracle';

            %%% inverse problem with TF dictionnary

            Gamma_est = op.analysis(Xsources_est');
            snr_oracle = snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2));
            for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
                lambda2number = lambda2number + 1;
                lambdaOnL = lambda/L;
                mu0nL = mu/L;
                z_est = Gamma_est;
                tau=1;
                for it=0:Nbit-1
                    Gamma_est_old = Gamma_est;
                    Gamma_est = z_est + op.analysis(transpose(lead_field'*(XXsensors-lead_field*op.synthesis(z_est)')/L));
                    Gamma_est = Gamma_est.*max(0,1-mu0nL./(abs(Gamma_est)+eps));
                    Gamma_est = Gamma_est.*max(0,1-lambdaOnL./sqrt(sum(abs(Gamma_est).^2,2)+eps));
                    tau0 = tau;
                    tau = (1+sqrt(1+4*tau^2))/2;
                    z_est = Gamma_est + (tau0-1)/(tau)*(Gamma_est-Gamma_est_old);
                    %z_est = Gamma_est + (it)/(it+5)*(Gamma_est-Gamma_est_old);
                end
                if snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2))>snr_oracle
                    snr_oracle = snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2));
                    it_oracle = it;
                    lambda2_oracle = lambda2Vect(lambda2number);
                    Gamma_est_oracle = Gamma_est;
                end
            end
            Xsources_est = op.synthesis(Gamma_est_oracle);


            G = dgtreal(Xsources_est, g, a, M);
            [nf,nt,nq] = size(G);

            Gamma = op.analysis(Xsources_est);
            G = abs(Gamma);

            k = 3;
            options.creation = 1;
            options.max_it = 1000;
            options.T = T;
            options.M = M;
            options.a = a;
            options.Xsensors = XXsensors;

            % Ini SVD
            [U_svd,D_svd,S_svd] = svd(sum(G,3),"econ");
            S_svd = sqrt(D_svd)*S_svd';
            U_svd = U_svd*sqrt(D_svd);
            W_init = abs(U_svd(:,1:k));
            H_init = abs(S_svd(1:k,:))';
            C_init = ones(nq,k);


            %%% IS-NTF %%%

            [W, H, C] = IS_NTF(G,W_init,H_init,C_init,options);

            [alphak,atom] = get_alphak_by_Wiener_filter(W, H', Gamma, k, options, op);
            Alphak = reshape(sum(alphak,3),[nf nt k]);
            Xk = zeros(T,k);
            for kk=1:k
                Xk(:,kk) = op.synthesis(Alphak(:,:,kk));
            end

            v_init = atom;
            normV = sqrt(sum(abs(v_init).^2,1));
            v_init = v_init./normV;
            % u_init=C;
            % normU = sqrt(sum(abs(u_init).^2,1));
            % u_init = u_init./normU;
            u_init =C_init;
            z_tmp = H;
            [~,nfr] = size(H');
            %z_init = ifftreal(fftreal(H),T);
            z_init = zeros(T,k);
            for l=2:nfr
                z_init(a*(l-2)+1,:) = z_tmp(l,:);
            end
            M = length(v_init(:,1));

            %maxLambda = 20;
            lambda = logspace(-1,-5,10);
            %lambda = 10;
            pos_z = 1;
            pos_u = 1;
            options.max_it = 1000;
            options.max_it_cdl = 5;
            options.max_it_csc = 15;
            options.oracle = 0;

            %%% CDl Multivariate rank-1

            [xk, v, u, z2] = M1csc_dl(Xsources_est,z_init,v_init, u_init,lambda,pos_z,pos_u,k,options);
            %u = max(u,0);
            cpu_time=toc;
            x = sum(xk,3);
            xtotk = reshape(sum(xk,2),[T k]);
            VV=zeros(M,K);
            VV(:,1)=[zeros(91,1);V(:,1)/norm(V(:,1));[zeros(37,1)]];
            VV(:,2)=[zeros(31,1);V(:,2)/norm(V(:,2));[zeros(97,1)]];
            VV(:,3)=[zeros(115,1);V(:,3)/norm(V(:,3));[zeros(13,1)]];
            V1=[zeros(91,1);V(:,1)/norm(V(:,1),'inf');[zeros(37,1)]];
            V2=[zeros(31,1);0.5*V(:,2)/norm(V(:,2),'inf');[zeros(97,1)]];
            V3=[zeros(115,1);V(:,3)/norm(V(:,3),'inf');[zeros(13,1)]];


            %%% Quant results

            SNR_motifs = zeros(k,1);

            optimal_transport_Uk = zeros(k,1);
            corr_atoms = zeros(K,1);

            SNR_motifs(1,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,2)),max(snr(sum(X(:,:,2,nub),2),Xk(:,2)),snr(sum(X(:,:,3,nub),2),Xk(:,2))));
            SNR_motifs_average(1,1) = SNR_motifs_average(1,1)+SNR_motifs(1,1)/Nubset;
            SNR_motifs(2,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,1)),max(snr(sum(X(:,:,2,nub),2),Xk(:,1)),snr(sum(X(:,:,3,nub),2),Xk(:,1))));
            SNR_motifs_average(2,1) = SNR_motifs_average(2,1)+SNR_motifs(2,1)/Nubset;
            SNR_motifs(3,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,3)),max(snr(sum(X(:,:,2,nub),2),Xk(:,3)),snr(sum(X(:,:,3,nub),2),Xk(:,3))));
            SNR_motifs_average(3,1) = SNR_motifs_average(3,1)+SNR_motifs(3,1)/Nubset;





            Fv = abs(fftreal(v));
            FV = abs(fftreal([V2 V1 V3]));
            Fv = Fv./max(Fv);
            FV = FV./max(FV);
            FV = FV(FV>0);
            Fv = Fv(FV>0);
            Tmp = (FV)./(Fv) - log((FV)./(Fv) ) -1;
            Tmp(isnan(Tmp)) = 0;

            for kk=1:K
                corr_atoms(kk)= max(max(max(xcorr(VV(:,1),v(:,kk)/norm(v(:,kk)))),max(xcorr(VV(:,2),v(:,kk)/norm(v(:,kk))))),max(xcorr(VV(:,3),v(:,kk)/norm(v(:,kk)))));
            end
            corr_atoms_average=sum(corr_atoms,1)/K;
            SNR_atoms_average=10*log10(1/(2-2*corr_atoms_average));


            flat = @(x)x(:);
            Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
                flat(reshape(1:n0*n1,n0,n1) ), ...
                ones(n0*n1,1) );
            Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
                flat(reshape(1:n0*n1,n0,n1)' ), ...
                ones(n0*n1,1) );
            Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];




            maxit = 1e4; tol = 1e-9;
            otransp = @(C,p0,p1)reshape( perform_linprog( ...
                Sigma(length(p0),length(p1)), ...
                [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );

            [a1,b1] = meshgrid(nn,nn);
            Cost_opti_transp = abs(a1-b1).^2;

            % opti transport U global

            p0 = sum(C,2)/norm(sum(C,2),1);
            p1 = sum(U,2)/norm(sum(U,2),1);

            gamma = otransp(Cost_opti_transp,p0,p1);

            fprintf('Number of non-zero: %d (N+N-1=%d)\n', full(sum(gamma(:)~=0)), N+N-1);
            fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));

            [I,J,gammaij] = find(gamma);
            Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;

            optimal_transport_U = Wasserstein_Distance_2;

            % opti transport Uk on each motif
            index = zeros(k,1);
            for kk=1:k
                WD2old = inf;
                for kkk=1:k
                    p0 = C(:,kkk)/norm(C(:,kkk),1);
                    p1 = U(:,kk)/norm(U(:,kk),1);
                    gamma = otransp(Cost_opti_transp,p0,p1);
                    [I,J,gammaij] = find(gamma);
                    Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;
                    if Wasserstein_Distance_2<WD2old
                        WD2old = Wasserstein_Distance_2;
                        index(kk) = kkk;
                    end
                end
                optimal_transport_Uk(kk) = WD2old;
                optimal_transport_Uk_average(kk,1) = optimal_transport_Uk_average(kk,1) + optimal_transport_Uk(kk)/Nubset;
            end


        end
        optimal_transport_Uk_average_average = sum(optimal_transport_Uk_average)/3;
        SNR_motifs_average_average = sum(SNR_motifs_average)/3;



    elseif corr == 0.8


        rng(42);
        load('dataw80.mat')
        SNR_motifs_average = zeros(k,1);
        optimal_transport_Uk_average = zeros(k,1);
        optimal_transport_U_average = 0;
        DIS_atoms_spectrum_average = 0;
        cpu_time_average=0;
        lead_field = G;
        Nubset=1;
        for nub=1:Nubset
            % Sensors Gaussian noising
            XXsensors = Xsensors(:,:,nub);
            snrlvl = iSNR;
            sigma2 = var(XXsensors(:))*10^(-snrlvl/10); % add noise
            XXsensors = XXsensors + sqrt(sigma2)*randn(size(XXsensors));

            % Gabor parameters
            M = 256;
            a = M/2;
            g = gabwin({'tight', 'hann'}, a, M);



            % TF operator
            op.analysis = @(x) dgtreal(x,g,a,M);
            op.synthesis = @(x) idgtreal(x,g,a,M,T);

            tic

            %%% Inverse problem



            Nbit = 300;
            L = norm(lead_field)^2;
            Nblambda2 = 15; %% lambda2 is for the l21 norm
            A=lead_field'*XXsensors;
            EnerA = zeros(20,1);
            for h=1:N
                EnerA(h,1) = norm(A(h,:));
            end
            lambda2Max = max(EnerA) ;
            lambda2Min = 1e-4*lambda2Max;
            lambda2Vect = logspace(log10(lambda2Max),log10(lambda2Min),Nblambda2);
            lambda2number = 0;
            mu = 0.001*lambda2Max; %% mu is for the l1 norm
            Xsources_est = lead_field'*XXsensors;

            % snr_oracle = snr(sum(Xsources,1)',sum(Xsources_est,1)');
            % for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
            %     lambda2number = lambda2number + 1;
            %     lambdaOnL = lambda/L;
            %     mu0nL = mu/L;
            %     z_est = Xsources_est;
            %     for it=0:Nbit-1
            %         Xsources_est_old = Xsources_est;
            %         Xsources_est = z_est + lead_field'*(XXsensors-lead_field*z_est)/L;
            %         Xsources_est = Xsources_est.*max(0,1-mu0nL./(abs(Xsources_est)+eps));
            %         Xsources_est = Xsources_est.*max(0,1-lambdaOnL./sqrt(sum(abs(Xsources_est).^2,2)+eps));
            %         z_est = Xsources_est + (it)/(it+5)*(Xsources_est-Xsources_est_old);
            %     end
            %     if snr(sum(Xsources,1)',sum(Xsources_est,1)')>snr_oracle
            %         snr_oracle = snr(sum(Xsources,1)',sum(Xsources_est,1)');
            %         it_oracle = it;
            %         lambda2_oracle = lambda2Vect(lambda2number);
            %         Xsources_est_oracle = Xsources_est;
            %     end
            % end
            % Xsources_est = Xsources_est_oracle';

            %%% inverse problem with TF dictionnary

            Gamma_est = op.analysis(Xsources_est');
            snr_oracle = snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2));
            for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
                lambda2number = lambda2number + 1;
                lambdaOnL = lambda/L;
                mu0nL = mu/L;
                z_est = Gamma_est;
                tau=1;
                for it=0:Nbit-1
                    Gamma_est_old = Gamma_est;
                    Gamma_est = z_est + op.analysis(transpose(lead_field'*(XXsensors-lead_field*op.synthesis(z_est)')/L));
                    Gamma_est = Gamma_est.*max(0,1-mu0nL./(abs(Gamma_est)+eps));
                    Gamma_est = Gamma_est.*max(0,1-lambdaOnL./sqrt(sum(abs(Gamma_est).^2,2)+eps));
                    tau0 = tau;
                    tau = (1+sqrt(1+4*tau^2))/2;
                    z_est = Gamma_est + (tau0-1)/(tau)*(Gamma_est-Gamma_est_old);
                    %z_est = Gamma_est + (it)/(it+5)*(Gamma_est-Gamma_est_old);
                end
                if snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2))>snr_oracle
                    snr_oracle = snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2));
                    it_oracle = it;
                    lambda2_oracle = lambda2Vect(lambda2number);
                    Gamma_est_oracle = Gamma_est;
                end
            end
            Xsources_est = op.synthesis(Gamma_est_oracle);


            G = dgtreal(Xsources_est, g, a, M);
            [nf,nt,nq] = size(G);

            Gamma = op.analysis(Xsources_est);
            G = abs(Gamma);

            k = 3;
            options.creation = 1;
            options.max_it = 1000;
            options.T = T;
            options.M = M;
            options.a = a;
            options.Xsensors = XXsensors;

            % Ini SVD
            [U_svd,D_svd,S_svd] = svd(sum(G,3),"econ");
            S_svd = sqrt(D_svd)*S_svd';
            U_svd = U_svd*sqrt(D_svd);
            W_init = abs(U_svd(:,1:k));
            H_init = abs(S_svd(1:k,:))';
            C_init = ones(nq,k);


            %%% IS-NTF %%%

            [W, H, C] = IS_NTF(G,W_init,H_init,C_init,options);

            [alphak,atom] = get_alphak_by_Wiener_filter(W, H', Gamma, k, options, op);
            Alphak = reshape(sum(alphak,3),[nf nt k]);
            Xk = zeros(T,k);
            for kk=1:k
                Xk(:,kk) = op.synthesis(Alphak(:,:,kk));
            end

            v_init = atom;
            normV = sqrt(sum(abs(v_init).^2,1));
            v_init = v_init./normV;
            % u_init=C;
            % normU = sqrt(sum(abs(u_init).^2,1));
            % u_init = u_init./normU;
            u_init =C_init;
            z_tmp = H;
            [~,nfr] = size(H');
            %z_init = ifftreal(fftreal(H),T);
            z_init = zeros(T,k);
            for l=2:nfr
                z_init(a*(l-2)+1,:) = z_tmp(l,:);
            end
            M = length(v_init(:,1));

            %maxLambda = 20;
            lambda = logspace(-1,-5,10);
            %lambda = 10;
            pos_z = 1;
            pos_u = 1;
            options.max_it = 1000;
            options.max_it_cdl = 5;
            options.max_it_csc = 15;
            options.oracle = 0;

            %%% CDl Multivariate rank-1

            [xk, v, u, z2] = M1csc_dl(Xsources_est,z_init,v_init, u_init,lambda,pos_z,pos_u,k,options);
            %u = max(u,0);
            cpu_time=toc;
            x = sum(xk,3);
            xtotk = reshape(sum(xk,2),[T k]);
            VV=zeros(M,K);
            VV(:,1)=[zeros(91,1);V(:,1)/norm(V(:,1));[zeros(37,1)]];
            VV(:,2)=[zeros(31,1);V(:,2)/norm(V(:,2));[zeros(97,1)]];
            VV(:,3)=[zeros(115,1);V(:,3)/norm(V(:,3));[zeros(13,1)]];
            V1=[zeros(91,1);V(:,1)/norm(V(:,1),'inf');[zeros(37,1)]];
            V2=[zeros(31,1);0.5*V(:,2)/norm(V(:,2),'inf');[zeros(97,1)]];
            V3=[zeros(115,1);V(:,3)/norm(V(:,3),'inf');[zeros(13,1)]];


            %%% Quant results

            SNR_motifs = zeros(k,1);

            optimal_transport_Uk = zeros(k,1);
            corr_atoms = zeros(K,1);

            SNR_motifs(1,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,2)),max(snr(sum(X(:,:,2,nub),2),Xk(:,2)),snr(sum(X(:,:,3,nub),2),Xk(:,2))));
            SNR_motifs_average(1,1) = SNR_motifs_average(1,1)+SNR_motifs(1,1)/Nubset;
            SNR_motifs(2,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,1)),max(snr(sum(X(:,:,2,nub),2),Xk(:,1)),snr(sum(X(:,:,3,nub),2),Xk(:,1))));
            SNR_motifs_average(2,1) = SNR_motifs_average(2,1)+SNR_motifs(2,1)/Nubset;
            SNR_motifs(3,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,3)),max(snr(sum(X(:,:,2,nub),2),Xk(:,3)),snr(sum(X(:,:,3,nub),2),Xk(:,3))));
            SNR_motifs_average(3,1) = SNR_motifs_average(3,1)+SNR_motifs(3,1)/Nubset;





            Fv = abs(fftreal(v));
            FV = abs(fftreal([V2 V1 V3]));
            Fv = Fv./max(Fv);
            FV = FV./max(FV);
            FV = FV(FV>0);
            Fv = Fv(FV>0);
            Tmp = (FV)./(Fv) - log((FV)./(Fv) ) -1;
            Tmp(isnan(Tmp)) = 0;

            for kk=1:K
                corr_atoms(kk)= max(max(max(xcorr(VV(:,1),v(:,kk)/norm(v(:,kk)))),max(xcorr(VV(:,2),v(:,kk)/norm(v(:,kk))))),max(xcorr(VV(:,3),v(:,kk)/norm(v(:,kk)))));
            end
            corr_atoms_average=sum(corr_atoms,1)/K;
            SNR_atoms_average=10*log10(1/(2-2*corr_atoms_average));


            flat = @(x)x(:);
            Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
                flat(reshape(1:n0*n1,n0,n1) ), ...
                ones(n0*n1,1) );
            Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
                flat(reshape(1:n0*n1,n0,n1)' ), ...
                ones(n0*n1,1) );
            Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];




            maxit = 1e4; tol = 1e-9;
            otransp = @(C,p0,p1)reshape( perform_linprog( ...
                Sigma(length(p0),length(p1)), ...
                [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );

            [a1,b1] = meshgrid(nn,nn);
            Cost_opti_transp = abs(a1-b1).^2;

            % opti transport U global

            p0 = sum(C,2)/norm(sum(C,2),1);
            p1 = sum(U,2)/norm(sum(U,2),1);

            gamma = otransp(Cost_opti_transp,p0,p1);

            fprintf('Number of non-zero: %d (N+N-1=%d)\n', full(sum(gamma(:)~=0)), N+N-1);
            fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));

            [I,J,gammaij] = find(gamma);
            Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;

            optimal_transport_U = Wasserstein_Distance_2;

            % opti transport Uk on each motif
            index = zeros(k,1);
            for kk=1:k
                WD2old = inf;
                for kkk=1:k
                    p0 = C(:,kkk)/norm(C(:,kkk),1);
                    p1 = U(:,kk)/norm(U(:,kk),1);
                    gamma = otransp(Cost_opti_transp,p0,p1);
                    [I,J,gammaij] = find(gamma);
                    Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;
                    if Wasserstein_Distance_2<WD2old
                        WD2old = Wasserstein_Distance_2;
                        index(kk) = kkk;
                    end
                end
                optimal_transport_Uk(kk) = WD2old;
                optimal_transport_Uk_average(kk,1) = optimal_transport_Uk_average(kk,1) + optimal_transport_Uk(kk)/Nubset;
            end


        end
        optimal_transport_Uk_average_average = sum(optimal_transport_Uk_average)/3;
        SNR_motifs_average_average = sum(SNR_motifs_average)/3;



    elseif corr == 0.99


        rng(42);
        load('dataw00.mat')
        SNR_motifs_average = zeros(k,1);
        optimal_transport_Uk_average = zeros(k,1);
        optimal_transport_U_average = 0;
        DIS_atoms_spectrum_average = 0;
        cpu_time_average=0;
        lead_field = G;
        Nubset=1;
        for nub=1:Nubset
            % Sensors Gaussian noising
            XXsensors = Xsensors(:,:,nub);
            snrlvl = iSNR;
            sigma2 = var(XXsensors(:))*10^(-snrlvl/10); % add noise
            XXsensors = XXsensors + sqrt(sigma2)*randn(size(XXsensors));

            % Gabor parameters
            M = 256;
            a = M/2;
            g = gabwin({'tight', 'hann'}, a, M);



            % TF operator
            op.analysis = @(x) dgtreal(x,g,a,M);
            op.synthesis = @(x) idgtreal(x,g,a,M,T);

            tic

            %%% Inverse problem



            Nbit = 300;
            L = norm(lead_field)^2;
            Nblambda2 = 15; %% lambda2 is for the l21 norm
            A=lead_field'*XXsensors;
            EnerA = zeros(20,1);
            for h=1:N
                EnerA(h,1) = norm(A(h,:));
            end
            lambda2Max = max(EnerA) ;
            lambda2Min = 1e-4*lambda2Max;
            lambda2Vect = logspace(log10(lambda2Max),log10(lambda2Min),Nblambda2);
            lambda2number = 0;
            mu = 0.001*lambda2Max; %% mu is for the l1 norm
            Xsources_est = lead_field'*XXsensors;

            % snr_oracle = snr(sum(Xsources,1)',sum(Xsources_est,1)');
            % for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
            %     lambda2number = lambda2number + 1;
            %     lambdaOnL = lambda/L;
            %     mu0nL = mu/L;
            %     z_est = Xsources_est;
            %     for it=0:Nbit-1
            %         Xsources_est_old = Xsources_est;
            %         Xsources_est = z_est + lead_field'*(XXsensors-lead_field*z_est)/L;
            %         Xsources_est = Xsources_est.*max(0,1-mu0nL./(abs(Xsources_est)+eps));
            %         Xsources_est = Xsources_est.*max(0,1-lambdaOnL./sqrt(sum(abs(Xsources_est).^2,2)+eps));
            %         z_est = Xsources_est + (it)/(it+5)*(Xsources_est-Xsources_est_old);
            %     end
            %     if snr(sum(Xsources,1)',sum(Xsources_est,1)')>snr_oracle
            %         snr_oracle = snr(sum(Xsources,1)',sum(Xsources_est,1)');
            %         it_oracle = it;
            %         lambda2_oracle = lambda2Vect(lambda2number);
            %         Xsources_est_oracle = Xsources_est;
            %     end
            % end
            % Xsources_est = Xsources_est_oracle';

            %%% inverse problem with TF dictionnary

            Gamma_est = op.analysis(Xsources_est');
            snr_oracle = snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2));
            for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
                lambda2number = lambda2number + 1;
                lambdaOnL = lambda/L;
                mu0nL = mu/L;
                z_est = Gamma_est;
                tau=1;
                for it=0:Nbit-1
                    Gamma_est_old = Gamma_est;
                    Gamma_est = z_est + op.analysis(transpose(lead_field'*(XXsensors-lead_field*op.synthesis(z_est)')/L));
                    Gamma_est = Gamma_est.*max(0,1-mu0nL./(abs(Gamma_est)+eps));
                    Gamma_est = Gamma_est.*max(0,1-lambdaOnL./sqrt(sum(abs(Gamma_est).^2,2)+eps));
                    tau0 = tau;
                    tau = (1+sqrt(1+4*tau^2))/2;
                    z_est = Gamma_est + (tau0-1)/(tau)*(Gamma_est-Gamma_est_old);
                    %z_est = Gamma_est + (it)/(it+5)*(Gamma_est-Gamma_est_old);
                end
                if snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2))>snr_oracle
                    snr_oracle = snr(sum(Xsources(:,:,nub),1)',sum(op.synthesis(Gamma_est),2));
                    it_oracle = it;
                    lambda2_oracle = lambda2Vect(lambda2number);
                    Gamma_est_oracle = Gamma_est;
                end
            end
            Xsources_est = op.synthesis(Gamma_est_oracle);


            G = dgtreal(Xsources_est, g, a, M);
            [nf,nt,nq] = size(G);

            Gamma = op.analysis(Xsources_est);
            G = abs(Gamma);

            k = 3;
            options.creation = 1;
            options.max_it = 1000;
            options.T = T;
            options.M = M;
            options.a = a;
            options.Xsensors = XXsensors;

            % Ini SVD
            [U_svd,D_svd,S_svd] = svd(sum(G,3),"econ");
            S_svd = sqrt(D_svd)*S_svd';
            U_svd = U_svd*sqrt(D_svd);
            W_init = abs(U_svd(:,1:k));
            H_init = abs(S_svd(1:k,:))';
            C_init = ones(nq,k);


            %%% IS-NTF %%%

            [W, H, C] = IS_NTF(G,W_init,H_init,C_init,options);

            [alphak,atom] = get_alphak_by_Wiener_filter(W, H', Gamma, k, options, op);
            Alphak = reshape(sum(alphak,3),[nf nt k]);
            Xk = zeros(T,k);
            for kk=1:k
                Xk(:,kk) = op.synthesis(Alphak(:,:,kk));
            end

            v_init = atom;
            normV = sqrt(sum(abs(v_init).^2,1));
            v_init = v_init./normV;
            % u_init=C;
            % normU = sqrt(sum(abs(u_init).^2,1));
            % u_init = u_init./normU;
            u_init =C_init;
            z_tmp = H;
            [~,nfr] = size(H');
            %z_init = ifftreal(fftreal(H),T);
            z_init = zeros(T,k);
            for l=2:nfr
                z_init(a*(l-2)+1,:) = z_tmp(l,:);
            end
            M = length(v_init(:,1));

            %maxLambda = 20;
            lambda = logspace(-1,-5,10);
            %lambda = 10;
            pos_z = 1;
            pos_u = 1;
            options.max_it = 1000;
            options.max_it_cdl = 5;
            options.max_it_csc = 15;
            options.oracle = 0;

            %%% CDl Multivariate rank-1

            [xk, v, u, z2] = M1csc_dl(Xsources_est,z_init,v_init, u_init,lambda,pos_z,pos_u,k,options);
            %u = max(u,0);
            cpu_time=toc;
            x = sum(xk,3);
            xtotk = reshape(sum(xk,2),[T k]);
            VV=zeros(M,K);
            VV(:,1)=[zeros(91,1);V(:,1)/norm(V(:,1));[zeros(37,1)]];
            VV(:,2)=[zeros(31,1);V(:,2)/norm(V(:,2));[zeros(97,1)]];
            VV(:,3)=[zeros(115,1);V(:,3)/norm(V(:,3));[zeros(13,1)]];
            V1=[zeros(91,1);V(:,1)/norm(V(:,1),'inf');[zeros(37,1)]];
            V2=[zeros(31,1);0.5*V(:,2)/norm(V(:,2),'inf');[zeros(97,1)]];
            V3=[zeros(115,1);V(:,3)/norm(V(:,3),'inf');[zeros(13,1)]];


            %%% Quant results

            SNR_motifs = zeros(k,1);

            optimal_transport_Uk = zeros(k,1);
            corr_atoms = zeros(K,1);

            SNR_motifs(1,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,2)),max(snr(sum(X(:,:,2,nub),2),Xk(:,2)),snr(sum(X(:,:,3,nub),2),Xk(:,2))));
            SNR_motifs_average(1,1) = SNR_motifs_average(1,1)+SNR_motifs(1,1)/Nubset;
            SNR_motifs(2,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,1)),max(snr(sum(X(:,:,2,nub),2),Xk(:,1)),snr(sum(X(:,:,3,nub),2),Xk(:,1))));
            SNR_motifs_average(2,1) = SNR_motifs_average(2,1)+SNR_motifs(2,1)/Nubset;
            SNR_motifs(3,1) = max(snr(sum(X(:,:,1,nub),2),Xk(:,3)),max(snr(sum(X(:,:,2,nub),2),Xk(:,3)),snr(sum(X(:,:,3,nub),2),Xk(:,3))));
            SNR_motifs_average(3,1) = SNR_motifs_average(3,1)+SNR_motifs(3,1)/Nubset;





            Fv = abs(fftreal(v));
            FV = abs(fftreal([V2 V1 V3]));
            Fv = Fv./max(Fv);
            FV = FV./max(FV);
            FV = FV(FV>0);
            Fv = Fv(FV>0);
            Tmp = (FV)./(Fv) - log((FV)./(Fv) ) -1;
            Tmp(isnan(Tmp)) = 0;

            for kk=1:K
                corr_atoms(kk)= max(max(max(xcorr(VV(:,1),v(:,kk)/norm(v(:,kk)))),max(xcorr(VV(:,2),v(:,kk)/norm(v(:,kk))))),max(xcorr(VV(:,3),v(:,kk)/norm(v(:,kk)))));
            end
            corr_atoms_average=sum(corr_atoms,1)/K;
            SNR_atoms_average=10*log10(1/(2-2*corr_atoms_average));


            flat = @(x)x(:);
            Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
                flat(reshape(1:n0*n1,n0,n1) ), ...
                ones(n0*n1,1) );
            Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
                flat(reshape(1:n0*n1,n0,n1)' ), ...
                ones(n0*n1,1) );
            Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];




            maxit = 1e4; tol = 1e-9;
            otransp = @(C,p0,p1)reshape( perform_linprog( ...
                Sigma(length(p0),length(p1)), ...
                [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );

            [a1,b1] = meshgrid(nn,nn);
            Cost_opti_transp = abs(a1-b1).^2;

            % opti transport U global

            p0 = sum(C,2)/norm(sum(C,2),1);
            p1 = sum(U,2)/norm(sum(U,2),1);

            gamma = otransp(Cost_opti_transp,p0,p1);

            fprintf('Number of non-zero: %d (N+N-1=%d)\n', full(sum(gamma(:)~=0)), N+N-1);
            fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));

            [I,J,gammaij] = find(gamma);
            Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;

            optimal_transport_U = Wasserstein_Distance_2;

            % opti transport Uk on each motif
            index = zeros(k,1);
            for kk=1:k
                WD2old = inf;
                for kkk=1:k
                    p0 = C(:,kkk)/norm(C(:,kkk),1);
                    p1 = U(:,kk)/norm(U(:,kk),1);
                    gamma = otransp(Cost_opti_transp,p0,p1);
                    [I,J,gammaij] = find(gamma);
                    Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;
                    if Wasserstein_Distance_2<WD2old
                        WD2old = Wasserstein_Distance_2;
                        index(kk) = kkk;
                    end
                end
                optimal_transport_Uk(kk) = WD2old;
                optimal_transport_Uk_average(kk,1) = optimal_transport_Uk_average(kk,1) + optimal_transport_Uk(kk)/Nubset;
            end


        end
        optimal_transport_Uk_average_average = sum(optimal_transport_Uk_average)/3;
        SNR_motifs_average_average = sum(SNR_motifs_average)/3;


    else
        fprintf('Data not generated for this corr please choose among the list')
    end



elseif method == 2
    if corr==0
        load('dataw00.mat')
        SNR_motifs_average = zeros(k,1);
        optimal_transport_Uk_average = zeros(k,1);
        optimal_transport_U_average = 0;
        DIS_atoms_spectrum_average = 0;
        cpu_time_average=0;
        lead_field = G;
        Nubset=1;
        for nub=1:Nubset
            % Sensors Gaussian noising
            snrlvl = iSNR;
            XXsensors = Xsensors(:,:,nub);
            sigma2 = var(XXsensors(:))*10^(-snrlvl/10); % add noise
            XXsensors = XXsensors' + (sqrt(sigma2)*randn(size(XXsensors)))';
            tic

            % Gabor parameters
            M = 256;
            a = M/2;
            g = gabwin({'tight', 'hann'}, a, M);
            G = dgtreal(XXsensors, g, a, M);
            [nf,nt,nq] = size(G);


            % TF operator
            op.analysis = @(x) dgtreal(x,g,a,M);
            op.synthesis = @(x) idgtreal(x,g,a,M,T);


            Gamma = op.analysis(XXsensors);
            G = abs(Gamma);

            k = 3;
            options.creation = 1;
            options.max_it = 1000;
            options.T = T;
            options.M = M;
            options.a = a;
            options.Xsensors = Xsensors(:,:,nub);

            % Ini SVD
            [U_svd,D_svd,S_svd] = svd(sum(G,3),"econ");
            S_svd = sqrt(D_svd)*S_svd';
            U_svd = U_svd*sqrt(D_svd);
            W_init = abs(U_svd(:,1:k));
            H_init = abs(S_svd(1:k,:))';
            C_init = ones(nq,k);


            %%% IS-NTF %%%

            [W, H, C] = IS_NTF(G,W_init,H_init,C_init,options);

            [alphak,atom] = get_alphak_by_Wiener_filter(W, H', Gamma, k, options, op);
            Alphak = reshape(sum(alphak,3),[nf nt k]);


            v_init = atom;
            normV = sqrt(sum(abs(v_init).^2,1));
            v_init = v_init./normV;
            % u_init=C;
            % normU = sqrt(sum(abs(u_init).^2,1));
            % u_init = u_init./normU;
            u_init =C_init;
            z_tmp = H;
            [~,nfr] = size(H');
            %z_init = ifftreal(fftreal(H),T);
            z_init = zeros(T,k);
            for l=2:nfr
                z_init(a*(l-2)+1,:) = z_tmp(l,:);
            end
            M = length(v_init(:,1));

            MaxLambda = MaxLambda_multi_cdl(XXsensors,z_init,u_init,v_init);
            lambda = logspace(log10(MaxLambda)-0.5,log10(MaxLambda)-5,10);
            %lambda = logspace(1,-4,15);
            pos_z = 1;
            pos_u = 0;
            options.max_it = 1000;
            options.max_it_cdl = 5;
            options.max_it_csc = 15;
            options.oracle = 1;

            %%% CDl Multivariate rank-1

            [xk, v, u, z2] = M1csc_dl(XXsensors,z_init,v_init, u_init,lambda,pos_z,pos_u,k,options);

            x = sum(xk,3);
            xtotk = reshape(sum(xk,2),[T k]);
            V1=[zeros(91,1);V(:,1)/norm(V(:,1),'inf');[zeros(37,1)]];
            V2=[zeros(31,1);0.5*V(:,2)/norm(V(:,2),'inf');[zeros(97,1)]];
            V3=[zeros(115,1);V(:,3)/norm(V(:,3),'inf');[zeros(13,1)]];
            %%% Results




            %% Inverse problem for u and reconstruction of the sources


            Nbit = 100;
            L = norm(lead_field)^2;
            Nblambda2 = 20;
            A=lead_field'*u;
            EnerA = zeros(20,1);
            for h=1:N
                EnerA(h,1) = norm(A(h,:));
            end
            lambda2Max = 0.5*norm(A,'inf');
            lambda2Min = 1e-5*lambda2Max;
            lambda2Vect = logspace(log10(lambda2Max),log10(lambda2Min),Nblambda2);
            lambda2number = 0;

            U_est = lead_field'*u;
            U_est = max(U_est,0);


            for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
                lambda2number = lambda2number + 1;
                lambdaOnL = lambda/L;
                z_est = U_est;
                for it=0:Nbit-1
                    U_est_old = U_est;
                    U_est = z_est + lead_field'*(u-lead_field*z_est)/L;
                    U_est = U_est.*max(0,1-lambdaOnL./(abs(U_est)+eps));
                    z_est = U_est + (it)/(it+5)*(U_est-U_est_old);
                end
                U_est = max(U_est,0);
            end

            cpu_time=toc;
            cpu_time_average = cpu_time_average + cpu_time/Nubset;
            VV=zeros(M,K);
            VV(:,1)=[zeros(91,1);V(:,1)/norm(V(:,1));[zeros(37,1)]];
            VV(:,2)=[zeros(31,1);V(:,2)/norm(V(:,2));[zeros(97,1)]];
            VV(:,3)=[zeros(115,1);V(:,3)/norm(V(:,3));[zeros(13,1)]];

            %%% Quant results


            SNR_motifs = zeros(K,1);
 
            optimal_transport_Uk = zeros(K,1);
            corr_atoms = zeros(K,1);


            SNR_motifs(1,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,2)),max(snr(sum(X2(:,:,nub),2),xtotk(:,2)),snr(sum(X3(:,:,nub),2),xtotk(:,2))));
            SNR_motifs_average(1,1) = SNR_motifs_average(1,1)+SNR_motifs(1,1)/Nubset;
            SNR_motifs(2,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,1)),max(snr(sum(X2(:,:,nub),2),xtotk(:,1)),snr(sum(X3(:,:,nub),2),xtotk(:,1))));
            SNR_motifs_average(2,1) = SNR_motifs_average(2,1)+SNR_motifs(2,1)/Nubset;
            SNR_motifs(3,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,3)),max(snr(sum(X2(:,:,nub),2),xtotk(:,3)),snr(sum(X3(:,:,nub),2),xtotk(:,3))));
            SNR_motifs_average(3,1) = SNR_motifs_average(3,1)+SNR_motifs(3,1)/Nubset;


            Fv = abs(fftreal(v));
            FV = abs(fftreal([V2 V1 V3]));
            Fv = Fv./max(Fv);
            FV = FV./max(FV);
            FV = FV(FV>0);
            Fv = Fv(FV>0);
            Tmp = (FV)./(Fv) - log((FV)./(Fv) ) -1;
            Tmp(isnan(Tmp)) = 0;
            DIS_atoms_spectrum = sum(Tmp(:));
            DIS_atoms_spectrum_average = DIS_atoms_spectrum_average + DIS_atoms_spectrum/Nubset;
            for kk=1:K
                corr_atoms(kk)= max(max(max(xcorr(VV(:,1),v(:,kk)/norm(v(:,kk)))),max(xcorr(VV(:,2),v(:,kk)/norm(v(:,kk))))),max(xcorr(VV(:,3),v(:,kk)/norm(v(:,kk)))));
            end
            corr_atoms_average=sum(corr_atoms,1)/K;
            SNR_atoms_average=10*log10(1/(2-2*corr_atoms_average));

            flat = @(x)x(:);
            Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
                flat(reshape(1:n0*n1,n0,n1) ), ...
                ones(n0*n1,1) );
            Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
                flat(reshape(1:n0*n1,n0,n1)' ), ...
                ones(n0*n1,1) );
            Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];




            maxit = 1e4; tol = 1e-9;
            otransp = @(C,p0,p1)reshape( perform_linprog( ...
                Sigma(length(p0),length(p1)), ...
                [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );

            [a1,b1] = meshgrid(nn,nn);
            Cost_opti_transp = abs(a1-b1).^2;

            % opti transport U global

            p0 = sum(U_est,2)/norm(sum(U_est,2),1);
            p1 = sum(U,2)/norm(sum(U,2),1);

            gamma = otransp(Cost_opti_transp,p0,p1);

            fprintf('Number of non-zero: %d (N+N-1=%d)\n', full(sum(gamma(:)~=0)), N+N-1);
            fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));

            [I,J,gammaij] = find(gamma);
            Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;

            optimal_transport_U = Wasserstein_Distance_2;
            optimal_transport_U_average = optimal_transport_U_average + optimal_transport_U/Nubset;
            % opti transport Uk on each motif
            index = zeros(k,1);
            for kk=1:k
                WD2old = inf;
                for kkk=1:k
                    p0 = U_est(:,kkk)/norm(U_est(:,kkk),1);
                    p1 = U(:,kk)/norm(U(:,kk),1);
                    gamma = otransp(Cost_opti_transp,p0,p1);
                    [I,J,gammaij] = find(gamma);
                    Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;
                    if Wasserstein_Distance_2<WD2old
                        WD2old = Wasserstein_Distance_2;
                        index(kk) = kkk;
                    end
                end
                optimal_transport_Uk(kk) = WD2old;
                optimal_transport_Uk_average(kk,1) = optimal_transport_Uk_average(kk,1) + optimal_transport_Uk(kk)/Nubset;
            end

        end
        optimal_transport_Uk_average_average = sum(optimal_transport_Uk_average)/3;
        SNR_motifs_average_average = sum(SNR_motifs_average)/3;

    elseif corr==0.2
        load('dataw20.mat')
        SNR_motifs_average = zeros(k,1);
        optimal_transport_Uk_average = zeros(k,1);
        optimal_transport_U_average = 0;
        DIS_atoms_spectrum_average = 0;
        cpu_time_average=0;
        lead_field = G;
        Nubset=1;
        for nub=1:Nubset
            % Sensors Gaussian noising
            snrlvl = iSNR;
            XXsensors = Xsensors(:,:,nub);
            sigma2 = var(XXsensors(:))*10^(-snrlvl/10); % add noise
            XXsensors = XXsensors' + (sqrt(sigma2)*randn(size(XXsensors)))';
            tic

            % Gabor parameters
            M = 256;
            a = M/2;
            g = gabwin({'tight', 'hann'}, a, M);
            G = dgtreal(XXsensors, g, a, M);
            [nf,nt,nq] = size(G);


            % TF operator
            op.analysis = @(x) dgtreal(x,g,a,M);
            op.synthesis = @(x) idgtreal(x,g,a,M,T);


            Gamma = op.analysis(XXsensors);
            G = abs(Gamma);

            k = 3;
            options.creation = 1;
            options.max_it = 1000;
            options.T = T;
            options.M = M;
            options.a = a;
            options.Xsensors = Xsensors(:,:,nub);

            % Ini SVD
            [U_svd,D_svd,S_svd] = svd(sum(G,3),"econ");
            S_svd = sqrt(D_svd)*S_svd';
            U_svd = U_svd*sqrt(D_svd);
            W_init = abs(U_svd(:,1:k));
            H_init = abs(S_svd(1:k,:))';
            C_init = ones(nq,k);


            %%% IS-NTF %%%

            [W, H, C] = IS_NTF(G,W_init,H_init,C_init,options);

            [alphak,atom] = get_alphak_by_Wiener_filter(W, H', Gamma, k, options, op);
            Alphak = reshape(sum(alphak,3),[nf nt k]);


            v_init = atom;
            normV = sqrt(sum(abs(v_init).^2,1));
            v_init = v_init./normV;
            % u_init=C;
            % normU = sqrt(sum(abs(u_init).^2,1));
            % u_init = u_init./normU;
            u_init =C_init;
            z_tmp = H;
            [~,nfr] = size(H');
            %z_init = ifftreal(fftreal(H),T);
            z_init = zeros(T,k);
            for l=2:nfr
                z_init(a*(l-2)+1,:) = z_tmp(l,:);
            end
            M = length(v_init(:,1));

            MaxLambda = MaxLambda_multi_cdl(XXsensors,z_init,u_init,v_init);
            lambda = logspace(log10(MaxLambda)-0.5,log10(MaxLambda)-5,10);
            %lambda = logspace(1,-4,15);
            pos_z = 1;
            pos_u = 0;
            options.max_it = 1000;
            options.max_it_cdl = 5;
            options.max_it_csc = 15;
            options.oracle = 1;

            %%% CDl Multivariate rank-1

            [xk, v, u, z2] = M1csc_dl(XXsensors,z_init,v_init, u_init,lambda,pos_z,pos_u,k,options);

            x = sum(xk,3);
            xtotk = reshape(sum(xk,2),[T k]);
            V1=[zeros(91,1);V(:,1)/norm(V(:,1),'inf');[zeros(37,1)]];
            V2=[zeros(31,1);0.5*V(:,2)/norm(V(:,2),'inf');[zeros(97,1)]];
            V3=[zeros(115,1);V(:,3)/norm(V(:,3),'inf');[zeros(13,1)]];
            %%% Results




            %% Inverse problem for u and reconstruction of the sources


            Nbit = 100;
            L = norm(lead_field)^2;
            Nblambda2 = 20;
            A=lead_field'*u;
            EnerA = zeros(20,1);
            for h=1:N
                EnerA(h,1) = norm(A(h,:));
            end
            lambda2Max = 0.5*norm(A,'inf');
            lambda2Min = 1e-5*lambda2Max;
            lambda2Vect = logspace(log10(lambda2Max),log10(lambda2Min),Nblambda2);
            lambda2number = 0;

            U_est = lead_field'*u;
            U_est = max(U_est,0);


            for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
                lambda2number = lambda2number + 1;
                lambdaOnL = lambda/L;
                z_est = U_est;
                for it=0:Nbit-1
                    U_est_old = U_est;
                    U_est = z_est + lead_field'*(u-lead_field*z_est)/L;
                    U_est = U_est.*max(0,1-lambdaOnL./(abs(U_est)+eps));
                    z_est = U_est + (it)/(it+5)*(U_est-U_est_old);
                end
                U_est = max(U_est,0);
            end

            cpu_time=toc;
            cpu_time_average = cpu_time_average + cpu_time/Nubset;
            VV=zeros(M,K);
            VV(:,1)=[zeros(91,1);V(:,1)/norm(V(:,1));[zeros(37,1)]];
            VV(:,2)=[zeros(31,1);V(:,2)/norm(V(:,2));[zeros(97,1)]];
            VV(:,3)=[zeros(115,1);V(:,3)/norm(V(:,3));[zeros(13,1)]];

            %%% Quant results


            SNR_motifs = zeros(K,1);

            optimal_transport_Uk = zeros(K,1);
            corr_atoms = zeros(K,1);


            SNR_motifs(1,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,2)),max(snr(sum(X2(:,:,nub),2),xtotk(:,2)),snr(sum(X3(:,:,nub),2),xtotk(:,2))));
            SNR_motifs_average(1,1) = SNR_motifs_average(1,1)+SNR_motifs(1,1)/Nubset;
            SNR_motifs(2,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,1)),max(snr(sum(X2(:,:,nub),2),xtotk(:,1)),snr(sum(X3(:,:,nub),2),xtotk(:,1))));
            SNR_motifs_average(2,1) = SNR_motifs_average(2,1)+SNR_motifs(2,1)/Nubset;
            SNR_motifs(3,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,3)),max(snr(sum(X2(:,:,nub),2),xtotk(:,3)),snr(sum(X3(:,:,nub),2),xtotk(:,3))));
            SNR_motifs_average(3,1) = SNR_motifs_average(3,1)+SNR_motifs(3,1)/Nubset;


            Fv = abs(fftreal(v));
            FV = abs(fftreal([V2 V1 V3]));
            Fv = Fv./max(Fv);
            FV = FV./max(FV);
            FV = FV(FV>0);
            Fv = Fv(FV>0);
            Tmp = (FV)./(Fv) - log((FV)./(Fv) ) -1;
            Tmp(isnan(Tmp)) = 0;
            DIS_atoms_spectrum = sum(Tmp(:));
            DIS_atoms_spectrum_average = DIS_atoms_spectrum_average + DIS_atoms_spectrum/Nubset;
            for kk=1:K
                corr_atoms(kk)= max(max(max(xcorr(VV(:,1),v(:,kk)/norm(v(:,kk)))),max(xcorr(VV(:,2),v(:,kk)/norm(v(:,kk))))),max(xcorr(VV(:,3),v(:,kk)/norm(v(:,kk)))));
            end
            corr_atoms_average=sum(corr_atoms,1)/K;
            SNR_atoms_average=10*log10(1/(2-2*corr_atoms_average));

            flat = @(x)x(:);
            Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
                flat(reshape(1:n0*n1,n0,n1) ), ...
                ones(n0*n1,1) );
            Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
                flat(reshape(1:n0*n1,n0,n1)' ), ...
                ones(n0*n1,1) );
            Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];




            maxit = 1e4; tol = 1e-9;
            otransp = @(C,p0,p1)reshape( perform_linprog( ...
                Sigma(length(p0),length(p1)), ...
                [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );

            [a1,b1] = meshgrid(nn,nn);
            Cost_opti_transp = abs(a1-b1).^2;

            % opti transport U global

            p0 = sum(U_est,2)/norm(sum(U_est,2),1);
            p1 = sum(U,2)/norm(sum(U,2),1);

            gamma = otransp(Cost_opti_transp,p0,p1);

            fprintf('Number of non-zero: %d (N+N-1=%d)\n', full(sum(gamma(:)~=0)), N+N-1);
            fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));

            [I,J,gammaij] = find(gamma);
            Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;

            optimal_transport_U = Wasserstein_Distance_2;
            optimal_transport_U_average = optimal_transport_U_average + optimal_transport_U/Nubset;
            % opti transport Uk on each motif
            index = zeros(k,1);
            for kk=1:k
                WD2old = inf;
                for kkk=1:k
                    p0 = U_est(:,kkk)/norm(U_est(:,kkk),1);
                    p1 = U(:,kk)/norm(U(:,kk),1);
                    gamma = otransp(Cost_opti_transp,p0,p1);
                    [I,J,gammaij] = find(gamma);
                    Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;
                    if Wasserstein_Distance_2<WD2old
                        WD2old = Wasserstein_Distance_2;
                        index(kk) = kkk;
                    end
                end
                optimal_transport_Uk(kk) = WD2old;
                optimal_transport_Uk_average(kk,1) = optimal_transport_Uk_average(kk,1) + optimal_transport_Uk(kk)/Nubset;
            end

        end
        optimal_transport_Uk_average_average = sum(optimal_transport_Uk_average)/3;
        SNR_motifs_average_average = sum(SNR_motifs_average)/3;

    elseif corr==0.4
        load('dataw40.mat')
        SNR_motifs_average = zeros(k,1);
        optimal_transport_Uk_average = zeros(k,1);
        optimal_transport_U_average = 0;
        DIS_atoms_spectrum_average = 0;
        cpu_time_average=0;
        lead_field = G;
        Nubset=1;
        for nub=1:Nubset
            % Sensors Gaussian noising
            snrlvl = iSNR;
            XXsensors = Xsensors(:,:,nub);
            sigma2 = var(XXsensors(:))*10^(-snrlvl/10); % add noise
            XXsensors = XXsensors' + (sqrt(sigma2)*randn(size(XXsensors)))';
            tic

            % Gabor parameters
            M = 256;
            a = M/2;
            g = gabwin({'tight', 'hann'}, a, M);
            G = dgtreal(XXsensors, g, a, M);
            [nf,nt,nq] = size(G);


            % TF operator
            op.analysis = @(x) dgtreal(x,g,a,M);
            op.synthesis = @(x) idgtreal(x,g,a,M,T);


            Gamma = op.analysis(XXsensors);
            G = abs(Gamma);

            k = 3;
            options.creation = 1;
            options.max_it = 1000;
            options.T = T;
            options.M = M;
            options.a = a;
            options.Xsensors = Xsensors(:,:,nub);

            % Ini SVD
            [U_svd,D_svd,S_svd] = svd(sum(G,3),"econ");
            S_svd = sqrt(D_svd)*S_svd';
            U_svd = U_svd*sqrt(D_svd);
            W_init = abs(U_svd(:,1:k));
            H_init = abs(S_svd(1:k,:))';
            C_init = ones(nq,k);


            %%% IS-NTF %%%

            [W, H, C] = IS_NTF(G,W_init,H_init,C_init,options);

            [alphak,atom] = get_alphak_by_Wiener_filter(W, H', Gamma, k, options, op);
            Alphak = reshape(sum(alphak,3),[nf nt k]);


            v_init = atom;
            normV = sqrt(sum(abs(v_init).^2,1));
            v_init = v_init./normV;
            % u_init=C;
            % normU = sqrt(sum(abs(u_init).^2,1));
            % u_init = u_init./normU;
            u_init =C_init;
            z_tmp = H;
            [~,nfr] = size(H');
            %z_init = ifftreal(fftreal(H),T);
            z_init = zeros(T,k);
            for l=2:nfr
                z_init(a*(l-2)+1,:) = z_tmp(l,:);
            end
            M = length(v_init(:,1));

            MaxLambda = MaxLambda_multi_cdl(XXsensors,z_init,u_init,v_init);
            lambda = logspace(log10(MaxLambda)-0.5,log10(MaxLambda)-5,10);
            %lambda = logspace(1,-4,15);
            pos_z = 1;
            pos_u = 0;
            options.max_it = 1000;
            options.max_it_cdl = 5;
            options.max_it_csc = 15;
            options.oracle = 1;

            %%% CDl Multivariate rank-1

            [xk, v, u, z2] = M1csc_dl(XXsensors,z_init,v_init, u_init,lambda,pos_z,pos_u,k,options);

            x = sum(xk,3);
            xtotk = reshape(sum(xk,2),[T k]);
            V1=[zeros(91,1);V(:,1)/norm(V(:,1),'inf');[zeros(37,1)]];
            V2=[zeros(31,1);0.5*V(:,2)/norm(V(:,2),'inf');[zeros(97,1)]];
            V3=[zeros(115,1);V(:,3)/norm(V(:,3),'inf');[zeros(13,1)]];
            %%% Results




            %% Inverse problem for u and reconstruction of the sources


            Nbit = 100;
            L = norm(lead_field)^2;
            Nblambda2 = 20;
            A=lead_field'*u;
            EnerA = zeros(20,1);
            for h=1:N
                EnerA(h,1) = norm(A(h,:));
            end
            lambda2Max = 0.5*norm(A,'inf');
            lambda2Min = 1e-5*lambda2Max;
            lambda2Vect = logspace(log10(lambda2Max),log10(lambda2Min),Nblambda2);
            lambda2number = 0;

            U_est = lead_field'*u;
            U_est = max(U_est,0);


            for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
                lambda2number = lambda2number + 1;
                lambdaOnL = lambda/L;
                z_est = U_est;
                for it=0:Nbit-1
                    U_est_old = U_est;
                    U_est = z_est + lead_field'*(u-lead_field*z_est)/L;
                    U_est = U_est.*max(0,1-lambdaOnL./(abs(U_est)+eps));
                    z_est = U_est + (it)/(it+5)*(U_est-U_est_old);
                end
                U_est = max(U_est,0);
            end

            cpu_time=toc;
            cpu_time_average = cpu_time_average + cpu_time/Nubset;
            VV=zeros(M,K);
            VV(:,1)=[zeros(91,1);V(:,1)/norm(V(:,1));[zeros(37,1)]];
            VV(:,2)=[zeros(31,1);V(:,2)/norm(V(:,2));[zeros(97,1)]];
            VV(:,3)=[zeros(115,1);V(:,3)/norm(V(:,3));[zeros(13,1)]];

            %%% Quant results


            SNR_motifs = zeros(K,1);

            optimal_transport_Uk = zeros(K,1);
            corr_atoms = zeros(K,1);


            SNR_motifs(1,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,2)),max(snr(sum(X2(:,:,nub),2),xtotk(:,2)),snr(sum(X3(:,:,nub),2),xtotk(:,2))));
            SNR_motifs_average(1,1) = SNR_motifs_average(1,1)+SNR_motifs(1,1)/Nubset;
            SNR_motifs(2,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,1)),max(snr(sum(X2(:,:,nub),2),xtotk(:,1)),snr(sum(X3(:,:,nub),2),xtotk(:,1))));
            SNR_motifs_average(2,1) = SNR_motifs_average(2,1)+SNR_motifs(2,1)/Nubset;
            SNR_motifs(3,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,3)),max(snr(sum(X2(:,:,nub),2),xtotk(:,3)),snr(sum(X3(:,:,nub),2),xtotk(:,3))));
            SNR_motifs_average(3,1) = SNR_motifs_average(3,1)+SNR_motifs(3,1)/Nubset;


            Fv = abs(fftreal(v));
            FV = abs(fftreal([V2 V1 V3]));
            Fv = Fv./max(Fv);
            FV = FV./max(FV);
            FV = FV(FV>0);
            Fv = Fv(FV>0);
            Tmp = (FV)./(Fv) - log((FV)./(Fv) ) -1;
            Tmp(isnan(Tmp)) = 0;
            DIS_atoms_spectrum = sum(Tmp(:));
            DIS_atoms_spectrum_average = DIS_atoms_spectrum_average + DIS_atoms_spectrum/Nubset;
            for kk=1:K
                corr_atoms(kk)= max(max(max(xcorr(VV(:,1),v(:,kk)/norm(v(:,kk)))),max(xcorr(VV(:,2),v(:,kk)/norm(v(:,kk))))),max(xcorr(VV(:,3),v(:,kk)/norm(v(:,kk)))));
            end
            corr_atoms_average=sum(corr_atoms,1)/K;
            SNR_atoms_average=10*log10(1/(2-2*corr_atoms_average));

            flat = @(x)x(:);
            Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
                flat(reshape(1:n0*n1,n0,n1) ), ...
                ones(n0*n1,1) );
            Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
                flat(reshape(1:n0*n1,n0,n1)' ), ...
                ones(n0*n1,1) );
            Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];




            maxit = 1e4; tol = 1e-9;
            otransp = @(C,p0,p1)reshape( perform_linprog( ...
                Sigma(length(p0),length(p1)), ...
                [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );

            [a1,b1] = meshgrid(nn,nn);
            Cost_opti_transp = abs(a1-b1).^2;

            % opti transport U global

            p0 = sum(U_est,2)/norm(sum(U_est,2),1);
            p1 = sum(U,2)/norm(sum(U,2),1);

            gamma = otransp(Cost_opti_transp,p0,p1);

            fprintf('Number of non-zero: %d (N+N-1=%d)\n', full(sum(gamma(:)~=0)), N+N-1);
            fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));

            [I,J,gammaij] = find(gamma);
            Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;

            optimal_transport_U = Wasserstein_Distance_2;
            optimal_transport_U_average = optimal_transport_U_average + optimal_transport_U/Nubset;
            % opti transport Uk on each motif
            index = zeros(k,1);
            for kk=1:k
                WD2old = inf;
                for kkk=1:k
                    p0 = U_est(:,kkk)/norm(U_est(:,kkk),1);
                    p1 = U(:,kk)/norm(U(:,kk),1);
                    gamma = otransp(Cost_opti_transp,p0,p1);
                    [I,J,gammaij] = find(gamma);
                    Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;
                    if Wasserstein_Distance_2<WD2old
                        WD2old = Wasserstein_Distance_2;
                        index(kk) = kkk;
                    end
                end
                optimal_transport_Uk(kk) = WD2old;
                optimal_transport_Uk_average(kk,1) = optimal_transport_Uk_average(kk,1) + optimal_transport_Uk(kk)/Nubset;
            end

        end
        optimal_transport_Uk_average_average = sum(optimal_transport_Uk_average)/3;
        SNR_motifs_average_average = sum(SNR_motifs_average)/3;
    elseif corr==0.6
        load('dataw60.mat')
        SNR_motifs_average = zeros(k,1);
        optimal_transport_Uk_average = zeros(k,1);
        optimal_transport_U_average = 0;
        DIS_atoms_spectrum_average = 0;
        cpu_time_average=0;
        lead_field = G;
        Nubset=1;
        for nub=1:Nubset
            % Sensors Gaussian noising
            snrlvl = iSNR;
            XXsensors = Xsensors(:,:,nub);
            sigma2 = var(XXsensors(:))*10^(-snrlvl/10); % add noise
            XXsensors = XXsensors' + (sqrt(sigma2)*randn(size(XXsensors)))';
            tic

            % Gabor parameters
            M = 256;
            a = M/2;
            g = gabwin({'tight', 'hann'}, a, M);
            G = dgtreal(XXsensors, g, a, M);
            [nf,nt,nq] = size(G);


            % TF operator
            op.analysis = @(x) dgtreal(x,g,a,M);
            op.synthesis = @(x) idgtreal(x,g,a,M,T);


            Gamma = op.analysis(XXsensors);
            G = abs(Gamma);

            k = 3;
            options.creation = 1;
            options.max_it = 1000;
            options.T = T;
            options.M = M;
            options.a = a;
            options.Xsensors = Xsensors(:,:,nub);

            % Ini SVD
            [U_svd,D_svd,S_svd] = svd(sum(G,3),"econ");
            S_svd = sqrt(D_svd)*S_svd';
            U_svd = U_svd*sqrt(D_svd);
            W_init = abs(U_svd(:,1:k));
            H_init = abs(S_svd(1:k,:))';
            C_init = ones(nq,k);


            %%% IS-NTF %%%

            [W, H, C] = IS_NTF(G,W_init,H_init,C_init,options);

            [alphak,atom] = get_alphak_by_Wiener_filter(W, H', Gamma, k, options, op);
            Alphak = reshape(sum(alphak,3),[nf nt k]);


            v_init = atom;
            normV = sqrt(sum(abs(v_init).^2,1));
            v_init = v_init./normV;
            % u_init=C;
            % normU = sqrt(sum(abs(u_init).^2,1));
            % u_init = u_init./normU;
            u_init =C_init;
            z_tmp = H;
            [~,nfr] = size(H');
            %z_init = ifftreal(fftreal(H),T);
            z_init = zeros(T,k);
            for l=2:nfr
                z_init(a*(l-2)+1,:) = z_tmp(l,:);
            end
            M = length(v_init(:,1));

            MaxLambda = MaxLambda_multi_cdl(XXsensors,z_init,u_init,v_init);
            lambda = logspace(log10(MaxLambda)-0.5,log10(MaxLambda)-5,10);
            %lambda = logspace(1,-4,15);
            pos_z = 1;
            pos_u = 0;
            options.max_it = 1000;
            options.max_it_cdl = 5;
            options.max_it_csc = 15;
            options.oracle = 1;

            %%% CDl Multivariate rank-1

            [xk, v, u, z2] = M1csc_dl(XXsensors,z_init,v_init, u_init,lambda,pos_z,pos_u,k,options);

            x = sum(xk,3);
            xtotk = reshape(sum(xk,2),[T k]);
            V1=[zeros(91,1);V(:,1)/norm(V(:,1),'inf');[zeros(37,1)]];
            V2=[zeros(31,1);0.5*V(:,2)/norm(V(:,2),'inf');[zeros(97,1)]];
            V3=[zeros(115,1);V(:,3)/norm(V(:,3),'inf');[zeros(13,1)]];
            %%% Results




            %% Inverse problem for u and reconstruction of the sources


            Nbit = 100;
            L = norm(lead_field)^2;
            Nblambda2 = 20;
            A=lead_field'*u;
            EnerA = zeros(20,1);
            for h=1:N
                EnerA(h,1) = norm(A(h,:));
            end
            lambda2Max = 0.5*norm(A,'inf');
            lambda2Min = 1e-5*lambda2Max;
            lambda2Vect = logspace(log10(lambda2Max),log10(lambda2Min),Nblambda2);
            lambda2number = 0;

            U_est = lead_field'*u;
            U_est = max(U_est,0);


            for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
                lambda2number = lambda2number + 1;
                lambdaOnL = lambda/L;
                z_est = U_est;
                for it=0:Nbit-1
                    U_est_old = U_est;
                    U_est = z_est + lead_field'*(u-lead_field*z_est)/L;
                    U_est = U_est.*max(0,1-lambdaOnL./(abs(U_est)+eps));
                    z_est = U_est + (it)/(it+5)*(U_est-U_est_old);
                end
                U_est = max(U_est,0);
            end

            cpu_time=toc;
            cpu_time_average = cpu_time_average + cpu_time/Nubset;
            VV=zeros(M,K);
            VV(:,1)=[zeros(91,1);V(:,1)/norm(V(:,1));[zeros(37,1)]];
            VV(:,2)=[zeros(31,1);V(:,2)/norm(V(:,2));[zeros(97,1)]];
            VV(:,3)=[zeros(115,1);V(:,3)/norm(V(:,3));[zeros(13,1)]];

            %%% Quant results


            SNR_motifs = zeros(K,1);

            optimal_transport_Uk = zeros(K,1);
            corr_atoms = zeros(K,1);


            SNR_motifs(1,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,2)),max(snr(sum(X2(:,:,nub),2),xtotk(:,2)),snr(sum(X3(:,:,nub),2),xtotk(:,2))));
            SNR_motifs_average(1,1) = SNR_motifs_average(1,1)+SNR_motifs(1,1)/Nubset;
            SNR_motifs(2,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,1)),max(snr(sum(X2(:,:,nub),2),xtotk(:,1)),snr(sum(X3(:,:,nub),2),xtotk(:,1))));
            SNR_motifs_average(2,1) = SNR_motifs_average(2,1)+SNR_motifs(2,1)/Nubset;
            SNR_motifs(3,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,3)),max(snr(sum(X2(:,:,nub),2),xtotk(:,3)),snr(sum(X3(:,:,nub),2),xtotk(:,3))));
            SNR_motifs_average(3,1) = SNR_motifs_average(3,1)+SNR_motifs(3,1)/Nubset;
 

            Fv = abs(fftreal(v));
            FV = abs(fftreal([V2 V1 V3]));
            Fv = Fv./max(Fv);
            FV = FV./max(FV);
            FV = FV(FV>0);
            Fv = Fv(FV>0);
            Tmp = (FV)./(Fv) - log((FV)./(Fv) ) -1;
            Tmp(isnan(Tmp)) = 0;
            DIS_atoms_spectrum = sum(Tmp(:));
            DIS_atoms_spectrum_average = DIS_atoms_spectrum_average + DIS_atoms_spectrum/Nubset;
            for kk=1:K
                corr_atoms(kk)= max(max(max(xcorr(VV(:,1),v(:,kk)/norm(v(:,kk)))),max(xcorr(VV(:,2),v(:,kk)/norm(v(:,kk))))),max(xcorr(VV(:,3),v(:,kk)/norm(v(:,kk)))));
            end
            corr_atoms_average=sum(corr_atoms,1)/K;
            SNR_atoms_average=10*log10(1/(2-2*corr_atoms_average));

            flat = @(x)x(:);
            Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
                flat(reshape(1:n0*n1,n0,n1) ), ...
                ones(n0*n1,1) );
            Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
                flat(reshape(1:n0*n1,n0,n1)' ), ...
                ones(n0*n1,1) );
            Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];




            maxit = 1e4; tol = 1e-9;
            otransp = @(C,p0,p1)reshape( perform_linprog( ...
                Sigma(length(p0),length(p1)), ...
                [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );

            [a1,b1] = meshgrid(nn,nn);
            Cost_opti_transp = abs(a1-b1).^2;

            % opti transport U global

            p0 = sum(U_est,2)/norm(sum(U_est,2),1);
            p1 = sum(U,2)/norm(sum(U,2),1);

            gamma = otransp(Cost_opti_transp,p0,p1);

            fprintf('Number of non-zero: %d (N+N-1=%d)\n', full(sum(gamma(:)~=0)), N+N-1);
            fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));

            [I,J,gammaij] = find(gamma);
            Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;

            optimal_transport_U = Wasserstein_Distance_2;
            optimal_transport_U_average = optimal_transport_U_average + optimal_transport_U/Nubset;
            % opti transport Uk on each motif
            index = zeros(k,1);
            for kk=1:k
                WD2old = inf;
                for kkk=1:k
                    p0 = U_est(:,kkk)/norm(U_est(:,kkk),1);
                    p1 = U(:,kk)/norm(U(:,kk),1);
                    gamma = otransp(Cost_opti_transp,p0,p1);
                    [I,J,gammaij] = find(gamma);
                    Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;
                    if Wasserstein_Distance_2<WD2old
                        WD2old = Wasserstein_Distance_2;
                        index(kk) = kkk;
                    end
                end
                optimal_transport_Uk(kk) = WD2old;
                optimal_transport_Uk_average(kk,1) = optimal_transport_Uk_average(kk,1) + optimal_transport_Uk(kk)/Nubset;
            end

        end
        optimal_transport_Uk_average_average = sum(optimal_transport_Uk_average)/3;
        SNR_motifs_average_average = sum(SNR_motifs_average)/3;

    elseif corr==0.8
        load('dataw80.mat')
        SNR_motifs_average = zeros(k,1);
        optimal_transport_Uk_average = zeros(k,1);
        optimal_transport_U_average = 0;
        DIS_atoms_spectrum_average = 0;
        cpu_time_average=0;
        lead_field = G;
        Nubset=1;
        for nub=1:Nubset
            % Sensors Gaussian noising
            snrlvl = iSNR;
            XXsensors = Xsensors(:,:,nub);
            sigma2 = var(XXsensors(:))*10^(-snrlvl/10); % add noise
            XXsensors = XXsensors' + (sqrt(sigma2)*randn(size(XXsensors)))';
            tic

            % Gabor parameters
            M = 256;
            a = M/2;
            g = gabwin({'tight', 'hann'}, a, M);
            G = dgtreal(XXsensors, g, a, M);
            [nf,nt,nq] = size(G);


            % TF operator
            op.analysis = @(x) dgtreal(x,g,a,M);
            op.synthesis = @(x) idgtreal(x,g,a,M,T);


            Gamma = op.analysis(XXsensors);
            G = abs(Gamma);

            k = 3;
            options.creation = 1;
            options.max_it = 1000;
            options.T = T;
            options.M = M;
            options.a = a;
            options.Xsensors = Xsensors(:,:,nub);

            % Ini SVD
            [U_svd,D_svd,S_svd] = svd(sum(G,3),"econ");
            S_svd = sqrt(D_svd)*S_svd';
            U_svd = U_svd*sqrt(D_svd);
            W_init = abs(U_svd(:,1:k));
            H_init = abs(S_svd(1:k,:))';
            C_init = ones(nq,k);


            %%% IS-NTF %%%

            [W, H, C] = IS_NTF(G,W_init,H_init,C_init,options);

            [alphak,atom] = get_alphak_by_Wiener_filter(W, H', Gamma, k, options, op);
            Alphak = reshape(sum(alphak,3),[nf nt k]);


            v_init = atom;
            normV = sqrt(sum(abs(v_init).^2,1));
            v_init = v_init./normV;
            % u_init=C;
            % normU = sqrt(sum(abs(u_init).^2,1));
            % u_init = u_init./normU;
            u_init =C_init;
            z_tmp = H;
            [~,nfr] = size(H');
            %z_init = ifftreal(fftreal(H),T);
            z_init = zeros(T,k);
            for l=2:nfr
                z_init(a*(l-2)+1,:) = z_tmp(l,:);
            end
            M = length(v_init(:,1));

            MaxLambda = MaxLambda_multi_cdl(XXsensors,z_init,u_init,v_init);
            lambda = logspace(log10(MaxLambda)-0.5,log10(MaxLambda)-5,10);
            %lambda = logspace(1,-4,15);
            pos_z = 1;
            pos_u = 0;
            options.max_it = 1000;
            options.max_it_cdl = 5;
            options.max_it_csc = 15;
            options.oracle = 1;

            %%% CDl Multivariate rank-1

            [xk, v, u, z2] = M1csc_dl(XXsensors,z_init,v_init, u_init,lambda,pos_z,pos_u,k,options);

            x = sum(xk,3);
            xtotk = reshape(sum(xk,2),[T k]);
            V1=[zeros(91,1);V(:,1)/norm(V(:,1),'inf');[zeros(37,1)]];
            V2=[zeros(31,1);0.5*V(:,2)/norm(V(:,2),'inf');[zeros(97,1)]];
            V3=[zeros(115,1);V(:,3)/norm(V(:,3),'inf');[zeros(13,1)]];
            %%% Results




            %% Inverse problem for u and reconstruction of the sources


            Nbit = 100;
            L = norm(lead_field)^2;
            Nblambda2 = 20;
            A=lead_field'*u;
            EnerA = zeros(20,1);
            for h=1:N
                EnerA(h,1) = norm(A(h,:));
            end
            lambda2Max = 0.5*norm(A,'inf');
            lambda2Min = 1e-5*lambda2Max;
            lambda2Vect = logspace(log10(lambda2Max),log10(lambda2Min),Nblambda2);
            lambda2number = 0;

            U_est = lead_field'*u;
            U_est = max(U_est,0);


            for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
                lambda2number = lambda2number + 1;
                lambdaOnL = lambda/L;
                z_est = U_est;
                for it=0:Nbit-1
                    U_est_old = U_est;
                    U_est = z_est + lead_field'*(u-lead_field*z_est)/L;
                    U_est = U_est.*max(0,1-lambdaOnL./(abs(U_est)+eps));
                    z_est = U_est + (it)/(it+5)*(U_est-U_est_old);
                end
                U_est = max(U_est,0);
            end

            cpu_time=toc;
            cpu_time_average = cpu_time_average + cpu_time/Nubset;
            VV=zeros(M,K);
            VV(:,1)=[zeros(91,1);V(:,1)/norm(V(:,1));[zeros(37,1)]];
            VV(:,2)=[zeros(31,1);V(:,2)/norm(V(:,2));[zeros(97,1)]];
            VV(:,3)=[zeros(115,1);V(:,3)/norm(V(:,3));[zeros(13,1)]];

            %%% Quant results


            SNR_motifs = zeros(K,1);
            optimal_transport_Uk = zeros(K,1);
            corr_atoms = zeros(K,1);


            SNR_motifs(1,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,2)),max(snr(sum(X2(:,:,nub),2),xtotk(:,2)),snr(sum(X3(:,:,nub),2),xtotk(:,2))));
            SNR_motifs_average(1,1) = SNR_motifs_average(1,1)+SNR_motifs(1,1)/Nubset;
            SNR_motifs(2,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,1)),max(snr(sum(X2(:,:,nub),2),xtotk(:,1)),snr(sum(X3(:,:,nub),2),xtotk(:,1))));
            SNR_motifs_average(2,1) = SNR_motifs_average(2,1)+SNR_motifs(2,1)/Nubset;
            SNR_motifs(3,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,3)),max(snr(sum(X2(:,:,nub),2),xtotk(:,3)),snr(sum(X3(:,:,nub),2),xtotk(:,3))));
            SNR_motifs_average(3,1) = SNR_motifs_average(3,1)+SNR_motifs(3,1)/Nubset;

            Fv = abs(fftreal(v));
            FV = abs(fftreal([V2 V1 V3]));
            Fv = Fv./max(Fv);
            FV = FV./max(FV);
            FV = FV(FV>0);
            Fv = Fv(FV>0);
            Tmp = (FV)./(Fv) - log((FV)./(Fv) ) -1;
            Tmp(isnan(Tmp)) = 0;
            DIS_atoms_spectrum = sum(Tmp(:));
            DIS_atoms_spectrum_average = DIS_atoms_spectrum_average + DIS_atoms_spectrum/Nubset;
            for kk=1:K
                corr_atoms(kk)= max(max(max(xcorr(VV(:,1),v(:,kk)/norm(v(:,kk)))),max(xcorr(VV(:,2),v(:,kk)/norm(v(:,kk))))),max(xcorr(VV(:,3),v(:,kk)/norm(v(:,kk)))));
            end
            corr_atoms_average=sum(corr_atoms,1)/K;
            SNR_atoms_average=10*log10(1/(2-2*corr_atoms_average));

            flat = @(x)x(:);
            Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
                flat(reshape(1:n0*n1,n0,n1) ), ...
                ones(n0*n1,1) );
            Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
                flat(reshape(1:n0*n1,n0,n1)' ), ...
                ones(n0*n1,1) );
            Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];




            maxit = 1e4; tol = 1e-9;
            otransp = @(C,p0,p1)reshape( perform_linprog( ...
                Sigma(length(p0),length(p1)), ...
                [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );

            [a1,b1] = meshgrid(nn,nn);
            Cost_opti_transp = abs(a1-b1).^2;

            % opti transport U global

            p0 = sum(U_est,2)/norm(sum(U_est,2),1);
            p1 = sum(U,2)/norm(sum(U,2),1);

            gamma = otransp(Cost_opti_transp,p0,p1);

            fprintf('Number of non-zero: %d (N+N-1=%d)\n', full(sum(gamma(:)~=0)), N+N-1);
            fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));

            [I,J,gammaij] = find(gamma);
            Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;

            optimal_transport_U = Wasserstein_Distance_2;
            optimal_transport_U_average = optimal_transport_U_average + optimal_transport_U/Nubset;
            % opti transport Uk on each motif
            index = zeros(k,1);
            for kk=1:k
                WD2old = inf;
                for kkk=1:k
                    p0 = U_est(:,kkk)/norm(U_est(:,kkk),1);
                    p1 = U(:,kk)/norm(U(:,kk),1);
                    gamma = otransp(Cost_opti_transp,p0,p1);
                    [I,J,gammaij] = find(gamma);
                    Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;
                    if Wasserstein_Distance_2<WD2old
                        WD2old = Wasserstein_Distance_2;
                        index(kk) = kkk;
                    end
                end
                optimal_transport_Uk(kk) = WD2old;
                optimal_transport_Uk_average(kk,1) = optimal_transport_Uk_average(kk,1) + optimal_transport_Uk(kk)/Nubset;
            end

        end
        optimal_transport_Uk_average_average = sum(optimal_transport_Uk_average)/3;
        SNR_motifs_average_average = sum(SNR_motifs_average)/3;
    elseif corr==0.99
        load('dataw99.mat')
        SNR_motifs_average = zeros(k,1);
        optimal_transport_Uk_average = zeros(k,1);
        optimal_transport_U_average = 0;
        DIS_atoms_spectrum_average = 0;
        cpu_time_average=0;
        lead_field = G;
        Nubset=1;
        for nub=1:Nubset
            % Sensors Gaussian noising
            snrlvl = iSNR;
            XXsensors = Xsensors(:,:,nub);
            sigma2 = var(XXsensors(:))*10^(-snrlvl/10); % add noise
            XXsensors = XXsensors' + (sqrt(sigma2)*randn(size(XXsensors)))';
            tic

            % Gabor parameters
            M = 256;
            a = M/2;
            g = gabwin({'tight', 'hann'}, a, M);
            G = dgtreal(XXsensors, g, a, M);
            [nf,nt,nq] = size(G);


            % TF operator
            op.analysis = @(x) dgtreal(x,g,a,M);
            op.synthesis = @(x) idgtreal(x,g,a,M,T);


            Gamma = op.analysis(XXsensors);
            G = abs(Gamma);

            k = 3;
            options.creation = 1;
            options.max_it = 1000;
            options.T = T;
            options.M = M;
            options.a = a;
            options.Xsensors = Xsensors(:,:,nub);

            % Ini SVD
            [U_svd,D_svd,S_svd] = svd(sum(G,3),"econ");
            S_svd = sqrt(D_svd)*S_svd';
            U_svd = U_svd*sqrt(D_svd);
            W_init = abs(U_svd(:,1:k));
            H_init = abs(S_svd(1:k,:))';
            C_init = ones(nq,k);


            %%% IS-NTF %%%

            [W, H, C] = IS_NTF(G,W_init,H_init,C_init,options);

            [alphak,atom] = get_alphak_by_Wiener_filter(W, H', Gamma, k, options, op);
            Alphak = reshape(sum(alphak,3),[nf nt k]);


            v_init = atom;
            normV = sqrt(sum(abs(v_init).^2,1));
            v_init = v_init./normV;
            % u_init=C;
            % normU = sqrt(sum(abs(u_init).^2,1));
            % u_init = u_init./normU;
            u_init =C_init;
            z_tmp = H;
            [~,nfr] = size(H');
            %z_init = ifftreal(fftreal(H),T);
            z_init = zeros(T,k);
            for l=2:nfr
                z_init(a*(l-2)+1,:) = z_tmp(l,:);
            end
            M = length(v_init(:,1));

            MaxLambda = MaxLambda_multi_cdl(XXsensors,z_init,u_init,v_init);
            lambda = logspace(log10(MaxLambda)-0.5,log10(MaxLambda)-5,10);
            %lambda = logspace(1,-4,15);
            pos_z = 1;
            pos_u = 0;
            options.max_it = 1000;
            options.max_it_cdl = 5;
            options.max_it_csc = 15;
            options.oracle = 1;

            %%% CDl Multivariate rank-1

            [xk, v, u, z2] = M1csc_dl(XXsensors,z_init,v_init, u_init,lambda,pos_z,pos_u,k,options);

            x = sum(xk,3);
            xtotk = reshape(sum(xk,2),[T k]);
            V1=[zeros(91,1);V(:,1)/norm(V(:,1),'inf');[zeros(37,1)]];
            V2=[zeros(31,1);0.5*V(:,2)/norm(V(:,2),'inf');[zeros(97,1)]];
            V3=[zeros(115,1);V(:,3)/norm(V(:,3),'inf');[zeros(13,1)]];
            %%% Results




            %% Inverse problem for u and reconstruction of the sources


            Nbit = 100;
            L = norm(lead_field)^2;
            Nblambda2 = 20;
            A=lead_field'*u;
            EnerA = zeros(20,1);
            for h=1:N
                EnerA(h,1) = norm(A(h,:));
            end
            lambda2Max = 0.5*norm(A,'inf');
            lambda2Min = 1e-5*lambda2Max;
            lambda2Vect = logspace(log10(lambda2Max),log10(lambda2Min),Nblambda2);
            lambda2number = 0;

            U_est = lead_field'*u;
            U_est = max(U_est,0);


            for lambda=lambda2Vect % warm restart: U_est is initialized by previous result
                lambda2number = lambda2number + 1;
                lambdaOnL = lambda/L;
                z_est = U_est;
                for it=0:Nbit-1
                    U_est_old = U_est;
                    U_est = z_est + lead_field'*(u-lead_field*z_est)/L;
                    U_est = U_est.*max(0,1-lambdaOnL./(abs(U_est)+eps));
                    z_est = U_est + (it)/(it+5)*(U_est-U_est_old);
                end
                U_est = max(U_est,0);
            end

            cpu_time=toc;
            cpu_time_average = cpu_time_average + cpu_time/Nubset;
            VV=zeros(M,K);
            VV(:,1)=[zeros(91,1);V(:,1)/norm(V(:,1));[zeros(37,1)]];
            VV(:,2)=[zeros(31,1);V(:,2)/norm(V(:,2));[zeros(97,1)]];
            VV(:,3)=[zeros(115,1);V(:,3)/norm(V(:,3));[zeros(13,1)]];

            %%% Quant results


            SNR_motifs = zeros(K,1);
            optimal_transport_Uk = zeros(K,1);
            corr_atoms = zeros(K,1);


            SNR_motifs(1,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,2)),max(snr(sum(X2(:,:,nub),2),xtotk(:,2)),snr(sum(X3(:,:,nub),2),xtotk(:,2))));
            SNR_motifs_average(1,1) = SNR_motifs_average(1,1)+SNR_motifs(1,1)/Nubset;
            SNR_motifs(2,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,1)),max(snr(sum(X2(:,:,nub),2),xtotk(:,1)),snr(sum(X3(:,:,nub),2),xtotk(:,1))));
            SNR_motifs_average(2,1) = SNR_motifs_average(2,1)+SNR_motifs(2,1)/Nubset;
            SNR_motifs(3,1) = max(snr(sum(X1(:,:,nub),2),xtotk(:,3)),max(snr(sum(X2(:,:,nub),2),xtotk(:,3)),snr(sum(X3(:,:,nub),2),xtotk(:,3))));
            SNR_motifs_average(3,1) = SNR_motifs_average(3,1)+SNR_motifs(3,1)/Nubset;

            Fv = abs(fftreal(v));
            FV = abs(fftreal([V2 V1 V3]));
            Fv = Fv./max(Fv);
            FV = FV./max(FV);
            FV = FV(FV>0);
            Fv = Fv(FV>0);
            Tmp = (FV)./(Fv) - log((FV)./(Fv) ) -1;
            Tmp(isnan(Tmp)) = 0;
            for kk=1:K
                corr_atoms(kk)= max(max(max(xcorr(VV(:,1),v(:,kk)/norm(v(:,kk)))),max(xcorr(VV(:,2),v(:,kk)/norm(v(:,kk))))),max(xcorr(VV(:,3),v(:,kk)/norm(v(:,kk)))));
            end
            corr_atoms_average=sum(corr_atoms,1)/K;
            SNR_atoms_average=10*log10(1/(2-2*corr_atoms_average));

            flat = @(x)x(:);
            Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
                flat(reshape(1:n0*n1,n0,n1) ), ...
                ones(n0*n1,1) );
            Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
                flat(reshape(1:n0*n1,n0,n1)' ), ...
                ones(n0*n1,1) );
            Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];




            maxit = 1e4; tol = 1e-9;
            otransp = @(C,p0,p1)reshape( perform_linprog( ...
                Sigma(length(p0),length(p1)), ...
                [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );

            [a1,b1] = meshgrid(nn,nn);
            Cost_opti_transp = abs(a1-b1).^2;

            % opti transport U global

            p0 = sum(U_est,2)/norm(sum(U_est,2),1);
            p1 = sum(U,2)/norm(sum(U,2),1);

            gamma = otransp(Cost_opti_transp,p0,p1);

            fprintf('Number of non-zero: %d (N+N-1=%d)\n', full(sum(gamma(:)~=0)), N+N-1);
            fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));

            [I,J,gammaij] = find(gamma);
            Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;

            optimal_transport_U = Wasserstein_Distance_2;
            optimal_transport_U_average = optimal_transport_U_average + optimal_transport_U/Nubset;
            % opti transport Uk on each motif
            index = zeros(k,1);
            for kk=1:k
                WD2old = inf;
                for kkk=1:k
                    p0 = U_est(:,kkk)/norm(U_est(:,kkk),1);
                    p1 = U(:,kk)/norm(U(:,kk),1);
                    gamma = otransp(Cost_opti_transp,p0,p1);
                    [I,J,gammaij] = find(gamma);
                    Wasserstein_Distance_2 = diag(Cost_opti_transp(I,J))'*gammaij;
                    if Wasserstein_Distance_2<WD2old
                        WD2old = Wasserstein_Distance_2;
                        index(kk) = kkk;
                    end
                end
                optimal_transport_Uk(kk) = WD2old;
                optimal_transport_Uk_average(kk,1) = optimal_transport_Uk_average(kk,1) + optimal_transport_Uk(kk)/Nubset;
            end

        end
        optimal_transport_Uk_average_average = sum(optimal_transport_Uk_average)/3;
        SNR_motifs_average_average = sum(SNR_motifs_average)/3;
    else
        fprintf('Data not generated for this corr please choose among the list')
    end
else
    frpintf('you have to choose a method between 1 (IPFirst) and 2 (CDLFirst)')
end

end






