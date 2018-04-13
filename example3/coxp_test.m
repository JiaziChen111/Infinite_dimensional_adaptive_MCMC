%%
%  testing preformance with joint proposal taking an example from
%  the lgcmp package, 
%  a log gaussian cox processes
%
%  to store the data set save=1; 
%  after the data is stored run coxp_figure to generate figures from
%  the article
%
% 
%   author: Jonas Wallin (2018-03-22)
%%

close all
clearvars
addpath util
rng(3)
save_data = 1;
methods = {'MALA'}; %{'CNL_cm','MALA','CN_cm_cs'};

debug = 0;
datadir = 'storedruns/';

adapt = 1;
joint = 0;

sim  = 10^5;
alpha= 2-0.5;
burnin = max(2*10^4, ceil(sim/5));

if debug==0
    N = 128/2;
    M = 256/2;
    Y      = load('../data/Y.dat');
    Yindex = load('../data/spatial_Y.dat');
    rl       = zeros(M, N);
    Yindex = sub2ind(size(rl), Yindex(:,1), Yindex(:,2));
    Ymat = zeros(M, N);
    Ymat(Yindex) =1;
    rl(Yindex) = Y;

    meanY    = load('../data/offset.txt');
    cellArea = 1;%90000; 
else
    N      = 2^4;
    M      = 2^3;
    Y      = poissrnd(2, M/2 * N/2, 1);
    rl     = reshape(Y, M/2, N/2);
    Ymat   = ones(M/2, N/2);
    meanY  = log(2) * ones(N*M/4, 1);
    
    cellArea = 1;
end
n1 = size(rl, 1)*2;
n2 = size(rl, 2)*2;
n1_1 = n1/4;
n2_1 = n2/4;
M  = n1;
N  = n2;
A = speye(n1*n2);
A_grid = speye(n1*n2);
A_mat = zeros(n1,n2);
A_mat((n1_1+1):(end-n1_1), (n2_1+1):(end-n2_1)) = Ymat;
A(A_mat(:)==0,:) = [];

A_mat((n1_1+1):(end-n1_1), (n2_1+1):(end-n2_1)) = 1;
A_grid(A_mat(:)==0,:) = [];

A_Y = zeros(n1,n2);
A_Y((n1_1+1):(end-n1_1), (n2_1+1):(end-n2_1)) = rl;
y = A * A_Y(:);



for m = 1:length(methods)
 

    method = methods{m};
    if strcmp(method, 'CN') || strcmp(method, 'CN_cm_cs')
       accepantce_rate = 0.2; 
    else
       accepantce_rate = 0.5; 
    end
    if strcmp(method, 'MALA')
        beta = 0.001;
    else
        beta = 0.1;
    end
    
    
   sigma_tau = [log(0.1) log(50)]';
    mean_st = [log(0.1);log(50)];

    sigma = exp(sigma_tau(1));
    tau   = exp(sigma_tau(2));


    Z = randn(n1*n2,1);
    [U, sqrt_lambda]= E(exp(sigma(1)), alpha,exp(sigma_tau(2)), reshape(Z, n1, n2));
    U = U(:);
    [phiU, grad] = potential(y, A, U(:), meanY, cellArea);

    AMCMC.count     = 0;
    AMCMC.E_var     = zeros(length(U), 1);
    AMCMC.n_adpt    = 20;
    AMCMC.D         = ones(length(U) , 1);
    AMCMC.Z_mean    = zeros(length(U), 1);
    AMCMC.E_mean    = zeros(length(U), 1);
    AMCMC.E2_mean    = zeros(length(U), 1);
    AMCMC.D2  = 1./AMCMC.D.^2;
    AMCMC.Q   = 1 - AMCMC.D2;
    AMCMC.batch = 5;
    AMCMC.beta = beta;
    AMCMC.a = 0 ;
      [K1, K2] = meshgrid(0:n1-1,0:n2-1);
    %K1 = K1;
    %K2 = K2;
    coeff = (pi^2*(K1.^2+K2.^2) + 1^2).^(-alpha/2);
    coeff = coeff(:);
    [B,I] = sort(coeff,'descend');

    if strcmp(method, 'CNL_cm')
        [gradZ] = Et(sigma, alpha,tau, reshape(grad, n1, n2));
        Zmean =  gradZ(:);
        Zmean_f = Zmean;
    elseif  strcmp(method, 'MALA')
        [gradZ] = Et(sigma, alpha,tau, reshape(grad, n1, n2));
        Zmean =  0.5 * gradZ(:);
        Zmean_f = Zmean;
        Z_m = Z + 0.5*AMCMC.beta^2*(AMCMC.D2.* gradZ(:));
        Zs = randn(n1*n2, 1);
        Z  =    Z_m + AMCMC.beta*(AMCMC.D.*Zs);
        [Uprop] = E(exp(sigma_tau(1)),    ...
                    alpha,                ...
                    exp(sigma_tau(2)),    ...
                    reshape(Z, n1, n2), ...
                    K1,                   ...
                    K2);
        [phiUProp, gradUprop] = potential(y, A, Uprop(:), meanY, cellArea);
        
        [gradZ] = Et(sigma,                      ...
                            alpha,                       ...
                            tau,                         ...
                            reshape(gradUprop, n1, n2),  ...
                            K1,                          ...
                            K2);
          Z_m     = Z + 0.5*AMCMC.beta^2*(AMCMC.D2.* gradZ(:));
    end

    acc = zeros(sim,1);
    %% simple pcN
    N = n1 * n2;
    count_s= 0 ;
    Sigma12_st = eye(2);
    %[d,I]=sort(Lambda_sqrt(:),'descend');
    acc_RR = 0;
  
    acc_RR2 = 0;

    U1  = [];
    pot = [];
    ts = zeros(2,2);
    ts_m = zeros(2,1);
    sigma_tau_vec  = zeros(sim, 2);
    count_acf = 0;
    acc_vec_hyper = zeros(sim,1);
    beta2=0.1;
    EUU_5  = zeros(n1/2, n2/2);
    EUU_1  = zeros(n1/2, n2/2);
    meanU  = zeros(n1/2, n2/2);
    meanU2 = meanU;
    Uprev = zeros(n1/2, n2/2);
    Uprev2 = Uprev;
    Uprev3 = Uprev;
    Uprev4 = Uprev;
    Uprev5 = Uprev;
    AMCMC.invLambda = 1./sqrt_lambda.^2;
    tic
    for i =1:sim
        if(mod(i,1000)==0)
            fprintf('(i,beta, acc) = (%d,[%.4f %.4f], %.2f)\n',i, beta, beta2, mean(acc(i-500:i)));
        end

        if joint
            proposal_joint;
        else
           proposal_gibbs; 
        end



        U1 = [U1;A(1,:)*U(:)];
        pot = [pot;phiU];


        if mod(i,10) == 0 
                if  i > burnin/4
                    count_s = count_s + 1;
                    w = 1/count_s;
                    X = sigma_tau;
                    ts_m = (1-w) * ts_m + w * X;
                    ts   = (1-w) * ts   + w * (X*X');
                end
                if  i > burnin/2
                        mean_ts = ts_m;
                        V = ts - ts_m * ts_m';
                        Sigma12_st =  chol(0.5*(V + V'))'; % sqrt(2.1608)*

                end
        end

        sigma_tau_vec(i,:) = sigma_tau;
        if adapt 
            [AMCMC] = AMCMC_D(Z, i, AMCMC, I,burnin);
        end


        %%
        % computing the ACF for the eniter grid
        %%
        if mod(i,50) == 0 && i > burnin 
            count_acf = count_acf + 1;
            AU =  reshape(A_grid*U(:),n1/2,n2/2);

            meanU = meanU + AU;
            meanU2 = meanU2 + AU.^2;
            if count_acf > 5
               EUU_5 =  EUU_5 +  AU.*Uprev5;
            end
            if count_acf > 2
               EUU_1 =  EUU_1 +  AU.*Uprev;
            end

            Uprev5 = Uprev4;
            Uprev4 = Uprev3;
            Uprev3 = Uprev2;
            Uprev2 = Uprev;
            Uprev = AU;
        end

    end
    %%
    % compute acf
    %%
    meanU = meanU/count_acf;
    meanU2 = meanU2/count_acf;
    EUU_5 = EUU_5/(count_acf-6);
    EUU_1 = EUU_1/(count_acf-2);
    ACF_5 = EUU_5 - meanU.^2;
    ACF_1 = EUU_1 - meanU.^2;
    ACF_5 = ACF_5./(meanU2 - meanU.^2);
    ACF_1 = ACF_1./(meanU2 - meanU.^2);
    time = toc;
    fprintf('acc = %.2f\n',mean(acc));
    

    figure()
    subplot(211)
    imagesc(ACF_1,[-1,1])
    subplot(212)
    imagesc(ACF_5,[-1,1])

 
    figure()
    subplot(311)
    autocorr(U1(burnin:end),1000);
    subplot(312)
    autocorr(exp(sigma_tau_vec(burnin:end,1)),1000);
    subplot(313)
    autocorr(exp(sigma_tau_vec(burnin:end,2)),1000);
    title(method);

    if save_data
        save([datadir,'sampleData_',method,'.mat'], ...
              'sigma_tau_vec', ...
              'ACF_1',         ...
              'ACF_5',         ...
              'time',          ...
              'burnin',        ...
              'U1',            ... 
              'sim');
    end

end
