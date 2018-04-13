%%
% The log Gaussian coxprocesses field
% 
%
%%

%%
% Sampling simple Gaussian models using CN with adapeted scale and
% centering
%
%%

close all
clear all
rng(1) % setting the seed
addpath ../likelihood
addpath ../sampler/
addpath ../GP/
addpath ../util/

methods = {'CN','CN_cm_cs','CNL_cm','AuxMarg','CNL_hm'};



sim    = 2 * 10^5;
Burnin = 2 * 10^4;

n_loc = 400; %20%200; %500
sigma = 1.5;
kappa  = 5;
loc = [0,0;0,1;1,0;1,1;rand(n_loc, 1), rand(n_loc, 1)];

model_0    = GPCreate('SPDE', loc, [sigma, kappa]);


% remove the four grid points
model_0.A = model_0.A(5:end,:);
model_0.loc = model_0.loc(5:end,:);%
trimesh(model_0.FV,model_0.P(:,1),model_0.P(:,2))
hold on
scatter(model_0.loc(:,1),model_0.loc(:,2),'r')
xlim([0,1])
ylim([0,1])

n = length(model_0.Q);
R = chol(model_0.Q);
X = R\randn(n,1);

model_0.mu  = 0.3;
p= exp(model_0.A*X + model_0.mu)./(1 + exp(model_0.A*X + model_0.mu));
model_0.nY = rndpois(3, length(p),1)+1;
model_0.Y  = rndbin(model_0.nY, p, 1);

model_0.X = zeros(size(X));
X_ = X;
X_(model_0.reo) = X;
figure()
trisurf(model_0.FV,model_0.P(:,1),model_0.P(:,2),X_);view(0,90);

    %%
    % option needed for scale adaption
    %%
    model_0.E_mean = zeros(size(model_0.X));
    model_0.D = ones(size(model_0.X));
    model_0.likelihood  = @ logLBinom;
    model_0.addapt_count_mean =0;
    model_0.addapt_count =  0;

    mcmcoptions.T = sim;
    mcmcoptions.Burnin = Burnin;
    mcmcoptions.StoreEvery = 10;
    mcmcoptions.Langevin = 0;
       model_0.adaptive= 'eigen';
    model_0.k = 0;
    summary_data =  zeros(length(methods),5);
    for i = 1:length(methods)
        method = methods{i};
        tic;
        if(strcmp(method, 'AuxMarg'))
         [model, samples, accRates] =  AuxMarg(model_0, mcmcoptions);
        elseif(strcmp(method, 'CN'))
            [model, samples, accRates] =  CN(model_0, mcmcoptions); 
        elseif(strcmp(method, 'CN_cm_cs'))
            [model, samples, accRates] =  CN_cmeasure_cs(model_0, mcmcoptions); 
        elseif(strcmp(method, 'CNL_cm'))
            [model, samples, accRates] =  LCN_cmeasure(model_0, mcmcoptions); 
        elseif(strcmp(method, 'CNL_hm'))
            [model, samples, accRates] =  LCN_hmeasure(model_0, mcmcoptions);
        end
         elapsedTime = toc;
        % 
        samples.F = samples.X;
        summarypCN = summaryStatistics(samples);
        summarypCN.elapsed = elapsedTime; 
        summarypCN.accRates = accRates;
        summarypCN.delta = model.delta; 
        summarypCN.eff_LogL = mcmc_ess(samples.LogL(mcmcoptions.Burnin+1:end));

         summary_data(i,:) = [min(summarypCN.eff_F)/elapsedTime, ...
                              min(summarypCN.eff_F)/length(samples.LogL),  ...
                              median(summarypCN.eff_F)/elapsedTime,  ...
                              median(summarypCN.eff_F)/length(samples.LogL), ...
                              model.beta];
        summary_data
    end
        
   figure()
   plot(model_0.A(7,:)*samples.X')
   hold on
   plot((model_0.A(7,:)*X)*ones(sim,1),'r');

   figure()
   plot(model_0.D)
   fprintf('mean(eff_F)/iter = %.2e\n',mean(summarypCN.eff_F)/length(samples.LogL));