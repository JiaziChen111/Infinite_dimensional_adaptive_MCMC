%%
%
%   Generating the tables in the first example in
%   Infinite dimensional AMCMC for Gaussian process, S Vadlamani and J Wallin
%   
%   The example is a simple loigt with Gaussian process prior
%   D: 2018-02-03
%%

close all
clear all
addpath ../likelihood
addpath ../sampler/
addpath ../GP/
addpath ../util/


methods = {'CN_cm_s'};{'CN','CN_cm_s','CN_pre_cs','CN_cm_cs' ,'CNL_pre','CNL_cm','AuxMarg'};
datasets = {'australian'};%{'australian','heart','german','pima','ripley'};
sim  = 300000;
%%
%
%%
summary_datas = cell(length(datasets),1);
for jj = 1:length(datasets)

    summary_data =  zeros(length(methods),5);
    for j = 1:length(methods);
        method = methods{j};
        dataset = datasets{jj};
        %Load and prepare train & test data
        if strcmp(dataset,'australian')
            load('../data/australian.mat');
            t=X(:,end);
            X(:,end)=[];  
            param = [exp(2.2433), 2*exp(3.5178)];

        elseif strcmp(dataset,'heart')
            load('../data/heart.mat');
            t=X(:,end);
            X(:,end)=[];

            t(t==1) = 0;
            t(t==2) = 1;   
            param = [exp(1.7622), 2*exp(2.6489)];

        elseif strcmp(dataset,'german')
            load('../data/german.mat');  
            t=X(:,end);
            X(:,end)=[];
            % German Credit - replace all 1s in t with 0s
            t(t==1) = 0;
            % German Credit - replace all 2s in t with 1s
            t(t==2) = 1;
            param = [exp(1.1269), 2*exp(3.38)];
        elseif strcmp(dataset,'pima')
            load('../data/pima.mat');
            t=X(:,end);
            X(:,end)=[];  
            param = [exp(0.04492), 2*exp(2.2538)];
        elseif strcmp(dataset,'ripley')
            load('../data/ripley.mat');
            t=X(:,end);
            X(:,end)=[];
            param = [exp(2.9285), 2*exp(-0.3725)];
        end
        Y = 2*t - 1;
        [n, D] = size(X);

        % normalize the data so that the inputs have unit variance and zero mean  
        meanX = mean(X);
        sqrtvarX = sqrt(var(X)); 
        X = X - repmat(meanX, n, 1);
        X = X./repmat(sqrtvarX, n, 1);

        model = GPCreate('gauss', X,param);
        %%
        % option needed for scale adaption
        %%
        model.Y = Y;
        model.X = zeros(n,1);
        model.A = speye(n);
        model.E_mean = zeros(size(model.X));
        model.addapt_count_mean = 1;
        model.addapt_count      = 1;
        model.D = ones(size(model.X));
        model.likelihood  = @ logLSigmoid;

        mcmcoptions.T = sim;
        mcmcoptions.Burnin = 20000;
        mcmcoptions.StoreEvery = 2;
        mcmcoptions.Langevin = 0;
        model.adaptive = 'eigen';   
        rng(5) % setting the seed
        tic
        
        if strcmp(method,'AuxMarg')
         [model_out, samples, accRates] = AuxMarg(model, mcmcoptions);
        elseif strcmp(method,'CN')
         [model_out, samples, accRates] = CN(model, mcmcoptions);   
        elseif strcmp(method,'CN_cm_s')
         [model_out, samples, accRates] = CN_cmeasure_s(model, mcmcoptions); 
        elseif strcmp(method,'CN_cm_cs')
         [model_out, samples, accRates] = CN_cmeasure_cs(model, mcmcoptions); 
        elseif strcmp(method,'CN_pre_cs')
         [model_out, samples, accRates] = CN_preconditoned_cs(model, mcmcoptions); 
        elseif strcmp(method,'CNL_pre')
         [model_out, samples, accRates] = LCN_precondtioned(model, mcmcoptions); 
        elseif strcmp(method,'CNL_cm') 
         [model_out, samples, accRates] = LCN_cmeasure(model, mcmcoptions); 
        else
            fprintf('error\n');
            samples = [];
        end
         elapsedTime = toc;
        samples.F = samples.X;
        summarypCN = summaryStatistics(samples);
        summarypCN.elapsed = elapsedTime; 
        summarypCN.accRates = accRates;
        summarypCN.delta = model_out.delta; 
        summarypCN.eff_LogL = mcmc_ess(samples.LogL(mcmcoptions.Burnin+1:end));
        fprintf('time = %.2f\n',elapsedTime);
         fprintf(' mean(logL) = %.2e, \n min(eff_F)/sec = %.2e\n', ...
                    mean(samples.LogL(mcmcoptions.Burnin+1:end)), ...
                    min(summarypCN.eff_F)/elapsedTime);
         summary_data(j,:) = [min(summarypCN.eff_F)/elapsedTime,
                              min(summarypCN.eff_F)/length(samples.LogL), 
                              median(summarypCN.eff_F)/elapsedTime, 
                              median(summarypCN.eff_F)/length(samples.LogL),
                              model_out.beta];
%          figure(1)
%          subplot(211)
%          plot(model_out.D)
%          hold on
%          title('scaling D');
%          subplot(212)
%          plot(model.A(1,:)*samples.X(mcmcoptions.Burnin+1:end,:)')
%          title('samples of first obs');
    end
    summary_datas{jj} = summary_data;
end
summary_datas{jj}
%AUS
%summary_data =
% 146.0573    0.0552  365.7625    0.1383    0.9177
%  113.2105    0.0342  212.0991    0.0641    1.0000
%   81.9706    0.0446  190.1171    0.1035    0.9477
%    2.7869    0.0008    7.2245    0.0020    0.1307