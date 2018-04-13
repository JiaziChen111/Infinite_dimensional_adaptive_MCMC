%%
% The log Gaussian coxprocesses field
% The figure shows the difference between the eigenvalue shrinkage of
% the estimated d, and the auxMarg  
%
%   to store figure set save_fig to one;
%
%%



close all
clear all
rng(10) % setting the seed
addpath ../likelihood
addpath ../sampler/
addpath ../GP/
addpath ../util/

save_fig  = 0;

methods = {'CN','AuxMarg','CNL_cm'};
 

load_data = 0;
if load_data == 0
    sim    = 10^4;
    Burnin = 10^3;

    n_loc = 20; %20%200; %500
    sigma = 1.5;
    kappa  = 5;
    loc = [0,0;0,1;1,0;1,1;rand(n_loc, 1), rand(n_loc, 1)];

    model    = GPCreate('SPDE', loc, [sigma, kappa]);


    % remove the four grid points
    model.A = model.A(5:end,:);
    model.loc = model.loc(5:end,:);%
    trimesh(model.FV,model.P(:,1),model.P(:,2))
    hold on
    scatter(model.loc(:,1),model.loc(:,2),'r')
    xlim([0,1])
    ylim([0,1])

    n = length(model.Q);
    R = chol(model.Q);
    X = R\randn(n,1);

    model.mu  = 0.3;
    p= exp(model.A*X + model.mu)./(1 + exp(model.A*X + model.mu));
    model.nY = rndpois(3, length(p),1)+1;
    model.Y  = rndbin(model.nY, p, 1);

    model.X = zeros(size(X));
    X_ = X;
    X_(model.reo) = X;
    figure()
    trisurf(model.FV,model.P(:,1),model.P(:,2),X_);view(0,90);

        %%
        % option needed for scale adaption
        %%
        model.E_mean = zeros(size(model.X));
        model.D = ones(size(model.X));
        model.likelihood  = @ logLBinom;
        model.addapt_count_mean =0;
        model.addapt_count =  0;

        mcmcoptions.T = sim;
        mcmcoptions.Burnin = Burnin;
        mcmcoptions.StoreEvery = 1;
        mcmcoptions.Langevin = 0;
           model.adaptive= 'eigen';
        model.k = 0;
        models =  cell(length(methods),1);
        for i = 1:length(methods)
            method = methods{i};
            tic;
            if(strcmp(method, 'AuxMarg'))
                 [modelAux, samples, accRates] =  AuxMarg(model, mcmcoptions);

            elseif(strcmp(method, 'CNL_cm'))
                [model_LCN, samples, accRates] =  LCN_cmeasure(model, mcmcoptions); 

            elseif(strcmp(method, 'CN'))
                [modelCN, samples, accRates] =  CN(model, mcmcoptions); 
            end
             elapsedTime = toc;
            % 
            samples.F = samples.X;
            summarypCN = summaryStatistics(samples);
            summarypCN.elapsed = elapsedTime; 
            summarypCN.accRates = accRates;
            summarypCN.eff_LogL = mcmc_ess(samples.LogL(mcmcoptions.Burnin+1:end));
             fprintf(' mean(logL) = %.2e, \n min(eff_F)/sec = %.2e min(eff_F)/iter = %.2e ESS_lik/sec = %.2e\n', ...
                        mean(samples.LogL(mcmcoptions.Burnin+1:end)), ...
                        min(summarypCN.eff_F)/elapsedTime, ...
                        min(summarypCN.eff_F)/length(samples.LogL),...
                    summarypCN.eff_LogL/elapsedTime);
        end
       close all
       save('fig2_a.mat','model_LCN','modelAux','modelCN')
else
   load('fig2_a.mat');  
end
   figure(1)
   plot(model_LCN.D_apt.^2*model_LCN.beta.^2, 'linewidth',1.1)
   
   hold on
   D_aux = modelAux.delta * ( modelAux.delta +  4*modelAux.Lambda)./(modelAux.delta + 2*modelAux.Lambda).^2;
   plot(D_aux,'r--' ,'linewidth',1.1)
   plot(modelCN.beta^2 * ones(length(D_aux),1),'k.' ,'linewidth',1.1)
   
    xlim([1,200])
    name = 'binomial_d_scaling';
    set(gca,'FontSize',20)
    xlabel('\sigma_i')
    ylabel('scaling')
    ylim([0,1])
    tightfig
    if save_fig == 1
        print(gcf, name, '-dpng')
        print(gcf, name, '-dpdf')
        print(gcf, name, '-deps')
        system(sprintf('%s%s%s','convert -trim ', name,'.png ', name,'.png'))
        system(sprintf('%s%s%s','convert -trim ', name,'.pdf ', name,'.pdf'))
    end