%%
% Sampling simple Gaussian models using CN with adapeted scale and
% centering
% to save the figure use save_fig = 1
%%

close all
clear all
rng(4) % setting the seed
addpath ../likelihood
addpath ../sampler/
addpath ../GP/
addpath ../util/



save_fig  = 0;
load_data = 0 ;

methods = {'CN','AuxMarg','CNL_cm'};
datasets = {'australian'};

%%
%
%%
if load_data==0
    sim  = 10000;
    Burnin = 1000;
    summary_datas = cell(length(datasets),1);
    models = cell(2,1);
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

                % German Credit - replace all 1s in t with 0s
                t(t==1) = 0;
                % German Credit - replace all 2s in t with 1s
                t(t==2) = 1;   
                param = [exp(1.7622), 2*exp(2.6489)];

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
            mcmcoptions.Burnin = Burnin;
            mcmcoptions.StoreEvery = 1;
            mcmcoptions.Langevin = 0;
            model.adaptive = 'eigen';   
            tic
            if strcmp(method,'AuxMarg')
             [modelAux, samples, accRates] = AuxMarg(model, mcmcoptions);
            elseif strcmp(method,'CN')
             [modelCN, samples, accRates] = CN(model, mcmcoptions);   
            elseif strcmp(method,'CNL_cm')
             [CNL_cm, samples, accRates] = LCN_cmeasure(model, mcmcoptions); 
            else
                fprintf('error\n');
                samples = [];
            end
             elapsedTime = toc;
            samples.F = samples.X;
            summarypCN = summaryStatistics(samples);
            summarypCN.elapsed = elapsedTime; 
            summarypCN.accRates = accRates;
            summarypCN.eff_LogL = mcmc_ess(samples.LogL(mcmcoptions.Burnin+1:end));


        end
    end
    save('fig2_b.mat','CNL_cm','modelAux','modelCN')
else
    load('fig2_b.mat'); 
end

   close all
   figure(1)
   plot(CNL_cm.D_apt.^2*CNL_cm.beta.^2, 'linewidth',1.1)
   
   hold on
   D_aux = modelAux.delta * ( modelAux.delta +  4*modelAux.Lambda)./(modelAux.delta + 2*modelAux.Lambda).^2;
   plot(D_aux,'r--' ,'linewidth',1.1)
  
   plot(modelCN.beta^2 * ones(length(D_aux),1),'k.' ,'linewidth',1.1)   
    xlim([1,200])
    name = 'binary_d_scaling';
    set(gca,'FontSize',20)
    xlabel('\sigma_i')
    ylabel('scaling')
    tightfig
    if save_fig == 1
        print(gcf, name, '-dpng')
        print(gcf, name, '-dpdf')
        print(gcf, name, '-deps')
        system(sprintf('%s%s%s','convert -trim ', name,'.png ', name,'.png'))
        system(sprintf('%s%s%s','convert -trim ', name,'.pdf ', name,'.pdf'))
    end