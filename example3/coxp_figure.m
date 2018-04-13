%%
%   After running coxp_test this produces the figures used in the data
%
%
%
%
% 
%   author: Jonas Wallin (2018-03-22)
%%
close all
addpath '../util/'
save_fig = 0;
datadir = 'storedruns/';
figdir  = 'figure/';

data_CN    =  load([datadir,'sampleData_CN_cm_cs.mat']);
data_CNL   =  load([datadir,'sampleData_CNL_cm.mat']);
data_MALA   =  load([datadir,'sampleData_MALA.mat']);
max = 1100; % longest acf
names = {'sigma','tau'};
data_CN.burnin= data_CN.burnin *1.5;
data_MALA.burnin= data_MALA.burnin *1.5;
data_CNL.burnin= data_CNL.burnin *1.5;
for i=1:2
    a1   = autocorr(exp(data_CN.sigma_tau_vec(data_CN.burnin:end, i)),max);
    a1_2 = autocorr(exp(data_CNL.sigma_tau_vec(data_CNL.burnin:end,i)),max);
    a1_3 = autocorr(exp(data_MALA.sigma_tau_vec(data_MALA.burnin:end,i)),max);
    figure()
    plot((0:max) * data_CN.time/data_CN.sim,a1','r')
    hold on
    plot((0:max) * data_CNL.time/data_CNL.sim, a1_2,'--b')
    plot((0:max) * data_MALA.time/data_MALA.sim, a1_3,'.k')
    ylim([0.0,1])
    xlim([0,6])
    %title(['ACF for ',names{i}],'Interpreter','latex');
    xlabel('seconds')
    ylabel('ACF')
    tightfig()
    if save_fig == 1
        name = [figdir, names{i}];
        print(gcf, name, '-dpng')
        print(gcf, name, '-dpdf')
        print(gcf, name, '-deps')
        system(sprintf('%s%s%s','convert -trim ', name,'.png ', name,'.png'))
        system(sprintf('%s%s%s','convert -trim ', name,'.pdf ', name,'.pdf'))
    end
end

datas = {data_CN,data_CNL, data_MALA};
names = {'CN','CNL','MALA'};
for i = 1:length(datas)
    fig = figure();
    imagesc(datas{i}.ACF_1,[-0.1,1])
    %ylabel('$ACF(50)$','interpreter','latex','fontsize',22)
    colorbar()
    %title(['ACF(100) for ', names{i}])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    fig2 = tightfig(fig);
    if save_fig == 1
        name = [figdir,'ACF_100_', names{i}];
        print(gcf, name, '-dpng')
        print(gcf, name, '-dpdf')
        print(gcf, name, '-deps')
        system(sprintf('%s%s%s','convert -trim ', name,'.png ', name,'.png'))
        system(sprintf('%s%s%s','convert -trim ', name,'.pdf ', name,'.pdf'))
    end
    fig = figure();
    imagesc(datas{i}.ACF_5,[-0.1,1])
    colorbar()
    %title(['ACF(500) for ', names{i}])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    %ylabel('$ACF(250)$','interpreter','latex','fontsize',22)
    fig2 = tightfig(fig);
    if save_fig == 1
        name = [figdir,'ACF_500_', names{i}];
        print(gcf, name, '-dpng')
        print(gcf, name, '-dpdf')
        print(gcf, name, '-deps')
        system(sprintf('%s%s%s','convert -trim ', name,'.png ', name,'.png'))
        system(sprintf('%s%s%s','convert -trim ', name,'.pdf ', name,'.pdf'))
    end
end
