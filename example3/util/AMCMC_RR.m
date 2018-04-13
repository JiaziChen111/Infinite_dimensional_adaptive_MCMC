function [beta,  acc_vec]  = AMCMC_RR(beta, acc_vec, count, batch, accepantce_rate, max_1, delta_min, delta_rate)
%%
% adapting sigma from Roberts a Rosenthal
%
%
%%

if nargin < 4
    batch = 50;
end
if nargin < 5
    accepantce_rate = 0.234;
end
if nargin < 6
    max_1 = 0;
end
if nargin < 7
    delta_min =0.01;
end
if nargin < 8
    delta_rate = 3/7 ;
end

if mod(count , batch) == 0
    delta_mod = min(delta_min, (count/batch)^(-delta_rate));
    if max_1==1
        delta = log(beta) - log(1 - beta);
    else
        delta = log(beta);
    end
    if abs(acc_vec/batch - accepantce_rate) > 0.06
        if acc_vec/batch > accepantce_rate
           delta = delta + delta_mod;
        else
            delta = delta - delta_mod;
        end
    end
    if max_1==1
        beta = exp(delta)/(1+exp(delta));
    else
        beta = exp(delta);
    end
    acc_vec = 0;  
end