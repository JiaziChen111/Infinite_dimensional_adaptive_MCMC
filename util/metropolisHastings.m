function [accept, A] = metropolisHastings(newLogPx, oldLogPx, newLogProp, oldLogProp)
%
%Desctiption:  The general Metropolis-Hastings step 
%

A = newLogPx + oldLogProp - oldLogPx - newLogProp;


accept = 0;
u = rand;
if log(u) < A
   accept = 1;
end

          