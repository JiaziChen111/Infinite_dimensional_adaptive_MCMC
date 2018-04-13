function A=build_A(FV,P,loc,FF)
% A=build_A(FV,P,loc,FF)
%	Function that evaluates FEM basis function at locations loc
%
%	INPUTS:
%	FV		: Facet-vertex matrix for 2D problem
%	P     : mesh nodes
% loc   : locations where the functions should be evaluated
%	FF    : Facet-facet matrix for 2D problems
%	OUTPUTS:
%	A	: Matrix containing elements A_ij = phi_i(loc_j)

if(min(size(P))==1)
    if(min(loc)< min(P) || max(loc) > max(P))
      error('locations outside support of basis')
    end

    nx  = length(P);
    nloc = length(loc);
    i = [1:nloc;1:nloc]';
    j = zeros(nloc,2);
    vals = ones(nloc,2);
    for ii =1:nloc
      j(ii,1) = sum(sum((loc(ii) - P)>=0));
      vals(ii,1) = loc(ii) - P(j(ii,1));
      j(ii,2) = j(ii,1) + 1;
      if(j(ii,2)<=nx)
        vals(ii,2) = P(j(ii,2)) - loc(ii);
      else
        j(ii,2) = j(ii,2) -2;
      end
    end
    vals = 1-bsxfun(@rdivide,vals,sum(vals,2));
    A = sparse(i(:),j(:),vals(:),nloc,nx);
  else
    P =  P';
    if (nargin<4), FF = []; end
    if isempty(FF), FF = fv2ff(FV); end
    nF = size(FV,1);

    Pinit = [0 0]';
    for k=1:size(Pinit,2)
      f = ceil(rand(1,1)*nF);
      finit(k,1) = point_locate(FV,P,Pinit(:,k),f,FF);
    end

    f_ = zeros(length(loc),1);
    f_dist = zeros(length(loc),1);
    w_ = zeros(length(loc),3);
    A_build = zeros(3*length(loc),3);

    fspecial = finit(1);
    Pspecial = Pinit(:,1);
    f_init = finit([1,1:end,1]);
    P_init = Pinit(:,[1,1:end,1]);
    for k=1:length(loc)
      P0 = loc(k,:)';
      [tmp,f_idx] = max(P0'*P_init);
      f = f_init(f_idx);
      [f,w,fdist] = point_locate(FV,P,P0,f,FF);

      if (f_idx>1)
        fspecial = f;
        Pspecial = P0;
      end
      f_init = [f;finit;fspecial];
      P_init = [P0,Pinit,Pspecial];

      f_(k) = f;
      f_dist(k) = fdist;
      w_(k,:) = w';
      A_build(3*(k-1)+(1:3),:) = [k*ones(3,1) FV(f,:)' w];
    end
    A = sparse(A_build(:,1),A_build(:,2),A_build(:,3),length(loc),size(P,2));
  end
end

function [f,w,f_seq_len,f_sequence]=point_locate(FV,P,pt,f,FF)

if (nargin<4), f = []; end
if (nargin<5), FF = []; end
if isempty(f), f = ceil(rand(1,1)*size(FV,1)); end
if isempty(FF), FF = fv2ff(FV); end

d = size(P,1);

p_search = pt;
f_sequence = f;

not_found = true;
while (not_found)
  PT = P(:,FV(f,:));
  ET = PT(:,[3,1,2])-PT(:,[2,3,1]);
  if (d==3)
    n = cross(ET(:,1),ET(:,2));
    n = n/sqrt(n'*n);
  end
  if (d==2)
    w = [PT;ones(1,3)]\[p_search;1];
  else
    w = [PT,n;ones(1,3),0]\[p_search;1];
    w = w(1:3);
  end
  [ws,i] = min(w);
  if (w(i(1)) >= -1e-8)
    not_found = false;
  else
    f_ = FF(f,rem(i(1),3)+1);
    if (any(f_sequence==f_))
      not_found = false;
    else
      f = f_;
      f_sequence = [f_sequence,f];
    end
  end
end

f_seq_len = length(f_sequence);
end
function C=cross(A,B)

C = [A(2)*B(3) - A(3)*B(2);...
     A(3)*B(1) - A(1)*B(3);...
     A(1)*B(2) - A(2)*B(1)];
end