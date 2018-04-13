function [A,P,FV,C,G,Aobs]=create_triangulation(xy,cutoff,maxedge,offset,minangle,obsloc)
%%
% wrapper function R to build matricies for FEM of a triangulation basis
% 
%
%
%%

dir_     = pwd;
dir_file = fileparts(mfilename('fullpath'));
cd(dir_file);
tmpdir = tempname;
system(sprintf('%s%s','mkdir ', tmpdir));
fid = fopen(sprintf('%s%s',tmpdir, '/points.bin'), 'w');
fwrite(fid,xy,'double');
fclose(fid);

pars = [cutoff maxedge offset minangle 0];

if nargin == 6
  pars(end) = 1;
  fid = fopen(sprintf('%s%s',tmpdir, '/obspoints.bin'), 'w');
  fwrite(fid,obsloc,'double');
  fclose(fid);
end

fid = fopen(sprintf('%s%s',tmpdir, '/parameters.bin'), 'w');
fwrite(fid,pars,'double');
fclose(fid);

path1 = getenv('PATH');
path1 = [path1 ':/usr/local/bin'];
setenv('PATH', path1);

system(sprintf('%s%s','unset DYLD_LIBRARY_PATH; Rscript create_triangulation.R ', tmpdir));

[A,P,FV,C,G]=load_matrices(tmpdir);
cd(dir_);

Aobs = [];
if nargin == 6
  Aobs = readData([tmpdir filesep 'Aobs.bin']);
  Aobs = convertToSparseMat(Aobs);
end

system(sprintf('%s%s','rm -r ', tmpdir));

function [A,P,FV,C,G]=load_matrices(file_path, prec)

%check if the path exists
e_path = exist(file_path);
if e_path==2
  %default precesion
  if nargin<2, prec='float64'; end
  %open file
  fid = fopen(file_path,'r');
  if fid==-1
    error('No data loaded')
  end
  %read data
  A = fread(fid, inf, prec);
  %close file
  fclose(fid);
  %...then create the sparse matrices
  A = convertToSparseMat(A);
elseif e_path==7
  %check if file exists, load the data
  A = readData([file_path filesep 'A.bin']);
  C = readData([file_path filesep 'C.bin']);
  G = readData([file_path filesep 'G.bin']);

  P = readData([file_path filesep 'P.bin']);
  P = reshape(P(3:end),int32([P(1) P(2)]));

  FV = readData([file_path filesep 'FV.int'],'int');
  FV = reshape(FV(3:end),[FV(1) FV(2)]);

  if isempty(A) && isempty(P) && isempty(FV)
    error('No data loaded')
  end
  %...then create the sparse matrices
  A = convertToSparseMat(A);
  C = convertToSparseMat(C);
  G = convertToSparseMat(G);
else
  error('file_path not valid')
end

function dat = readData(fname,prec)
fid = fopen([fname '64'],'r');
if fid==-1
  %no such file, attempt to open 32-bit version
  fid = fopen([fname '64'],'r');
  if fid==-1
    dat=[];
    return
  end
  if nargin < 2;  prec = 'float32'; end
else
 if nargin <2;  prec = 'float64'; end
end
%read data
dat = fread(fid, inf, prec);
%close file
fclose(fid);

function mat = convertToSparseMat(vec)
%see that the vector is non-empty and contains the right amount of elements
if isempty(vec) || ((vec(3)+1)*3) ~= numel(vec)
  mat=[];
  return
end
%extract data and reshape
m = vec(1); n=vec(2);
vec = reshape(vec(4:end),[vec(3) 3]);
mat = sparse(vec(:,1),vec(:,2),vec(:,3),m,n);

