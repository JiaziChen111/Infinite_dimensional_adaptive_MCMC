source("writeSparseMatrix.R")
library(INLA)
library(methods)

tmpdir = commandArgs(TRUE)[1]

to.read = file(paste(tmpdir,"/points.bin",sep=""), "rb")
xyc = readBin(to.read, double(), n = 10000000)
close(to.read)

to.read = file(paste(tmpdir,"/parameters.bin",sep=""), "rb")
p = readBin(to.read, double(), n = 10000000)
close(to.read)

dim(xyc) <- c(length(xyc)/2,2)

m1 <- inla.mesh.create.helper(as.matrix(xyc),
                              cutoff=p[1],
                              max.edge=c(p[2],p[3]),
                              offset=c(p[4],p[5]),
                              min.angle=p[6])

A <- inla.spde.make.A(m1, loc=as.matrix(xyc))

fem = inla.fmesher.smorg(m1$loc, m1$graph$tv, fem = 2,
                         output = list("c0", "c1", "g1", "g2"))

loc <- c(dim(m1$loc),m1$loc)
writeBin(loc, con = paste(tmpdir,"/P.bin64",sep=""), size=8)
FV <-  as.integer(c(dim(m1$graph$tv),m1$graph$tv))
writeBin(FV,con = paste(tmpdir,"/FV.int64",sep=""))
writeSparseMatrix(A, paste(tmpdir,"/A.bin64",sep=""))

writeSparseMatrix(fem$c1, paste(tmpdir,"/C.bin64",sep=""))
writeSparseMatrix(fem$g1, paste(tmpdir,"/G.bin64",sep=""))

if(p[7] == 1){
  to.read = file(paste(tmpdir,"/obspoints.bin",sep=""), "rb")
  obs.loc = readBin(to.read, double(), n = 10000000)
  close(to.read)
  dim(obs.loc) <- c(length(obs.loc)/2,2)
  A.obs <- inla.spde.make.A(m1, loc=as.matrix(obs.loc))
  writeSparseMatrix(A.obs, paste(tmpdir,"/Aobs.bin64",sep=""))
}



