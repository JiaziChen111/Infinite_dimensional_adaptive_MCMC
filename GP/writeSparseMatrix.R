writeSparseMatrix <- function(Q, file.name, size=8){
  Q.tmp <- as(Q,"dgTMatrix")
   ##add everything into a big vector
  Q <- c(dim(Q.tmp),length(Q.tmp@i),Q.tmp@i+1,Q.tmp@j+1,Q.tmp@x);
  ##write binary file
  writeBin(Q, con = file.name, size=size)
}