myFunction=function(roads,car,packages) {
  nextMove=0
  toGo=0
  offset=0
  if (car$load==0) {
    toGo = findClosestGoal(roads,car,packages)#goes to first undelivered package in array
  } else {
    toGo=car$load
    offset=2 #change from package pick up to package delivery
  }
  car$nextMove <- aStarNextMove(roads, car, packages,offset, toGo)
  return (car)
}

manhattanDistance <- function(xGoal,yGoal,xNode,yNode){
  return(abs(xGoal-xNode)+abs(yGoal-yNode))
}

findClosestGoal <- function(roads, car, packages){
  i <- 0
  distances <- list(999,999,999,999,999)
  for (j in packages[,5]){
    i= i +1
    if (j == 0){
      absx <- abs(packages[i,1]-car$x)
      absy <- abs(packages[i,2]-car$y)
      distances[i] = absx + absy
    }
  }
  return(which.min(distances))
}

aStarNextMove <- function(roads, car, packages,offset, toGo){
  expanded <- c(car$x,car$y,0,0,999)
  frontier <- identifyFrontier(expanded, roads, car, packages,offset, toGo)
  goalExpanded = FALSE
  while(!goalExpanded){
    minFIndex <- getMinF(frontier) #get min index of
    expanded <- frontier[,minFIndex] 
    frontier <- frontier[,-minFIndex]#expand and pop first item of frontier
    newFrontier <- identifyFrontier(expanded, roads, car, packages, offset, toGo)
    frontier <- appendNewFrontier(frontier, newFrontier)#append new frontier items
    lengthToGoal <- manhattanDistance(packages[toGo,1+offset],packages[toGo,2+offset], expanded[1], expanded[2])
    if(lengthToGoal == 0) {goalExpanded = TRUE}
  }
  
  nextMove <- expanded[5]
  return(nextMove)
}

identifyFrontier <- function(expanded, roads, car, packages, offset, toGo){
  xStart <- expanded[1]
  yStart <- expanded[2]
  gStart <- expanded[3]
  nextMove <- expanded[5]
  minEdgeCost <-  min(c(min(roads$hroads),min(roads$vroads)))
  frontier <- c()
  if(xStart >1) {
    x <- xStart-1
    y <- yStart
    g <- gStart + roads$hroads[xStart-1,yStart]
    h <- minEdgeCost * manhattanDistance(packages[toGo,1+offset],packages[toGo,2+offset], xStart-1,yStart) #just manhattan distance as of now...
    if(nextMove == 999) {node <- c(x,y,g,h,4)} #nextMove = 4
    else {node <- c(x,y,g,h,nextMove)}
    frontier <- array(c(frontier,node),dim = c(5,(length(frontier)/5)+1))
  }
  if(xStart <= 9) {
    x <- xStart+1
    y <- yStart
    g <- gStart + roads$hroads[xStart,yStart]
    h <- minEdgeCost*manhattanDistance(packages[toGo,1+offset],packages[toGo,2+offset], xStart+1,yStart) #just manhattan distance as of now...
    if(nextMove == 999) {node <- c(x,y,g,h,6)} #nextMove <- 6
    else {node <- c(x,y,g,h,nextMove)}
    frontier <- array(c(frontier,node),dim = c(5,(length(frontier)/5)+1))
  }
  if(yStart > 1) {
    x <- xStart
    y <- yStart-1
    g <- gStart + roads$vroads[xStart, yStart-1]
    h <- minEdgeCost*manhattanDistance(packages[toGo,1+offset],packages[toGo,2+offset], xStart, yStart-1)
    if(nextMove == 999) {node <- c(x,y,g,h,2)} #nextMove <- 2
    else {node <- c(x,y,g,h,nextMove)}
    frontier <- array(c(frontier,node),dim = c(5,(length(frontier)/5)+1))
  }
  if(yStart <=9) {
    x <- xStart
    y <- yStart+1
    g <- gStart + roads$vroads[xStart,yStart]
    h <- minEdgeCost*manhattanDistance(packages[toGo,1+offset],packages[toGo,2+offset], xStart,yStart+1)
    if(nextMove == 999) {node <- c(x,y,g,h,8)} #nextMove <- 8
    else {node <- c(x,y,g,h,nextMove)}
    frontier <- array(c(frontier,node),dim = c(5,(length(frontier)/5)+1))
  }
  return(frontier)
}

appendNewFrontier <- function(frontierIn, newfrontierIn){
  frontier <- frontierIn
  newFrontier <- newfrontierIn
  newFrontierItems <- length(newFrontier)/5
  
  for(i in 1:newFrontierItems){
    #create 
    if(length(frontier) == 5){
      xCol <- frontier[1]
      yCol <- frontier[2]
    }
    else{
      xCol <- frontier[1,]
      yCol <- frontier[2,]
    }
    xInVector <- (xCol == newFrontier[1,i])
    yInVector <- (yCol == newFrontier[2,i])
    
    duplicateVector <- xInVector + yInVector
    if(2 %in% duplicateVector){
      #there is a duplicate
      indexDuplicate <- which.max(duplicateVector) #get index of 2/where duplicate exists
      fFrontier <- frontier[3,indexDuplicate] + frontier[4,indexDuplicate]
      fNew <- newFrontier[3,i] + newFrontier[4,i]
      if(fNew < fFrontier){
        frontier <- frontier[,-indexDuplicate]
        frontier <- array(c(frontier,newFrontier[,i]),dim = c(5,(length(frontier)/5)+1))
      }
      #else do not append new node
    }
    else{#else if it is not duplicate, append
      frontier <- array(c(frontier,newFrontier[,i]),dim = c(5,(length(frontier)/5)+1))
    }
  }
  return(frontier)
}

getMinF <- function(frontier){
  f <- frontier[3,] + frontier[4,]
  return(which.min(f))
}
