## Loading required packages.
library(bnlearn)
library(Rgraphviz)


## Create the dag
game_net <- model2network("[AC][RC][AT|AC][RT|RC][AS|AT:AC][AD|AT:AC][AA|AT:AC][RS|RT:RC][RD|RT:RC][RA|RT:RC][AACT|AS:AD:AA][RRCT|RS:RD:RA:AACT]")

## Plot the dag
graphviz.plot(game_net)


# Fit a custom dag with probabilities

# Character Nodes (Actor and Reactor)

cptAC = matrix(c(0.5, 0.5), ncol = 2, dimnames = list(NULL, c("Satyr", "Golem")))
cptRC = matrix(c(0.5, 0.5), ncol = 2, dimnames = list(NULL, c("Satyr", "Golem")))

# Type nodes conditioned on Character (Actor and Reactor)

cptAT = c(0.33, 0.34, 0.33, 0.33, 0.34, 0.33)
dim(cptAT) <- c(3,2)
dimnames(cptAT) <- list("AT"=c("Type1", "Type2", "Type3"), "AC" = c("Satyr", "Golem"))

cptRT = c(0.33, 0.34, 0.33, 0.33, 0.34, 0.33)
dim(cptRT) <- c(3,2)
dimnames(cptRT) <- list("RT"=c("Type1", "Type2", "Type3"), "RC" = c("Satyr", "Golem"))

# Attack, strength and defense conditioned on Type and Character (Actor and Reactor)
cptAA <- c(0.2, 0.8, 0.6, 0.4, 0.8, 0.2, 0.75, 0.25, 0.4, 0.6, 0.9, 0.1)
dim(cptAA) <- c(2,3,2)
dimnames(cptAA) <- list("AA"= c("LOW", "HIGH"), "AT"= c("Type1", "Type2", "Type3"), "AC"= c("Satyr", "Golem"))

cptAD <- c(0.9, 0.1, 0.3, 0.7, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, 0.6, 0.4)
dim(cptAD) <- c(2,3,2)
dimnames(cptAD) <- list("AD"= c("LOW", "HIGH"), "AT"= c("Type1", "Type2", "Type3"), "AC"= c("Satyr", "Golem"))

cptAS <- c(0.4, 0.6, 0.2, 0.8, 0.5, 0.5, 0.6, 0.4, 0.5, 0.5, 0.8, 0.2)
dim(cptAS) <- c(2,3,2)
dimnames(cptAS) <- list("AS"= c("LOW", "HIGH"), "AT"= c("Type1", "Type2", "Type3"), "AC"= c("Satyr", "Golem"))

cptRA <- c(0.2, 0.8, 0.6, 0.4, 0.8, 0.2, 0.75, 0.25, 0.4, 0.6, 0.9, 0.1)
dim(cptRA) <- c(2,3,2)
dimnames(cptRA) <- list("RA"= c("LOW", "HIGH"), "RT"= c("Type1", "Type2", "Type3"), "RC"= c("Satyr", "Golem"))

cptRD <- c(0.9, 0.1, 0.3, 0.7, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, 0.6, 0.4)
dim(cptRD) <- c(2,3,2)
dimnames(cptRD) <- list("RD"= c("LOW", "HIGH"), "RT"= c("Type1", "Type2", "Type3"), "RC"= c("Satyr", "Golem"))

cptRS <- c(0.4, 0.6, 0.2, 0.8, 0.5, 0.5, 0.6, 0.4, 0.5, 0.5, 0.8, 0.2)
dim(cptRS) <- c(2,3,2)
dimnames(cptRS) <- list("RS"= c("LOW", "HIGH"), "RT"= c("Type1", "Type2", "Type3"), "RC"=c("Satyr", "Golem"))

# Action conditioned on Actor Strength, Attack and defense

cptAACT <- c(0.1, 0.3, 0.6, 0.3, 0.5, 0.2, 0.3, 0.3, 0.4, 0.5,0.4,0.1,0.1, 0.2, 0.7, 0.4,0.3,0.3, 0.2, 0.4, 0.4, 0.6, 0.3, 0.1)
dim(cptAACT)<- c(3,2,2,2)
dimnames(cptAACT)<- list("AACT"= c("Attack", "Taunt", "Walk"), "AA"= c("LOW", "HIGH"), "AD"= c("LOW", "HIGH"), "AS"= c("LOW", "HIGH"))




cptRRCT <- c(0.5, 0.4, 0.05, 0.05, 0.2, 0.6, 0.1, 0.1, 0.001, 0.001, 0.997, 0.001,0.4, 0.3,0.1, 0.2, 0.1, 0.5, 0.2, 0.2, 0.001, 0.001, 0.99, 0.008,0.1, 0.3, 0.55, 0.05, 0.1, 0.2, 0.65, 0.05, 0.001, 0.001, 0.997, 0.001, 0.3, 0.2, 0.3, 0.2,0.1, 0.3, 0.4, 0.2,0.001, 0.001, 0.99, 0.008,0.3, 0.3, 0.399, 0.001,0.2, 0.4, 0.399, 0.001,0.001, 0.001, 0.997, 0.001,  0.3, 0.4, 0.1, 0.2,0.3, 0.3, 0.1, 0.3,0.001, 0.001, 0.99, 0.008, 0.2, 0.3, 0.49, 0.01,0.2, 0.2, 0.59, 0.01,0.001, 0.001, 0.997, 0.001,  0.2, 0.2, 0.4, 0.2,0.1, 0.1, 0.4, 0.4,0.001, 0.001, 0.99, 0.008)
dim(cptRRCT)<- c(4,2,2,2,3)
dimnames(cptRRCT)<- list("RRCT"=c("Dying", "Hurt", "Idle", "Attack"), "RA"= c("LOW", "HIGH"), "RD"= c("LOW", "HIGH"), "RS"= c("LOW", "HIGH"), "AACT"= c("Attack", "Taunt", "Walk"))


dfit <- custom.fit(game_net, dist = list(AC = cptAC, RC=cptRC, AT= cptAT, RT=cptRT, AA=cptAA, AD=cptAD, AS=cptAS, RA=cptRA, RS=cptRS, RD=cptRD, AACT=cptAACT, RRCT=cptRRCT))

#dfit

getinvProb<- function(event, evidence) {
  cpstmt <- paste("cpquery(dfit, ",event, ",", evidence, ")", sep = "")
  expr <- parse(text= cpstmt)
  return(eval(expr))
}

actionNodes <- c("AA", "AD", "AS")
reactionNodes <- c("RA", "RD", "RS")


getActionInvCpt <- function(){
  cpt <- list()
  for(node in actionNodes){
    cpt[[node]] <- getActCptNode(node)
    dim(cpt[[node]])<- c(2,3,3,2)
    dimnames(cpt[[node]])<- list(`node`=  c("LOW", "HIGH"), "AACT"=c("Attack", "Taunt", "Walk"),"AT"= c("Type1", "Type2", "Type3"), "AC"= c("Satyr", "Golem"))
  }
  
  return(cpt)
}

getActCptNode <- function(node){
  idx <- 1
  probs <- c()
  for(act in evidence1Values){
    for(typ in evidence2Values){
      for(chr in evidence3Values){
        for(ev in eventValues){
          event = paste("(", node, "==", "'", ev, "')", sep = "")
          evidence = constructEvidence(act, typ, chr)
          probs[[idx]] <- getinvProb(event, evidence)
          idx<- idx+1
        }
      }
    }
  }
  return(probs)
}

constructEvidence <- function(act, typ, char){
  st<- paste("(AACT==", "'", act, "'", " & ", "AT==", "'",typ, "'", " & ", "AC==", "'",char, "')", sep="")
  return(st)
}

eventValues <- c("LOW", "HIGH")
evidence1Values <- c("Attack", "Taunt", "Walk")
evidence2Values <- c("Type1", "Type2", "Type3")
evidence3Values <- c("Satyr", "Golem")




# Actor nodes will be indexed by Action, Type and character



