#!/usr/bin/env Rscript

plot_bar <- function(d) {
    col <- rep('grey',nrow(d))
    col[d[,2]=='EasyExploration'] = 'blue'
    col[d[,2]=='HardExplorationDenseReward'] = 'red'
    col[d[,2]=='HardExplorationSparseReward'] = 'brown'
    barplot(ifelse(d[,3]>2,2,d[,3]), ylim=c(-1,2), border=FALSE, ylab='Relative Performance', yaxt='n', col=col)
    axis(2, at=pretty(seq(-1,2,length.out=100)), lab=paste(pretty(seq(-1,2,length.out=100)) * 100, '%'), las=TRUE, cex.axis=0.7)
    num_neg <- sum(d[,3] < 0)
    text(labels=sprintf("%4d%% %s",as.integer(d[,3]*100),d[,1])[seq(1,num_neg)],
         x=1.2*(seq(1,num_neg)-1)-0.1, y=0.05, srt=90, cex=0.7, pos=4)
    text(labels=sprintf("%s %4d%%",d[,1],as.integer(d[,3]*100))[seq(num_neg+1,nrow(d))],
         x=1.2*(seq(num_neg+1,nrow(d)))+0.6, y=-0.05, srt=90, cex=0.7, pos=2)
    legend('topleft', border=FALSE,
           legend=c('Easy Exploration','Hard Exploration (Dense Reward)','Hard Exploration (Sparse Reward)','Unknown'),
           fill=c('blue','red','brown','grey'))
}

if(TRUE) {
  d <- read.table("relative_score.txt", header=FALSE)
  d <- d[order(d[,3]),]

  pdf(file="relative_score.pdf", width=10, height=7);
  plot_bar(d);
  dev.off();

}
