#!/usr/bin/R

library(Rcpp)
sourceCpp("~/darl/tools/lib.cpp")

prepare_plot <- function() {
  plot_ylim <<- c()
  confidence_ylim <<- c()
}

grp_avg <- function(e) {
  x = e$dt-e$dt[1]
  y = e[,4] # 4 for Score, 5 for Policy Performance
  return(group_avg(x,y,60))
}

parse_res <- function(dirname) {
  e <- read.csv(paste(dirname, 'out/logfile', sep='/'), header=FALSE)
  d <- read.csv(paste(dirname, 'out/progress.csv', sep='/'))
	e$dt <- strptime(e[,1], "%Y-%m-%d %H:%M:%OS")
	e <- e[e$dt >= e$dt[1], ]
  plot_ylim <<- range(c(plot_ylim, range(grp_avg(e)$y)))
  return(list(e=e,d=d))
}

parse_csv <- function(filename) {
	e <- read.csv(filename, header=FALSE)
	e$dt <- strptime(e[,1], "%Y-%m-%d %H:%M:%OS")
	e <- e[e$dt >= e$dt[1], ]
	return(e)
}

parse_multi <- function(dirbasename) {
  dirnames <- paste(dirbasename, c('_t1', '_t2', '_t3', '_t4', '_t5'), sep='') 
  d <- NULL
  for(dir in dirnames) {
    e <- parse_csv(paste(dir,'out/logfile',sep='/'))
    rt <- grp_avg(e)
    newd <- data.frame(x=rt$x, y=rt$y) 
    colnames(newd)[2] <- paste(dir,'y',sep='.')
    if(is.null(d)) {
      d <- newd
    } else {
      d <- merge(d, newd, by='x', all=TRUE)
    }
  }
  # Post-process
  selected <- !is.na(rowMeans(d[,2:ncol(d)]))
  d <- d[selected,]
  apply(d[,2:ncol(d)],1,quantile,0.25,na.rm=TRUE) -> lower
  apply(d[,2:ncol(d)],1,median,na.rm=TRUE) -> med
  apply(d[,2:ncol(d)],1,quantile,0.75,na.rm=TRUE) -> upper
  confidence_ylim <<- range(c(confidence_ylim, range(lower), range(upper)), na.rm=TRUE)
  return(data.frame(x=d$x, lower=lower, med=med, upper=upper))
}

make_plot <- function(res, ...) {
  rt = grp_avg(res$e)
  ylab='Score' # e[,4]
  #ylab='Performance' # e[,5]
  if(TRUE) {
    plot(x=rt$x/3600, y=rt$y, type='n', xlab='Training Time/hour', ylab=ylab, ylim=plot_ylim, ...)
    abline(v=seq(0,10,0.5), col='grey', lty=2)
  }
  if(FALSE) {
    xlim = c(-max(res$d$policy_entropy), 0)
    plot(x=rt$x/3600, y=rt$y, type='n', xlab='Negative Entropy', ylab=ylab, ylim=plot_ylim, xlim=xlim, ...)
    abline(v=seq(xlim[1],0,length.out=4), col='grey', lty=2)
  }
  abline(h=seq(plot_ylim[1], plot_ylim[2], length.out=5), col='grey', lty=2)
  prepare_plot()
}

make_confidence_plot <- function(d, ...) {
  ylab='Score' # e[,4]
  #ylab='Performance' # e[,5]
  # d has f columns: x, lower, med, upper
  plot(x=d$x/3600, y=d$med, type='n', xlab='Training Time/hour', ylab=ylab, ylim=confidence_ylim, ...)
  abline(v=seq(0,3,0.5), col='grey', lty=2)
  abline(h=seq(confidence_ylim[1], confidence_ylim[2], length.out=5), col='grey', lty=2)
  prepare_plot()
}

draw_line <- function(res, ...) {
	#lines(filter(e[,4], rep(1/300, 300)), x=(e$dt-e$dt[1])/3600, ...)
  rt = grp_avg(res$e)
  ent = group_avg(res$d$time_elapsed, res$d$policy_entropy, 60)
  d1 <- data.frame(x=rt$x, rt=rt$y)
  d2 <- data.frame(x=ent$x, ent=ent$y)
  d <- merge(x=d1, y=d2, by="x")
  if(TRUE) {
    lines(d$rt, x=d$x/3600, ...)
  }
  if(FALSE) {
	  lines(d$rt, x=-d$ent, ...)
  }
}

draw_confidence_line <- function(d, col, ...) {
  alpha_col = rgb(t(col2rgb(col)), alpha=50, maxColorValue = 255)
  polygon(x=c(rev(d$x), d$x)/3600, y=c(rev(d$upper),d$lower), col=alpha_col, border=FALSE)
  lines(x=d$x/3600, d$med, col=col, ...)
}

old_get_dir_name <- function() {
  pwds <- unlist(strsplit(getwd(), split='/'))[5]
  return(pwds[length(pwds)])
}

get_dir_name <- function() {
  return(basename(getwd())[1])
}

get_starting_score <- function(res) {
  rt = grp_avg(res$e)
  return(mean(rt$y[1], na.rm=TRUE))
}

get_ending_score <- function(res) {
  rt = grp_avg(res$e)
  return(mean(rev(rt$y)[1:min(length(rt$y),1)], na.rm=TRUE))
}

get_relative_score <- function(proposed, baseline, random, human=NA) {
  # The formula follows from Z.Wang et al., Dueling network architectures for deep reinforcement learning
  return ((proposed-baseline) / (ifelse(is.na(human),baseline,max(human,baseline)) - random))
}

get_human_score <- function() {
  game <- get_dir_name()
  dat <- read.table("../atari_human_score.txt", header=FALSE, stringsAsFactors=FALSE)
  ret <- dat[dat[,1]==game,]
  if(nrow(ret)==0) {
    return(NA)
  } else {
    return(ret[,2])
  }
}

get_game_type <- function() {
  game <- get_dir_name()
  dat <- read.table("../atari_human_score.txt", header=FALSE, stringsAsFactors=FALSE)
  ret <- dat[dat[,1]==game,]
  if(nrow(ret)==0) {
    return('Unknown')
  } else {
    return(ret[,3])
  }
}

main <- function(f, exp_name, make_pdf=TRUE, make_png=FALSE, raw=FALSE) {
  if(raw) { f() }
  if(make_pdf) { pdf(file=paste(get_dir_name(), '_', exp_name, ".pdf", sep=''), width=5,   height=5);   f(); dev.off() }
  if(make_png) { png(file=paste(get_dir_name(), '_', exp_name, ".png", sep=''), width=300, height=300); f(); dev.off() }
}

