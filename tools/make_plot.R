#!/usr/bin/Rscript
source("~/darl/tools/atari.R")

draw_confidence <- function() {
  prepare_plot()

  ppo_on     <- parse_multi("ppo_on_2e8_rnn_32_0_16k_a16_lr1e3")
  mkl_on_ns  <- parse_multi("mkl_on_nsD_2e8_rnn_32_0_16k_a16_lr1e3")
  pkl_on_ns  <- parse_multi("pkl_on_nsD_2e8_rnn_32_0_16k_a16_lr1e3")
  acer_on_ns <- parse_multi("acer_on_nsD_2e8_rnn_32_0_16k_a16_lr1e3")

  cols <- c(rainbow(3), 'black')
  legends <- c('MKL ns', 'PKL ns', 'ACER ns', 'PPO')

  make_confidence_plot(ppo_on, main=basename(getwd())[1])
  draw_confidence_line(mkl_on_ns, col=cols[1])
  draw_confidence_line(pkl_on_ns, col=cols[2])
  draw_confidence_line(acer_on_ns,col=cols[3])
  draw_confidence_line(ppo_on, col=cols[4])

  s <- 1:4
  legend('bottomright', col=cols[s], legend=legends[s], lty=1)
}

draw_omega <- function() {
  prepare_plot()

  mkl_1s_pgis2_f <- parse_res("mkl_on_1sD_pgis2_2e8_rnn_32_0_16k_a16_lr1e3_full")
  mkl_1s_pgis2_p <- parse_res("mkl_on_1sD_pgis2_2e8_rnn_32_0_16k_a16_lr1e3_partial")
  mkl_1s_pgis2_k <- parse_res("mkl_on_1sD_pgis2_2e8_rnn_32_0_16k_a16_lr1e3_kernel")
  mkl_1s_pgis2_0 <- parse_res("mkl_on_1sD_pgis2_2e8_rnn_32_0_16k_a16_lr1e3_full0")

  mkl_ns_pgis2_f <- parse_res("mkl_on_nsD_pgis2_2e8_rnn_32_0_16k_a16_lr1e3_full")
  mkl_ns_pgis2_p <- parse_res("mkl_on_nsD_pgis2_2e8_rnn_32_0_16k_a16_lr1e3_partial")
  mkl_ns_pgis2_k <- parse_res("mkl_on_nsD_pgis2_2e8_rnn_32_0_16k_a16_lr1e3_kernel")
  mkl_ns_pgis2_0 <- parse_res("mkl_on_nsD_pgis2_2e8_rnn_32_0_16k_a16_lr1e3_full0")

  cols <- rainbow(8)
  make_plot(mkl_1s_pgis2_f, main=basename(getwd())[1])
  draw_line(mkl_1s_pgis2_f, col=cols[1])
  draw_line(mkl_1s_pgis2_p, col=cols[2])
  draw_line(mkl_1s_pgis2_k, col=cols[3])
  draw_line(mkl_1s_pgis2_0, col=cols[4])
  draw_line(mkl_ns_pgis2_f, col=cols[5])
  draw_line(mkl_ns_pgis2_p, col=cols[6])
  draw_line(mkl_ns_pgis2_k, col=cols[7])
  draw_line(mkl_ns_pgis2_0, col=cols[8])

  legends <- c('1s Full', '1s Partial', '1s Kernel', '1s Full0',
               'ns Full', 'ns Partial', 'ns Kernel', 'ns Full0')

  s <- 1:8

  legend('bottomright', col=cols[s], legend=legends[s], lty=1)
}

draw_off <- function() {
  prepare_plot()

  ppo_on <- parse_res("ppg_on_2e8_rnn_32_0_16k_a16")
  acer_on <- parse_res("acer_on_2e8_rnn_32_0_16k_a16_lr1e3_wf0")
  #ppo_on_a64 <- parse_res("ppg_on_2e8_rnn_32_0_16k_a64")
  mkl_1s <- parse_res("mkl_on_1sD_2e8_rnn_32_0_16k_a16")
  mkl_ns <- parse_res("mkl_on_nsD_2e8_rnn_32_0_16k_a16")
  acer_1s <- parse_res("acer_on_1sD_2e8_rnn_32_0_16k_a16_lr1e3")
  acer_ns <- parse_res("acer_on_nsD_2e8_rnn_32_0_16k_a16_lr1e3")
  acerinf_1s <- parse_res("acerinf_on_1sD_2e8_rnn_32_0_16k_a16_lr1e3_wf0")
  acerinf_ns <- parse_res("acerinf_on_nsD_2e8_rnn_32_0_16k_a16_lr1e3_wf0")
  #acer_ns_a64_wf0 <- parse_res("acer_on_nsD_2e8_rnn_32_0_16k_a64_lr1e3_wf0")
  akl_1s <- parse_res("akl_on_1sD_2e8_rnn_32_0_16k_a16_lr1e3_wf0")
  akl_ns <- parse_res("akl_on_nsD_2e8_rnn_32_0_16k_a16_lr1e3")
  aif_1s <- parse_res("aif_on_1sD_2e8_rnn_32_0_16k_a16_lr1e3_wf0")
  aif_ns <- parse_res("aif_on_nsD_2e8_rnn_32_0_16k_a16_lr1e3_wf0")
  pkl_1s <- parse_res("pkl_on_1sD_2e8_rnn_32_0_16k_a16_lr1e3")
  pkl_ns <- parse_res("pkl_on_nsD_2e8_rnn_32_0_16k_a16_lr1e3")
  pif_1s <- parse_res("pif_on_1sD_2e8_rnn_32_0_16k_a16_lr1e3_wf0")
  pif_ns <- parse_res("pif_on_nsD_2e8_rnn_32_0_16k_a16_lr1e3_wf0")
  #mkl2_1s <- parse_res("mkl2_on_1sD_2e8_rnn_32_0_16k_a16_lr1e4")
  #mkl2_ns <- parse_res("mkl2_on_nsD_2e8_rnn_32_0_16k_a16_lr1e4")
  mkl_1s_pgis2 <- parse_res("mkl_on_1sD_pgis2_2e8_rnn_32_0_16k_a16_lr1e3_kernel")
  mkl_ns_pgis2 <- parse_res("mkl_on_nsD_pgis2_2e8_rnn_32_0_16k_a16_lr1e3_kernel")

  cols <- c(rainbow(16), 'black', 'grey')
  make_plot(mkl_1s, main=basename(getwd())[1])
  #draw_line(mkl_1s, col=cols[1])
  draw_line(mkl_ns, col=cols[2])
  #draw_line(mkl2_1s, col=cols[3])
  #draw_line(mkl2_ns, col=cols[4])
  #draw_line(mkl_1s_pgis2, col=cols[5])
  #draw_line(mkl_ns_pgis2, col=cols[6])
  #draw_line(acer_1s, col=cols[7])
  draw_line(acer_ns, col=cols[8])
  #draw_line(acer_ns_a64_wf0, col=cols[8], lwd=2, lty=2)
  #draw_line(ppo_on_a64, col=cols[9], lwd=2, lty=2)
  #draw_line(akl_1s, col=cols[9])
  draw_line(akl_ns, col=cols[10])
  #draw_line(pkl_1s, col=cols[11])
  draw_line(pkl_ns, col=cols[12])
  #draw_line(aif_1s, col=cols[13])
  draw_line(aif_ns, col=cols[14])
  #draw_line(pif_1s, col=cols[15])
  draw_line(pif_ns, col=cols[16])
  draw_line(ppo_on, col='black')
  draw_line(acer_on, col='grey')

  legends <- c('MKL 1s', 'MKL ns', 'MKL2 1s', 'MKL2 ns', 'MKL 1s2', 'MKL ns2',
               'ACER 1s', 'ACER ns', 'AKL 1s','AKL ns','PKL 1s','PKL ns',
               'AIF 1s', 'AIF ns', 'PIF 1s', 'PIF ns', 'PPO on','ACER on')

  s <- c(2,8,10,12,14,16,17,18)

  legend('bottomright', col=cols[s], legend=legends[s], lty=1)
}

draw_new <- function() {
  prepare_plot()

  ppo_on     <- parse_res("ppo_on_4e8_rnn_32_30_16k")
  ppg_on     <- parse_res("ppg_on_4e8_rnn_32_30_16k")
  inf_vtc_ns <- parse_res("inf_on_nsD_4e8_rnn_32_30_16k")
  mkl_vtc_ns <- parse_res("mkl_on_nsD_4e8_rnn_32_30_16k")
  pkl_vtc_ns <- parse_res("pkl_on_nsD_4e8_rnn_32_30_16k")
  pif_vtc_ns <- parse_res("pif_on_nsD_4e8_rnn_32_30_16k")
  mkl2_vtc_ns <- parse_res("mkl2_on_nsD_sq_4e8_rnn_32_30_16k")
  mkl_vtc_ns_pgis2 <- parse_res("mkl_on_nsD_pgis2_4e8_rnn_32_30_16k")

  cols <- rainbow(20)
  make_plot(ppo_on, main=basename(getwd())[1])
  draw_line(ppo_on, col=cols[2])
  draw_line(inf_vtc_ns, col=cols[6])
  draw_line(mkl_vtc_ns, col=cols[8])
  draw_line(pkl_vtc_ns, col=cols[12])
  draw_line(ppg_on, col=cols[14])
  draw_line(pif_vtc_ns, col=cols[16])
  draw_line(mkl2_vtc_ns, col=cols[18])
  draw_line(mkl_vtc_ns_pgis2, col=cols[20])

  legends <- c('PPO vanilla', 'PPO vtrace',
               'PPO vtrace 1step', 'PPO vtrace nstep',
               'INF vtrace 1step', 'INF vtrace nstep',
               'MKL vtrace 1step', 'MKL vtrace nstep',
               'MTV vtrace 1step', 'MTV vtrace nstep',
               'PKL vtrace 1step', 'PKL vtrace nstep',
               'PPG vanilla', 'PPG vtrace',
               'PIF vtrace 1step', 'PIF vtrace nstep',
               'MKL2 vtrace 1step', 'MKL2 vtrace nstep',
               'MKL nsD PGIS2', 'MKL nsD PGIS2')

  s <- c(2,6,8,12,14,16,18,20)

  legend('bottomright', col=cols[s], legend=legends[s], lty=1)
}

draw_robot <- function() {
  prepare_plot()

  #ppo <- parse_res("ppo")
  #mkl <- parse_res("mkl")
  #ppo_ent <- parse_res("ppo_ent")
  #mkl_ent <- parse_res("mkl_ent")
  #ent <- parse_res("ent_on_nsD")
  mkl_nsD <- parse_res("mkl_on_nsD")
  mkl2_nsD <- parse_res("mkl2_on_nsD")

  cols <- rainbow(2)
  legends <- c('MKL', 'MKL2')
  make_plot(mkl_nsD, main=basename(getwd())[1])
  draw_line(mkl_nsD, col=cols[1])
  draw_line(mkl2_nsD, col=cols[2])

  s <- c(1,2)

  legend('bottomright', col=cols[s], legend=legends[s], lty=1)
}

draw_nstep_confidence <- function() {
  prepare_plot()

  mdkl_rnn_nstep <- parse_multi("mdkl_rnn_large")
  mdkl_rnn_1step <- parse_multi("mdkl_rnn_1step")

  make_confidence_plot(mdkl_rnn_nstep, main=basename(getwd())[1])
  draw_confidence_line(mdkl_rnn_1step, col='green')
  draw_confidence_line(mdkl_rnn_nstep, col='blue')
  legend('bottomright', col=c('green','blue'),
         legend=c('KL 1-step', 'KL n-step'), lty=1)
}

draw_episode <- function() {
  prepare_plot()

  e4  <- parse_res("mkl_on_nsD_4e8_rnn_32_4_16k")
  e16 <- parse_res("mkl_on_nsD_4e8_rnn_32_16_16k")
  e64 <- parse_res("mkl_on_nsD_4e8_rnn_32_64_16k")
  e0  <- parse_res("mkl_on_nsD_4e8_rnn_32_0_16k")
  inf <- parse_res("inf_on_nsD_4e8_rnn_32_30_16k")

  cols <- rainbow(5)
  legends <- c('e4', 'e16', 'e64', 'e0', 'inf30')
  make_plot(inf, main=basename(getwd())[1])
  draw_line(e4,  col=cols[1])
  draw_line(e16, col=cols[2])
  draw_line(e64, col=cols[3])
  draw_line(e0,  col=cols[4])
  draw_line(inf, col=cols[5])
  
  s <- c(1,2,3,4,5)
  legend('bottomright', col=cols[s], legend=legends[s], lty=1)
}

compute_stats <- function() {
  prepare_plot()
  ppo_rnn_large <- parse_res("ppo_on_4e8_rnn_32_30_16k")
  mkl_rnn_large <- parse_res("mkl_on_nsD_4e8_rnn_32_30_16k")

  proposed <- get_ending_score(mkl_rnn_large)
  baseline <- get_ending_score(ppo_rnn_large)
  random   <- get_starting_score(ppo_rnn_large)
  human    <- get_human_score()

  print(paste(proposed, baseline, random, human))

  line <- paste(get_dir_name(), get_game_type(), get_relative_score(proposed, baseline, random, human))
  write.table(line,file="../relative_score.txt",row.names=FALSE, col.names=FALSE, quote=FALSE, append=TRUE)
}

main(draw_confidence, "conf_lr1e3")
#main(draw_nstep_confidence, "nstep_conf")
#main(compute_stats, "stats", make_pdf=FALSE, make_png=FALSE, raw=TRUE)
#main(draw_comp, "comp")
#main(draw_off, "off_lr1e3")
#main(draw_omega, "omega_lr1e3")

