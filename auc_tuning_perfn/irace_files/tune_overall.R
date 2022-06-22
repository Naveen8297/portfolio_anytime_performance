#!/usr/bin/env Rscript
args = commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {         
  stop("At least one argument must be supplied (idx)", call. = FALSE)
} else if (length(args) > 2) {
  stop("At most three arguments must be supplied")
}



idx  <- as.integer(args[1])+1

library(irace)

log_dir = "log4/"

s <- readScenario("scenario.txt")
p <- readParameters("parameters.txt")

#fids <- c("ng_hm")
dim <- 10
#budget <- 5000

# seeds <- seq(idx, 100+idx, 10)

#if (idx %% 6 < 3) {
#    dim <- 10
#} else {
#    dim <- 50
#}
if (idx %% 3 == 1) {
    budget <- 5000
} else if (idx %% 3 == 2) {
    budget <- 1000
} else {
    budget <- 3200
}
#fids <- c('ng_cigar')
#if (idx <= 6) {
#    fids <- c('ng_hm',
# 'ng_rastrigin',
# 'ng_griewank',
# 'ng_rosenbrock',
# 'ng_ackley',
# 'ng_lunacek',
# 'ng_deceptivemultimodal',
# 'ng_bucherastrigin',
# 'ng_multipeak',
# 'ng_sphere',
# 'ng_doublelinearslope')
#} else {
#    fids <- c('ng_stepdoublelinearslope',
# 'ng_cigar',
# 'ng_altcigar',
# 'ng_ellipsoid',
# 'ng_altellipsoid',
# 'ng_stepellipsoid',
# 'ng_discus',
# 'ng_bentcigar',
# 'ng_deceptiveillcond',
# 'ng_deceptivepath')
#}


for (seed in seq(5)) {
    scen <- s
    scen$logFile <- paste0(log_dir, "log_F", "_", dim, "D_seed", seed, "_B", budget)
    scen$boundMax <- budget
    scen$instances <- c("1 --fid ng_hm", "2 --fid ng_rastrigin", "3 --fid ng_griewank",
                        "4 --fid ng_rosenbrock", "5 --fid ng_ackley", "6 --fid ng_lunacek",
                        "7 --fid ng_deceptivemultimodal", "8 --fid ng_bucherastrigin", "9 --fid ng_multipeak",
                        "10 --fid ng_sphere", "11 --fid ng_doublelinearslope",
                        "12 --fid ng_cigar", "13 --fid ng_altcigar", "14 --fid ng_ellipsoid",
                        "15 --fid ng_altellipsoid", "16 --fid ng_discus",
                        "17 --fid ng_bentcigar", "18 --fid ng_deceptiveillcond", "19 --fid ng_deceptivepath")
    #p$domain$fid <- fid
    p$domain$dim <- dim
    scen$seed <- seed
    scen <- checkScenario(scen)
    irace(scen, p)
    #for (fid in fids) {
    #    scen <- s
    #    scen$logFile <- paste0(log_dir, "log_F", fid, "_", dim, "D_seed", seed, "_B", budget)
    #    scen$boundMax <- budget
    #    p$domain$fid <- fid
    #    p$domain$dim <- dim
    #    scen$seed <- seed
    #    scen <- checkScenario(scen)
    #    irace(scen, p)
    #}
}
