import nevergrad as ng
#from nevergrad.functions import ArtificialFunction
import ioh
import numpy as np
from ioh import logger
import pandas as pd

from ioh import RealConstraint
from nevergrad import functions
import inspect
import typing
import os, contextlib
from sklearn.metrics import auc

dimension = 10
evals = [1, 223, 445, 668, 890, 1112, 1334, 1557, 1779, 2001, 2223, 2445, 2668, 2890,
         3112, 3334, 3557, 3779, 4001, 4223, 4445, 4668, 4890, 5001]
targets = [2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8]
targets = [10**x for x in targets]

                  #p_cma, p_pso, scale, random_init, transform
configurations = [[8,40,1.0,False,'arctan']
                  #[16, 50, 3.4778, True, 'identity'],[18, 40, 4.7983, False, 'gaussian'],[15, 34, 1.5662, True, 'arctan'], #hm

                  #[29, 49, 3.486, True, 'identity'],[28, 31, 3.4958, True, 'identity'],[30, 49, 3.2658, True, 'identity'], #rastrigin
                  #[14, 30, 1.7049, True, 'identity'],[30, 31, 1.9816, False, 'identity'],[20, 32, 1.1169, False, 'identity'], #griewank
                  #[5, 33, 3.1159, False, 'identity'],[5, 36, 3.7158, False, 'identity'],[5, 32, 3.2513, True, 'identity'], #rosenbrock
                  #[10, 33, 1.9528, False, 'identity'],[24, 30, 3.8499, False, 'identity'],  #ackley
                  #[19, 39, 4.3254, True, 'arctan'],[24, 34, 1.1049, False, 'identity'], #lunacek
                  #[21, 30, 4.4569, False, 'identity'],[29, 30, 2.0299, True, 'identity'] #deceptivemultimodal
#                  [12, 37, 1.9121, True, 'identity'],[19, 42, 4.402, True, 'identity'], #bucherast
#                  [16, 32, 2.6862, True, 'identity'],[12, 30, 3.3726, True, 'identity'],[28, 31, 3.4958, True, 'identity'], #multipeak
#                  [5, 30, 2.5372, False, 'identity'], #sphere
#                  [5, 37, 3.8272, False, 'arctan'],[14, 35, 1.7129, False, 'arctan'],#dlinear
#                  [5, 30, 1.5128, False, 'identity'],[12, 32, 1.0583, False, 'identity'],[13, 32, 2.7954, True, 'identity'],#cigar
                  #altcigar

 #                 [13, 31, 2.6896, True, 'identity'],[27, 34, 2.512, True, 'identity'],  #ellipsoid
                  #altellip
                  #discus
 #                 [21, 32, 2.9533, False, 'identity'],#bentcigar
 #                 [5, 30, 4.3751, True, 'arctan'],[7, 31, 4.0938, False, 'gaussian'] #decepillcond
                  #deceptivepath

                 ]



names = [
        "hm",
        "rastrigin",
        "griewank",
        "rosenbrock",
        "ackley",
        "lunacek",
        "deceptivemultimodal",
        "bucherastrigin",
        "multipeak"] # "sphere", "cigar", "ellipsoid"]

names += ["sphere", "doublelinearslope"]  #, "stepdoublelinearslope"]
names += ["cigar", "altcigar", "ellipsoid", "altellipsoid"] #, "stepellipsoid", "discus", "bentcigar"]
names += ["discus", "bentcigar"]
names += ["deceptiveillcond", "deceptivepath"]  #list of objective functions
#names = ['deceptivemultimodal']

#algorithms = ['DiagonalCMA', 'CMA', 'NGO', 'Shiwa', 'PSO']

budget = 5000 #[10, 100, 1000]

def name_algo_tuple(config):
    algo1 = 'DCMA_'+str(config[0])+'_'+str(config[2])+'_'+str(config[3])[0]
    algo2 = "PSO_"+str(config[1])+"_"+config[4][:2]
    return algo1, algo2

algo_names = []
for config in configurations:
    algo_n1, algo_n2 = name_algo_tuple(config)
    tuple_name = "("+algo_n1+", "+algo_n2+")"
    algo_names.append(tuple_name)

df = pd.DataFrame(columns=algo_names, index=names)
df = df.fillna(0)
df = df.astype(float)

def ecdf(y, targets=targets):
    ecdf = 0
    for t in targets:

        if y <= t:
            ecdf+=1
    ecdf = ecdf/11
    return ecdf


def ecdf_auc(ecdf_all_runs, evals=evals, targets=targets):
    pts = []
    #for feval in evals:
    #    ind = evals.index(feval)
    #    total = 0
    #    for target in targets:
    #        tar_total = 0
    #        for i in range(25):
    #            if y_all_runs[i][ind] <= target:
    #                tar_total+=1
    #        tar_total = tar_total/25
    #        total = total + tar_total
    #    total = total/11
    #    pts.append(total)


    #for y in y_avg_list:
    #    total = 0
    #    for t in targets:
    #        if y <= t:
    for feval in evals:
        ind = evals.index(feval)
        total = 0
        for i in range(5):
            total += ecdf_all_runs[i][ind]
        total = total/5
        pts.append(total)
    ar = auc(evals, pts)
    return ar

def param(dimension):
    x = ng.p.Array(shape=(dimension,))
    parametrization = ng.p.Instrumentation(x)
    return parametrization

def main():

    for name in names:
        #new_f =  ArtificialFunction(name, block_dimension=dimension, rotation=False, noise_level=0, split=False) #translation_factor = 4.0)
        #f = ioh.problem.wrap_real_problem(lambda x: new_f(np.array(x)), name+'_ng',
        #                  optimization_type=ioh.OptimizationType.Minimization, n_variables = dimension)
        #f = ioh.get_problem(name+'_ng', 0, dim=dimension)
        #for budget in budgets:
        fname = "ng_"+name
        def temp(x):
            return functions.corefuncs.registry[fname[3:]](np.array(x))
        ioh.problem.wrap_real_problem(temp, fname[3:]+"_ng", n_variables=dimension, 
                                      optimization_type=ioh.OptimizationType.Minimization)
        func1 = ioh.get_problem(fname[3:]+"_ng", iid=0, dim=dimension)
        func2 = ioh.get_problem(fname[3:]+"_ng", iid=0, dim=dimension)

        for config in configurations:    #configurations: list of lists
            algo1, algo2 = name_algo_tuple(config)
            combo_name = "("+algo1+", "+algo2+")"
            folder = combo_name+"_"
            #algo1 = algo[1:].partition(",")[0]
            #algo2 = algo.partition(",")[2][1:-1]
            lc = logger.Analyzer(root=r"tune_final2_"+str(budget)+"/data_"+name, folder_name=folder+algo1, algorithm_name=algo1)
                        #store_positions = True)
            func1.attach_logger(lc)

            lp = logger.Analyzer(root=r"tune_final2_"+str(budget)+"/data_"+name, folder_name=folder+algo2, algorithm_name=algo2)
                        #store_positions = True)
            func2.attach_logger(lp)

            

                    #y_all_runs = []
            ecdf_all_runs = []

            for runs in range(5):
                        #y_per_run = []

                p = ng.optimizers.registry['PSO'](parametrization=param(dimension), budget=budget)
                c = ng.optimizers.registry['DiagonalCMA'](parametrization=param(dimension), budget=budget)

                c.__dict__["_popsize"] = config[0]   #p_cma
                p.__dict__["llambda"] =  config[1]   #p_pso
                c._scale = config[2]    #scale
                c._random_init = config[3]   #random_init
                if config[4] == 'arctan':    #transform
                    pass
                elif config[4] == 'identity':
                    c._transform = ng.parametrization.transforms.Affine(1,0)
                elif config[4] == 'gaussian':
                    c._transform = ng.parametrization.transforms.CumulativeDensity()

                ecdf_per_run = []
                for u in range(1, budget+2):
                    xc = c.ask()
                    yc = func1(*xc.args, **xc.kwargs)
                    c.tell(xc,yc)
                    xp = p.ask()
                    yp = func2(*xp.args, **xp.kwargs)
                    p.tell(xp,yp)

                    y = min(func1.state.current_best.y, func2.state.current_best.y)
                    if u in evals:
                        ecdf_per_run.append(ecdf(y))
                                #y_per_run.append(y)
                        #y_all_runs.append(y_per_run)
                print("Function:",name,"Algo tuple:",combo_name,"Run:",runs)
                ecdf_all_runs.append(ecdf_per_run)
                #print(len(ecdf_per_run))
                func1.reset()
                func2.reset()
            #print(len(ecdf_all_runs))
            del lc
            del lp
            #algo = name_algo_tuple(config)
            df.loc[name][combo_name] = ecdf_auc(ecdf_all_runs)
    df.to_csv("tune_final3.csv")

if __name__ == "__main__":
    main()
