import nevergrad as ng
from nevergrad.functions import ArtificialFunction
import ioh
import numpy as np
from ioh import logger
import pandas as pd 

dimension = 10

names = [
        "hm",
        "rastrigin",
        "griewank",
        "rosenbrock",
        "ackley",
        "lunacek",
        "deceptivemultimodal",
        "bucherastrigin",
        "multipeak"]
        
names += ["sphere", "doublelinearslope"]  #, "stepdoublelinearslope"]
names += ["cigar", "altcigar", "ellipsoid", "altellipsoid"] #, "stepellipsoid", "discus", "bentcigar"]"""
names += ["discus", "bentcigar"]
names += ["deceptiveillcond", "deceptivepath"]   #list of objective functions

#algorithms = ["CMA", "DiagonalCMA", "Powell", "SQP"]
#algorithms = ['CMA']
algorithms = ['DiagonalCMA', 'CMA', 'NGO', 'Shiwa', 'PSO']  #,  'NaiveTBPSA', 'PSO', 'DE', 'LhsDE', 'RandomSearch', 'OnePlusOne', 'TwoPointsDE']
#['CMA', 'DiagonalCMA', 'PSO', 'Shiwa', 'NGO']
budget = 10000 #[10, 100, 1000]

def param(dimension):
    x = ng.p.Array(shape=(dimension,))
    parametrization = ng.p.Instrumentation(x)
    return parametrization

def main():
        for name in names:
                new_f =  ArtificialFunction(name, block_dimension=dimension, rotation=False, noise_level=0, split=False) #translation_factor = 4.0)
                f = ioh.problem.wrap_real_problem(lambda x: new_f(np.array(x)), name+'_ng',
                                  optimization_type=ioh.OptimizationType.Minimization, n_variables = dimension)
                #f = ioh.get_problem(name+'_ng', 0, dim=dimension)
                #for budget in budgets:
                for algo in algorithms:
                    l = logger.Analyzer(root=r"data1_new_"+str(budget)+"/data_"+name, folder_name=algo, algorithm_name=algo) 
                        #store_positions = True)
                    f.attach_logger(l)
                    for runs in range(25):
                        
                        optim = ng.optimizers.registry[algo](parametrization=param(dimension), budget=budget)
                        for u in range(budget):

                            x1 = optim.ask()
                            
                            y1 = f(*x1.args, **x1.kwargs)
                            
                            optim.tell(x1, y1)
                            

                        recommendation = optim.recommend()
                        print("* ", "Dimension:", dimension, "Problem:",  name, "Run:", runs, "Budget:", budget, "Algorithm:", algo,
                                  " provides a vector of parameters with test error ",
                             f(*recommendation.args, **recommendation.kwargs))
                        f.reset()
                    del l

if __name__ == "__main__":
    main()