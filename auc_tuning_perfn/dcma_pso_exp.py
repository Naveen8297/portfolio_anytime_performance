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
names += ["cigar", "altcigar", "ellipsoid", "altellipsoid"] #, "stepellipsoid", "discus", "bentcigar"]
names += ["discus", "bentcigar"]
names += ["deceptiveillcond", "deceptivepath"]  #list of objective functions

algorithms = ['DiagonalCMA','PSO']


#set these acc. irace results
configurations_dcma = []  #each config: [p_cma, scale, random_init]
configurations_pso = []   #each config: [p_pso, transform]
budget = 5000 #[10, 100, 1000]

def param(dimension):
    x = ng.p.Array(shape=(dimension,))
    parametrization = ng.p.Instrumentation(x)
    return parametrization

def main():
		
        for name in names:
                new_f =  ArtificialFunction(name, block_dimension=dimension, rotation=False, noise_level=0, split=False) 
                f = ioh.problem.wrap_real_problem(lambda x: new_f(np.array(x)), name+'_ng',
                                  optimization_type=ioh.OptimizationType.Minimization, n_variables = dimension)
                
                for algo in algorithms:
                    if algo == 'DiagonalCMA':
                        for config in configurations_dcma:
                            folder = str(config[0])+'p'+str(config[1]).replace('.','')+'s'+str(config[2]) #folder and algo name based on configuration
                            algo_name = str(config[0])+'_'+str(config[1])+'_'+str(config[2])[0]
                            
                        
                            l = logger.Analyzer(root=r"filepath/data_"+name, folder_name=folder, algorithm_name='DCMA_'+algo_name) 
                        
                            f.attach_logger(l)
                            for runs in range(25):
                        
                                optim = ng.optimizers.registry[algo](parametrization=param(dimension), budget=budget)
                                optim._popsize = config[0]
                                optim._scale = config[1]
                                optim._random_init = config[2]    #change parameter values

                                for u in range(budget):

                                    x1 = optim.ask()   ####


                            ####
                                    y1 = f(*x1.args, **x1.kwargs)
                            
                                    optim.tell(x1, y1)
                            

                                recommendation = optim.recommend()
                                print("* ", "Dimension:", dimension, "Problem:",  name, "Run:", runs, "Budget:", budget, "Algorithm:", algo,
                                          " provides a vector of parameters with test error ",
                                    f(*recommendation.args, **recommendation.kwargs))
                                f.reset()
                            del l

                    elif algo == 'PSO':
                        for config in configurations_pso:
                            folder = str(config[0])+'p'+str(config[1])+'t'
                            algo_name = str(config[0])+'_'+str(config[1]).upper()[0]
                            

                            l = logger.Analyzer(root=r"dcma_pso10/data_"+name, folder_name=folder, algorithm_name='PSO_'+algo_name) 
                        
                            f.attach_logger(l)
                            for runs in range(25):
                        
                                optim = ng.optimizers.registry[algo](parametrization=param(dimension), budget=budget)
                                optim.llambda = config[0]
                                if config[1] == 'arctan':
                                    pass
                                elif config[1] == 'identity':
                                    optim._transform = ng.parametrization.transforms.Affine(1,0)
                                elif config[1] == 'gaussian':
                                    optim._transform = ng.parametrization.transforms.CumulativeDensity()

                                

                                for u in range(budget):

                                    x1 = optim.ask()   ####


                            ####
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
