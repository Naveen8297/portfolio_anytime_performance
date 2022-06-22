import nevergrad as ng
from nevergrad.functions import ArtificialFunction
import ioh
import numpy as np
from ioh import logger
import pandas as pd
from datetime import datetime

dimension = 10

#list of objective functions
names = [
        "hm",
        "rastrigin",
        "griewank",
        "rosenbrock",
        "ackley",
        "lunacek",
        "deceptivemultimodal",
        "bucherastrigin",
        "multipeak",
    ]
names += ["sphere", "doublelinearslope"]  #, "stepdoublelinearslope"]
names += ["cigar", "altcigar", "ellipsoid", "altellipsoid"] #, "stepellipsoid", "discus", "bentcigar"]
names += ["discus", "bentcigar"]
names += ["deceptiveillcond", "deceptivepath"]      


  

algorithms = ["DE","NelderMead","NaiveIsoEMNA"]
algorithms += ['CMA', 'DiagonalCMA', 'PSO', 'Shiwa', 'NGO','EDA'] 
#budgets = [2000, 5000, 10000] 
budgets = [5000]     #change to desired budget(s)

def param(dimension):
    x = ng.p.Array(shape=(dimension,))
    parametrization = ng.p.Instrumentation(x)
    return parametrization

def main():
	
	for name in names:
		new_f =  ArtificialFunction(name, block_dimension=dimension, rotation=False, noise_level=0, split=False)  #nevergrad instance of function
		f = ioh.problem.wrap_real_problem(lambda x: new_f(np.array(x)), name+'_ng',
                                  optimization_type=ioh.OptimizationType.Minimization, n_variables = dimension)  #wrapping, add _ng at the end to make it IOHanalyzer compatible
		
		for budget in budgets:
			for algo in algorithms:
				l = logger.Analyzer(root=r"file_path"+str(budget)+"/data_"+name, folder_name=algo, algorithm_name=algo)
				f.attach_logger(l)
				for runs in range(25):   #25 runs
				
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






