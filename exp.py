from ioh import get_problem
import numpy as np
from ioh import problem, OptimizationType
from ioh import Experiment
import ioh
from ioh import logger
from nevergrad import functions
import nevergrad as ng

dimension = 20

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
names += ["cigar", "altcigar", "ellipsoid", "altellipsoid", "stepellipsoid", "discus", "bentcigar"]
names += ["deceptiveillcond", "deceptivepath"]   #list of objective functions

#algorithms = ["ECMA"] #["CMandAS2", "ECMA", "Cobyla", "MetaModel", "CMA", "DiagonalCMA", "FCMA", "BOBYQA", "F127CMA", "Powell", "SQP"]
algorithms = ["CMA", "DiagonalCMA", "FCMA"]

budget = 1000 #[10, 100, 1000]

def param(dimension):
    x = ng.p.Array(shape=(dimension,))
    parametrization = ng.p.Instrumentation(x)
    return parametrization

def main():
    for name in names:
        f = ioh.problem.wrap_real_problem(lambda x: getattr(functions.corefuncs, name)(np.array(x)), name+'_ng', optimization_type=ioh.OptimizationType.Minimization, n_variables = dimension)
        for algo in algorithms:
            l = logger.Analyzer(root=r"data_new/data_"+name, folder_name="run", algorithm_name = algo)
            f.attach_logger(l)		
            for runs in range(10):
                optim = ng.optimizers.registry[algo](parametrization=param(dimension), budget=budget)
                for u in range(budget):
                    x1 = optim.ask()
                    y1 = f(*x1.args, **x1.kwargs)  
                    optim.tell(x1, y1)
                f.reset()
            del l 

if __name__ == "__main__":
    main()