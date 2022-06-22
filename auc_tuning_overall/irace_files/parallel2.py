#!/usr/bin/env python3

import sys
import argparse
import warnings
import ioh
from ioh import RealConstraint
#from modcma.modularcmaes import ModularCMAES
#from modcma.parameters import Parameters
import nevergrad as ng
import numpy as np
from nevergrad import functions
import inspect
import typing
import os, contextlib
from sklearn.metrics import auc
#from multiprocess import Process
#import multiprocess

NoneType = type(None)

target_pow = [2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8]
targets = [10**x for x in target_pow]
area = 0

 


_base_bounds = {'popsize_cma': [5, 30],
                'popsize_pso': [30, 50], 
                'scale': [1.0, 5.0]
                }

_supported_parameters = ["popsize_cma", "popsize_pso", "scale", "random_init", "transform"]


def _cmaes_param_to_irace_dict(param_list, fid=1, dim=5, fixed_budget=False):
    """
    Helper function for generate_parameter_file
    """
    #c_func = ng.optimizers._CMA.__init__

    #param_dict = inspect.getfullargspec(c_func).annotations
    irace_params = []
    for p in param_list:
        curr_param = []
        opts = None

        if p not in param_dict:
            continue

        temp = param_dict[p]
        if type(temp) == bool:
            curr_param1 = [f"{p}", f'"--{p} "', 'c', '(True, False)']
        elif temp is None and p == "popsize":
            if p in _base_bounds:
                curr_param1 = [f"{p}", f'"--{p} "', 'i', tuple(_base_bounds[p])]
        elif type(temp) == float and p in _base_bounds:
            curr_param1 = [f"{p}", f'"--{p} "', 'r', tuple(_base_bounds[p])]

        
        irace_params.append(curr_param1)
    # Fixed parameters, currently categorical since fixed-int didn't work
    #irace_params.append(['eval_type', '"--eval_type "', 'c', f'({eval_type})'])
    irace_params.append(['fid', '"--fid "', 'c', f'({fid})'])
    irace_params.append(['dim', '"--dim "', 'c', f'({dim})'])
    return irace_params


def generate_parameter_file(filename, fid=1, dim=5, parameters=None, fixed_budget=False):
    """
    Function to generate an irace-readable parameter file

    Parameters
    ----------
    filename:
        Where to store this parameter file
    fid:
        Fixed parameter indicating the bbob-functions ID
    dim:
        Fixed parameter indicating the bbob-functions dimension
    parameters:
        Optional, the list of the parameters to use. For all available parameters, use get_valid_parameters
    fixed_budget:
        Whether to view this as fixed-budget instead of fixed-target optimization. Currently not fully implemented
    Notes
    -----
    None of the parameters are conditional, and support for conditional parameters is not implemented
    """
    if parameters is None:
        parameters = _supported_parameters
    irace_params = _cmaes_param_to_irace_dict(parameters, fid=fid, dim=dim,
                                               fixed_budget=fixed_budget)
    with open(filename, 'a') as f:
        for line in irace_params:
            f.write(f"{line[0]} {line[1]} {line[2]} {line[3]}\n")


def parse_paramters(configuration_parameters):
    """
    Function to parse parameters into usable dictionaries

    Parameters
    ----------
    configuration_paramters:
        The command line arguments, without the executables name
    Notes
    -----
    Returns two dicts: one for the problem-specific settings and one for the algorithm settings.
    This algorithm settings contains the splitpoint, C1 as dict and C2 as dict
    """
    parser = argparse.ArgumentParser(description="Run CMA-ES configuration",
                                     argument_default=argparse.SUPPRESS)
    #print("configuration_parameters")
    #print(configuration_parameters)
    # Positional parameters
    parser.add_argument('configuration_id', type=str)
    parser.add_argument('instance_name', type=str)
    parser.add_argument('seed', type=int)
    parser.add_argument('iid', type=int)
    parser.add_argument('budget', type=int)

    # 'global' parameters
    parser.add_argument('--fid', dest='fid', type=str, required=True)
    parser.add_argument('--dim', dest='dim', type=int, required=True)
    parser.add_argument('--eval_type', dest='eval_type', type=str, required=False, default='PAR4')

    # C1 and C2 paramters
    c_func = ng.optimizers.CMA.__init__
    p_func = ng.optimizers.PSO.__init__

    param_dict_cma = inspect.getfullargspec(c_func).annotations
    #print(param_dict_cma)
    param_dict_pso = inspect.getfullargspec(p_func).annotations

    #param_dict = ng.optimizers.CMA._config
    #print("param_dict_1")
    #print(param_dict)
    for p in _supported_parameters:
        opts = None
        if p=="popsize_cma":
            #print(param_dict_cma)
            temp = param_dict_cma[p[:-4]]
        elif p == "popsize_pso":
            temp = param_dict_pso[p[:-4]]
        elif p not in param_dict_cma and p not in param_dict_pso:
            continue
        elif p in param_dict_cma:
            temp = param_dict_cma[p]
        elif p in param_dict_pso:
            temp = param_dict_pso[p]
        if temp == bool or temp == str:
            parser.add_argument(f"--{p}", dest=f"{p}", type=str)
        elif temp == int or temp == float:
            parser.add_argument(f"--{p}", dest=f"{p}", type=temp)
        elif temp == typing.Union[int, NoneType]:
            parser.add_argument(f"--{p}", dest=f"{p}", type=int)
    #print("param_dict_2")
    #print(param_dict)

     
    # Process into dicts
    argsdict = parser.parse_args(configuration_parameters).__dict__
    #print("argsdict")
    #print(argsdict)
    alg_config = {}
    for k, v in argsdict.items():
        if v in ["False", "True", "None"]:
            v = eval(v)
        if k in param_dict_cma or k in param_dict_pso:
            alg_config[k] = v
        elif k in ["popsize_cma","popsize_pso","budget"]:
            alg_config[k] = v
    problem_config = {k: argsdict[k] for k in ('fid', 'dim', 'iid', 'seed', 'budget', 'eval_type')}
    #print("alg_config")
    #print(alg_config)
    return [problem_config, alg_config]

def param(dimension):
    x = ng.p.Array(shape=(dimension,))
    
    parametrization = ng.p.Instrumentation(x)
    return parametrization


def ecdf_auc(y_, evals, targets=targets):
    pts = []
    for y in y_:
        total = 0
        for t in targets:
            if y <= t:
                total += 1
        total = total/11
        pts.append(total)
    ar = auc(evals, pts)
    return ar


def run_target(func1, func2, seed, parameters, budget=None, penalty_factor=4,
                      fixed_budget=False, target_precision=0, verbose=False):

    global area
    y_ = []
    evals = []

    np.random.seed(seed)
    if budget is None:
        budget = 10e4 * dim

    parameters['budget'] = int(budget)

    #c = ModularCMAES(func, d=func.meta_data.n_variables, **parameters)
    #c.parameters.target = func.objective.y
    d = func1.meta_data.n_variables
    p = ng.optimizers.registry['PSO'](parametrization=param(d), budget=budget)
    c = ng.optimizers.registry['CMA'](parametrization=param(d), budget=budget)

    c._diagonal = True
    for k in parameters.keys():
        if k=='budget':
            continue
        if k=='popsize_pso':
            p.__dict__["llambda"] = parameters[k]
            #c.__dict__["_popsize"] = parameters[k]
            continue
        if k=='popsize_cma':
            c.__dict__["_popsize"] = parameters[k]
            continue
        if k=='transform':
            if parameters[k]=='arctan':
                p.__dict__["_"+k] = ng.parametrization.transforms.ArctanBound(0,1)
            elif parameters[k]=='identity':
                p.__dict__["_"+k] = ng.parametrization.transforms.Affine(1,0)
            elif parameters[k]=='gaussian':
                p.__dict__["_"+k] = ng.parametrization.transforms.CumulativeDensity()
            continue
        #p.__dict__["_"+k] = parameters[k]
        c.__dict__["_"+k] = parameters[k]
    #print(p.__dict__)
    #print(c.__dict__)
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
                #recom = p.minimize(func)
    	        for u in range(budget):
                     xc = c.ask()
                     yc = func1(*xc.args, **xc.kwargs)
                     c.tell(xc,yc)
                     xp = p.ask()
                     yp = func2(*xp.args, **xp.kwargs)
                     p.tell(xp,yp)

                     y = min(func1.state.current_best.y, func2.state.current_best.y)
                	#if u%100==0:
                   		#print(func1.state.current_best.y)
                   		#print(func2.state.current_best.y)
                     y_.append(y)
                     evals.append(u)
    #print(func1.state.evaluations)
    #print(func2.state.evaluations)
    area-=ecdf_auc(y_, evals)

        
    #c.minimize(func)  ####
    """for u in range(budget):
        x = c.ask()
        y = func(*x.args, **x.kwargs)
        c.tell(x,y)"""




def get_problem(problem_config):
    fname = problem_config['fid']
    if fname[:2] == 'ng':
        def temp(x):
            return functions.corefuncs.registry[fname[3:]](np.array(x))
        return ioh.problem.wrap_real_problem(temp, fname[3:], n_variables=problem_config['dim'], optimization_type=ioh.OptimizationType.Minimization)
    else:
        return ioh.get_problem(fname, problem_config['iid'], problem_config['dim'])


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    if len(sys.argv) == 3:
        if sys.argv[1] == "--generate_parameters":
            generate_parameter_file(sys.argv[2])
            sys.exit(0)
        else:
            print("Unknown arguments, either use --generate_parameters filename or use" +
                  "--help to see usage for running the dynamic ccmaes")
            sys.exit(1)

    problem_config, alg_config = parse_paramters(sys.argv[1:])
    #print(problem_config)
    #print(alg_config)
    budget = int(problem_config['budget'])

#     func = IOH_function(problem_config['fid'], problem_config['dim'],
#                         problem_config['iid'], target_precision=1e-8, suite="BBOB")
    fname = problem_config['fid']
    if fname[:2] == 'ng':
        def temp(x):
            return functions.corefuncs.registry[fname[3:]](np.array(x))
        ioh.problem.wrap_real_problem(temp, fname[3:], n_variables=problem_config['dim'], optimization_type=ioh.OptimizationType.Minimization)
        func1 = ioh.get_problem(fname[3:], iid=problem_config['iid'], dim=problem_config['dim'])
        func2 = ioh.get_problem(fname[3:], iid=problem_config['iid'], dim=problem_config['dim'])
        #func1 =  ioh.problem.wrap_real_problem(temp, fname[3:], n_variables=problem_config['dim'], optimization_type=ioh.OptimizationType.Minimization)
        #func2 =  ioh.problem.wrap_real_problem(temp, fname[3:], n_variables=problem_config['dim'], optimization_type=ioh.OptimizationType.Minimization)
    #else:
        #func1 =  ioh.get_problem(fname, problem_config['iid'], problem_config['dim'])
        #func2 =  ioh.get_problem(fname, problem_config['iid'], problem_config['dim'])
#     func = get_problem(problem_config)
                                                       
    eval_type = problem_config['eval_type']
#     if eval_type == 'ECDF':
#         func = auc_func(problem_config['fid'], problem_config['dim'],
#                         problem_config['iid'], target_precision=1e-8, suite="BBOB", budget=budget)

    
    run_target(func1, func2, problem_config['seed'], parameters=alg_config, 
                budget=problem_config['budget'], verbose=False)

    
    res = area
#     if eval_type == 'ECDF':
#         fraction = sum(func.best_so_far_precision > func.target_values) / 51
#         unused_budget = max(budget - func.evaluations, 0)
#         res = budget - (func.auc - (unused_budget * fraction))

#     elif func.final_target_hit:
#         res = min(func.evaluations, budget)
#     else:
#         if eval_type == 'PAR4':
#             res = budget * penalty_factor
#         elif eval_type == 'PAR4-inf':
#             res = "Inf"
#         elif eval_type == 'PAR4-prec':
#             res = budget * penalty_factor + np.log10(func.best_so_far_precision)

    print(res)
    sys.exit(0)
