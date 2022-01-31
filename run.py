import pandas as pd
import json
import pso
import ga as ga
import logapso
import hgapso
import functions
import multiprocessing as mp
import time

def read_json(file_name):
    """
    Read a Json file with the parameters for optimization algorithms.

    Parameters
    ----------
    file_name: string

    Return
    ------
    params: dict
        Dictionary with parameters.
    """
    with open(file_name) as json_file:
        params = json.load(json_file)
    return params


def save_results(algorithm, benchmark_func, df_results):
    """
    Save the results of experiments to an Excel file.
    
    Parameters
    ----------
    algorithm: string
        Name of optimization algorithm.
    benchmark_func: string
        Name of benchmark function.
    df_results: Dataframe
        Dataframe containing the results of experiments.
    """
    file_name = f"{algorithm}_{benchmark_func}.xlsx"
    iters = df_results.get("max_iters")[0]
    runs = len(df_results.get("run")) // iters

    writer = pd.ExcelWriter(f'exp_results/{file_name}', engine='xlsxwriter')
    df_results.to_excel(writer, sheet_name='Data', index=False)

    workbook = writer.book
    # add formulas to the excel worksheet
    worksheet = workbook.get_worksheet_by_name('Data')
    average_time = ''
    average_convergence = ''
    for i in range(0, runs):
        average_time = average_time + f'B{str((i * iters) + 2)},'
        average_convergence = average_convergence + f'A{str((i + 1) * iters + 1)},'

    # average formulas for the convergence and time between all runs
    worksheet.write('M1', 'Average Convergence')
    worksheet.write_formula('M2', f'=AVERAGE({average_convergence[:-1]})')
    worksheet.write('O1', 'Average Time')
    worksheet.write_formula('O2', f'=AVERAGE({average_time[:-1]})')

    # min formulas for the convergence and time between all runs
    worksheet.write('M3', 'Min Convergence')
    worksheet.write_formula('M4', f'=MIN({average_convergence[:-1]})')
    worksheet.write('O3', 'Min Time')
    worksheet.write_formula('O4', f'=MIN({average_time[:-1]})')

    # add line chart for the fitness-iteration
    worksheet = workbook.add_chartsheet()
    chart_fitness = workbook.add_chart({'type': 'line'})
    chart_fitness.set_y_axis({'name': 'Fitness'})
    chart_fitness.set_x_axis({'name': 'Iteration'})
    chart_fitness.set_title({'name': file_name[:-5]})
    chart_fitness.set_size({'width': 720, 'height': 576})

    for i in range(0, runs):
        chart_fitness.add_series({
            'values': ['Data', ((i * iters) + 1), 0, ((i + 1) * iters), 0],
            'name': "run " + str(i + 1)
        })
    worksheet.set_chart(chart_fitness)

    # add column chart for the time-runs
    worksheet = workbook.add_chartsheet()
    chart_time = workbook.add_chart({'type': 'column'})
    chart_time.set_y_axis({'name': 'Time'})
    chart_time.set_title({'name': file_name[:-5]})
    chart_time.set_size({'width': 720, 'height': 576})

    for i in range(0, runs):
        chart_time.add_series({
            'values': ['Data', ((i * iters) + 1), 1, ((i * iters) + 1), 1],
            'name': "run " + str(i + 1),
            'categories': str(i + 1)
        })
    worksheet.set_chart(chart_time)

    workbook.close()
    print(f'{file_name} succesfully saved.')


def create_grid_params(dict_params):
    """
    Transform a dictionary that returns lists to a list
    of dictionaries containing all possible combination of 
    parameters (cartesian product).

    Parameters
    ----------
    dict_params: dict
        Dictionary containing parameters for experiments.

    Returns
    -------
    final_params: list[dict]
        List containing a dict for each permutation of parameters.
    """
    cartesian_params = [[]]
    # Generate the cartesian product of all possible parameters
    for vals in dict_params.values():
        cartesian_params = [p + [v] for p in cartesian_params for v in vals]

    keys = dict_params.keys()
    final_params = [dict(zip(keys, params)) for params in cartesian_params]
    return final_params


def merge_and_clean_params(lists_of_params_dicts, algorithm):
    """
    Merge the parameters used in an experiment.

    Parameters
    ----------
    lists_of_params_dicts: list[dict]
        List containing all dicts of params for an algorithm.
    algorithm: string
        Algorithm for the parameteres.

    Returns
    -------
    all_params: dict
        Dictionary containing all parameters for the current experiment.
    """
    all_params = {}
    for dict_params in lists_of_params_dicts:
        all_params.update(dict_params.copy())

    if algorithm == 'hgapso':
        del all_params['pop_size_ga']
        del all_params['max_iters_ga']
        del all_params['elite']
        del all_params['prob_cross']
        del all_params['c']
        del all_params['n_gens']
    return all_params


def run_experiment(algorithm, parameters, func_name, n_runs,
                   df_results):
    """
    Run single experiment.

    Parameters
    ----------
    algorithm: string
        Name of algorithm.
    parameters: dict
        Parameters for the algorithm in the current experiment.
    func_name: string
    n_runs: int
    df_results: DataFrame
        Dataframe for saving the experiments results.

    Returns
    -------
    df_results: DataFrame
        DataFrame containing the results of experiments.
    """
    eval_func, l_bound, u_bound, task = functions.get_function(func_name)
    print(f'======== Benchmark function: {func_name} ========')

    func_params = {"eval_func": eval_func, "l_bound": l_bound,
                   "u_bound": u_bound, "task": task}
    # create all permutations of parameters if needed
    grid_params = create_grid_params(parameters)

    n_params = len(grid_params)
    index_params = 1

    for p in grid_params:
        print(f'======== Parameters {index_params} of {n_params} ========')
        for run in range(n_runs):
            print(f'-------- {algorithm} - {func_name} - run {run + 1} --------')
            inicio = time.time()
            if algorithm == 'pso':
                _, _, best_evals = pso.run_pso(
                    eval_func=eval_func, consts=p['consts'],
                    max_iters=p['max_iters_pso'], pop_size=p['pop_size_pso'],
                    particle_size=p['particle_size'], l_bound=l_bound,
                    u_bound=u_bound, task=task)
            elif algorithm == 'ga':
                best_evals = ga.run_full_ga(
                    eval_func=eval_func, max_iters=p['max_iters_ga'],
                    pop_size=p['pop_size_ga'], elite=p['elite'], l_bound=l_bound,
                    u_bound=u_bound)
            elif algorithm == 'hgapso':
                _, _, best_evals = hgapso.run_hgapso(alg_params=p, func_params=func_params)
            elif algorithm == 'logapso':
                _, _, best_evals = logapso.run_logapso(
                    alg_params=p, func_params=func_params,
                    prob_run_ga=p['prob_run_ga'],
                    step_size=p['step_size'])

            timed = time.time() - inicio
            print(timed)
            n_iters = len(best_evals)

            df_results = add_results_to_df(p, df_results, n_iters,
                                           best_evals, run, algorithm, timed)
        index_params += 1
    return df_results


def add_results_to_df(params, df_results, n_iters, best_evals, run,
                      algorithm, time):
    """
    Add the results of current experiments to the final Dataframe.

    Parameters
    ----------
    params: dict
        Dictionary containing the parameters of experiments.
    df_results: Dataframe
        Dataframe containing the results of experiments.
    n_iters: int
        Number of iterations.
    best_evals: list[float]
        Fitness of best candidate solution for each iteration.
    run: int
        Current execution of the algorithm.
    algorithm: string
        Name of algorithm.

    Returns
    -------
    df_results: Dataframe
        Dataframe containg the results of experiments so far.
    """
    if algorithm == 'ga':
        info_to_input = {'prob_mut': [params['prob_mut']] * n_iters,
                         'prob_cross': [params['prob_cross']] * n_iters,
                         'fitness': best_evals,
                         'max_iters': [params['max_iters_ga'] + 1] * n_iters,
                         'pop_size': [params['pop_size_ga']] * n_iters,
                         'run': [run + 1] * n_iters,
                         'time': [time] * n_iters
                         }

        return df_results.append(pd.DataFrame(info_to_input), ignore_index=True)

    info_to_input = {'w': [params['consts'][0]] * n_iters,
                     'c1': [params['consts'][1]] * n_iters,
                     'c2': [params['consts'][2]] * n_iters,
                     'fitness': best_evals,
                     'max_iters': [params['max_iters_pso']] * n_iters,
                     'pop_size': [params['pop_size_pso']] * n_iters,
                     'run': [run + 1] * n_iters,
                     'time': [time] * n_iters}
    if algorithm == 'logapso':
        info_to_input.update({'prob_mut': [params['prob_mut']] * n_iters,
                              'prob_run_ga': [params['prob_run_ga']] * n_iters,
                              'step_size': [params['step_size']] * n_iters})
    elif algorithm == 'hgapso':
        info_to_input.update({'prob_mut': [params['prob_mut']] * n_iters})

    info_to_input.update(
        {'particle_size': [params['particle_size']] * n_iters})

    return df_results.append(pd.DataFrame(info_to_input), ignore_index=True)


def run_pso_experiments(list_params, func_name, n_runs):
    """
    Execute experiments with the PSO algorithm 'n_runs' times for each
    group of PSO parameters and a given benchmark function.

    Parameters
    ----------
    list_params: dict
        Dictionary containing lists of different values for parameters.
    func_name: string
        Name of benchmark function.
    n_runs: int
        Number of runs for same parameter.
    """
    df_results = pd.DataFrame(columns=['fitness', 'time', 'w', 'c1', 'c2', 'max_iters',
                                       'pop_size', 'run', 'particle_size'])
    df_results = run_experiment('pso', list_params, func_name,
                                n_runs, df_results)
    save_results('pso', func_name, df_results)


def run_ga_experiments(list_params, func_name, n_runs):
    """
        Run experiments with the GA algorithm 'n_run' times for
        each set of parameters.

        Parameters
        ----------
        list_params: dict
            Dictionary containing all GA parameters.
        func_name: string
            Name of function.
        n_runs: int
            Number of times the experiment is executed.
        """

    df_results = pd.DataFrame(
        columns=['fitness', 'time', 'max_iters',
                 'pop_size', 'run', 'prob_mut', 'prob_cross'])
    iters = []
    for i in list_params.get('max_iters_ga'):
        iters.append(i - 1)
    list_params.update({'max_iters_ga': iters})
    df_results = run_experiment('ga', list_params, func_name,
                                n_runs, df_results)
    save_results('ga', func_name, df_results)


def run_hgapso_experiments(list_pso_params, list_ga_params, func_name,
                           n_runs):
    """
    Run experiments with the HGAPSO algorithm 'n_run' times for
    each set of parameters.

    Parameters
    ----------
    list_pso_params: dict
        Dictionary containing all PSO parameters tested.
    list_ga_params: dict
        Dictionary containing all GA parameters.
    func_name: string
        Name of function.
    n_runs: int
        Number of times the experiment is executed.
    """
    all_params = merge_and_clean_params(
        [list_pso_params, list_ga_params], 'hgapso')

    df_results = pd.DataFrame(
        columns=['fitness', 'time', 'w', 'c1', 'c2', 'max_iters',
                 'pop_size', 'run', 'prob_mut', 'particle_size'])

    df_results = run_experiment('hgapso', all_params, func_name,
                                n_runs, df_results)
    save_results('hgapso', func_name, df_results)


def run_logapso_experiments(list_pso_params, list_ga_params,
                            list_logapso_params, func_name, n_runs):
    """
    Execute experiments with the LOGAPSO 'n_runs' times for each
    combination of PSO and GA parameters for a given benchmark function.

    Parameters
    ----------
    list_pso_params: dict
        Dictionary containing the PSO parameters.
    list_ga_params: dict
        Dictionary containing the GA parameters.
    list_logapso_params: dict
        Dictionary containing the LOGAPSO parameters.
    func_name: string
    n_runs: int
    """
    all_params = merge_and_clean_params(
        [list_pso_params, list_ga_params, list_logapso_params], 'logapso')

    df_results = pd.DataFrame(
        columns=['fitness', 'time', 'w', 'c1', 'c2', 'max_iters', 'pop_size',
                 'run', 'prob_mut', 'prob_run_ga', 'step_size', 'particle_size'])

    df_results = run_experiment('logapso', all_params, func_name, n_runs,
                                df_results)
    save_results('logapso', func_name, df_results)


def run_processes(processes, n_cpus):
    '''
    Run all processes in list.

    Parameters
    ----------
    processes: list
        List containing the processes to run.
    n_cpus: int
        Number of cpus used.
    '''
    # while there is any process in list.

    for i in range(n_cpus):
        if i < len(processes):
            processes[i].start()
        else:
            break
    if len(processes) > n_cpus:
        run_processes(processes[n_cpus:], n_cpus)


def run_parallel_experiments(params, n_cpus):
    """
    Break the experiments in different processes and run them
    using all avalilable cores.

    Parameters
    ----------
    n_runs: int
        Number of executions of the same algorithm with the same set
        of parameters.
    params: dict
        Dictionary containing the parameters for the experiments.
    n_cpus: int
        Number of avalilable cpus.
    """
    algorithms = ['pso', 'ga', 'hgapso', 'logapso']
    benchmark_funcs = params['function']
    n_runs = params['n_runs']
    # List containg the processes to run in parallel
    processes = []

    for alg in algorithms:
        for func in benchmark_funcs:
            if alg == 'pso':
                processes.append(
                    mp.Process(target=run_pso_experiments,
                               args=(params['pso'], func, n_runs,))
                )
            elif alg == 'ga':
                processes.append(
                    mp.Process(target=run_ga_experiments,
                               args=(params['ga'], func, n_runs,))
                )
            elif alg == 'hgapso':
                processes.append(
                    mp.Process(target=run_hgapso_experiments,
                               args=(params['pso'], params['ga'],
                                     func, n_runs,))
                )
            elif alg == 'logapso':
                processes.append(
                    mp.Process(target=run_logapso_experiments,
                               args=(params['pso'], params['ga'],
                                     params['logapso'], func, n_runs,))
                )
        run_processes(processes, n_cpus)
        processes.clear()


if __name__ == '__main__':
    params = read_json('parameters.json')
    run_parallel_experiments(params, mp.cpu_count())
