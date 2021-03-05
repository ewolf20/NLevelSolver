import numpy as np 
from scipy.integrate import RK45, quad_vec
import scipy.linalg 



"""A function which uses a fast algorithm to evolve a constant-coupling system

Given parameters which specify an N-level Hamiltonian with no more than N couplings of 
constant magnitude and fixed detuning, uses a fast algorithm to solve the system exactly.

Parameters:
    number_levels: The number of levels N in the system 
    level_gammas: The linewidths of each of the levels
    level_couplings: An NxN numpy array containing the coupling magnitudes for the system. Supports nonzero entries 
    on the diagonals. Should be symmetric, with behavior undefined if this is not fulfilled.
    level_detunings: An NxN numpy array containing the detunings of each coupling. Array should be antisymmetric,
    with behavior undefined if this is not fulfilled. 
    initial_state: A numpy array of length N specifying the initial state of the system.
    times: Either a float or an iterable of floats indicating the times at which the state of the system should be returned. 

Returns:
    If times is a single float, a single length N array specifying the final state of the system at the specified time. 
    Otherwise, a list of length N arrays specifying the state of the system at every time specified by times.

Remarks:
    This function respects convention and assumes that the entry in the Hamiltonian corresponding to a specified coupling 
    has an additional factor of 1/2; for a two level system, if the coupling were set to 1, the Hamiltonian would be 1/2[[0 1],[1 0]]. 

    All dimensionful quantities (time, inverse time) should be expressed in a common set of dimensionless units. 
"""
def fast_evolve_constant_system(number_levels, level_gammas, level_couplings, level_detunings, initial_state, times):
    constant_coefficient_hamiltonian, basis_change_detunings_vector = _give_constant_system_transformed_hamiltonian(number_levels, level_gammas, level_couplings, level_detunings)
    basis_transform_matrix_function = _fast_constant_hamiltonian_basis_change_function_factory(basis_change_detunings_vector)
    eigenvalues, eigenvector_list = _diagonalize_hamiltonian(constant_coefficient_hamiltonian)
    eigenvector_coefficients_list = _get_eigvec_coefficients(initial_state, eigenvector_list) 
    #Handle the different cases of time
    try:
        final_state_array = []
        for time in times:
            final_state = _fast_evolve_constant_system_helper(number_levels, eigenvector_list, eigenvalues, eigenvector_coefficients_list, basis_transform_matrix_function, time)
            final_state_array.append(final_state) 
        return final_state_array
    except TypeError:
        time = times 
        final_state = _fast_evolve_constant_system_helper(number_levels, eigenvector_list, eigenvalues, eigenvector_coefficients_list, basis_transform_matrix_function, time) 
        return final_state 

    
def _fast_evolve_constant_system_helper(number_levels, eigenvector_list, eigenvalues, eigenvector_coefficients_list, basis_transform_matrix_function, time):
    final_state_new_basis = np.zeros(number_levels, dtype = complex)
    for eigenvector, eigenvalue, eigenvector_coefficient in zip(eigenvector_list, eigenvalues, eigenvector_coefficients_list):
        final_state_new_basis += eigenvector_coefficient * eigenvector * np.exp(time * eigenvalue) 
    basis_transform_matrix = basis_transform_matrix_function(time) 
    #The basis transform matrix is diagonal, so its conjugate transpose is its conjugate
    basis_transform_matrix_conjugate = np.conjugate(basis_transform_matrix)
    final_state_old_basis = np.matmul(basis_transform_matrix_conjugate, final_state_new_basis) 
    return final_state_old_basis 

"""Diagonalize an arbitrary, possibly non-hermitian hamiltonian 

Given a hamiltonian, diagonalize -1j * hamiltonian using numpy's built-in 
Schur decomposition and return the eigenvalues and eigenvectors, suitable 
for time-evolving a system.

Parameters:
    The hamiltonian to be diagonalized

Returns:
    (eigenvalues, eigenvector_list): The eigenvalues and eigenvectors of -1j * hamiltonian 

Remarks:
    The function is prone to a numerical error where eigenvalues with small negative real parts 
    become eigenvalues with small positive real parts. This can violate a sub-unitarity condition 
    (i.e. population always decreases). To address this, any eigenvalues with positive real parts 
    automatically have their real parts set to 0. This prevents any unphysical exponential growth,
    though it cannot fully address the issue. Look into performing schur decomposition with more 
    accuracy. 
"""
def _diagonalize_hamiltonian(hamiltonian):
    number_levels = len(hamiltonian) 
    schur_results = scipy.linalg.schur(-1j * hamiltonian) 
    schur_matrix = schur_results[0] 
    eigenvalues = np.zeros(number_levels, dtype = complex) 
    for i in range(number_levels):
        eigenvalues[i] = schur_matrix[i][i] 
    eigenvector_list = [] 
    transform_matrix = schur_results[1] 
    for i in range(number_levels):
        eigenvector_list.append(transform_matrix[:, i]) 
    #WARNING: This is a hack. See documentation
    # eigenvalues = np.array([x if np.real(x) <= 0 else 1j * np.imag(x) for x in eigenvalues])
    return (eigenvalues, eigenvector_list)

def _get_eigvec_coefficients(initial_state, eigenvector_list):
    eigenvector_coefficients_list = [] 
    for eigenvector in eigenvector_list:
        #Vdot automatically takes complex conjugate of first argument
        eigenvector_coefficient = np.vdot(eigenvector, initial_state) 
        eigenvector_coefficients_list.append(eigenvector_coefficient)
    return eigenvector_coefficients_list
        
    





"""Constructs a basis-changed, constant-coefficient Hamiltonian for a constant-coupling, fixed detuning system.

Given parameters specifying the system, returns a constant-coefficient Hamiltonian which represents 
the constant-coupling, fixed detuning system in a particular basis, along with a transformation matrix 
for converting to and from the new basis.

Parameters:
    number_levels: The number of levels in the constant-coupling system 
    level_gammas: A length N array containing linewidths of the levels, in order, of the constant coupling system 
    level_couplings: An NxN array containing the couplings between levels 
    level_detunings: An NxN array containing the detunings of the couplings between levels

Returns:
    A tuple (hamiltonian, basis_change_matrix)

    hamiltonian: An NxN array representing the Hamiltonian of the system; this includes the factor of 1/2, but not the -i which 
    will appear in the Schrodinger equation. 

    basis_change_matrix_function: The unitary matrix function (of t) 
    which transforms vectors _from_ the original basis _to_ the basis of the returned Hamiltonian

Notes: The choice of basis change to a constant-coefficient system is not unique if there are fewer than N nonzero couplings in the system. 
If N - 1 couplings are specified, the final condition will be provided by stipulating that the sum of the detunings in the change of basis matrix 
is zero. If there are even fewer couplings - an atypical edge case - conditions are generated at random. 
"""

#TODO Check if the hamiltonian explicitly needs complex dtype 
def _give_constant_system_transformed_hamiltonian(number_levels, level_gammas, level_couplings, level_detunings):
    #Construct the initial Hamiltonian
    hamiltonian = np.zeros((number_levels, number_levels), dtype = complex) 
    hamiltonian += 1.0/2.0 * level_couplings
    for i in range(number_levels):
        hamiltonian[i][i] += -1j * 1.0/2.0 * level_gammas[i] 
    nonzero_coupling_positions_list = [] 
    for i in range(number_levels):
        for j in range(i):
            if(hamiltonian[i][j] != 0):
                nonzero_coupling_positions_list.append((i, j)) 
    #The condition matrix establishes the conditions on the deltas in the change of basis matrix
    condition_matrix = np.zeros((number_levels, number_levels))
    condition_vector = np.zeros(number_levels)
    condition_counter = 0
    if(number_levels < len(nonzero_coupling_positions_list)):
        raise ValueError("The fast constant coefficient algorithm works only if there are at most N nonzero couplings.")
    for ij_tuple in nonzero_coupling_positions_list:
        i, j = ij_tuple 
        condition_matrix[condition_counter][i] = -1 
        condition_matrix[condition_counter][j] = 1 
        condition_vector[condition_counter] = level_detunings[i][j]
        condition_counter += 1
    if(condition_counter < number_levels):
        condition_matrix[condition_counter] = np.ones(number_levels) 
        condition_vector[condition_counter] = 0 
        condition_counter += 1 
    #Populate matrix with additional random conditions - highly likely to work
    while(condition_counter < number_levels):
        condition_matrix[condition_counter] = np.random.normal(loc = 0.0, scale = 1.0, size = number_levels) 
        condition_vector[condition_counter] = 0 
        condition_counter += 1 
    inverse_condition_matrix = np.linalg.inv(condition_matrix) 
    basis_change_detunings_vector = np.matmul(inverse_condition_matrix, condition_vector)
    for i in range(number_levels):
        hamiltonian[i][i] -= basis_change_detunings_vector[i] 
    return (hamiltonian, basis_change_detunings_vector) 



def _fast_constant_hamiltonian_basis_change_function_factory(basis_change_detunings_vector):
    def basis_change_function(t):
        basis_change_matrix = np.zeros((len(basis_change_detunings_vector), len(basis_change_detunings_vector)), dtype = complex) 
        for i in range(len(basis_change_detunings_vector)):
            basis_change_matrix[i][i] = np.exp(1j * t * basis_change_detunings_vector[i]) 
        return basis_change_matrix 
    return basis_change_function 
    




"""Evolves a dynamic system via naive Runge-Kutta
Solves the dynamics of a system with non-constant couplings or detunings by using a built-in scipy Runge-Kutta solver. 
Parameters:
    number_levels: The number of levels in the system 
    level_gammas: A length N numpy array containing the linewidths of each level
    level_couplings_function: A function of scalar time t which returns an NxN numpy array containing the complex couplings - i.e. 
    the entries of the hamiltonian matrix - as a function of time
    initial_state: A length N (complex) numpy array containing the initial state of the system at t = 0. 
    times: Either a float or a length M array of floats stipulating the time(s) at which the final state should be evaluated. 
        If an array, should have times in ascending order for maximum efficiency. 
Returns:
    If times is a single float, a single length N np array specifying the final state of the system at the given time. 
    Otherwise, a list of M arrays of length N specifying the state of the system at the times specified by times.
"""



def rk_evolve_dynamic_system(number_levels, level_gammas, level_couplings_function, initial_state, times):
    diff_function = _rk_evolve_dynamic_system_diff_function_factory(number_levels, level_gammas, level_couplings_function)
    starting_time = 0 
    starting_state = initial_state
    #Hacky way to handle scalar or array time input  
    try: 
        states_list = [] 
        for time in times:
            current_state = _rk_evolve_dynamic_system_rk_evolve_helper(diff_function, starting_state, starting_time, time) 
            states_list.append(current_state) 
            starting_state = current_state 
            starting_time = time 
        return states_list 
    except TypeError:
        return _rk_evolve_dynamic_system_rk_evolve_helper(diff_function, initial_state, 0, times) 
    


    


"""A helper function that generates the diff function for RK45""" 
def _rk_evolve_dynamic_system_diff_function_factory(number_levels, level_gammas, level_couplings_function):
    hamiltonian_function = _dynamic_hamiltonian_function_factory(number_levels, level_gammas, level_couplings_function)
    def diff_function(t, y):
        diff_matrix = -1j * hamiltonian_function(t)
        diff_array = np.matmul(diff_matrix, y)
        return diff_array 
    return diff_function 


"""A helper function that generates a function which returns the hamiltonian as a function of t, given the couplings and gammas etc."""
def _dynamic_hamiltonian_function_factory(number_levels, level_gammas, level_couplings_function):
    def hamiltonian(t):
        system_hamiltonian = np.zeros((number_levels, number_levels), dtype = complex)
        instantaneous_couplings = level_couplings_function(t)
        system_hamiltonian += instantaneous_couplings
        for i in range(number_levels):
            system_hamiltonian[i][i] += -1j * 1.0/2.0 * level_gammas[i]
        return system_hamiltonian
    return hamiltonian


"""A helper function that evolves an rk solver from initial to final state."""
def _rk_evolve_dynamic_system_rk_evolve_helper(diff_function, initial_state, initial_time, final_time):
    #Default tolerances are usually ok
    rk_solver = RK45(diff_function, initial_time, initial_state, final_time)
    while(rk_solver.status == "running"):
        rk_solver.step() 
    return rk_solver.y


"""Evolves a dynamic hamiltonian system using the naive matrix exponential technique

Uses the classic evolution operator exp(int(-i H)) to evolve an initial state to a final state under an arbitrary hamiltonian. 
Mathematically exactly the same as RK evolve; can be considerably faster or slower depending on exact function being integrated.

Parameters:

    number_levels: The number N of levels in the system
    level_gammas: A length N array of values indicating the linewidth for each level 
    level_couplings_function: A function of scalar time t which returns an NxN numpy array containing the coupling 
        strengths at any time 
    coupling_phases_function: A function of scalar time t which returns the (real) phases of each coupling as a function of time t
    initial_state: A length N (complex) numpy array containing the initial state of the system at t = 0. 
    times: Either a float or a length M array of floats stipulating the time(s) at which the final state should be evaluated. 
        If an array, should have times in ascending order for maximum efficiency. 
Returns:
    If times is a single float, a single length N np array specifying the final state of the system at the given time. 
    Otherwise, a list of M arrays of length N specifying the state of the system at the times specified by times."""

def exponential_evolve_dynamic_system(number_levels, level_gammas, level_couplings_function, level_phases_function, initial_state, times):
    hamiltonian_function = _dynamic_hamiltonian_function_factory(number_levels, level_gammas, level_couplings_function, level_phases_function)
    def evolution_function(t):
        return -1j * hamiltonian_function(t)
    starting_time = 0
    starting_state = initial_state 
    try: 
        states_list = [] 
        for time in times:
            current_state = _exponential_evolve_helper(evolution_function, starting_state, starting_time, time) 
            states_list.append(current_state) 
            starting_state = current_state 
            starting_time = time 
        return states_list
    except TypeError:
        return _exponential_evolve_helper(evolution_function, initial_state, 0, times) 


def _exponential_evolve_helper(evolution_function, initial_state, initial_time, final_time):
    integrated_matrix = quad_vec(evolution_function, initial_time, final_time)[0]
    evolution_matrix = scipy.linalg.expm(integrated_matrix)
    final_state = np.matmul(evolution_matrix, initial_state) 
    return final_state


"""Evolves a suitable constant system where the beams couple manifolds of states

Solves the dynamics of a system where beams of constant coupling and (reference) 
detuning couple not individual states, but manifolds. Manifolds are collections 
of states coupled by a single beam which have defined energy relationships between 
them which establish relationships between the detunings of effective couplings:
a higher-up state within a manifold will have a detuning which is more positive 
than a lower one.

Solves the system by an analogous technique to fast_evolve_constant_system, but 
uses the manifold relations to work where that method would be over-determined.

Parameters:

    number_manifolds: the number of manifolds N in the system
    manifold_couplings_array: the "base couplings" which exist between manifolds
    manifold_detunings_array: the "base detunings" of the beams between manifolds
    manifold_positions_list: A 2D iterable [[manifold1level1, manifold1level2, ...], [manifold2level1, manifold2level2, ...], ...]
        which contains the positions of levels within manifolds
    manifold_gammas_list: A 2D iterable with the same syntax as manifold_positions_list which specifies 
        the linewidth of each level within each manifold
    manifold_coupling_proportionality_list: A 4D iterable which specifies proportionality constants for couplings between levels:
        the actual coupling between two levels is the "base coupling" which applies to their manifolds times the relevant constant. 
        Index convention: i: first manifold, j: level in first manifold, k: second manifold, l: level in second manifold. Symmetric 
        under i <-> k, j <-> l
    initial_state: A 2D iterable with complex entries [[manifold1level1coefficient, manifold1level2coefficient, ...], [manifold2level1coefficient, ...],...]
    times: The amount of time to evolve the system. Can be either a single float or an iterable of floats.


Returns:

    If times is a single float, a single list of lists in the format of initial_state, containing the final coefficients for each level
    Otherwise, a list of length equal to times, each element of which is a list of lists as above appropriate to the state at the given time.

Remarks:

    Implementation with numpy array input is made infeasible by the fact that numpy doesn't do ragged arrays.
"""

def fast_evolve_constant_manifold_system(number_manifolds, manifold_couplings_array, manifold_detunings_array, manifold_positions_list, 
                                            manifold_gammas_list, manifold_coupling_proportionality_list, initial_state, times):
    full_transformed_hamiltonian, full_basis_change_detunings_vector = _give_constant_manifold_system_transformed_hamiltonian(number_manifolds,
                                                                                                                            manifold_couplings_array, 
                                                                                                                            manifold_detunings_array,
                                                                                                                            manifold_positions_list,
                                                                                                                            manifold_gammas_list, 
                                                                                                                            manifold_coupling_proportionality_list)
    basis_transform_matrix_function = _fast_constant_hamiltonian_basis_change_function_factory(full_basis_change_detunings_vector)
    #Flatten the initial state 
    total_number_levels = len(full_transformed_hamiltonian) 
    flattened_initial_state = np.zeros(total_number_levels, dtype = complex) 
    initial_state_index = 0
    for manifold_state_list in initial_state:
        for state_coefficient in manifold_state_list:
            flattened_initial_state[initial_state_index] = state_coefficient 
            initial_state_index += 1
    eigenvalues, eigenvector_list = _diagonalize_hamiltonian(full_transformed_hamiltonian) 
    eigenvector_coefficients_list = _get_eigvec_coefficients(flattened_initial_state, eigenvector_list)
    try:
        final_states_list = [] 
        for time in times:
            flattened_final_state = _fast_evolve_constant_system_helper(total_number_levels, eigenvector_list, eigenvalues, 
                                                                        eigenvector_coefficients_list, basis_transform_matrix_function, time)
            final_state = []
            flattened_final_state_index = 0
            for i, manifold_state_list in zip(initial_state, range(len(initial_state))):
                final_state.append([])
                for j in range(len(manifold_state_list)):
                    final_state[i].append(flattened_final_state[flattened_final_state_index])
                    flattened_final_state_index += 1
            final_states_list.append(final_state) 
        return final_states_list
    except TypeError:
        time = times 
        flattened_final_state = _fast_evolve_constant_system_helper(total_number_levels, eigenvector_list, eigenvalues, 
                                                                    eigenvector_coefficients_list,  basis_transform_matrix_function, time)
        final_state = []
        flattened_final_state_index = 0
        for manifold_state_list, i in zip(initial_state, range(len(initial_state))):
            final_state.append([])
            for j in range(len(manifold_state_list)):
                final_state[i].append(flattened_final_state[flattened_final_state_index])
                flattened_final_state_index += 1 
        return final_state

def _give_constant_manifold_system_transformed_hamiltonian(number_manifolds, manifold_couplings_array, manifold_detunings_array, manifold_positions_list,
                                                    manifold_gammas_list, manifold_coupling_proportionality_list):
    foo, manifold_basis_change_detunings_vector = _give_constant_system_transformed_hamiltonian(number_manifolds, np.zeros(number_manifolds), 
                                                                                                manifold_couplings_array, manifold_detunings_array)
    total_number_levels = 0
    for manifold_positions, manifold_detuning in zip(manifold_positions_list, manifold_basis_change_detunings_vector):
        number_manifold_levels = len(manifold_positions)
        total_number_levels += number_manifold_levels
    full_hamiltonian = np.zeros((total_number_levels, total_number_levels), dtype = complex)
    initial_index = 0
    full_basis_change_detunings_vector = np.zeros(total_number_levels)
    for initial_manifold_row, initial_manifold_proportionalities, initial_manifold_positions, initial_manifold_gammas, initial_manifold_basis_change_detuning in zip(manifold_couplings_array, 
                                                                                                                                                                    manifold_coupling_proportionality_list,
                                                                                                                                                                    manifold_positions_list,
                                                                                                                                                                    manifold_gammas_list,
                                                                                                                                                                    manifold_basis_change_detunings_vector):
        for initial_level_proportionalities, initial_level_position, initial_level_gamma in zip(initial_manifold_proportionalities, initial_manifold_positions, 
                                                                                        initial_manifold_gammas):
            total_level_detuning = initial_manifold_basis_change_detuning - initial_level_position
            full_basis_change_detunings_vector[initial_index] = total_level_detuning
            full_hamiltonian[initial_index][initial_index] -= total_level_detuning
            full_hamiltonian[initial_index][initial_index] += -1j * 1.0/2.0 * initial_level_gamma 
            final_index = 0
            for final_manifold_coupling_cell, final_manifold_proportionalities in zip(initial_manifold_row, initial_level_proportionalities):
                for final_level_proportionality in final_manifold_proportionalities:
                    effective_coupling = final_manifold_coupling_cell * final_level_proportionality 
                    full_hamiltonian[initial_index][final_index] += 1.0/2.0 * effective_coupling 
                    final_index += 1
            initial_index += 1
    return (full_hamiltonian, full_basis_change_detunings_vector)





    