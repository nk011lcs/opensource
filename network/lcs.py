subword_matching_t = 0.5
exp_matching_t = 0.75

def func_match(func1, func2):
    if len(func1) == 0 and len(func2) == 0:
        return True
    elif len(func1) == 0 or len(func2) == 0:
        return False
    return len(func1 & func2) / min(len(func1), len(func2)) > subword_matching_t


def expression_match(exp1, exp2):
    if len(exp1) == 0 and len(exp2) == 0:
        return True
    elif len(exp1) == 0 or len(exp2) == 0:
        return False
    return len(exp1 & exp2) / min(len(exp1), len(exp2)) > exp_matching_t


def match(e1, e2):
    if e1[0] == e2[0]:
        if e1[0] == 'ExpressionStatement':
            func1 = set(e1[2])
            func2 = set(e2[2])
            if func_match(func1, func2):
                return True
        elif e1[0] == 'SwitchStatement':
            func1 = set(e1[2])
            func2 = set(e2[2])
            if func_match(func1, func2):
                return True
        elif e1[0] == 'WhileStatement':
            return True
        elif e1[0] == 'IfStatement':
            if expression_match(set(e1[1]), set(e2[1])) and func_match(set(e1[2]), set(e2[2])):
                return True
        elif e1[0] == 'ReturnStatement':
            if expression_match(set(e1[1]), set(e2[1])) and func_match(set(e1[2]), set(e2[2])):
                return True
        elif e1[0] == 'ForStatement':
            if expression_match(set(e1[1]), set(e2[1])) and expression_match(set(e1[1]), set(e2[1])) and func_match(
                # if expression_match(set(e1[1][0]), set(e2[1][0])) and expression_match(e1[1][1], e2[1][1]) and func_match(
                    set(e1[2]),
                    set(e2[2])):
                return True

    elif (e1[0] == 'SwitchStatement' and e2[0] == 'IfStatement') or (
            e2[0] == 'SwitchStatement' and e1[0] == 'IfStatement'):
        if e2[0] == 'IfStatement' and expression_match(set(e2[1]), set('==')) and func_match(set(e1[2]), set(e2[2])):
            return True
        elif e1[0] == 'IfStatement' and expression_match(set(e1[1]), set('==')) and func_match(set(e1[2]), set(e2[2])):
            return True
    return False


# get the matrix of LCS lengths at each sub-step of the recursive process
# (m+1 by n+1, where m=len(list1) & n=len(list2) ... it's one larger in each direction
# so we don't have to special-case the x-1 cases at the first elements of the iteration
def lcs_mat(list1, list2):
    m = len(list1)
    n = len(list2)
    # construct the matrix, of all zeroes
    mat = [[0] * (n + 1) for row in range(m + 1)]
    # populate the matrix, iteratively
    for row in range(1, m + 1):
        for col in range(1, n + 1):
            if match(list1[row - 1], list2[col - 1]):
                # if it's the same element, it's one longer than the LCS of the truncated lists
                mat[row][col] = mat[row - 1][col - 1] + 1
            else:
                # they're not the same, so it's the the maximum of the lengths of the LCSs of the two options (different list truncated in each case)
                mat[row][col] = max(mat[row][col - 1], mat[row - 1][col])
    # the matrix is complete
    return mat


# backtracks all the LCSs through a provided matrix
def all_lcs(lcs_dict, mat, list1, list2, index1, index2):
    # if we've calculated it already, just return that
    if ((index1, index2) in lcs_dict):
        return lcs_dict[(index1, index2)]
    # otherwise, calculate it recursively
    if (index1 == 0) or (index2 == 0):  # base case
        return [[]]
    elif match(list1[index1 - 1], list2[index2 - 1]):
        # elements are equal! Add it to all LCSs that pass through these indices
        lcs_dict[(index1, index2)] = [prevs + [list1[index1 - 1]] for prevs in
                                      all_lcs(lcs_dict, mat, list1, list2, index1 - 1, index2 - 1)]
        return lcs_dict[(index1, index2)]
    else:
        lcs_list = []  # set of sets of LCSs from here
        # not the same, so follow longer path recursively
        if mat[index1][index2 - 1] >= mat[index1 - 1][index2]:
            before = all_lcs(lcs_dict, mat, list1, list2, index1, index2 - 1)
            for series in before:  # iterate through all those before
                if series not in lcs_list:
                    lcs_list.append(
                        series)  # and if it's not already been found, append to lcs_list
        if mat[index1 - 1][index2] >= mat[index1][index2 - 1]:
            before = all_lcs(lcs_dict, mat, list1, list2, index1 - 1, index2)
            for series in before:
                if series not in lcs_list:
                    lcs_list.append(series)
        lcs_dict[(index1, index2)] = lcs_list
        return lcs_list


# return a set of the sets of longest common subsequences in list1 and list2
def lcs(list1, list2):
    # mapping of indices to list of LCSs, so we can cut down recursive calls enormously
    mapping = dict()
    # start the process...
    m = lcs_mat(list1, list2)
    return all_lcs(mapping, m, list1, list2, len(list1), len(list2))[0]
