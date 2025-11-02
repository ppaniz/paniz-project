def merge(L, R):
    i = 0
    j = 0
    A = []
    comparisons = 0

    while i < len(L) and j < len(R):
        comparisons += 1
        if L[i] <= R[j]:
            A.append(L[i])
            i += 1
        else:
            A.append(R[j])
            j += 1
    while i < len(L):
        A.append(L[i])
        i += 1

    while j < len(R):
        A.append(R[j])
        j += 1

    return A, comparisons

L = [2, 5, 8, 12]
R = [1, 3, 7, 10, 15]

A, count = merge(L, R)
print("last array : ", A)
print("number of merges : ", count)
