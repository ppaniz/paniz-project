def insertion_sort(A):
    n = len(A)
    for i in range(1, n):
        key = A[i]
        j = i - 1
        while j >= 0 and A[j] > key:
            A[j+1] = A[j]
            j -= 1
        A[j+1] = key
    return A

nums = input("write the number(with space) : ")
arr = list(map(int, nums.split()))
print("before sorting : ", arr)
print("after sorting : ", insertion_sort(arr))