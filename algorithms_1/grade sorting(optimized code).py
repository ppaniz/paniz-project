import bisect

def insertion_sort_binary(A):
    A = A.copy()
    for i in range(1, len(A)):
        key = A[i]
        pos = bisect.bisect_left(A, key, 0, i)
        A = A[:pos] + [key] + A[pos:i] + A[i+1:]
    return A

nums = input("write the number(with space) : ")
arr = list(map(int, nums.split()))
print("before sorting : ", arr)
print("after sorting : ", insertion_sort_binary(arr))
