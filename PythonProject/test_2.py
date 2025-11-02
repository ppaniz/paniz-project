n = int(input("یه عدد وارد کن: "))

if n < 2:
    print("عدد اول نیست")
else:
    is_prime = True
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            is_prime = False
            break
    if is_prime:
        print("عدد اول هست")
    else:
        print("عدد اول نیست")
