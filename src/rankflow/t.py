def func1(a, b, **kwargs):
    return a + b


def func2(c, d, **kwargs):
    return c * d


config = {"a": 1, "b": 2, "c": 3, "d": 4}
result = func1(a=100, **config)  # Output: 3
print(result)
result = func2(**config)  # Output: 12
print(result)
