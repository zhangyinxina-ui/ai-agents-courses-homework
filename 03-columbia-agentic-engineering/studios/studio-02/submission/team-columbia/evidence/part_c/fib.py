#!/usr/bin/env python3
"""Print the first 20 Fibonacci numbers iteratively starting from 0."""


def main() -> None:
    a, b = 0, 1
    for _ in range(20):
        print(a)
        a, b = b, a + b


if __name__ == "__main__":
    main()
