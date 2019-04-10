using LinearAlgebra
using StatsModels
using Random
using Distributions
# zad1
A = [1 2 -1 0; 3 -2 4 5; 2 6 5 -3; 0 1 5 -4];
B = [3 6; 4 0; 2 -1; 1 1];

det(A)
A*B
B'
inv(A)
A^3

# zad2
A = [3 1 -2; -3 2 1; -2 6 3];
B = [1 2 -2; 0 2 -1];
C = [1 2; 3 4];

det(A)
det(C)
A*B'
C'
inv(C)
inv(A)*B'*C^2

#zad3
A=[2 3; 1 4];
Imat = Matrix(I,2,2);
A^2 + 6A + 4Imat

#zad8
X = [1 2 3 4 5 6 7 8 9 10]';
Y = [10 15 13 22 23 20 18 25 27 22]';
cor(X,Y)
X = hcat(fill(1, 10),X)
inv(X'X)*X'Y

## linear regression
Random.seed!(1234);
n  = 1000^2;
x = rand(Normal(),n);
m = reshape(x, 1000, 1000);

@time b = m * m;
