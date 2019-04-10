using LinearAlgebra
using SparseArrays
using DataFrames
using StatsModels
using BenchmarkTools

# dane z brakami
dane_oryg = [
1 1 1 missing 1
2 2 2 1 2
3 2 2 2 missing
4 1 2 1 3
5 1 1 missing 3
6 1 1 2 3
7 2 1 2 3
8 2 2 1 3
9 2 1 1 2
10 1 1 1 2
11 2 2 2 2
12 1 2 1 3
13 2 1 1 1
14 1 2 1 missing
15 1 1 2 2
16 2 2 1 2
17 2 1 2 3
18 1 2 2 1
19 1 1 2 missing
20 2 1 1 2]

dane = repeat(dane_oryg, 10000);

# zmieniamy na df -- czy konieczne? categoricalArrays?
dane_df = convert(DataFrame, dane)
categorical!(dane_df, :2)
categorical!(dane_df, :3)
categorical!(dane_df, :4)
categorical!(dane_df, :5)

# wyrzucamy braki danych
dane_df_nm = dropmissing(dane_df)

## standard (timing)
full = ModelMatrix(ModelFrame(@formula(x1 ~ 1 + x2 + x3), dane_df));
X̂ = sum(full.m, dims = 1);

function calib()
    full_nm = ModelMatrix(ModelFrame(@formula(x1 ~ 1 + x2 + x3), dane_df_nm));
    d = fill(1, size(full_nm.m,1));
    X = full_nm.m;
    D = Diagonal(d);
    w = d + D*X*inv(X'D*X)*(X̂ - d'X)';
    w
end

function calibSparse()
    full_nm = ModelMatrix(ModelFrame(@formula(x1 ~ 1 + x2 + x3), dane_df_nm));
    d = fill(1, size(full_nm.m,1));
    X = sparse(full_nm.m);
    D = Diagonal(d);
    w = d + D*X*inv(Matrix(X'D*X))*(X̂ - d'X)';
    w
end

BLAS.set_num_threads(2);
@btime calib();
@btime calibSparse();

BLAS.set_num_threads(1);
@btime calib();
@btime calibSparse();


@time full_nm = ModelMatrix(ModelFrame(@formula(x1 ~ 1 + x2 + x3), dane_df_nm));
@time d = fill(1, size(full_nm.m,1));
@time X = full_nm.m;
@time D = Diagonal(d);
@time w = d + D*X*inv(X'D*X)*(X̂ - d'X)';
